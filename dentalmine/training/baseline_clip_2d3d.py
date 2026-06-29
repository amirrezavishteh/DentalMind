"""UNIFIED BASELINE — one shared Med-CLIP backbone trained across 2D and 3D.

This is the project's central training entry point. A SINGLE Med-CLIP image
encoder (MMKD-CLIP -> BiomedCLIP) is the shared backbone for every task:

  2D branch (always, if 2D data present)
      Fine-tune the CLIP image tower on 2D crops (FDI teeth + caries/healthy)
      against frozen text-prototype embeddings — adapts CLIP to dental imagery.
      [reuses the leak-free, held-out-validated loop in clip_finetune.py]

  3D branch (runs when CBCT volumes exist; synthetic smoke otherwise)
      The SAME (now dentally-adapted) CLIP encoder embeds each CBCT slice, and a
      C1 cross-slice attention module fuses a window of neighbouring slices into
      the central slice — giving the 2D CLIP backbone 3D awareness without a
      separate 3D backbone. A small head is supervised per-slice.

So "the CLIP backbone is shared between 2D and 3D" is literal: identical weights
encode both; 3D only adds the lightweight C1 fusion on top. The 3D branch is
data-gated and never fabricates training when CBCT data is absent.

Run:
  # 2D + (3D if CBCT present). Quick code-path check:
  python -m training.baseline_clip_2d3d --data-root ../data/2D --fast-dev-run
  # Real run on the A100:
  python -m training.baseline_clip_2d3d --data-root ../data/2D --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.madclip.c1_slice_attention import CrossSliceAttentionAdapter
from training.clip_finetune import finetune_2d, load_shared_encoder
from training.trainer_base import resolve_device, save_checkpoint


class SliceHead(nn.Module):
    def __init__(self, dim: int, n_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, n_classes))

    def forward(self, x):                      # x: [B, 1, 1, D] -> [B, n_classes]
        return self.head(x.flatten(1))


def _find_volumes(cbct_dir: Path):
    if not cbct_dir.exists():
        return []
    return sorted(list(cbct_dir.rglob("*.nii.gz")) + list(cbct_dir.rglob("*.nii")))


def _synthetic_3d_smoke(encoder, args, device):
    """Validate the CLIP-slice -> C1 -> head -> loss path without CBCT data."""
    print("3D branch --fast-dev-run: SYNTHETIC (no CBCT data needed)", flush=True)
    dim = encoder.image_embed_dim
    k = args.k
    c1 = CrossSliceAttentionAdapter(embed_dim=dim, num_heads=8).to(device)
    head = SliceHead(dim).to(device)
    opt = torch.optim.AdamW(list(c1.parameters()) + list(head.parameters()), lr=args.lr)
    # each slice embedding reshaped to a 1x1 spatial map so C1 attends over slices
    window = [torch.randn(2, 1, 1, dim, device=device) for _ in range(2 * k + 1)]
    target = torch.randint(0, 2, (2, 1), device=device).float()
    for it in range(3):
        enriched = c1(window, target_idx=k)
        loss = F.binary_cross_entropy_with_logits(head(enriched), target)
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"  3D smoke iter {it+1}/3 loss={loss.item():.4f}", flush=True)
    return c1, head


def _train_3d(encoder, args, device):
    cbct_dir = Path(args.cbct_dir)
    volumes = _find_volumes(cbct_dir)
    if not volumes:
        if args.fast_dev_run:
            c1, head = _synthetic_3d_smoke(encoder, args, device)
            save_checkpoint(args.output_3d, c1_state_dict=c1.state_dict(),
                            head_state_dict=head.state_dict(), phase="baseline_3d_smoke")
            print(f"Saved 3D C1 (smoke) -> {args.output_3d}", flush=True)
            return
        print(
            f"3D branch SKIPPED: no CBCT volumes under {cbct_dir} (data/3D empty).\n"
            f"  The shared CLIP backbone is fully trained on 2D above. The moment you\n"
            f"  add CBCT (*.nii.gz) + --labels-dir, this branch trains C1 on the SAME\n"
            f"  encoder. Validate the 3D code path now: add --fast-dev-run.", flush=True)
        return

    import numpy as np
    import SimpleITK as sitk
    from PIL import Image

    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    if labels_dir is None or not labels_dir.exists():
        print("3D branch SKIPPED: --labels-dir (per-volume masks) required for supervision.",
              flush=True)
        return

    dim, k = encoder.image_embed_dim, args.k
    c1 = CrossSliceAttentionAdapter(embed_dim=dim, num_heads=8).to(device)
    head = SliceHead(dim).to(device)
    opt = torch.optim.AdamW(list(c1.parameters()) + list(head.parameters()), lr=args.lr)
    print(f"3D branch: {len(volumes)} CBCT volumes on shared {encoder.name} encoder", flush=True)

    def clip_embed_slice(slice2d: np.ndarray) -> torch.Tensor:
        mn, mx = float(slice2d.min()), float(slice2d.max())
        norm = (slice2d - mn) / (mx - mn + 1e-6) * 255.0
        pil = Image.fromarray(norm.astype("uint8")).convert("RGB")
        with torch.no_grad():
            return encoder.encode_image(pil).reshape(1, 1, 1, dim)  # [1,1,1,D]

    for epoch in range(args.epochs_3d):
        t0 = time.perf_counter()
        for vpath in volumes:
            lpath = labels_dir / vpath.name
            if not lpath.exists():
                continue
            vol = sitk.GetArrayFromImage(sitk.ReadImage(str(vpath))).astype("float32")
            lab = sitk.GetArrayFromImage(sitk.ReadImage(str(lpath)))
            z = vol.shape[0]
            for ci in range(k, z - k):
                window = [clip_embed_slice(vol[ci + off]) for off in range(-k, k + 1)]
                enriched = c1(window, target_idx=k)
                target = torch.tensor([[float((lab[ci] > 0).any())]], device=device)
                loss = F.binary_cross_entropy_with_logits(head(enriched), target)
                opt.zero_grad(); loss.backward(); opt.step()
            if args.fast_dev_run:
                break
        print(f"3D epoch {epoch+1}/{args.epochs_3d} done in {time.perf_counter()-t0:.1f}s", flush=True)
        if args.fast_dev_run:
            break

    save_checkpoint(args.output_3d, c1_state_dict=c1.state_dict(),
                    head_state_dict=head.state_dict(), phase="baseline_3d")
    print(f"Saved 3D C1 adapter -> {args.output_3d}", flush=True)


def train(args):
    device = resolve_device(args.device)
    encoder = load_shared_encoder(args, device)

    print("\n=== 2D BRANCH (shared CLIP backbone) ===", flush=True)
    finetune_2d(encoder, args, device)

    print("\n=== 3D BRANCH (same CLIP backbone + C1 cross-slice) ===", flush=True)
    _train_3d(encoder, args, device)

    print("\nBaseline complete: shared CLIP backbone trained on 2D"
          + (" + 3D" if _find_volumes(Path(args.cbct_dir)) or args.fast_dev_run else " (3D pending data)"),
          flush=True)


def build_argparser():
    ap = argparse.ArgumentParser()
    # 2D branch (consumed by clip_finetune.finetune_2d)
    ap.add_argument("--data-root", default="../data/2D")
    ap.add_argument("--output", default="../weights/baseline_clip_2d.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=None)
    # 3D branch
    ap.add_argument("--cbct-dir", default="../data/3D/cbct")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--output-3d", default="../weights/baseline_clip_c1_3d.pt")
    ap.add_argument("--epochs-3d", type=int, default=50)
    ap.add_argument("--k", type=int, default=2)
    # shared
    ap.add_argument("--device", default=None)
    ap.add_argument("--fast-dev-run", action="store_true")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
