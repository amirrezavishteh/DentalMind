"""Phase 2 — train the C1 cross-slice attention adapter (DentVFM frozen).

Needs CBCT volumes (sequence data) + per-slice supervision. No CBCT dataset is
present yet (data/3D is empty), so on real data this exits with an actionable
message rather than faking training. ``--fast-dev-run`` exercises the full
C1 + head + loss + backward path on SYNTHETIC tensors so the code is verifiable
today; it does not learn anything real.

Expected real-data layout (when you have CBCT):
  <cbct-dir>/*.nii.gz                      CBCT volumes
  <labels-dir>/<same-stem>.nii.gz         per-voxel lesion/structure masks
                                          (e.g. from nnU-Net / DentalSegmentator)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from models.madclip.c1_slice_attention import CrossSliceAttentionAdapter
from training.trainer_base import resolve_device, save_checkpoint


class SliceDetectionHead(nn.Module):
    """Tiny per-slice detector on C1-enriched features -> [B, n_classes] logits."""

    def __init__(self, embed_dim: int, n_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, n_classes),
        )

    def forward(self, feat_hw_d: torch.Tensor) -> torch.Tensor:
        # [B, H, W, D] -> global-avg-pool -> [B, n_classes]
        pooled = feat_hw_d.mean(dim=(1, 2))
        return self.head(pooled)


def _find_volumes(cbct_dir: Path):
    if not cbct_dir.exists():
        return []
    return sorted(list(cbct_dir.rglob("*.nii.gz")) + list(cbct_dir.rglob("*.nii")))


def _synthetic_smoke(args, device):
    """Validate the C1 + head + loss + backward path without any data."""
    print("Phase 2 --fast-dev-run: SYNTHETIC smoke (no CBCT data needed)", flush=True)
    embed_dim, h, w, k = args.embed_dim, 16, 16, 2
    c1 = CrossSliceAttentionAdapter(embed_dim=embed_dim).to(device)
    head = SliceDetectionHead(embed_dim).to(device)
    opt = torch.optim.AdamW(list(c1.parameters()) + list(head.parameters()), lr=args.lr)

    feats = [torch.randn(2, h, w, embed_dim, device=device) for _ in range(2 * k + 1)]
    target = torch.randint(0, 2, (2, 1), device=device).float()
    for it in range(3):
        enriched = c1(feats, target_idx=k)
        logits = head(enriched)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"  smoke iter {it+1}/3 loss={loss.item():.4f}", flush=True)
    save_checkpoint(args.output, c1_state_dict=c1.state_dict(), phase="phase2_smoke")
    print(f"Saved C1 adapter (smoke) -> {args.output}", flush=True)


def train(args):
    device = resolve_device(args.device)
    cbct_dir = Path(args.cbct_dir)
    volumes = _find_volumes(cbct_dir)

    if not volumes:
        if args.fast_dev_run:
            return _synthetic_smoke(args, device)
        print(
            f"Phase 2 BLOCKED: no CBCT volumes found under {cbct_dir}.\n"
            f"  This phase trains the C1 cross-slice adapter on 3D sequence data,\n"
            f"  which has not been downloaded yet (data/3D is empty).\n"
            f"  Provide CBCT volumes (*.nii.gz) + per-slice labels, then rerun.\n"
            f"  To validate the training code path now without data:\n"
            f"    python -m training.phase2_c1_adapter --fast-dev-run", flush=True)
        return

    # ---- real-data path (runs once CBCT + labels exist) ----------------
    import numpy as np
    import SimpleITK as sitk

    from models.backbone.dentvfm import DentVFMEncoder

    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    if labels_dir is None or not labels_dir.exists():
        print(f"Phase 2 BLOCKED: --labels-dir with per-volume masks is required for "
              f"supervision (e.g. nnU-Net/DentalSegmentator output).", flush=True)
        return

    print(f"Phase 2: {len(volumes)} CBCT volumes, device={device}", flush=True)
    backbone = DentVFMEncoder(model_name=args.model, pretrained=True, freeze=True,
                              device=device, img_size=args.img_size)
    c1 = CrossSliceAttentionAdapter(embed_dim=backbone.embed_dim).to(device)
    head = SliceDetectionHead(backbone.embed_dim).to(device)
    opt = torch.optim.AdamW(list(c1.parameters()) + list(head.parameters()), lr=args.lr)
    k = args.k

    def load_slices(path):
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype("float32")
        vol = (vol - vol.mean()) / (vol.std() + 1e-6)
        return vol  # [Z, H, W]

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        for vi, vpath in enumerate(volumes):
            lpath = labels_dir / vpath.name
            if not lpath.exists():
                continue
            vol = load_slices(vpath)
            lab = sitk.GetArrayFromImage(sitk.ReadImage(str(lpath)))
            z = vol.shape[0]
            for ci in range(k, z - k):
                window = []
                for off in range(-k, k + 1):
                    sl = torch.from_numpy(vol[ci + off]).to(device)
                    sl = sl.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
                    sl = torch.nn.functional.interpolate(
                        sl, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
                    with torch.no_grad():
                        window.append(backbone.encode_spatial(sl))
                enriched = c1(window, target_idx=k)
                target = torch.tensor([[float((lab[ci] > 0).any())]], device=device)
                loss = nn.functional.binary_cross_entropy_with_logits(head(enriched), target)
                opt.zero_grad(); loss.backward(); opt.step()
            if args.fast_dev_run:
                break
        print(f"epoch {epoch+1}/{args.epochs} done in {time.perf_counter()-t0:.1f}s", flush=True)
        if args.fast_dev_run:
            break

    save_checkpoint(args.output, c1_state_dict=c1.state_dict(), phase="phase2")
    print(f"Saved C1 adapter -> {args.output}", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbct-dir", default="../data/3D/cbct")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--backbone-ckpt", default=None)
    ap.add_argument("--output", default="../weights/c1_adapter.pt")
    ap.add_argument("--model", default="vit_large_patch14_dinov2")
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fast-dev-run", action="store_true")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
