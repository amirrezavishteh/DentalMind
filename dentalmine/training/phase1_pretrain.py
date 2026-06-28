"""Phase 1 — self-supervised pretraining of the shared DentVFM-2D backbone.

PROMPT.md specifies DINOv2 SSL on ALL dental images. A faithful DINOv2 run needs
the heavy facebookresearch/dinov2 stack (EMA teacher, multi-crop, iBOT); to keep
this self-contained and runnable today this uses SimCLR-style NT-Xent contrastive
SSL over two augmented views — same goal (a label-free shared image encoder),
fewer moving parts. Swap in the dinov2 loss later without changing the I/O.

Runnable now on your 2D data. For a fast check use:
  python -m training.phase1_pretrain --data-dir ../data/2D --fast-dev-run \
      --model vit_base_patch16_224 --img-size 224
Spec target (needs a GPU): --model vit_large_patch14_dinov2 --img-size 518
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.backbone.dentvfm import DentVFMEncoder
from training.trainer_base import (
    cosine_lr, list_2d_images, resolve_device, save_checkpoint, set_lr,
)


class TwoViewDataset(Dataset):
    def __init__(self, paths, img_size: int):
        self.paths = paths
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.aug(img), self.aug(img)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def nt_xent(z1, z2, temperature: float = 0.2):
    """SimCLR NT-Xent contrastive loss over a batch of paired views."""
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                     # [2B, D]
    sim = z @ z.t() / temperature                      # [2B, 2B]
    sim.fill_diagonal_(float("-inf"))
    targets = torch.arange(b, device=z.device)
    targets = torch.cat([targets + b, targets], dim=0)  # positive = the other view
    return F.cross_entropy(sim, targets)


def train(args):
    device = resolve_device(args.device)
    data_root = Path(args.data_dir)
    paths = list_2d_images(data_root)
    if not paths:
        raise FileNotFoundError(
            f"No 2D images found under {data_root}. Expected tufts_yolo/, tufts/, "
            f"caries_zenodo/, mandible_mendeley/ or dentex/ subfolders."
        )
    print(f"Phase 1 SSL: {len(paths)} images, model={args.model}, device={device}", flush=True)

    ds = TwoViewDataset(paths, args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)

    backbone = DentVFMEncoder(model_name=args.model, pretrained=args.pretrained,
                              freeze=False, device=device, img_size=args.img_size)
    backbone.train()
    head = ProjectionHead(backbone.embed_dim).to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)

    n_batches = len(loader)
    total_steps = args.epochs * n_batches
    warmup = min(args.warmup_steps, max(1, total_steps // 10))
    step = 0
    print(f"Starting: {args.epochs} epochs x {n_batches} batches", flush=True)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        running = 0.0
        for bi, (v1, v2) in enumerate(loader):
            lr = cosine_lr(step, total_steps, args.lr, warmup)
            set_lr(opt, lr)
            v1, v2 = v1.to(device), v2.to(device)
            z1 = head(backbone.encode_cls(v1))
            z2 = head(backbone.encode_cls(v2))
            loss = nt_xent(z1, z2, args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); step += 1
            if bi == 0 or (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
                print(f"  epoch {epoch+1}/{args.epochs} batch {bi+1}/{n_batches} "
                      f"loss={loss.item():.4f} lr={lr:.2e} "
                      f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
            if args.fast_dev_run:
                break
        print(f"epoch {epoch+1}/{args.epochs} avg_loss={running/max(1,(bi+1)):.4f}", flush=True)
        if args.fast_dev_run:
            break

    save_checkpoint(args.output, model_state_dict=backbone.model.state_dict(),
                    model_name=args.model, img_size=args.img_size, phase="phase1_ssl")
    print(f"Saved DentVFM backbone -> {args.output}", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data/2D")
    ap.add_argument("--output", default="../weights/dentvfm_phase1.pt")
    ap.add_argument("--model", default="vit_large_patch14_dinov2")
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--pretrained", action="store_true",
                    help="Continue from ImageNet/DINOv2 weights instead of from scratch.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fast-dev-run", action="store_true")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
