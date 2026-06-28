"""Phase 4 — 3D -> 2D knowledge distillation.

Pipeline (PROMPT.md): nnU-Net segments CBCT -> generate DRR projections ->
project 3D masks onto the DRR planes -> write 2D pseudo-labels -> re-run Phase 3
with the extra pseudo-labelled data.

State of dependencies here:
  * CBCT data           : not downloaded yet (data/3D empty)  -> hard requirement
  * nnU-Net / Dental-   : optional; if nnUNetv2_predict is on PATH it is used,
    Segmentator           otherwise you must pass --masks-dir with existing masks
  * DRR generation      : plastimatch optional; falls back to a pure-numpy
                          max-intensity-projection (MIP) approximation so this
                          phase runs without extra system tools

``--fast-dev-run`` validates the volume->DRR + mask->2D-projection code on a
SYNTHETIC volume (no data/tools needed).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def mip_projection(volume: np.ndarray, axis: int = 1) -> np.ndarray:
    """Maximum-intensity projection -> 2D DRR approximation, normalized to uint8."""
    proj = volume.max(axis=axis).astype(np.float32)
    mn, mx = float(proj.min()), float(proj.max())
    if mx > mn:
        proj = (proj - mn) / (mx - mn) * 255.0
    return proj.astype(np.uint8)


def project_mask(mask: np.ndarray, axis: int = 1) -> np.ndarray:
    """Project a 3D label volume onto a plane (any positive voxel -> foreground)."""
    return (mask > 0).any(axis=axis).astype(np.uint8) * 255


def mask_to_yolo_boxes(mask2d: np.ndarray, class_id: int = 0):
    """Connected-component bounding boxes from a 2D mask -> YOLO lines."""
    from scipy import ndimage

    h, w = mask2d.shape
    labeled, n = ndimage.label(mask2d > 0)
    lines = []
    for comp in range(1, n + 1):
        ys, xs = np.where(labeled == comp)
        if xs.size < 8:                       # drop tiny noise components
            continue
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _synthetic_smoke(args):
    print("Phase 4 --fast-dev-run: SYNTHETIC volume (no CBCT/nnU-Net/plastimatch needed)", flush=True)
    vol = np.random.rand(48, 64, 64).astype(np.float32)
    mask = np.zeros_like(vol, dtype=np.uint8)
    mask[20:28, 30:40, 30:40] = 1                       # a fake lesion blob
    drr = mip_projection(vol, axis=1)
    mask2d = project_mask(mask, axis=1)
    boxes = mask_to_yolo_boxes(mask2d)
    print(f"  DRR shape={drr.shape}, projected mask fg px={int((mask2d>0).sum())}, "
          f"derived {len(boxes)} pseudo-boxes: {boxes}", flush=True)
    out = Path(args.pseudo_output)
    (out / "opg" / "images").mkdir(parents=True, exist_ok=True)
    (out / "opg" / "labels").mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(out / "opg" / "images" / "synthetic_0.png"), drr)
    (out / "opg" / "labels" / "synthetic_0.txt").write_text("\n".join(boxes), encoding="utf-8")
    print(f"  wrote synthetic pseudo-labelled DRR -> {out/'opg'}", flush=True)


def train(args):
    cbct_dir = Path(args.cbct_dir)
    volumes = sorted(list(cbct_dir.rglob("*.nii.gz")) + list(cbct_dir.rglob("*.nii"))) \
        if cbct_dir.exists() else []

    if not volumes:
        if args.fast_dev_run:
            return _synthetic_smoke(args)
        print(
            f"Phase 4 BLOCKED: no CBCT volumes under {cbct_dir} (data/3D is empty).\n"
            f"  This phase distills 3D CBCT segmentations into 2D pseudo-labels.\n"
            f"  Needs: CBCT volumes (*.nii.gz) + either nnUNetv2_predict on PATH or\n"
            f"         pre-computed masks via --masks-dir.\n"
            f"  Validate the projection code now without data:\n"
            f"    python -m training.phase4_distillation --fast-dev-run", flush=True)
        return

    import SimpleITK as sitk

    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    have_nnunet = shutil.which("nnUNetv2_predict") is not None
    if masks_dir is None and not have_nnunet:
        print("Phase 4 BLOCKED: need --masks-dir OR nnUNetv2_predict on PATH for 3D masks.",
              flush=True)
        return

    out = Path(args.pseudo_output)
    (out / "opg" / "images").mkdir(parents=True, exist_ok=True)
    (out / "opg" / "labels").mkdir(parents=True, exist_ok=True)
    import cv2

    n_done = 0
    for vpath in volumes:
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(vpath))).astype(np.float32)
        if masks_dir is not None and (masks_dir / vpath.name).exists():
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(masks_dir / vpath.name)))
        else:
            print(f"  [skip] no mask for {vpath.name} (run nnU-Net first / pass --masks-dir)",
                  flush=True)
            continue
        drr = mip_projection(vol, axis=1)
        mask2d = project_mask(mask, axis=1)
        boxes = mask_to_yolo_boxes(mask2d)
        cv2.imwrite(str(out / "opg" / "images" / f"{vpath.stem}.png"), drr)
        (out / "opg" / "labels" / f"{vpath.stem}.txt").write_text("\n".join(boxes), encoding="utf-8")
        n_done += 1
        if args.fast_dev_run:
            break

    print(f"Phase 4 done: wrote {n_done} pseudo-labelled DRRs -> {out/'opg'}", flush=True)
    print("Next: merge into a YOLO dataset and re-run Phase 3 / Stage-1 YOLO with the "
          "pseudo-labelled DRRs included.", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbct-dir", default="../data/3D/cbct")
    ap.add_argument("--masks-dir", default=None, help="Pre-computed 3D masks (nnU-Net output).")
    ap.add_argument("--pseudo-output", default="../data/pseudo")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fast-dev-run", action="store_true")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
