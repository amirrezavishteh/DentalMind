"""Convert Tufts Dental Database tooth bounding boxes -> YOLO FDI dataset.

Source (already downloaded, no extra download needed):
  ../data/tufts/radiographs/*.JPG                      (1000 panoramic X-rays)
  ../data/tufts/segmentation/teeth_bbox.json           (per-image tooth boxes,
                                                         Universal Numbering 1-32
                                                         for permanent teeth +
                                                         letters A-T for primary
                                                         teeth in pediatric scans)

This script keeps only permanent teeth (numeric titles 1-32), maps Universal
Numbering -> FDI notation (matching models/heads/panoramic_head.py's vocabulary),
and writes a standard YOLO detection dataset:

  data/tufts_yolo/
  ├── data.yaml
  ├── images/{train,val}/*.jpg
  └── labels/{train,val}/*.txt

Usage:
  python scripts/tufts_to_yolo.py --val-fraction 0.1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

# Universal Numbering System (1-32) -> FDI notation.
UNIVERSAL_TO_FDI = {
    1: "18", 2: "17", 3: "16", 4: "15", 5: "14", 6: "13", 7: "12", 8: "11",
    9: "21", 10: "22", 11: "23", 12: "24", 13: "25", 14: "26", 15: "27", 16: "28",
    17: "38", 18: "37", 19: "36", 20: "35", 21: "34", 22: "33", 23: "32", 24: "31",
    25: "41", 26: "42", 27: "43", 28: "44", 29: "45", 30: "46", 31: "47", 32: "48",
}
FDI_CLASSES = [UNIVERSAL_TO_FDI[i] for i in range(1, 33)]  # fixed class order
FDI_TO_IDX = {fdi: i for i, fdi in enumerate(FDI_CLASSES)}

MIN_BOX_PX = 3  # drop degenerate/zero-area annotation artifacts


def unrotate_box_ccw(ax, ay, bx, by, orig_w: int, orig_h: int):
    """Map a box from the Labelbox annotation canvas back to original pixel space.

    The Tufts bbox JSON coordinates were captured on a canvas rotated 90deg
    counter-clockwise relative to the published radiograph orientation. Verified
    by checking that same-side tooth pairs (e.g. Universal #1 upper-right-3rd-molar
    and #32 lower-right-3rd-molar; #16 upper-left-3rd-molar and #17 lower-left-3rd-
    molar) land in matching x-ranges with correct upper/lower y-halves only under
    this transform (the alternative 90deg CW transform fails that consistency
    check). This is the inverse of a 90 deg CCW rotation.
    """
    y1, y2 = ax, bx
    x1, x2 = orig_w - by, orig_w - ay
    return x1, y1, x2, y2


def find_image(radiographs_dir: Path, external_id: str) -> Path | None:
    base = external_id.rsplit(".", 1)[0]
    for ext in (".JPG", ".jpg", ".png", ".PNG"):
        p = radiographs_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def convert(
    bbox_json: Path,
    radiographs_dir: Path,
    out_dir: Path,
    val_fraction: float,
    seed: int,
    visualize_n: int = 0,
):
    with open(bbox_json, "r", encoding="utf-8") as fh:
        entries = json.load(fh)

    random.seed(seed)
    random.shuffle(entries)
    n_val = max(1, int(len(entries) * val_fraction))

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    n_written, n_skipped, n_boxes = 0, 0, 0
    for i, item in enumerate(entries):
        split = "val" if i < n_val else "train"
        src = find_image(radiographs_dir, item["External ID"])
        if src is None:
            n_skipped += 1
            continue

        with Image.open(src) as im:
            w, h = im.size

        lines = []
        for obj in item["Label"]["objects"]:
            title = obj.get("title", "")
            if not title.isdigit():
                continue  # skip primary teeth (A-T) — out of Phase 1 FDI scope
            tooth_num = int(title)
            if tooth_num not in UNIVERSAL_TO_FDI:
                continue
            box = obj.get("bounding box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = unrotate_box_ccw(*box, orig_w=w, orig_h=h)
            if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
                continue
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cls_idx = FDI_TO_IDX[UNIVERSAL_TO_FDI[tooth_num]]
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            n_skipped += 1
            continue

        stem = src.stem
        dst_img = images_dir / split / f"{stem}.jpg"
        if src.suffix.lower() != ".jpg":
            Image.open(src).convert("RGB").save(dst_img, quality=95)
        else:
            shutil.copy2(src, dst_img)
        (labels_dir / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        n_written += 1
        n_boxes += len(lines)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(out_dir.resolve()).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(FDI_CLASSES)}\n"
        "names: " + json.dumps(FDI_CLASSES) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {n_written} images ({n_boxes} tooth boxes), skipped {n_skipped} "
          f"(no match / no usable boxes).")
    print(f"Classes: {len(FDI_CLASSES)} FDI teeth -> {data_yaml}")

    if visualize_n > 0:
        _write_debug_visualizations(images_dir / "train", labels_dir / "train",
                                     out_dir / "_debug_viz", visualize_n)


def _write_debug_visualizations(images_dir: Path, labels_dir: Path, out_dir: Path, n: int):
    """Draw decoded YOLO boxes back onto images for a quick visual sanity check."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    img_files = sorted(images_dir.glob("*.jpg"))[:n]
    for img_path in img_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            cls_idx, cx, cy, bw, bh = line.split()
            cls_idx, cx, cy, bw, bh = int(cls_idx), float(cx), float(cy), float(bw), float(bh)
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, FDI_CLASSES[cls_idx], (x1, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / img_path.name), img)
    print(f"Wrote {len(img_files)} debug visualizations -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-json", default="../data/tufts/segmentation/teeth_bbox.json")
    ap.add_argument("--radiographs-dir", default="../data/tufts/radiographs")
    ap.add_argument("--out-dir", default="../data/tufts_yolo")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visualize-n", type=int, default=0,
                     help="Write N debug images with decoded boxes drawn on, for sanity checking.")
    args = ap.parse_args()

    convert(
        Path(args.bbox_json), Path(args.radiographs_dir), Path(args.out_dir),
        args.val_fraction, args.seed, args.visualize_n,
    )
