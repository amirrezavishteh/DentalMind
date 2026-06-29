"""Tests for pure training/preprocessing helpers (no GPU, no data, no downloads)."""
import numpy as np

from training.phase4_distillation import (
    mask_to_yolo_boxes, mip_projection, project_mask,
)
from training.trainer_base import cosine_lr


def test_mip_projection_shape_and_range():
    vol = np.random.rand(20, 32, 48).astype(np.float32)
    drr = mip_projection(vol, axis=1)          # project out the H axis
    assert drr.shape == (20, 48)
    assert drr.dtype == np.uint8
    assert drr.max() <= 255 and drr.min() >= 0


def test_project_mask_and_boxes():
    mask = np.zeros((20, 32, 48), dtype=np.uint8)
    mask[5:15, 10:20, 10:25] = 1               # one solid blob
    mask2d = project_mask(mask, axis=1)
    assert (mask2d > 0).any()
    boxes = mask_to_yolo_boxes(mask2d, class_id=0)
    assert len(boxes) == 1
    cls, cx, cy, bw, bh = boxes[0].split()
    assert cls == "0"
    assert 0.0 <= float(cx) <= 1.0 and 0.0 <= float(cy) <= 1.0
    assert 0.0 < float(bw) <= 1.0 and 0.0 < float(bh) <= 1.0


def test_mask_to_yolo_drops_tiny_noise():
    mask2d = np.zeros((40, 40), dtype=np.uint8)
    mask2d[0, 0] = 255                          # 1-px speck -> below min size
    assert mask_to_yolo_boxes(mask2d) == []


def test_cosine_lr_warmup_and_decay():
    base = 1e-3
    assert cosine_lr(0, 100, base, warmup_steps=10) < base      # warmup ramps up
    assert abs(cosine_lr(10, 100, base, warmup_steps=10) - base) < 1e-9
    assert cosine_lr(100, 100, base, warmup_steps=10) < base    # decayed by the end


def test_preprocess_cbct_outputs_rgb_slices():
    from pipeline.preprocessor import Preprocessor
    slices = [np.random.rand(40, 40).astype(np.float32) for _ in range(4)]
    display, spacing = Preprocessor().preprocess_cbct(slices, spacing_mm=0.4)
    assert len(display) == 4
    assert display[0].shape == (40, 40, 3)      # uint8 BGR/RGB
    assert spacing == 0.4
