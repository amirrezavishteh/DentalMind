"""Tests for the nnU-Net wrapper post-processing + C3 3D merge. No nnU-Net needed."""
import numpy as np

from models.madclip.c3_clustering import FindingClusterer
from models.segmentation.nnunet_wrapper import NNUNetWrapper
from schema.finding import Detection


def _synthetic_seg():
    seg = np.zeros((20, 40, 40), dtype=np.int16)
    seg[5:9, 15:19, 8:12] = 3       # upper-right -> Q1
    seg[5:9, 15:19, 28:32] = 3      # upper-left  -> Q2
    seg[12:16, 15:19, 8:12] = 4     # lower-right -> Q4
    seg[12:16, 15:19, 28:32] = 4    # lower-left  -> Q3
    return seg


def test_get_tooth_instances_quadrants():
    inst = NNUNetWrapper.get_tooth_instances(_synthetic_seg())
    assert sorted(k[0] for k in inst) == ["1", "2", "3", "4"]
    assert all(m.dtype == bool for m in inst.values())


def test_get_tooth_instances_drops_noise():
    seg = np.zeros((20, 40, 40), dtype=np.int16)
    seg[0, 0, 0] = 3                # single voxel -> below MIN_TOOTH_VOXELS
    assert NNUNetWrapper.get_tooth_instances(seg) == {}


def test_c3_merge_with_3d_flags_matching_fdi():
    det = Detection(class_name="periapical_lesion", confidence=0.7,
                    calibrated_confidence=0.7, bbox_xyxy=[0.4, 0.4, 0.6, 0.6], tooth_fdi="11")
    clusters = FindingClusterer().cluster([det], fdi_confidence=1.0)
    assert clusters[0].has_3d_mask is False
    merged = FindingClusterer().merge_with_3d(clusters, {"11": np.ones((2, 2, 2), bool)})
    assert merged[0].has_3d_mask is True
