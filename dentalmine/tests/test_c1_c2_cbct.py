"""Tests for the 3D path: C1 cross-slice attention, C2 consistency decision,
and the CBCT inference routing / reasoning chain.

All CPU-only and dependency-light (no SimpleITK / no model downloads): the CBCT
reasoning is exercised on synthetic detections, and routing on path strings.
"""
import torch

from models.madclip.c1_slice_attention import CrossSliceAttentionAdapter
from models.madclip.c2_consistency import ConsistencyFilter
from models.madclip.c3_clustering import FindingClusterer
from schema.finding import Detection, ModalityType


def _det(cls, box, slice_idx, conf=0.6):
    return Detection(class_name=cls, confidence=conf, calibrated_confidence=conf,
                     bbox_xyxy=box, slice_idx=slice_idx)


# ---- C1 cross-slice attention -------------------------------------------
def test_c1_preserves_shape():
    c1 = CrossSliceAttentionAdapter(embed_dim=64, num_heads=8)
    feats = [torch.randn(2, 8, 8, 64) for _ in range(5)]
    out = c1(feats, target_idx=2)
    assert out.shape == (2, 8, 8, 64)


def test_c1_single_slice_is_identity():
    c1 = CrossSliceAttentionAdapter(embed_dim=64, num_heads=8)
    one = torch.randn(1, 4, 4, 64)
    out = c1([one], target_idx=0)
    assert torch.equal(out, one)


def test_c1_select_k_by_modality():
    assert CrossSliceAttentionAdapter.select_k("cbct", 0.5) >= 2
    assert CrossSliceAttentionAdapter.select_k("full_mouth_series") == 1
    assert CrossSliceAttentionAdapter.select_k("serial_bw") == 2
    assert CrossSliceAttentionAdapter.select_k("panoramic") == 0


# ---- C2 neighbour-consistency decision ----------------------------------
def test_c2_keeps_persistent_removes_singleslice_noise():
    lesion = [0.4, 0.4, 0.6, 0.6]
    per_slice = {i: [_det("periapical_lesion", lesion, i)] for i in range(2, 6)}
    per_slice[0], per_slice[1] = [], []
    per_slice[3].append(_det("bone_loss", [0.0, 0.0, 0.1, 0.1], 3))  # 1-slice noise

    kept = ConsistencyFilter(min_support=2, iou_threshold=0.4, physical_window_mm=3.0)\
        .filter(per_slice, spacing_mm=1.0)
    classes = {d.class_name for dets in kept.values() for d in dets}
    assert "periapical_lesion" in classes
    assert "bone_loss" not in classes


def test_c2_single_slice_volume_unchanged():
    per_slice = {0: [_det("deep_caries", [0.3, 0.3, 0.5, 0.5], 0)]}
    out = ConsistencyFilter(min_support=3).filter(per_slice, spacing_mm=1.0)
    assert out == per_slice


def test_c2_boosts_confidence_of_survivors():
    box = [0.4, 0.4, 0.6, 0.6]
    per_slice = {i: [_det("periapical_lesion", box, i, conf=0.5)] for i in range(5)}
    kept = ConsistencyFilter(min_support=2, iou_threshold=0.4, physical_window_mm=5.0)\
        .filter(per_slice, spacing_mm=1.0)
    survivor = next(d for dets in kept.values() for d in dets)
    assert survivor.calibrated_confidence > 0.5     # boosted by neighbour support


# ---- CBCT routing + end-to-end reasoning chain --------------------------
def test_cbct_input_routing():
    from pipeline.inference_engine import _is_cbct_input
    assert _is_cbct_input("scan.nii.gz", None) is True
    assert _is_cbct_input("scan.nii", None) is True
    assert _is_cbct_input("opg.png", None) is False
    assert _is_cbct_input("opg.png", ModalityType.CBCT) is True


def test_2d_heads_route_by_modality():
    """BW/PA/OPG each dispatch to their own head with modality-appropriate classes."""
    import numpy as np
    from models.medclip.stub import StubEncoder
    from models.heads.bitewing_head import BitewingHead, _BW_KEYS
    from models.heads.periapical_head import PeriapicalHead, _PA_KEYS

    enc = StubEncoder()
    img = (np.random.rand(200, 400, 3) * 255).astype("uint8")
    bw = BitewingHead(enc, score_threshold=0.0)
    pa = PeriapicalHead(enc, score_threshold=0.0)
    # periapical_lesion is a PA target but never a bitewing target
    assert "periapical_lesion" in _PA_KEYS and "periapical_lesion" not in _BW_KEYS
    assert all(d.class_name in _BW_KEYS for d in bw.detect(img))
    assert all(d.class_name in _PA_KEYS for d in pa.detect(img))


def test_cbct_chain_consistent_lesion_to_cluster():
    """C2 decision -> C3 cluster across slices yields a per-region finding."""
    box = [0.4, 0.4, 0.6, 0.6]
    per_slice = {i: [_det("periapical_lesion", box, i)] for i in range(4)}
    kept = ConsistencyFilter(min_support=2, iou_threshold=0.4, physical_window_mm=3.0)\
        .filter(per_slice, spacing_mm=1.0)
    flat = [d for dets in kept.values() for d in dets]
    clusters = FindingClusterer().cluster(flat, fdi_confidence=0.0)  # DBSCAN (no FDI)
    assert len(clusters) >= 1
    assert clusters[0].severity.value in {"HIGH", "CRITICAL"}
