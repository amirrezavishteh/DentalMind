"""Tests for the MadClip integration layer (C3 clustering, C4 prompts)."""
import pytest

from models.madclip.c3_clustering import PATTERN_LIBRARY, FindingClusterer
from models.madclip.c4_prompts import TreatmentPromptGenerator
from schema.finding import Detection


def _det(cls, fdi, x=0.1):
    return Detection(
        class_name=cls, confidence=0.7, calibrated_confidence=0.7,
        bbox_xyxy=[x, 0.1, x + 0.05, 0.3], tooth_fdi=fdi,
    )


def test_c3_groups_by_fdi():
    dets = [
        _det("dentin_caries", "26"), _det("bone_loss", "26"),
        _det("dentin_caries", "27"),
        _det("periapical_lesion", "36"),
    ]
    clusters = FindingClusterer().cluster(dets, image_height_px=1000)
    fdis = {c.tooth_fdi for c in clusters}
    assert fdis == {"26", "27", "36"}


def test_c3_pattern_compounding():
    # caries + bone on same tooth -> caries_with_bone
    dets = [_det("dentin_caries", "26"), _det("bone_loss", "26")]
    clusters = FindingClusterer().cluster(dets, image_height_px=1000)
    assert clusters[0].pattern == "caries_with_bone"

    # caries + periapical lesion -> caries_with_pal (critical)
    dets = [_det("dentin_caries", "11"), _det("periapical_lesion", "11")]
    clusters = FindingClusterer().cluster(dets, image_height_px=1000)
    assert clusters[0].pattern == "caries_with_pal"
    assert clusters[0].severity.value == "CRITICAL"


def test_c3_dbscan_fallback_low_fdi_conf():
    dets = [_det("dentin_caries", None, x=0.1), _det("dentin_caries", None, x=0.8)]
    clusters = FindingClusterer().cluster(dets, image_height_px=1000, fdi_confidence=0.2)
    assert len(clusters) == 2  # two spatially separate detections


def test_c4_all_patterns_produce_valid_prompt():
    from schema.finding import Severity, ToothCluster, Urgency
    gen = TreatmentPromptGenerator()
    for pattern in PATTERN_LIBRARY:
        cluster = ToothCluster(
            tooth_fdi="26", severity=Severity.HIGH, urgency=Urgency.SOON,
            pattern=pattern, findings=[_det("dentin_caries", "26")],
            cluster_confidence=0.7, surfaces_affected=["distal"], bone_loss_mm=2.0,
        )
        out = gen.generate_from_template(cluster)
        assert out.tooth_fdi == "26"
        assert out.pattern == pattern
        assert out.options, f"no options for {pattern}"
        assert "AI second opinion" in out.disclaimer
        assert "AI second opinion" in out.to_text()


def test_c3_empty():
    assert FindingClusterer().cluster([]) == []
