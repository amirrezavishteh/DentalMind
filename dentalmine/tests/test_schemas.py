"""Schema validation tests."""
import json

from schema.finding import (
    Detection, InferenceResult, ModalityType, Severity, ToothCluster, Urgency,
)
from schema.prompt import DISCLAIMER, PromptOutput, TreatmentOption


def _make_cluster(fdi="26", sev=Severity.HIGH):
    det = Detection(
        class_name="dentin_caries", confidence=0.8, calibrated_confidence=0.75,
        bbox_xyxy=[0.1, 0.2, 0.3, 0.4], tooth_fdi=fdi, surface="distal",
    )
    return ToothCluster(
        tooth_fdi=fdi, severity=sev, urgency=Urgency.SOON, pattern="caries_dentin",
        findings=[det], cluster_confidence=0.75, surfaces_affected=["distal"],
    )


def test_inference_result_serializes():
    result = InferenceResult(
        modality=ModalityType.OPG, image_paths=["x.png"], clusters=[_make_cluster()],
    ).recompute_summary()
    s = result.model_dump_json()
    parsed = json.loads(s)
    assert parsed["modality"] == "panoramic"
    assert parsed["total_teeth_affected"] == 1
    assert parsed["highest_severity"] == "HIGH"


def test_highest_severity_is_max():
    result = InferenceResult(
        modality=ModalityType.OPG, image_paths=["x.png"],
        clusters=[_make_cluster(sev=Severity.LOW), _make_cluster(sev=Severity.CRITICAL)],
    ).recompute_summary()
    assert result.highest_severity == Severity.CRITICAL


def test_prompt_always_has_disclaimer():
    p = PromptOutput(
        tooth_fdi="26", pattern="caries_dentin",
        options=[TreatmentOption(rank=1, text="restore", when_to_use="vital")],
        tests_suggested=["cold test"], urgency=Urgency.SOON, red_flags=["pain"],
        tier="template",
    )
    assert p.disclaimer == DISCLAIMER
    assert DISCLAIMER in p.to_text()


def test_detection_bbox_in_unit_range():
    det = Detection(
        class_name="bone_loss", confidence=0.5, calibrated_confidence=0.5,
        bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
    )
    assert all(0.0 <= v <= 1.0 for v in det.bbox_xyxy)
    cx, cy = det.bbox_center
    assert (cx, cy) == (0.5, 0.5)
