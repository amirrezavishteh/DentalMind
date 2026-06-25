"""End-to-end pipeline test (CPU-only, offline via Stub Med-CLIP)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

# Force offline stub encoder so the test needs no network / GPU.
os.environ["DENTALMIND_MEDCLIP"] = "stub"

from config_loader import load_config              # noqa: E402
from schema.finding import InferenceResult          # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "test_opg.jpg"


def _make_fixture_if_missing():
    if FIXTURE.exists():
        return
    import cv2
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    # synthetic wide panoramic-like gradient with some structure
    img = np.random.default_rng(0).integers(40, 200, size=(700, 1400), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imwrite(str(FIXTURE), img)


@pytest.fixture(scope="module")
def engine():
    from pipeline.inference_engine import InferenceEngine
    cfg = load_config(None)
    return InferenceEngine(cfg, device="cpu")


def test_e2e_infer_produces_outputs(engine, tmp_path):
    _make_fixture_if_missing()
    out_dir = tmp_path / "smoke"
    result = engine.infer(
        input_path=str(FIXTURE), modality="opg", output_dir=str(out_dir),
    )
    assert isinstance(result, InferenceResult)
    assert result.error is None
    assert result.processing_time_ms > 0
    assert result.modality.value == "panoramic"

    for fname in ("findings.json", "original.png", "annotated.png",
                  "treatment_card.txt", "summary.txt"):
        assert (out_dir / fname).exists(), f"missing {fname}"

    # findings.json validates against the schema
    data = json.loads((out_dir / "findings.json").read_text(encoding="utf-8"))
    InferenceResult.model_validate(data)

    # treatment card ends with disclaimer (when there are findings)
    card = (out_dir / "treatment_card.txt").read_text(encoding="utf-8")
    assert "AI second opinion" in card


def test_e2e_auto_modality(engine, tmp_path):
    _make_fixture_if_missing()
    result = engine.infer(str(FIXTURE), modality="auto", output_dir=str(tmp_path / "auto"))
    assert result.error is None
    # wide aspect ratio should route to panoramic
    assert result.modality.value == "panoramic"
