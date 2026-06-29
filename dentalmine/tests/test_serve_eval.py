"""Tests for the serve (FastAPI) and eval (batch report) layers. CPU/stub only."""
import os
from pathlib import Path

import pytest

os.environ.setdefault("DENTALMIND_MEDCLIP", "stub")  # offline deterministic encoder

_FIXTURE = Path(__file__).parent / "fixtures" / "test_opg.jpg"


def test_eval_dir_report():
    from pipeline.evaluate import evaluate_dir
    rep = evaluate_dir(str(_FIXTURE.parent), modality="opg", device="cpu")
    assert rep["n_images"] >= 1
    assert rep["n_errors"] == 0
    assert "severity_histogram" in rep
    assert rep["mean_ms_per_image"] > 0


def test_serve_health_and_infer():
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient
    from pipeline.serve import create_app

    client = TestClient(create_app(device="cpu"))
    assert client.get("/health").json()["status"] == "ok"

    with open(_FIXTURE, "rb") as f:
        resp = client.post("/infer?modality=opg", files={"file": ("t.jpg", f, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "panoramic"
    assert body["error"] is None
