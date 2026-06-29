"""FastAPI serving layer — wraps the InferenceEngine behind an HTTP API.

Endpoints:
  GET  /health            -> {status, modalities, medclip}
  POST /infer             -> multipart file upload (2D image) -> InferenceResult JSON
                             query params: modality=auto|opg|bw|pa|cbct, use_vlm, patient_age

The engine (and its Med-CLIP encoder) is loaded once at app startup and reused.
"""
import tempfile
from pathlib import Path
from typing import Optional

from config_loader import load_config
from schema.finding import ModalityType
from schema.patient import PatientContext

# Imported at module top so FastAPI can resolve the UploadFile type hint on the
# route handler (function-local imports break pydantic-v2 forward-ref resolution).
# Guarded so the rest of the package imports fine without fastapi installed.
try:
    from fastapi import FastAPI, File, Query, UploadFile
    from fastapi.responses import JSONResponse
    _FASTAPI_OK = True
except ImportError:  # pragma: no cover
    _FASTAPI_OK = False


def create_app(config_path: Optional[str] = None, checkpoint: Optional[str] = None,
               device: Optional[str] = None):
    if not _FASTAPI_OK:
        raise ImportError("fastapi is required for `serve` (pip install fastapi uvicorn python-multipart).")

    from pipeline.inference_engine import InferenceEngine

    app = FastAPI(title="DentalMind", version="0.1.0")
    cfg = load_config(config_path)
    engine = InferenceEngine(cfg, checkpoint_dir=checkpoint, device=device)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "modalities": [m.value for m in ModalityType],
            "medclip": engine.model_versions.get("medclip", "unknown"),
            "device": engine.device,
        }

    @app.post("/infer")
    async def infer(
        file: UploadFile = File(...),
        modality: str = Query("auto"),
        use_vlm: bool = Query(False),
        patient_age: Optional[int] = Query(None),
    ):
        suffix = Path(file.filename or "upload.png").suffix or ".png"
        data = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            pc = PatientContext(age=patient_age) if patient_age is not None else None
            result = engine.infer(tmp_path, modality=modality, patient_context=pc,
                                  use_vlm=use_vlm)
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
        status = 200 if not result.error else 422
        return JSONResponse(status_code=status, content=result.model_dump(mode="json"))

    return app


def serve(host: str = "0.0.0.0", port: int = 8080, config_path: Optional[str] = None,
          checkpoint: Optional[str] = None, device: Optional[str] = None):
    import uvicorn
    app = create_app(config_path, checkpoint, device)
    uvicorn.run(app, host=host, port=port)
