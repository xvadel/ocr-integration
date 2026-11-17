"""
FastAPI service for RegOCR: Egyptian Commercial Register extraction API.
"""

from __future__ import annotations

import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from regocr import SCHEMA_VERSION, __version__
from regocr.models import ExtractionResult
from regocr.pipeline import RegOCRPipeline


class EngineChoice(str, Enum):
    auto = "auto"
    paddleocr = "paddleocr"
    easyocr = "easyocr"
    tesseract = "tesseract"


app = FastAPI(
    title="RegOCR API",
    description="High-accuracy OCR and structured extraction for Egyptian Commercial Register documents.",
    version=__version__,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
_pipelines: dict[str, RegOCRPipeline] = {}


def get_pipeline(engine: str = "auto") -> RegOCRPipeline:
    if engine not in _pipelines:
        _pipelines[engine] = RegOCRPipeline(engine=engine)
    return _pipelines[engine]


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "regocr",
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
    }


@app.post("/extract", response_model=ExtractionResult)
async def extract_document(
    file: UploadFile = File(...),
    engine: EngineChoice = Query(EngineChoice.auto, description="OCR engine choice"),
):
    """
    Extract structured data from an uploaded commercial register file (Image or PDF).
    """
    ext = Path(file.filename or "file").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: jpg, png, pdf, tiff, webp.",
        )

    pipeline = get_pipeline(engine=engine.value)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = pipeline.process_file(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
