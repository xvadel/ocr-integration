"""Pydantic v2 output models for the Egyptian Commercial Register OCR pipeline (schema v0.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Field-level models
# ---------------------------------------------------------------------------

class FieldCandidate(BaseModel):
    """A single OCR-engine candidate for a field value."""

    value: str
    raw: Optional[str] = None
    engine: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[list[float]] = None  # [x1, y1, x2, y2] normalized [0, 1]
    page: int = 1
    variant: str = "A"


class FieldResult(BaseModel):
    """Extraction result for one document field."""

    value: Optional[Union[str, int, float]] = None
    raw: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    engines_agree: Optional[bool] = None
    validated: Optional[bool] = None
    needs_review: bool = False
    candidates: list[FieldCandidate] = Field(default_factory=list)
    page: int = 1
    bbox: Optional[list[float]] = None  # [x1, y1, x2, y2] normalized [0, 1]
    status: Literal["ok", "suspect", "missing"] = "missing"
    grounded: bool = False
    issues: list[str] = Field(default_factory=list)
    code: Optional[str] = None  # e.g. legal_form code like 'llc'
    unit: Optional[str] = None  # e.g. 'EGP' for capital


# ---------------------------------------------------------------------------
# Preprocessing & Document-level models
# ---------------------------------------------------------------------------

class PreprocessingReport(BaseModel):
    """Per-page CV preprocessing telemetry."""

    page: int = Field(ge=1)
    steps_applied: list[str] = Field(default_factory=list)
    skew_angle: float = 0.0
    blur_score: float = 0.0
    illumination_score: float = 1.0
    perspective_corrected: bool = False
    orientation_angle: int = 0
    stamp_suppressed: bool = False
    quality_flags: list[str] = Field(default_factory=list)


class QualityInfo(BaseModel):
    """Image / scan quality assessment."""

    score: float = Field(ge=0.0, le=1.0)
    issues: list[
        Literal[
            "blurry",
            "low_resolution",
            "skewed",
            "dark",
            "cropped",
            "noisy",
            "uneven_illumination",
        ]
    ] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Metadata about the source document."""

    type: Literal["commercial_register", "tax_card", "unknown"] = "commercial_register"
    file_format: Literal["image", "pdf"]
    pdf_kind: Optional[Literal["digital", "scanned", "digital_garbled", "mixed"]] = None
    pages: int = Field(default=1, ge=1)
    page_used: int = Field(default=1, ge=1)
    pages_processed: list[int] = Field(default_factory=lambda: [1])
    quality: QualityInfo


class ExtractionInfo(BaseModel):
    """Details about how text was extracted."""

    text_method: Literal["text_layer", "ocr", "mixed", "vision_llm", "none"]
    engine: Optional[str] = None
    languages: list[str] = Field(default_factory=lambda: ["ar", "en"])
    ocr_confidence_mean: Optional[float] = Field(None, ge=0.0, le=1.0)
    duration_ms: int = Field(ge=0)


class ReviewInfo(BaseModel):
    """Human-review status."""

    required: bool
    fields_to_review: list[str] = Field(default_factory=list)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    """Complete extraction result for a single document submission."""

    schema_version: str = "0.4"
    country: Optional[str] = "eg"
    document_type: Optional[str] = "commercial_register"
    template_id: Optional[str] = None
    document_id: str
    submission_id: str
    created_at: datetime
    test_data: bool = False
    status: Literal["accepted", "needs_review", "rejected"]
    reasons: list[str] = Field(default_factory=list)
    reupload_hint: Optional[str] = None
    document: DocumentInfo
    preprocessing: list[PreprocessingReport] = Field(default_factory=list)
    extraction: ExtractionInfo
    fields: dict[str, FieldResult]
    raw_text: Optional[str] = None
    review: ReviewInfo
