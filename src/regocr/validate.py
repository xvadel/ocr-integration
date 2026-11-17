"""
Validation, calibrated confidence scoring, re-upload hints, and decision routing.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

from regocr.models import FieldResult, QualityInfo, ReviewInfo
from regocr.normalize import norm_arabic, to_ascii_digits

CORE_REQUIRED_FIELDS = ("company_name", "register_number", "issue_date")
NUMERIC_ID_FIELDS = (
    "register_number",
    "unified_register_number",
    "national_entity_number",
    "tax_id",
    "capital",
    "issue_date",
    "expiry_date",
)


def validate_field_value(
    name: str,
    value: Any,
    raw: Optional[str] = None,
    enum_values: Optional[Dict[str, str]] = None,
) -> Tuple[bool, List[str]]:
    """Run deterministic format/domain validators for a single extracted field."""
    if value is None:
        return False, ["missing"]

    issues: List[str] = []

    if name == "register_number":
        s = to_ascii_digits(str(value)).strip()
        if not re.fullmatch(r"\d{3,10}", s):
            issues.append("format_mismatch")

    elif name == "unified_register_number":
        digits = re.sub(r"\D", "", to_ascii_digits(str(value)))
        if not (10 <= len(digits) <= 20):
            issues.append("format_mismatch")

    elif name == "national_entity_number":
        digits = re.sub(r"\D", "", to_ascii_digits(str(value)))
        if not (10 <= len(digits) <= 16):
            issues.append("format_mismatch")

    elif name == "tax_id":
        s = to_ascii_digits(str(value)).strip()
        if not re.fullmatch(r"\d{3}-\d{3}-\d{3}", s):
            issues.append("format_mismatch")

    elif name in ("issue_date", "expiry_date"):
        s = str(value).strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            issues.append("format_mismatch")
        else:
            try:
                y, m, d = map(int, s.split("-"))
                dt = datetime.date(y, m, d)
                if not (1950 <= dt.year <= 2060):
                    issues.append("date_out_of_range")
            except ValueError:
                issues.append("invalid_calendar_date")

    elif name == "capital":
        try:
            num = float(value)
            if num < 1000.0 or num > 1e12:
                issues.append("capital_out_of_range")
        except (TypeError, ValueError):
            issues.append("format_mismatch")

    elif name == "company_name":
        s = str(value).strip()
        if len(s) < 6:
            issues.append("too_short")
        if any(tok in s for tok in ["نموذج وهمي", "للاختبار فقط", "اسم الشركة"]):
            issues.append("label_leakage")

    elif name in ("legal_form", "governorate") and enum_values:
        norm_v = norm_arabic(str(value))
        valid_norms = {norm_arabic(v) for v in enum_values.values()}
        if norm_v not in valid_norms:
            issues.append("invalid_enum_value")

    elif name == "manager":
        s = str(value).strip()
        if len(s.split()) < 2 or re.search(r"\d{3,}", s):
            issues.append("invalid_person_name")

    return len(issues) == 0, issues


def compute_field_confidence(
    engine_confidence: Optional[float],
    engines_agree: Optional[bool],
    validated: Optional[bool],
    grounded: bool,
    quality_score: float,
    is_text_layer: bool = False,
) -> Optional[float]:
    """Compute a calibrated confidence score in [0.05, 0.99] as a function of:
    engine confidence, cross-engine agreement, validator outcome, grounding, and image quality.
    """
    if engine_confidence is None and not is_text_layer:
        return None

    if is_text_layer:
        base = 0.96 if (validated is not False and grounded) else 0.55
        if validated is False:
            base -= 0.25
        return round(max(0.05, min(0.99, base)), 4)

    eng_c = max(0.0, min(1.0, float(engine_confidence if engine_confidence is not None else 0.5)))
    q = max(0.0, min(1.0, float(quality_score)))

    # Weighted base from real OCR confidence + image quality
    score = 0.65 * eng_c + 0.20 * q + 0.05

    # Cross-engine agreement signal
    if engines_agree is True:
        score += 0.08
    elif engines_agree is False:
        score -= 0.22

    # Deterministic validator signal
    if validated is True:
        score += 0.05
    elif validated is False:
        score -= 0.30

    # Grounding check
    if not grounded:
        score -= 0.15

    return round(max(0.05, min(0.99, score)), 4)


def build_reupload_hint(status: str, quality: QualityInfo, reasons: List[str]) -> Optional[str]:
    """Generate clear, actionable user-facing guidance when a document is rejected or low quality."""
    if status == "accepted":
        return None

    hints: List[str] = []
    if "blurry" in quality.issues:
        hints.append("The image is too blurry — hold the camera steady or scan at 300 DPI.")
    if "low_resolution" in quality.issues:
        hints.append("Resolution is too low — upload a higher-resolution photo or PDF (at least 1000px wide).")
    if "dark" in quality.issues or "uneven_illumination" in quality.issues:
        hints.append("Lighting is dark or uneven — capture in even daylight without heavy shadows.")
    if "noisy" in quality.issues:
        hints.append("Heavy scan noise detected — clean the scanner glass or re-export as a digital PDF.")
    if "skewed" in quality.issues:
        hints.append("The page is tilted — align the document flat inside the frame.")

    missing_core = [r.split(":", 1)[1] for r in reasons if r.startswith("critical_field_missing:")]
    if missing_core:
        hints.append(f"Required fields could not be read clearly ({', '.join(missing_core)}). Ensure the full page is visible.")

    if not hints and status == "rejected":
        hints.append("Document could not be processed reliably. Please upload a clearer image or original PDF.")

    return " ".join(hints) if hints else None


def validate_document_status(
    quality: QualityInfo,
    fields: Dict[str, FieldResult],
    ocr_confidence: Optional[float] = None,
    min_doc_quality_reject: float = 0.35,
    min_doc_quality_accept: float = 0.65,
    min_field_confidence: float = 0.68,
) -> Tuple[str, List[str], ReviewInfo, Optional[str]]:
    """Evaluates priority status: rejected > needs_review > accepted, plus user reupload_hint."""
    reasons: List[str] = []
    fields_to_review: List[str] = []

    # 1. Critical Quality Gate (Direct Rejection)
    if quality.score < min_doc_quality_reject or (
        "blurry" in quality.issues and "noisy" in quality.issues and "low_resolution" in quality.issues
    ):
        reasons.append("image_quality_too_low")
        hint = build_reupload_hint("rejected", quality, reasons)
        return "rejected", reasons, ReviewInfo(required=False, fields_to_review=[]), hint

    # 2. Critical Core Fields Missing (Rejection)
    missing_cores = [
        k for k in CORE_REQUIRED_FIELDS
        if fields.get(k) is None or fields[k].status == "missing" or fields[k].value is None
    ]
    if missing_cores:
        reasons.append("core_fields_missing")
        for k in missing_cores:
            reasons.append(f"critical_field_missing:{k}")
        hint = build_reupload_hint("rejected", quality, reasons)
        return "rejected", reasons, ReviewInfo(required=False, fields_to_review=[]), hint

    # 3. Logical Timeline Cross-Field Check
    issue_fld = fields.get("issue_date")
    expiry_fld = fields.get("expiry_date")
    if issue_fld and expiry_fld and issue_fld.value and expiry_fld.value:
        if str(expiry_fld.value) <= str(issue_fld.value):
            expiry_fld.status = "suspect"
            expiry_fld.validated = False
            expiry_fld.needs_review = True
            if "expiry_precedes_issue" not in expiry_fld.issues:
                expiry_fld.issues.append("expiry_precedes_issue")
            reasons.append("timeline_error:expiry_precedes_issue")
            fields_to_review.append("expiry_date")

    # 4. Field-level Suspect / Disagreement / Validation / Confidence Checks
    for k, fld in fields.items():
        if fld.needs_review:
            fields_to_review.append(k)
        if fld.status == "suspect":
            fields_to_review.append(k)
            fld.needs_review = True
            for iss in fld.issues:
                reasons.append(f"{k}:{iss}")
        elif fld.validated is False and fld.value is not None:
            fields_to_review.append(k)
            fld.needs_review = True
            reasons.append(f"{k}:validation_failed")
        elif fld.engines_agree is False:
            fields_to_review.append(k)
            fld.needs_review = True
            reasons.append(f"{k}:engine_disagreement")
        elif fld.value is not None and not fld.grounded:
            fields_to_review.append(k)
            fld.needs_review = True
            reasons.append(f"{k}:not_grounded")
        elif fld.value is not None and fld.confidence is not None and fld.confidence < min_field_confidence:
            fields_to_review.append(k)
            fld.needs_review = True
            reasons.append(f"{k}:low_confidence")

    # 5. Borderline Quality Check
    if quality.score < min_doc_quality_accept:
        reasons.append("borderline_image_quality")
        for k in CORE_REQUIRED_FIELDS:
            if k not in fields_to_review:
                fields_to_review.append(k)

    # 6. Borderline OCR Mean Confidence
    if ocr_confidence is not None and ocr_confidence < 0.65:
        reasons.append("borderline_ocr_confidence")
        for k in CORE_REQUIRED_FIELDS:
            if k not in fields_to_review and fields.get(k) and fields[k].value is not None:
                fields_to_review.append(k)

    if fields_to_review or reasons:
        fields_to_review = sorted(list(set(fields_to_review)))
        reasons = sorted(list(set(reasons)))
        hint = build_reupload_hint("needs_review", quality, reasons)
        return (
            "needs_review",
            reasons,
            ReviewInfo(required=True, fields_to_review=fields_to_review),
            hint,
        )

    return "accepted", [], ReviewInfo(required=False, fields_to_review=[]), None
