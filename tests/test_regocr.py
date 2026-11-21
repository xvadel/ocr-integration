"""
Comprehensive unit & integration test suite for RegOCR v0.3.
Covers:
  1. Normalization: digits, bidi, Arabic, visual-RTL reordering (zero LGPL), dates, tax IDs, capital.
  2. Enforced preprocessing gate: OCR engines reject raw arrays + AST audit of call sites.
  3. Cropping helper & PreprocessingReport telemetry.
  4. Multi-page PDF routing & cross-page merging (synthetic 2-page PDF where fields live on Page 2).
  5. Strict JSON Schema validation against registration_schema.json (v0.3) and merged_company.schema.json.
  6. Bad-quality rejection with clear reasons and user-facing reupload_hint.
  7. Digital PDF & Digital Visual PDF routing and exact extraction.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from regocr import SCHEMA_VERSION
from regocr.models import ExtractionResult
from regocr.normalize import (
    norm_arabic,
    parse_iso_date,
    parse_numeric_capital,
    parse_tax_id,
    reorder_visual_rtl,
    strip_bidi,
    to_ascii_digits,
)
from regocr.ocr_engines import EasyOCREngine, PaddleOCREngine, TesseractEngine
from regocr.pipeline import RegOCRPipeline
from regocr.preprocess import PreprocessConfig, PreprocessResult, crop_field_image, preprocess_image


def test_to_ascii_digits():
    assert to_ascii_digits("١٢٣٤٥٦٧٨٩٠") == "1234567890"
    assert to_ascii_digits("۱۲۳۴۵۶۷۸۹۰") == "1234567890"
    assert to_ascii_digits("621097") == "621097"


def test_strip_bidi():
    bidi_str = "\u200fشركة\u200e النور\u202a\u2066"
    cleaned = strip_bidi(bidi_str)
    for ch in ("\u200f", "\u200e", "\u202a", "\u2066"):
        assert ch not in cleaned


def test_norm_arabic():
    raw = "إِسْمُ الشَّرِكَةِ / الْمُنْشَأَةِ"
    norm = norm_arabic(raw)
    assert "إ" not in norm
    assert "ة" not in norm
    assert "اسم الشركه" in norm


def test_parse_iso_date_strict_calendar():
    assert parse_iso_date("02/02/2024") == "2024-02-02"
    assert parse_iso_date("١٨/٠٧/٢٠١٥") == "2015-07-18"
    assert parse_iso_date("2021-10-10") == "2021-10-10"
    # Invalid calendar dates must return None
    assert parse_iso_date("31/02/2024") is None
    assert parse_iso_date("99/99/2024") is None


def test_parse_tax_id_western_and_indic_rtl():
    assert parse_tax_id("940-648-196") == "940-648-196"
    # Arabic-Indic visual RTL order: ٦٧٩-٩٤٦-١٦٠ -> logical 160-946-679
    assert parse_tax_id("٦٧٩-٩٤٦-١٦٠", reverse_indic_groups=True) == "160-946-679"
    # Text layer (already logical): do not reverse
    assert parse_tax_id("١٦٠-٩٤٦-٦٧٩", reverse_indic_groups=False) == "160-946-679"


def test_parse_numeric_capital():
    assert parse_numeric_capital("100,000 جنيه مصري") == 100000.0
    assert parse_numeric_capital("١٬٠٠٠٬٠٠٠ جنيه مصري") == 1000000.0
    assert parse_numeric_capital("٠١,٠٠٠,٠٠٠") == 1000000.0
    assert parse_numeric_capital("250,000.50") == 250000.5


def test_enforced_preprocessing_gate_rejects_raw_arrays():
    """Priority 5.1: OCR engines MUST reject raw numpy arrays and only accept PreprocessResult."""
    dummy = np.full((100, 100), 255, dtype=np.uint8)
    for cls in (TesseractEngine, EasyOCREngine, PaddleOCREngine):
        eng = cls()
        raised = False
        try:
            eng.run(dummy)  # type: ignore[arg-type]
        except TypeError:
            raised = True
        assert raised, f"{cls.__name__}.run() failed to reject raw numpy array!"


def test_ast_no_ocr_call_bypasses_preprocess_gate():
    """Static AST check: ensure no module in src/regocr calls raw OCR libraries outside ocr_engines.py."""
    src_dir = ROOT / "src" / "regocr"
    forbidden = {"pytesseract", "easyocr", "paddleocr"}
    for py_file in src_dir.glob("*.py"):
        if py_file.name == "ocr_engines.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Allow pytesseract OSD inside preprocess.py only
                    if py_file.name == "preprocess.py" and alias.name == "pytesseract":
                        continue
                    assert alias.name.split(".")[0] not in forbidden, (
                        f"{py_file.name} imports {alias.name} directly, bypassing ocr_engines/preprocess gate!"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, (
                    f"{py_file.name} imports from {node.module} directly!"
                )


def test_no_pymupdf_or_python_bidi_in_regocr():
    """License check: ensure zero PyMuPDF (AGPL) or python-bidi (LGPL) imports in src/regocr/."""
    src_dir = ROOT / "src" / "regocr"
    banned = {"pymupdf", "fitz", "bidi"}
    for py_file in src_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned, f"Banned import {alias.name} in {py_file.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, f"Banned import {node.module} in {py_file.name}"


def test_crop_field_image_and_preprocess_report():
    img = np.full((1200, 900, 3), 240, dtype=np.uint8)
    cv2.putText(img, "123456", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 3)
    prep = preprocess_image(img, config=PreprocessConfig(fix_orientation=False), page=1)
    assert isinstance(prep, PreprocessResult)
    assert prep.report["page"] == 1
    assert "steps_applied" in prep.report
    assert "blur_score" in prep.report
    assert "illumination_score" in prep.report
    crop = crop_field_image(prep, [0.05, 0.20, 0.45, 0.30], upscale=2.0)
    assert crop.ndim == 2 and crop.shape[0] > 50 and crop.shape[1] > 50


def test_bad_quality_rejection_and_reupload_hint():
    """Priority 7.2: low-res heavily blurred dark image is rejected with a clear reupload_hint."""
    bad = np.full((300, 220, 3), 40, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tmp_path = Path(tf.name)
    cv2.imwrite(str(tmp_path), bad)
    try:
        pipe = RegOCRPipeline(engine="tesseract")  # fast offline check
        res = pipe.process_file(tmp_path)
        assert res.status == "rejected"
        assert "image_quality_too_low" in res.reasons
        assert res.reupload_hint and len(res.reupload_hint) > 20
        assert res.document_id.startswith("doc_")
        assert res.schema_version == SCHEMA_VERSION
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_multipage_synthetic_pdf_field_on_page_2():
    """Priority 4: synthetic 2-page PDF where tax_id and capital appear ONLY on page 2."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if not font_path.exists():
        print("Skipping multipage test: tahoma.ttf not found")
        return

    pdfmetrics.registerFont(TTFont("TahomaTest", str(font_path)))

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        pdf_path = Path(tf.name)

    try:
        c = canvas.Canvas(str(pdf_path))
        # Page 1: core fields only (NO tax_id, NO capital)
        c.setFont("TahomaTest", 14)
        c.drawString(50, 760, "نموذج وهمي - سجل تجاري")
        c.drawString(50, 720, "اسم الشركة / المنشأة: شركة النور للخدمات اللوجستية")
        c.drawString(50, 680, "الشكل القانوني: شركة مساهمة مصرية")
        c.drawString(50, 640, "رقم القيد بالسجل التجاري: 621097")
        c.drawString(50, 600, "تاريخ القيد: 02/02/2024")
        c.showPage()

        # Page 2: tax_id and capital ONLY on page 2!
        c.setFont("TahomaTest", 14)
        c.drawString(50, 760, "تابع صحيفة السجل التجاري")
        c.drawString(50, 720, "رقم البطاقة الضريبية: 940-648-196")
        c.drawString(50, 680, "رأس المال: 500,000 جنيه مصري")
        c.drawString(50, 640, "تاريخ انتهاء الصلاحية: 02/02/2029")
        c.showPage()
        c.save()

        pipe = RegOCRPipeline(engine="auto")
        res = pipe.process_file(pdf_path)

        assert res.document.pages == 2
        assert res.document.pages_processed == [1, 2]
        assert res.fields["register_number"].value == "621097"
        assert res.fields["register_number"].page == 1
        assert res.fields["tax_id"].value == "940-648-196"
        assert res.fields["tax_id"].page == 2
        assert res.fields["capital"].value == 500000.0
        assert res.fields["capital"].page == 2
        assert res.fields["expiry_date"].value == "2029-02-02"
        assert res.fields["expiry_date"].page == 2
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


def test_digital_and_visual_pdfs_and_json_schema():
    """Verify digital + digital_visual PDFs extract accurately and validate against registration_schema.json v0.3."""
    schema = json.loads((ROOT / "registration_schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    pipe = RegOCRPipeline(engine="auto")
    for rel in (
        "ocr_pdfs/pdfs/digital/sample_01_digital.pdf",
        "ocr_pdfs/pdfs/digital_visual/sample_01_visual.pdf",
    ):
        p = ROOT / rel
        if not p.exists():
            continue
        res = pipe.process_file(p)
        payload = res.model_dump(mode="json")
        errors = list(validator.iter_errors(payload))
        assert not errors, f"Schema errors on {rel}: {[e.message for e in errors]}"
        assert res.status == "accepted", f"{rel} unexpected status {res.status} reasons={res.reasons}"
        assert res.fields["register_number"].value == "621097"
        assert res.fields["issue_date"].value == "2024-02-02"
        assert res.fields["tax_id"].value == "940-648-196", f"{rel} tax_id={res.fields['tax_id'].value}"
        assert res.fields["capital"].value == 100000.0


def test_merged_handoff_schema_valid():
    schema = json.loads((ROOT / "schemas" / "merged_company.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    sample = {
        "schema_version": "0.3",
        "company_id": "8f4b6c1e-3d52-4a2b-9f1a-7c8d9e0f1a2b",
        "document_id": "doc_a1b2c3d4e5f67890",
        "pipeline_status": "enriched",
        "updated_at": "2026-11-17T12:10:05Z",
        "external_keys": {
            "unified_register_number": "17066 77763 29141",
            "national_entity_number": "07439150008063",
            "register_number": "083778",
            "issuing_office": "مكتب سجل تجاري طنطا",
            "tax_id": None,
        },
        "ocr": {
            "schema_version": "0.3",
            "submission_id": "sub_1794913502_egypt_00001",
            "document_id": "doc_a1b2c3d4e5f67890",
            "status": "accepted",
            "fields": {},
        },
        "enrichment": None,
        "merged": {
            "fields": {
                "register_number": {
                    "value": "083778",
                    "provenance": "ocr",
                    "confidence": 0.94,
                    "verified_by_human": False,
                }
            },
            "conflicts": [],
            "needs_review": False,
        },
    }
    errors = list(validator.iter_errors(sample))
    assert not errors, [e.message for e in errors]


def test_post_decision_guards():
    from regocr.guards import apply_guards
    # Synthetic doc representing a tax certificate rather than a commercial register
    doc = {
        "status": "accepted",
        "raw_text": "هيئة الزكاة والضريبة والجمارك شهادة تسجيل ضريبة القيمة المضافة vat",
        "fields": {
            "company_name": {"value": "شركة تجريبية", "status": "ok", "bbox": [0.1, 0.2, 0.5, 0.25], "engines_agree": True, "candidates": [{"value": "شركة تجريبية", "confidence": 0.9}]},
            "register_number": {"value": "123456", "status": "ok", "bbox": [0.1, 0.3, 0.3, 0.35], "engines_agree": True, "candidates": [{"value": "123456", "confidence": 0.95}]},
            "issue_date": {"value": "2024-01-01", "status": "ok", "bbox": [0.1, 0.4, 0.3, 0.45], "engines_agree": True, "candidates": [{"value": "2024-01-01", "confidence": 0.95}]}
        }
    }
    guarded, applied = apply_guards(doc, expected_type="commercial_register")
    assert guarded["status"] == "rejected", f"Expected rejected, got {guarded['status']}"
    assert any("wrong_document_type" in x for x in applied)

    # Synthetic doc with a core field in suspicious footer region (> 0.92)
    doc_cr = {
        "status": "accepted",
        "raw_text": "وزارة التجارة مستخرج السجل التجاري رقم القيد 101010",
        "fields": {
            "company_name": {"value": "شركة الفا", "status": "ok", "bbox": [0.1, 0.95, 0.5, 0.98]},
            "register_number": {"value": "101010", "status": "ok", "bbox": [0.1, 0.2, 0.3, 0.25]},
            "issue_date": {"value": "2024-01-01", "status": "ok", "bbox": [0.1, 0.3, 0.3, 0.35]}
        }
    }
    guarded_cr, applied_cr = apply_guards(doc_cr, expected_type="commercial_register")
    assert guarded_cr["status"] == "needs_review"
    assert "company_name:suspicious_region" in applied_cr


if __name__ == "__main__":
    tests = [
        test_to_ascii_digits,
        test_strip_bidi,
        test_norm_arabic,
        test_parse_iso_date_strict_calendar,
        test_parse_tax_id_western_and_indic_rtl,
        test_parse_numeric_capital,
        test_enforced_preprocessing_gate_rejects_raw_arrays,
        test_ast_no_ocr_call_bypasses_preprocess_gate,
        test_no_pymupdf_or_python_bidi_in_regocr,
        test_crop_field_image_and_preprocess_report,
        test_bad_quality_rejection_and_reupload_hint,
        test_multipage_synthetic_pdf_field_on_page_2,
        test_digital_and_visual_pdfs_and_json_schema,
        test_merged_handoff_schema_valid,
        test_post_decision_guards,
    ]
    for fn in tests:
        fn()
        print(f"  [PASS] {fn.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED!")
