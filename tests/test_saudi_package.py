"""
test_saudi_package.py - Unit & Contract Tests for Saudi Arabia Package (v0.4).

Covers:
  1. Both pages OK -> accepted, validates against schemas/sa/sa_registration.schema.json
  2. One page only -> needs_review with missing_document reason
  3. Swapped order -> order-independent acceptance
  4. Duplicate page type -> needs_review: duplicate_document_type
  5. Masked unified number -> needs_review (never rejected)
  6. Company name mismatch across documents -> cross_doc_mismatch:company_name
  7. VAT number format and TIN prefix check
  8. Date orders (DD/MM/YYYY vs YYYY/MM/DD) & Arabic-Indic digits
  9. Footer serial regression -> footer serial never misattributed as VAT / CR number
  10. Wrong country (Egyptian document uploaded with country=sa) -> rejected / missing doc
  11. Private fixtures test (auto-skipped when PRIVATE_FIXTURES_DIR is unset)
"""

import os
import json
from pathlib import Path
from jsonschema import Draft202012Validator

from regocr.countries.sa.module import SaudiCountryModule, parse_flexible_date
from regocr.ocr_engines import OCRWord

ROOT = Path(__file__).resolve().parent.parent


def _make_word(text: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.95):
    return OCRWord(text=text, confidence=conf, bbox=[x1, y1, x2, y2])


def _synthetic_cr_page(company_name="شركة الرياض للتقنية", unified="7012345678"):
    text = (
        "المملكة العربية السعودية\n"
        "وزارة التجارة\n"
        "شهادة السجل التجاري\n"
        f"{company_name}\n"
        "نوع الكيان: مؤسسة فردية\n"
        f"الرقم الوطني الموحد: {unified}\n"
        "تاريخ الإصدار: 15/02/2023\n"
        "حالة السجل: نشط\n"
        "البيانات الأساسية للسجل التجاري\n"
    )
    words = [
        _make_word("المملكة", 0.40, 0.05, 0.60, 0.08),
        _make_word("وزارة", 0.45, 0.09, 0.55, 0.12),
        _make_word("التجارة", 0.45, 0.09, 0.55, 0.12),
        _make_word("شهادة", 0.40, 0.13, 0.60, 0.16),
        _make_word("السجل", 0.40, 0.13, 0.60, 0.16),
        _make_word("التجاري", 0.40, 0.13, 0.60, 0.16),
        _make_word(company_name, 0.30, 0.18, 0.70, 0.22),
        _make_word("نوع الكيان: مؤسسة", 0.20, 0.24, 0.80, 0.27),
        _make_word(f"الرقم الوطني الموحد: {unified}", 0.20, 0.29, 0.80, 0.32),
        _make_word("تاريخ الإصدار: 15/02/2023", 0.20, 0.34, 0.80, 0.37),
        _make_word("حالة السجل: نشط", 0.20, 0.39, 0.80, 0.42),
    ]
    return {"page_num": 1, "raw_text": text, "words": words, "image": None}


def _synthetic_vat_page(taxpayer="شركة الرياض للتقنية", vat_num="310123456700003", tin="1012345670"):
    text = (
        "هيئة الزكاة والضريبة والجمارك\n"
        "شهادة تسجيل في ضريبة القيمة المضافة\n"
        f"الرقم المميز TIN: {tin}\n"
        "رقم الشهادة Certificate No.: 100200300400\n"
        "تاريخ الشهادة Certificate date: 15/02/2023\n"
        f"اسم المكلف: {taxpayer}\n"
        f"رقم التسجيل الضريبي: {vat_num}\n"
        "تاريخ نفاذ التسجيل: 2023/02/15\n"
        "عنوان المكلف: الرياض، المملكة العربية السعودية\n"
        "رقم السجل التجاري / الرخصة: 7012345678\n"
        "الفترة الضريبية: ربع سنوي\n"
        "تاريخ استحقاق أول إقرار ضريبي: 2023/05/31\n"
        "zatca.gov.sa 9988776655\n"
    )
    words = [
        _make_word("هيئة الزكاة والضريبة والجمارك", 0.30, 0.05, 0.70, 0.08),
        _make_word("شهادة تسجيل في ضريبة القيمة المضافة", 0.25, 0.09, 0.75, 0.12),
        _make_word(f"TIN: {tin}", 0.10, 0.15, 0.40, 0.18),
        _make_word("Certificate No: 100200300400", 0.10, 0.19, 0.40, 0.22),
        _make_word("Certificate date: 15/02/2023", 0.10, 0.23, 0.40, 0.26),
        _make_word(taxpayer, 0.35, 0.35, 0.65, 0.38),
        _make_word(vat_num, 0.35, 0.39, 0.65, 0.42),
        _make_word("2023/02/15", 0.35, 0.43, 0.65, 0.46),
        _make_word("الرياض", 0.35, 0.47, 0.65, 0.50),
        _make_word("7012345678", 0.35, 0.50, 0.65, 0.53),
        _make_word("ربع سنوي", 0.35, 0.54, 0.65, 0.57),
        _make_word("2023/05/31", 0.35, 0.58, 0.65, 0.61),
        # footer with serial number that should be ignored
        _make_word("zatca.gov.sa 9988776655", 0.10, 0.94, 0.90, 0.98),
    ]
    return {"page_num": 2, "raw_text": text, "words": words, "image": None}


def test_saudi_both_pages_accepted_and_schema_valid():
    """Requirement: Both pages present and valid -> accepted, schema conforms to 2020-12."""
    mod = SaudiCountryModule()
    p1 = _synthetic_cr_page()
    p2 = _synthetic_vat_page()
    pkg = mod.process_package([p1, p2])

    assert pkg["status"] == "accepted", f"Expected accepted, got {pkg['status']} reasons={pkg['reasons']}"
    assert pkg["package_status"] == "accepted"
    assert len(pkg["documents"]) == 2

    # Schema validation
    schema_path = ROOT / "schemas" / "sa" / "sa_registration.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(pkg))
    assert not errors, f"Schema validation errors: {[e.message for e in errors]}"


def test_saudi_one_page_only_needs_review():
    """Requirement: Missing one of the required documents returns needs_review with precise reason."""
    mod = SaudiCountryModule()
    p1 = _synthetic_cr_page()
    pkg = mod.process_package([p1])

    assert pkg["status"] == "needs_review"
    assert "missing_document:SA_VAT_CERT" in pkg["reasons"]
    assert pkg["reupload_hint"] is not None

    p2 = _synthetic_vat_page()
    pkg2 = mod.process_package([p2])
    assert pkg2["status"] == "needs_review"
    assert "missing_document:SA_CR_CERT" in pkg2["reasons"]


def test_saudi_swapped_order():
    """Requirement: Page order is independent."""
    mod = SaudiCountryModule()
    p1 = _synthetic_cr_page()
    p2 = _synthetic_vat_page()
    # Feed VAT first, then CR
    pkg = mod.process_package([p2, p1])
    assert pkg["status"] == "accepted"
    assert len(pkg["documents"]) == 2


def test_saudi_duplicate_page_type():
    """Requirement: Duplicate pages of same type -> needs_review."""
    mod = SaudiCountryModule()
    p1 = _synthetic_cr_page()
    p1_dup = _synthetic_cr_page()
    pkg = mod.process_package([p1, p1_dup])

    assert pkg["status"] == "needs_review"
    assert any("duplicate_document_type:SA_CR_CERT" in r for r in pkg["reasons"])


def test_saudi_masked_unified_number_needs_review_not_rejected():
    """Requirement: Genuine but partially masked document ends in needs_review, not rejected."""
    mod = SaudiCountryModule()
    # Mask unified number with asterisks
    p1 = _synthetic_cr_page(unified="70123****")
    p2 = _synthetic_vat_page()
    pkg = mod.process_package([p1, p2])

    assert pkg["status"] == "needs_review", f"Expected needs_review, got {pkg['status']}"
    assert pkg["status"] != "rejected"


def test_saudi_company_name_mismatch():
    """Requirement: Mismatched company names between CR and VAT triggers review."""
    mod = SaudiCountryModule()
    p1 = _synthetic_cr_page(company_name="شركة النور اللوجستية")
    p2 = _synthetic_vat_page(taxpayer="شركة الأمل للتقنية والبرمجيات")
    pkg = mod.process_package([p1, p2])

    assert "cross_doc_mismatch:company_name" in pkg["reasons"]
    assert "company_name" in pkg["handoff"]["suspect_fields"]


def test_saudi_vat_number_format_and_tin_prefix():
    """Requirement: VAT number must be 15 digits; starts with TIN prefix."""
    mod = SaudiCountryModule()
    # VAT number starting with 3999 while TIN is 10123...
    p1 = _synthetic_cr_page()
    p2 = _synthetic_vat_page(vat_num="399999999900003", tin="1012345670")
    pkg = mod.process_package([p1, p2])

    assert any("vat_number_tin_prefix_mismatch" in r for r in pkg["reasons"])


def test_saudi_date_orders_and_arabic_indic_digits():
    """Requirement: Date parsing in both orders and Arabic-Indic digits."""
    # Western YYYY/MM/DD
    d1, c1 = parse_flexible_date("2024/05/15")
    assert d1 == "2024-05-15"
    assert c1 == "gregorian"

    # Western DD/MM/YYYY
    d2, c2 = parse_flexible_date("15/05/2024")
    assert d2 == "2024-05-15"
    assert c2 == "gregorian"

    # Arabic-Indic digits
    d3, c3 = parse_flexible_date("١٥/٠٥/٢٠٢٤")
    assert d3 == "2024-05-15"
    assert c3 == "gregorian"

    # Hijri date (1300-1499)
    d4, c4 = parse_flexible_date("1445/10/07")
    assert d4 == "1445-10-07"
    assert c4 == "hijri"


def test_saudi_footer_serial_regression():
    """Requirement: A footer serial number next to zatca.gov.sa is never misextracted as VAT / CR number."""
    mod = SaudiCountryModule()
    p2 = _synthetic_vat_page()
    doc_vat = mod.extract_vat_cert(p2["raw_text"], p2["words"])

    # Footer serial was 9988776655 in our fixture
    vat_val = doc_vat["fields"]["vat_registration_number"]["value"]
    cr_val = doc_vat["fields"]["cr_license_id_no"]["value"]
    assert vat_val != "9988776655"
    assert cr_val != "9988776655"


def test_wrong_country_egyptian_uploaded_as_saudi():
    """Requirement: Egyptian synthetic document uploaded as Saudi returns missing_document / unrecognized."""
    mod = SaudiCountryModule()
    eg_text = "جمهورية مصر العربية وزارة التموين والتجارة الداخلية السجل التجاري رقم القيد 621097"
    classified = mod.classify_page(eg_text)
    # Egyptian document must not classify as SA_CR_CERT or SA_VAT_CERT
    assert classified is None


def test_private_fixtures_saudi_if_env_set():
    """Assertion that real Saudi files in PRIVATE_FIXTURES_DIR are not rejected."""
    priv_dir = os.environ.get("PRIVATE_FIXTURES_DIR")
    if not priv_dir or not os.path.exists(priv_dir):
        return  # Auto-skip cleanly

    p1_path = Path(priv_dir) / "page1-real.jpeg"
    p2_path = Path(priv_dir) / "page2-real.jpeg"
    if not (p1_path.exists() and p2_path.exists()):
        return

    from regocr.pipeline import RegOCRPipeline
    pipe = RegOCRPipeline(country="sa")
    res = pipe.process_files([p1_path, p2_path])

    # Core acceptance criterion 1:
    assert res.status != "rejected", f"Real Saudi pages must not be rejected! Reasons: {res.reasons}"
    assert res.package_status != "rejected"
    assert res.documents is not None
    assert len(res.documents) == 2


if __name__ == "__main__":
    tests = [
        test_saudi_both_pages_accepted_and_schema_valid,
        test_saudi_one_page_only_needs_review,
        test_saudi_swapped_order,
        test_saudi_duplicate_page_type,
        test_saudi_masked_unified_number_needs_review_not_rejected,
        test_saudi_company_name_mismatch,
        test_saudi_vat_number_format_and_tin_prefix,
        test_saudi_date_orders_and_arabic_indic_digits,
        test_saudi_footer_serial_regression,
        test_wrong_country_egyptian_uploaded_as_saudi,
        test_private_fixtures_saudi_if_env_set,
    ]
    for fn in tests:
        fn()
        print(f"  [PASS] {fn.__name__}")
    print(f"\nALL {len(tests)} SAUDI PACKAGE TESTS PASSED!")
