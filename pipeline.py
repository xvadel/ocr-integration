#!/usr/bin/env python3
"""
pipeline.py - Production-Grade Registration Flow & OCR Processing Pipeline.

Processes Egyptian Commercial Register documents (Images and PDFs) through:
  1. File Input & Type Detection (detect_pdf_kind.py)
  2. OpenCV Image Preprocessing & Quality Gate (preprocess.py)
  3. Text Extraction (Direct Text Layer for digital PDFs, or Tesseract/EasyOCR)
  4. Field Extraction (Regex, Anchors, and Structural Matchers)
  5. Normalization (Eastern/Indic to Western digits, ISO-8601 dates, numeric capital)
  6. Grounding Verification (Ensuring every extracted value physically exists in raw text)
  7. Strict JSON Schema Validation against registration_schema.json
  8. Decision Routing (rejected > needs_review > accepted)

CLI Usage:
  python pipeline.py path/to/document.pdf
  python pipeline.py images/ --out output_results.json
  python pipeline.py file.jpg --engine tesseract
  python pipeline.py file.jpg --engine easyocr
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from jsonschema import Draft202012Validator
from PIL import Image

from detect_pdf_kind import detect_pdf_kind, norm_arabic
from preprocess import assess_image_quality, preprocess_image, to_cv2_image

# Configure default Tesseract path on Windows if not already in PATH
if sys.platform == "win32":
    default_win_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_win_tess):
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = default_win_tess
        except ImportError:
            pass

HERE = Path(__file__).parent
SCHEMA_PATH = HERE / "registration_schema.json"
if not SCHEMA_PATH.exists():
    SCHEMA_PATH = HERE / "registration_output.schema.json"

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

GOVERNORATES = [
    "القاهرة", "الجيزة", "الإسكندرية", "الدقهلية", "الشرقية", "الغربية",
    "دمياط", "القليوبية", "بورسعيد", "السويس", "الإسماعيلية", "البحيرة",
    "كفر الشيخ", "المنوفية", "بني سويف", "الفيوم", "المنيا", "أسيوط",
    "سوهاج", "قنا", "الأقصر", "أسوان", "مطروح", "الوادي الجديد",
    "البحر الأحمر", "شمال سيناء", "جنوب سيناء"
]

LEGAL_FORM_CODES = {
    "شركة ذات مسئولية محدودة": "llc",
    "شركة مساهمة مصرية": "jsc",
    "منشأة فردية": "sole_proprietorship",
    "شركة تضامن": "partnership",
    "شركة توصية بسيطة": "limited_partnership",
}


# -----------------------------------------------------------------------------
# Normalization Utilities
# -----------------------------------------------------------------------------
def to_ascii_digits(text: str) -> str:
    """Converts Arabic-Indic & Persian numerals to standard Latin/Western digits."""
    if not text:
        return ""
    return str(text).translate(AR_DIGITS)


def parse_iso_date(raw_date: str) -> Optional[str]:
    """
    Standardizes dates in DD/MM/YYYY, DD-MM-YYYY, YYYY/MM/DD, or YYYY-MM-DD to ISO 'YYYY-MM-DD'.
    Handles both Latin and Arabic-Indic numerals.
    """
    if not raw_date:
        return None
    cleaned = to_ascii_digits(raw_date).strip()
    cleaned = re.sub(r"[^\d/\-.]", "", cleaned)
    parts = [p for p in re.split(r"[/.\-]", cleaned) if p]
    if len(parts) != 3:
        return None

    # Determine day, month, year
    try:
        if len(parts[0]) == 4:
            # YYYY-MM-DD
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts[2]) == 4:
            # DD-MM-YYYY
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts[2]) == 2:
            # DD-MM-YY -> assume 20YY
            day, month, year = int(parts[0]), int(parts[1]), 2000 + int(parts[2])
        else:
            return None

        # Validate calendar bounds
        dt = datetime.date(year, month, day)
        return dt.isoformat()
    except (ValueError, OverflowError):
        return None


def parse_numeric_capital(raw_val: str) -> Optional[float]:
    """
    Extracts numeric capital value from strings like '1,000,000 جنيه مصري' or '١٬٠٠٠٬٠٠٠'.
    Returns float or int conforming to jsonschema type 'number'.
    """
    if not raw_val:
        return None
    norm_txt = to_ascii_digits(raw_val)
    # Remove comma, arabic thousand separator, and text
    cleaned = norm_txt.replace(",", "").replace("٬", "")
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if match:
        val = float(match.group(1))
        return int(val) if val.is_integer() else val
    return None


# -----------------------------------------------------------------------------
# Grounding Check
# -----------------------------------------------------------------------------
def check_grounding(extracted_value: Any, raw_value: Optional[str], raw_text: Optional[str]) -> bool:
    """
    Verifies that the extracted and normalized value physically exists in the raw text.
    Checks:
      1. Exact match of raw_value in raw_text
      2. Normalized Arabic match
      3. Digit sequence matching for numbers, dates, tax IDs
    """
    if not raw_text:
        return False
    if extracted_value is None and raw_value is None:
        return False

    raw_text_clean = raw_text
    raw_text_norm = norm_arabic(raw_text)
    raw_digits = re.sub(r"\D", "", to_ascii_digits(raw_text))

    # 1. Direct raw check
    if raw_value and raw_value in raw_text_clean:
        return True

    # 2. Check normalized text
    if raw_value:
        norm_raw = norm_arabic(raw_value)
        if norm_raw and norm_raw in raw_text_norm:
            return True

    # 3. For numeric strings / numbers (registration number, capital, tax ID)
    if extracted_value is not None:
        ext_digits = re.sub(r"\D", "", to_ascii_digits(str(extracted_value)))
        if ext_digits and len(ext_digits) >= 3 and ext_digits in raw_digits:
            return True

    # 4. For dates: check if day, month, year appear together or nearby
    if isinstance(extracted_value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", extracted_value):
        y, m, d = extracted_value.split("-")
        # Check DD/MM/YYYY or YYYY/MM/DD in text
        patterns = [
            f"{d}/{m}/{y}", f"{int(d)}/{int(m)}/{y}",
            f"{y}/{m}/{d}", f"{y}-{m}-{d}", f"{d}-{m}-{y}"
        ]
        text_ascii = to_ascii_digits(raw_text_clean)
        for p in patterns:
            if p in text_ascii:
                return True

    return False


# -----------------------------------------------------------------------------
# Field Extractor
# -----------------------------------------------------------------------------
class DocumentFieldExtractor:
    """Extracts structured fields from Egyptian Commercial Register document texts."""

    def __init__(self, raw_text: str):
        self.raw_text = raw_text or ""
        self.lines = [line.strip() for line in self.raw_text.splitlines() if line.strip()]
        self.text_norm = norm_arabic(self.raw_text)
        self.text_ascii = to_ascii_digits(self.raw_text)

    def extract_all(self) -> Dict[str, Dict[str, Any]]:
        fields = {}

        # 1. Company Name
        fields["company_name"] = self._extract_company_name()

        # 2. Legal Form
        fields["legal_form"] = self._extract_legal_form()

        # 3. Register Number
        fields["register_number"] = self._extract_register_number()

        # 4. Issue Date & Expiry Date
        issue_d, expiry_d = self._extract_dates()
        fields["issue_date"] = issue_d
        fields["expiry_date"] = expiry_d

        # 5. Activity
        fields["activity"] = self._extract_activity()

        # 6. Governorate
        fields["governorate"] = self._extract_governorate()

        # 7. Address
        fields["address"] = self._extract_address()

        # 8. Capital
        fields["capital"] = self._extract_capital()

        # 9. Manager
        fields["manager"] = self._extract_manager()

        # 10. Tax ID / Tax Card Number
        fields["tax_id"] = self._extract_tax_id()

        # 11. Issuing Office
        fields["issuing_office"] = self._extract_office()

        return fields

    def _extract_company_name(self) -> Dict[str, Any]:
        # Search for company name patterns
        anchors = ["اسم الشركة / المنشأة", "اسم الشركة", "اسم المنشأة", "الاسم التجاري", "اسم الشركه"]
        raw = self._find_value_near_anchor(anchors)
        is_legal_form = any(norm_arabic(lf) in norm_arabic(raw or "") for lf in ["مسئولية محدودة", "مساهمة مصرية", "منشأة فردية", "تضامن"])
        if is_legal_form or any(a in (raw or "") for a in ["رقم القيد", "تاريخ", "الشكل القانوني"]):
            raw = None

        # Fallback: look for lines starting with 'شركة' or 'مؤسسة'
        if not raw or len(raw) < 5:
            for ln in self.lines:
                clean = re.sub(r"^[^\u0621-\u064A]*", "", ln).strip()
                if (clean.startswith("شركة ") or clean.startswith("مؤسسة ")) and len(clean) >= 8:
                    if not any(norm_arabic(stop) in norm_arabic(clean) for stop in ["مساهمة", "مسئولية محدودة", "تضامن", "وهمي", "للاختبار"]):
                        raw = clean
                        break

        if raw:
            # Clean unwanted artifacts
            raw = re.sub(r"^(?:اسم الشركة / المنشأة|اسم الشركة|الاسم التجاري)\s*[:\-]?\s*", "", raw).strip()
            # Remove test notices if present
            raw = re.sub(r"\s*(?:نموذج وهمي|للاختبار فقط).*$", "", raw).strip()
            grounded = check_grounding(raw, raw, self.raw_text)
            return {
                "value": raw if raw else None,
                "raw": raw if raw else None,
                "status": "ok" if (raw and grounded) else ("suspect" if raw else "missing"),
                "confidence": 0.95 if grounded else 0.5,
                "grounded": grounded,
                "issues": [] if grounded else (["not_grounded"] if raw else ["not_found"]),
                "page": 1
            }

        return {
            "value": None, "raw": None, "status": "missing",
            "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1
        }

    def _extract_legal_form(self) -> Dict[str, Any]:
        anchors = ["الشكل القانوني", "الكيان القانوني", "نوع الشركة"]
        raw = self._find_value_near_anchor(anchors)
        found_form = None
        code = None

        search_corpus = f"{raw or ''} {self.raw_text}"
        for form_name, form_code in LEGAL_FORM_CODES.items():
            if norm_arabic(form_name) in norm_arabic(search_corpus):
                found_form = form_name
                code = form_code
                break

        if found_form:
            grounded = check_grounding(found_form, raw or found_form, self.raw_text)
            return {
                "value": found_form,
                "raw": raw or found_form,
                "code": code,
                "status": "ok" if grounded else "suspect",
                "confidence": 0.95 if grounded else 0.5,
                "grounded": grounded,
                "issues": [] if grounded else ["not_grounded"],
                "page": 1
            }

        return {
            "value": None, "raw": None, "status": "missing",
            "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1
        }

    def _extract_register_number(self) -> Dict[str, Any]:
        anchors = ["رقم القيد بالسجل التجاري", "رقم القيد", "رقم السجل التجاري", "رقم السجل"]
        raw = self._find_value_near_anchor(anchors)

        cand_digits = None
        if raw and not any(kw in raw for kw in ["تاريخ", "بطاقة", "إصدار", "انتهاء"]):
            clean_raw = re.sub(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}", "", raw)
            clean_raw = re.sub(r"\d{3}-\d{3}-\d{3}", "", clean_raw)
            m = re.search(r"([0-9٠-٩]{3,12})", clean_raw)
            if m:
                cand_digits = m.group(1)

        # Fallback: scan lines near 'رقم القيد' (strictly excluding 'تاريخ')
        if not cand_digits:
            for i, line in enumerate(self.lines):
                if ("رقم القيد" in line or ("السجل" in line and "رقم" in line)) and "تاريخ" not in line:
                    for j in range(i, min(len(self.lines), i + 3)):
                        cand_ln = self.lines[j]
                        if "تاريخ" in cand_ln or "/" in cand_ln:
                            continue
                        clean_ln = re.sub(r"\d{3}-\d{3}-\d{3}", "", cand_ln)
                        m = re.search(r"\b([0-9٠-٩]{4,10})\b", clean_ln)
                        if m:
                            cand_digits = m.group(1)
                            raw = cand_digits
                            break
                    if cand_digits:
                        break

        # Standalone search for 5-8 digit sequence
        if not cand_digits:
            for token in re.findall(r"\b([0-9٠-٩]{5,7})\b", self.raw_text):
                # Ensure it's not part of a date or tax id
                if not re.search(r"[/\-.]", token):
                    cand_digits = token
                    raw = token
                    break

        if cand_digits:
            val = to_ascii_digits(cand_digits)
            is_valid_format = bool(re.fullmatch(r"\d{3,12}", val))
            grounded = check_grounding(val, raw, self.raw_text)
            status = "ok" if (is_valid_format and grounded) else "suspect"
            issues = []
            if not is_valid_format:
                issues.append("format_mismatch")
            if not grounded:
                issues.append("not_grounded")

            return {
                "value": val,
                "raw": raw,
                "status": status,
                "confidence": 0.95 if status == "ok" else 0.5,
                "grounded": grounded,
                "issues": issues,
                "page": 1
            }

        return {
            "value": None, "raw": None, "status": "missing",
            "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1
        }

    def _extract_dates(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Find all date candidates in text
        date_pattern = r"(?<!\d)([0-9٠-٩]{1,4}[/\-.][0-9٠-٩]{1,2}[/\-.][0-9٠-٩]{1,4})(?!\d)"
        all_matches = re.findall(date_pattern, self.raw_text)

        parsed_dates = []
        for d_str in all_matches:
            iso = parse_iso_date(d_str)
            if iso:
                parsed_dates.append((d_str, iso))

        # Anchor searches
        issue_raw = self._find_value_near_anchor(["تاريخ القيد", "تاريخ الإصدار", "تاريخ التسجيل"])
        expiry_raw = self._find_value_near_anchor(["تاريخ انتهاء الصلاحية", "تاريخ الانتهاء", "تاريخ انتهاء القيد"])

        issue_iso = parse_iso_date(issue_raw) if issue_raw else None
        expiry_iso = parse_iso_date(expiry_raw) if expiry_raw else None

        # Disambiguate dates using chronological logic if needed
        if not issue_iso or not expiry_iso:
            if len(parsed_dates) >= 2:
                # Sort unique dates chronologically
                sorted_dates = sorted(list(dict.fromkeys(parsed_dates)), key=lambda x: x[1])
                if not issue_iso:
                    issue_raw, issue_iso = sorted_dates[0]
                if not expiry_iso:
                    expiry_raw, expiry_iso = sorted_dates[-1]
            elif len(parsed_dates) == 1:
                if not issue_iso:
                    issue_raw, issue_iso = parsed_dates[0]

        # Build issue_date field
        if issue_iso:
            grounded = check_grounding(issue_iso, issue_raw, self.raw_text)
            issue_field = {
                "value": issue_iso,
                "raw": issue_raw,
                "status": "ok" if grounded else "suspect",
                "confidence": 0.92 if grounded else 0.45,
                "grounded": grounded,
                "issues": [] if grounded else ["not_grounded"],
                "page": 1
            }
        else:
            issue_field = {
                "value": None, "raw": issue_raw,
                "status": "suspect" if issue_raw else "missing",
                "confidence": 0.3 if issue_raw else None,
                "grounded": False,
                "issues": ["format_mismatch"] if issue_raw else ["not_found"],
                "page": 1
            }

        # Build expiry_date field
        if expiry_iso:
            grounded = check_grounding(expiry_iso, expiry_raw, self.raw_text)
            expiry_field = {
                "value": expiry_iso,
                "raw": expiry_raw,
                "status": "ok" if grounded else "suspect",
                "confidence": 0.90 if grounded else 0.45,
                "grounded": grounded,
                "issues": [] if grounded else ["not_grounded"],
                "page": 1
            }
        else:
            expiry_field = {
                "value": None, "raw": expiry_raw,
                "status": "suspect" if expiry_raw else "missing",
                "confidence": 0.3 if expiry_raw else None,
                "grounded": False,
                "issues": ["format_mismatch"] if expiry_raw else ["not_found"],
                "page": 1
            }

        # Logical timeline check: expiry must be after issue date
        if issue_field["value"] and expiry_field["value"]:
            if expiry_field["value"] <= issue_field["value"]:
                expiry_field["status"] = "suspect"
                expiry_field["issues"].append("expiry_precedes_issue")

        return issue_field, expiry_field

    def _extract_activity(self) -> Dict[str, Any]:
        raw = self._find_value_near_anchor(["النشاط", "الغرض", "نوع النشاط"])
        if raw:
            raw = re.sub(r"^(?:النشاط|الغرض)\s*[:\-]?\s*", "", raw).strip()
            grounded = check_grounding(raw, raw, self.raw_text)
            return {
                "value": raw, "raw": raw, "status": "ok" if grounded else "suspect",
                "confidence": 0.90, "grounded": grounded, "issues": [] if grounded else ["not_grounded"], "page": 1
            }
        return {"value": None, "raw": None, "status": "missing", "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1}

    def _extract_governorate(self) -> Dict[str, Any]:
        raw = self._find_value_near_anchor(["المحافظة", "محافظة"])
        gov_match = None
        for gov in GOVERNORATES:
            if gov in (raw or "") or gov in self.raw_text:
                gov_match = gov
                break
        if gov_match:
            grounded = check_grounding(gov_match, raw or gov_match, self.raw_text)
            return {
                "value": gov_match, "raw": raw or gov_match, "status": "ok" if grounded else "suspect",
                "confidence": 0.95, "grounded": grounded, "issues": [] if grounded else ["not_grounded"], "page": 1
            }
        return {"value": None, "raw": None, "status": "missing", "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1}

    def _extract_address(self) -> Dict[str, Any]:
        raw = self._find_value_near_anchor(["العنوان", "المقر"])
        if raw:
            raw = re.sub(r"^(?:العنوان|المقر)\s*[:\-]?\s*", "", raw).strip()
            grounded = check_grounding(raw, raw, self.raw_text)
            return {
                "value": to_ascii_digits(raw), "raw": raw, "status": "ok" if grounded else "suspect",
                "confidence": 0.88, "grounded": grounded, "issues": [] if grounded else ["not_grounded"], "page": 1
            }
        return {"value": None, "raw": None, "status": "missing", "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1}

    def _extract_capital(self) -> Dict[str, Any]:
        anchors = ["رأس المال", "رأس مال الشركة", "رأس مال"]
        raw = self._find_value_near_anchor(anchors)
        if not raw:
            m = re.search(r"([0-9٠-٩,٬.]+\s*(?:جنيه|جنيه مصري|EGP|LE))", self.raw_text)
            if m:
                raw = m.group(1)

        if raw:
            val = parse_numeric_capital(raw)
            if val is not None:
                grounded = check_grounding(val, raw, self.raw_text)
                return {
                    "value": val,
                    "raw": raw,
                    "unit": "EGP",
                    "status": "ok" if grounded else "suspect",
                    "confidence": 0.92 if grounded else 0.5,
                    "grounded": grounded,
                    "issues": [] if grounded else ["not_grounded"],
                    "page": 1
                }
            return {
                "value": None, "raw": raw, "status": "suspect", "unit": "EGP",
                "confidence": 0.4, "grounded": False, "issues": ["format_mismatch"], "page": 1
            }

        return {"value": None, "raw": None, "status": "missing", "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1}

    def _extract_manager(self) -> Dict[str, Any]:
        raw = self._find_value_near_anchor(["المدير / المالك", "المدير المسئول", "المدير", "صاحب المنشأة", "المالك"])
        if raw:
            raw = re.sub(r"^(?:المدير / المالك|المدير|المالك)\s*[:\-]?\s*", "", raw).strip()
            # Clean unwanted tokens
            raw = re.sub(r"\s*(?:رقم البطاقة|تاريخ|السجل).*$", "", raw).strip()
            grounded = check_grounding(raw, raw, self.raw_text)
            return {
                "value": raw, "raw": raw, "status": "ok" if grounded else "suspect",
                "confidence": 0.90, "grounded": grounded, "issues": [] if grounded else ["not_grounded"], "page": 1
            }
        return {"value": None, "raw": None, "status": "missing", "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1}

    def _extract_tax_id(self) -> Dict[str, Any]:
        anchors = ["رقم البطاقة الضريبية", "البطاقة الضريبية", "رقم التسجيل الضريبي"]
        raw = self._find_value_near_anchor(anchors)

        tax_cand = None
        # Pattern XXX-XXX-XXX
        tax_re = r"([0-9٠-٩]{3}[-\s][0-9٠-٩]{3}[-\s][0-9٠-٩]{3})"
        if raw:
            m = re.search(tax_re, raw)
            if m:
                tax_cand = m.group(1)

        if not tax_cand:
            m = re.search(tax_re, self.raw_text)
            if m:
                tax_cand = m.group(1)
                raw = tax_cand

        if tax_cand:
            val = to_ascii_digits(tax_cand).replace(" ", "-")
            grounded = check_grounding(val, raw, self.raw_text)
            return {
                "value": val,
                "raw": raw,
                "status": "ok" if grounded else "suspect",
                "confidence": 0.95 if grounded else 0.5,
                "grounded": grounded,
                "issues": [] if grounded else ["not_grounded"],
                "page": 1
            }

        return {"value": None, "raw": None, "status": "missing", "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1}

    def _extract_office(self) -> Dict[str, Any]:
        raw = self._find_value_near_anchor(["جهة الإصدار"])
        if not raw:
            for ln in self.lines:
                if ln.startswith("مكتب السجل التجاري"):
                    raw = ln
                    break
        if raw:
            raw = re.sub(r"^جهة الإصدار\s*[:\-]?\s*", "", raw).strip()
            grounded = check_grounding(raw, raw, self.raw_text)
            return {
                "value": raw, "raw": raw, "status": "ok" if grounded else "suspect",
                "confidence": 0.90, "grounded": grounded, "issues": [] if grounded else ["not_grounded"], "page": 1
            }
        return {"value": None, "raw": None, "status": "missing", "confidence": None, "grounded": False, "issues": ["not_found"], "page": 1}

    def _find_value_near_anchor(self, anchors: List[str]) -> Optional[str]:
        doc_labels = [
            "اسم الشركة", "الاسم التجاري", "اسم المنشأة", "الشكل القانوني", "الكيان القانوني",
            "رقم القيد", "رقم السجل", "السجل التجاري", "تاريخ القيد", "تاريخ الإصدار",
            "النشاط", "الغرض", "المحافظة", "العنوان", "المقر", "رأس المال", "رأس مال",
            "المدير", "المالك", "رقم البطاقة الضريبية", "البطاقة الضريبية",
            "تاريخ انتهاء", "جهة الإصدار"
        ]
        delimiters = " :-\t|/،,؛"
        for i, line in enumerate(self.lines):
            for anchor in anchors:
                if anchor in line:
                    # 1. Check if value is on the same line after delimiter
                    rem = line.split(anchor, 1)[1].strip(delimiters)
                    if rem and len(rem) >= 2 and not any(lbl in rem for lbl in doc_labels if lbl != anchor):
                        return rem
                    # 2. Check if value is on the same line before delimiter (visual RTL order)
                    pre = line.split(anchor, 1)[0].strip(delimiters)
                    if pre and len(pre) >= 2 and not any(lbl in pre for lbl in doc_labels if lbl != anchor):
                        # Make sure pre is not just a document header
                        if not any(hdr in pre for hdr in ["نموذج وهمي", "سجل تجاري"]):
                            return pre
                    # 3. Else check subsequent lines (skip label parts like 'التجاري' or 'المنشأة')
                    for j in range(i + 1, min(len(self.lines), i + 4)):
                        cand = self.lines[j].strip(delimiters)
                        if cand:
                            # Skip if this line is just continuation of anchor title
                            if cand in ("التجاري", "المنشأة", "الصلاحية"):
                                continue
                            # Skip if this line is another known field label
                            if any(lbl in cand for lbl in doc_labels if cand.startswith(lbl)):
                                break
                            return cand
        return None


# -----------------------------------------------------------------------------
# Decision Logic & Status Routing
# -----------------------------------------------------------------------------
def evaluate_document_status(
    quality: Dict[str, Any],
    fields: Dict[str, Dict[str, Any]],
    ocr_confidence: Optional[float] = None
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Evaluates priority status: rejected > needs_review > accepted.

    Returns:
      (status, reasons, review_object)
    """
    reasons: List[str] = []
    fields_to_review: List[str] = []
    q_score = quality.get("score", 1.0)
    q_issues = quality.get("issues", [])

    core_keys = ["company_name", "register_number", "issue_date"]

    # 1. Critical Quality Check (Direct Rejection Gate)
    if q_score < 0.35 or ("blurry" in q_issues and "noisy" in q_issues and "low_resolution" in q_issues):
        reasons.append("image_quality_too_low")
        return "rejected", reasons, {"required": False, "fields_to_review": [], "reviewed_by": None, "reviewed_at": None}

    # 2. Critical Core Fields Missing (Rejection)
    missing_cores = [k for k in core_keys if fields.get(k, {}).get("status") == "missing" or fields.get(k, {}).get("value") is None]
    if missing_cores:
        reasons.append("core_fields_missing")
        for k in missing_cores:
            reasons.append(f"critical_field_missing:{k}")
        return "rejected", reasons, {"required": False, "fields_to_review": [], "reviewed_by": None, "reviewed_at": None}

    # 3. Grounding & Field Validity Checks (Needs Review)
    for k, v in fields.items():
        if v.get("status") == "suspect":
            fields_to_review.append(k)
            for iss in v.get("issues", []):
                reasons.append(f"{k}:{iss}")
        elif v.get("value") is not None and not v.get("grounded"):
            fields_to_review.append(k)
            reasons.append(f"{k}:not_grounded")

    # 4. Logical Timeline Error Check
    issue_val = fields.get("issue_date", {}).get("value")
    expiry_val = fields.get("expiry_date", {}).get("value")
    if issue_val and expiry_val and expiry_val <= issue_val:
        if "expiry_date" not in fields_to_review:
            fields_to_review.append("expiry_date")
        reasons.append("timeline_error:expiry_precedes_issue")

    # 5. Borderline Quality Check
    if q_score < 0.65:
        reasons.append("borderline_image_quality")
        if not fields_to_review:
            # Require review of core fields if quality is borderline
            fields_to_review.extend(core_keys)

    # 6. Borderline OCR Confidence Check
    if ocr_confidence is not None and ocr_confidence < 0.65:
        reasons.append("borderline_ocr_confidence")
        for k in core_keys:
            if k not in fields_to_review and fields.get(k, {}).get("value") is not None:
                fields_to_review.append(k)

    # Status Determination
    if fields_to_review or reasons:
        # Deduplicate
        fields_to_review = sorted(list(set(fields_to_review)))
        if not fields_to_review:
            fields_to_review = ["company_name"]
        return "needs_review", sorted(list(set(reasons))), {
            "required": True,
            "fields_to_review": fields_to_review,
            "reviewed_by": None,
            "reviewed_at": None
        }

    # Accepted: All clear
    return "accepted", [], {
        "required": False,
        "fields_to_review": [],
        "reviewed_by": None,
        "reviewed_at": None
    }


# -----------------------------------------------------------------------------
# Main Registration Pipeline
# -----------------------------------------------------------------------------
class RegistrationPipeline:
    """Full end-to-end processing pipeline for commercial register documents."""

    def __init__(self, engine: str = "auto", dpi: int = 250, tesseract_cmd: Optional[str] = None):
        self.engine = engine  # 'auto', 'tesseract', 'easyocr', 'paddleocr', 'pymupdf'
        self.dpi = dpi
        if tesseract_cmd:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._easyocr_reader = None
        self._paddle_ocr = None

        # Load Schema
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.validator = Draft202012Validator(self.schema)

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            import easyocr
            self._easyocr_reader = easyocr.Reader(["ar", "en"], gpu=False)
        return self._easyocr_reader

    def _get_paddle_ocr(self):
        """Initialize PaddleOCR once and reuse the model."""
        if self._paddle_ocr is None:
            try:
                import torch  # Prevents Windows DLL loading order issue with albumentations/shm.dll
            except ImportError:
                pass
            from paddleocr import PaddleOCR

            self._paddle_ocr = PaddleOCR(
                lang="ar",
                ocr_version="PP-OCRv5",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )

        return self._paddle_ocr

    def _run_paddleocr(self, image):
        """Run PaddleOCR and return normalized text + mean confidence."""
        ocr = self._get_paddle_ocr()
        prediction = ocr.predict(input=image)

        extracted_text = []
        confidence_scores = []

        for page in prediction:
            payload = page.json

            if isinstance(payload, str):
                payload = json.loads(payload)

            data = payload.get("res", payload)

            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])

            for index, text in enumerate(texts):
                if not text or not str(text).strip():
                    continue

                extracted_text.append(str(text).strip())

                if index < len(scores):
                    try:
                        score = float(scores[index])
                        if 0 <= score <= 1:
                            confidence_scores.append(score)
                    except (TypeError, ValueError):
                        pass

        raw_text = "\n".join(extracted_text)

        confidence = (
            round(sum(confidence_scores) / len(confidence_scores), 4)
            if confidence_scores else 0.0
        )

        return raw_text, confidence

    def _run_easyocr(self, image):
        """Run EasyOCR and return normalized text + mean confidence."""
        reader = self._get_easyocr_reader()

        results = reader.readtext(
            image,
            detail=1,
            paragraph=False
        )

        texts = []
        scores = []

        for result in results:
            if len(result) < 3:
                continue

            text = str(result[1]).strip()
            if text:
                texts.append(text)

            try:
                score = float(result[2])
                if score > 0:
                    scores.append(score)
            except (TypeError, ValueError):
                pass

        confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        return "\n".join(texts), confidence

    def _run_tesseract(self, image):
        """Run Tesseract and return normalized text + mean confidence."""
        import pytesseract

        raw_text = pytesseract.image_to_string(
            image,
            lang="ara+eng",
            config="--psm 4"
        )

        data = pytesseract.image_to_data(
            image,
            lang="ara+eng",
            config="--psm 4",
            output_type=pytesseract.Output.DICT
        )

        scores = []

        for confidence, text in zip(data["conf"], data["text"]):
            if str(confidence) == "-1" or not str(text).strip():
                continue

            try:
                score = float(confidence) / 100.0
                if score > 0:
                    scores.append(score)
            except (TypeError, ValueError):
                pass

        mean_confidence = (
            round(sum(scores) / len(scores), 4)
            if scores else 0.0
        )

        return raw_text, mean_confidence

    def _get_core_fields_count(self, raw_text):
        """Count how many required fields can be detected in OCR output."""
        if not raw_text or not raw_text.strip():
            return 0

        extractor = DocumentFieldExtractor(
            unicodedata.normalize("NFKC", raw_text)
        )
        fields = extractor.extract_all()

        core_keys = [
            "company_name",
            "register_number",
            "issue_date"
        ]

        return sum(
            1 for key in core_keys
            if fields.get(key, {}).get("value") is not None
            and fields.get(key, {}).get("status") != "missing"
        )

    def process_file(self, file_path: Union[str, Path], submission_id: Optional[str] = None) -> Dict[str, Any]:
        """Processes a single Image or PDF file and returns strict JSON output."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sub_id = submission_id or f"sub_{int(time.time())}_{file_path.stem[:12]}"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        is_pdf = file_path.suffix.lower() == ".pdf"
        t0 = time.perf_counter()

        # Step 1 & 2: File type check and detection
        if is_pdf:
            pdf_info = detect_pdf_kind(file_path)
            pdf_kind = pdf_info["pdf_kind"]
            pages_count = pdf_info["pages"]
            file_format = "pdf"
        else:
            pdf_kind = None
            pages_count = 1
            file_format = "image"

        # Determine extraction strategy
        raw_text = ""
        text_method = "ocr"
        used_engine = self.engine
        ocr_confidence_mean = None
        quality_score = 1.0
        quality_issues = []

        # -------------------------------------------------------------
        # Branch A: Digital PDF with valid text layer
        # -------------------------------------------------------------
        if is_pdf and pdf_kind in ("digital", "digital_garbled") and self.engine in ("auto", "pymupdf"):
            text_method = "text_layer"
            used_engine = pdf_info["recommended_engine"]
            if used_engine == "pdfplumber_bidi":
                from detect_pdf_kind import extract_bidi_text
                raw_text = extract_bidi_text(file_path)
            else:
                doc = pymupdf.open(str(file_path))
                raw_text = "\n".join(p.get_text() for p in doc)
                doc.close()

            raw_text = unicodedata.normalize("NFKC", raw_text)
            quality_score = 0.98
            quality_issues = []
            ocr_confidence_mean = None

        # -------------------------------------------------------------
        # Branch B: Scanned PDF or Standard Image -> OpenCV Preprocess + OCR
        # -------------------------------------------------------------
        else:
            text_method = "ocr"
            # Choose OCR engine
            if self.engine == "auto":
                chosen_ocr = "tesseract"
            else:
                chosen_ocr = self.engine

            used_engine = chosen_ocr

            # Render or Load Image
            if is_pdf:
                doc = pymupdf.open(str(file_path))
                page = doc[0]  # Use first page of document
                pix = page.get_pixmap(dpi=self.dpi, alpha=False)
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                doc.close()
            else:
                pil_img = Image.open(file_path)

            # OpenCV Preprocessing & Quality Gate
            preprocessed_cv, quality = preprocess_image(pil_img)
            quality_score = quality["score"]
            quality_issues = quality["issues"]

            # -------------------------------------------------------------
            # OCR Engine Selection + Fallback
            # -------------------------------------------------------------

            ocr_engines = {
                "paddleocr": self._run_paddleocr,
                "easyocr": self._run_easyocr,
                "tesseract": self._run_tesseract
            }

            # Auto mode:
            # PaddleOCR -> EasyOCR -> Tesseract
            if self.engine == "auto":
                engine_order = [
                    "paddleocr",
                    "easyocr",
                    "tesseract"
                ]
            else:
                # Explicit engine selection: run only the requested engine.
                engine_order = [self.engine]

            best_result = None
            best_engine = None
            best_score = (-1, -1.0)

            for engine_name in engine_order:
                print(f"[OCR] Trying {engine_name}...")

                try:
                    candidate_text, candidate_confidence = ocr_engines[engine_name](
                        preprocessed_cv
                    )

                    candidate_text = unicodedata.normalize(
                        "NFKC",
                        candidate_text or ""
                    )

                    core_count = self._get_core_fields_count(candidate_text)

                    # Prefer more detected core fields.
                    # If tied, prefer the higher OCR confidence.
                    candidate_score = (
                        core_count,
                        candidate_confidence
                    )

                    print(
                        f"[OCR] {engine_name}: "
                        f"core_fields={core_count}/3, "
                        f"confidence={candidate_confidence:.3f}, "
                        f"text_length={len(candidate_text)}"
                    )

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_result = candidate_text
                        best_engine = engine_name

                    # All three core fields found: no need for fallback.
                    if core_count == 3:
                        print(f"[OCR] {engine_name} succeeded. Fallback skipped.")
                        break

                    if self.engine != "auto":
                        break

                    print(
                        f"[OCR] {engine_name} did not detect all core fields. "
                        "Trying fallback..."
                    )

                except Exception as error:
                    print(f"[OCR ERROR] {engine_name}: {error}")

                    if self.engine != "auto":
                        raise

                    print(f"[OCR] Moving to next fallback engine...")

            # Use the best result among the attempted engines.
            if best_result is None:
                raise RuntimeError("All OCR engines failed to process the document.")

            raw_text = best_result
            ocr_confidence_mean = best_score[1]
            used_engine = best_engine

            print(
                f"[OCR] Selected engine: {used_engine} "
                f"(core fields: {best_score[0]}/3)"
            )

            raw_text = unicodedata.normalize("NFKC", raw_text)

        duration_ms = int((time.perf_counter() - t0) * 1000)

        # Step 3 & 4: Field Extraction & Grounding
        extractor = DocumentFieldExtractor(raw_text)
        fields = extractor.extract_all()

        # Step 6: Decision Logic & Status Routing
        status, reasons, review_obj = evaluate_document_status(
            quality={"score": quality_score, "issues": quality_issues},
            fields=fields,
            ocr_confidence=ocr_confidence_mean
        )

        # Build Final Schema-Conforming Document
        output_doc = {
            "schema_version": "0.1",
            "submission_id": sub_id,
            "created_at": now_iso,
            "test_data": True,
            "status": status,
            "reasons": reasons,
            "document": {
                "type": "commercial_register",
                "file_format": file_format,
                "pdf_kind": pdf_kind,
                "pages": pages_count,
                "page_used": 1,
                "quality": {
                    "score": quality_score,
                    "issues": quality_issues
                }
            },
            "extraction": {
                "text_method": text_method,
                "engine": used_engine,
                "languages": ["ar", "en"] if text_method == "ocr" else ["ar"],
                "ocr_confidence_mean": ocr_confidence_mean,
                "duration_ms": duration_ms
            },
            "fields": fields,
            "raw_text": raw_text,
            "review": review_obj
        }

        # Step 5: JSON Schema Validation
        validation_errors = list(self.validator.iter_errors(output_doc))
        if validation_errors:
            err_msgs = [f"Schema validation error at {list(e.path)}: {e.message}" for e in validation_errors]
            raise ValueError(f"Output failed JSON schema validation: {'; '.join(err_msgs)}")

        return output_doc


def main():
    parser = argparse.ArgumentParser(description="Process Commercial Register documents with OCR and Validation.")
    parser.add_argument("path", type=Path, help="Input file (Image or PDF) or directory")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON file or directory")
    parser.add_argument(
        "--engine",
        choices=["auto", "paddleocr", "tesseract", "easyocr", "pymupdf"],
        default="auto"
    )
    parser.add_argument("--dpi", type=int, default=250, help="Rendering DPI for PDF scans (200-300)")
    parser.add_argument("--tesseract-cmd", default=None, help="Path to tesseract binary")
    args = parser.parse_args()

    pipeline = RegistrationPipeline(engine=args.engine, dpi=args.dpi, tesseract_cmd=args.tesseract_cmd)

    target = args.path
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(list(target.glob("*.jpg")) + list(target.glob("*.png")) + list(target.glob("*.pdf")))
    else:
        sys.exit(f"Target path does not exist: {target}")

    results = {}
    print(f"\nProcessing {len(files)} file(s) with engine='{args.engine}'...\n")

    for f in files:
        try:
            doc = pipeline.process_file(f)
            results[f.name] = doc
            st = doc["status"].upper()
            q = doc["document"]["quality"]["score"]
            reg = doc["fields"]["register_number"]["value"]
            comp = doc["fields"]["company_name"]["value"]
            rev = doc["review"]["required"]
            print(f"[{st:<12}] {f.name:<28} | Score: {q:.2f} | Reg#: {str(reg):<8} | Company: {str(comp)[:30]} | Review: {rev}")
            if doc["reasons"]:
                print(f"    Reasons: {', '.join(doc['reasons'])}")
        except Exception as e:
            print(f"[ERROR       ] {f.name:<28} | Error: {e}")

    # Output to stdout or file
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved structured output to: {args.out}")
    elif len(files) == 1:
        # Single file -> print JSON directly
        single_doc = list(results.values())[0] if results else {}
        print("\n=== JSON OUTPUT ===")
        print(json.dumps(single_doc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
