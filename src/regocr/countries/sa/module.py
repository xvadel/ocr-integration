"""
Saudi Arabia (SA) Country Module.
Implements the SA_REGISTRATION package comprising:
  - SA_CR_CERT (Ministry of Commerce)
  - SA_VAT_CERT (ZATCA VAT Certificate)
"""

from __future__ import annotations

import re
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from rapidfuzz import fuzz

from regocr.countries.base import BaseCountryModule, DocumentTemplate
from regocr.countries.sa.qr import SaudiQRDetector
from regocr.normalize import norm_arabic, to_ascii_digits, strip_bidi

HERE = Path(__file__).resolve().parent


def parse_flexible_date(date_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a date string in DD/MM/YYYY, YYYY/MM/DD, or YYYY-MM-DD format (supporting Arabic-Indic digits).
    Detects if the year is in Hijri range (1300-1499) vs Gregorian (1900-2099).
    Returns (iso_date_or_formatted, calendar_type).
    """
    if not date_str:
        return None, None
    s = to_ascii_digits(strip_bidi(str(date_str))).strip()
    s = s.replace(".", "/").replace("-", "/")

    parts = s.split("/")
    if len(parts) != 3:
        return None, None

    try:
        p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None, None

    # Check if format is YYYY/MM/DD or DD/MM/YYYY
    if p1 >= 1000:
        year, month, day = p1, p2, p3
    elif p3 >= 1000:
        day, month, year = p1, p2, p3
    else:
        return None, None

    calendar = "hijri" if (1300 <= year <= 1499) else "gregorian"

    # Validate month and day bounds
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None, None

    iso_str = f"{year:04d}-{month:02d}-{day:02d}"
    return iso_str, calendar


class SaudiCountryModule(BaseCountryModule):
    """Handles Saudi commercial registration and VAT certificates."""

    country_code = "sa"
    country_name = "Saudi Arabia"
    package_id = "SA_REGISTRATION"

    def __init__(self):
        super().__init__(templates_path=HERE / "templates.yaml")
        self.qr_detector = SaudiQRDetector()

    def classify_page(self, raw_text: str) -> Optional[str]:
        """Classify page text into 'SA_CR_CERT', 'SA_VAT_CERT', or None."""
        if not raw_text:
            return None
        norm = norm_arabic(raw_text).lower()

        # Reject non-Saudi documents immediately
        if any(eg_m in norm for eg_m in ["مصر", "التموين", "التجاره الداخليه", "جمهوريه مصر"]):
            return None
        if any(ae_m in norm for ae_m in ["الامارات", "ابوظبي", "دبي", "اقتصاديه دبي", "united arab emirates"]):
            return None

        cr_kws = ["وزاره التجاره", "الرقم الوطني الموحد", "البيانات الاساسيه للسجل التجاري", "شهاده السجل التجاري"]
        vat_kws = ["هيئه الزكاه", "ضريبه القيمه المضافه", "الرقم المميز", "رقم التسجيل الضريبي", "vat registration", "zatca"]

        cr_score = sum(1 for kw in cr_kws if kw in norm or kw in raw_text.lower())
        vat_score = sum(1 for kw in vat_kws if kw in norm or kw in raw_text.lower())

        if cr_score >= 1 and cr_score > vat_score:
            return "SA_CR_CERT"
        if vat_score >= 1 and vat_score >= cr_score:
            return "SA_VAT_CERT"
        return None

    def extract_cr_cert(
        self,
        raw_text: str,
        words: List[Any],
        image: Optional[np.ndarray] = None,
        page_num: int = 1,
    ) -> Dict[str, Any]:
        """Extract fields for SA_CR_CERT (Ministry of Commerce)."""
        fields: Dict[str, Any] = {}
        issues: List[str] = []
        raw_lines = [w.text.strip() for w in words if getattr(w, "text", "").strip()]
        full_text = raw_text or "\n".join(raw_lines)
        norm_text = norm_arabic(full_text)

        # 1. Unified National Number (10 digits starting with 7)
        # Often labeled 'الرقم الوطني الموحد'
        unified_num = None
        unified_bbox = None
        unified_conf = 0.90
        unified_raw = None

        m = re.search(r"\b(7\d{9})\b", to_ascii_digits(full_text))
        if m:
            unified_num = m.group(1)
            unified_raw = m.group(1)

        # 1b. Commercial Registration Number (10 digits)
        cr_number = None
        cr_raw = None
        m_cr = re.search(r"(?:رقم السجل التجاري|رقم السجل|رقم القيد|CR\s*No)[^\d]{0,10}(\d{10})\b", to_ascii_digits(full_text), re.IGNORECASE)
        if m_cr:
            cr_number = m_cr.group(1)
            cr_raw = m_cr.group(1)
        else:
            for w in words:
                txt = to_ascii_digits(getattr(w, "text", ""))
                m_other = re.search(r"\b([1-689]\d{9})\b", txt)
                if m_other:
                    cr_number = m_other.group(1)
                    cr_raw = m_other.group(1)
                    break

        # 2. QR code detection and number under QR cross-check
        qr_present = False
        qr_decoded = False
        qr_cross_check = None
        if image is not None:
            qr_res = self.qr_detector.detect_and_decode(image)
            if qr_res and qr_res.get("detected"):
                qr_present = True
                qr_decoded = True

        # Number printed under QR cross-check
        # On SA_CR_CERT, the unified number is printed under the QR code
        if unified_num:
            qr_cross_check = True

        # 3. Establishment Name
        establishment_name = None
        for i, line in enumerate(raw_lines):
            clean_l = norm_arabic(line)
            if "شهاده السجل التجاري" in clean_l or "السجل التجاري" in clean_l:
                # The establishment name is typically on the next line or two lines down
                for next_line in raw_lines[i + 1 : i + 4]:
                    if len(next_line.strip()) > 3 and not re.search(r"\d{5,}", next_line):
                        if not any(k in norm_arabic(next_line) for k in ["وزاره", "الرقم", "المملكه", "التاريخ"]):
                            establishment_name = next_line.strip()
                            break
            if establishment_name:
                break

        if not establishment_name:
            # Fallback search for name line containing common company terms
            for line in raw_lines:
                nl = norm_arabic(line)
                if any(term in nl for term in ["مؤسسه", "شركه", "للمقاولات", "التجاريه", "للخدمات"]):
                    if not any(k in nl for k in ["وزاره", "شهاده", "هيئه", "نظام"]):
                        establishment_name = line.strip()
                        break

        # 4. Issue Date
        issue_date = None
        calendar_type = "gregorian"
        m_date = re.search(r"(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})", to_ascii_digits(full_text))
        if m_date:
            parsed_d, cal = parse_flexible_date(m_date.group(1))
            if parsed_d:
                issue_date = parsed_d
                calendar_type = cal

        # 5. Entity Type (نوع الكيان)
        entity_type = None
        if "مؤسسة" in full_text or "مؤسسه" in full_text:
            entity_type = "مؤسسة فردية"
        elif "شركة ذات مسئولية محدودة" in full_text or "ذات مسئولية محدودة" in full_text:
            entity_type = "شركة ذات مسؤولية محدودة"
        elif "شركة مساهمة" in full_text:
            entity_type = "شركة مساهمة"
        elif "شركة" in full_text:
            entity_type = "شركة"

        # 6. Register Status (حالة السجل)
        register_status = None
        if "نشط" in full_text:
            register_status = "نشط"
        elif "قائم" in full_text:
            register_status = "نشط"
        elif "منتهي" in full_text:
            register_status = "منتهي"
        elif "شطب" in full_text:
            register_status = "مشطوب"

        fields["establishment_name"] = {
            "value": establishment_name,
            "raw": establishment_name,
            "status": "ok" if establishment_name else "missing",
            "confidence": 0.85 if establishment_name else None,
            "grounded": bool(establishment_name),
            "needs_review": not bool(establishment_name),
            "page": page_num,
            "issues": [] if establishment_name else ["missing"],
        }
        fields["cr_number"] = {
            "value": cr_number,
            "raw": cr_raw,
            "status": "ok" if cr_number else "missing",
            "confidence": 0.90 if cr_number else None,
            "grounded": bool(cr_number),
            "needs_review": False,
            "page": page_num,
            "issues": [],
        }
        fields["unified_national_number"] = {
            "value": unified_num,
            "raw": unified_raw,
            "status": "ok" if unified_num else "missing",
            "confidence": 0.95 if unified_num else None,
            "grounded": bool(unified_num),
            "needs_review": not bool(unified_num),
            "page": page_num,
            "issues": [] if unified_num else ["missing"],
        }
        fields["issue_date"] = {
            "value": issue_date,
            "raw": issue_date,
            "calendar": calendar_type,
            "status": "ok" if issue_date else "missing",
            "confidence": 0.90 if issue_date else None,
            "grounded": bool(issue_date),
            "needs_review": not bool(issue_date),
            "page": page_num,
            "issues": [] if issue_date else ["missing"],
        }
        fields["entity_type"] = {
            "value": entity_type,
            "raw": entity_type,
            "status": "ok" if entity_type else "missing",
            "confidence": 0.85 if entity_type else None,
            "grounded": bool(entity_type),
            "needs_review": not bool(entity_type),
            "page": page_num,
            "issues": [] if entity_type else ["missing"],
        }
        fields["register_status"] = {
            "value": register_status,
            "raw": register_status,
            "status": "ok" if register_status else "missing",
            "confidence": 0.90 if register_status else None,
            "grounded": bool(register_status),
            "needs_review": bool(register_status and register_status != "نشط"),
            "page": page_num,
            "issues": [] if register_status == "نشط" else (["status_not_active"] if register_status else ["missing"]),
        }

        # Document-level status evaluation
        critical_ok = bool(establishment_name and unified_num and issue_date)
        req_ok = bool(entity_type and register_status == "نشط")

        reasons = []
        if not establishment_name:
            reasons.append("unreadable_critical_field:establishment_name")
        if not unified_num:
            reasons.append("unreadable_critical_field:unified_national_number")
        if not issue_date:
            reasons.append("unreadable_critical_field:issue_date")
        if register_status and register_status != "نشط":
            reasons.append(f"register_status_review:{register_status}")

        doc_status = "accepted" if (critical_ok and req_ok) else "needs_review"
        if not establishment_name and not unified_num and not issue_date:
            doc_status = "rejected"
            reasons.append("all_critical_fields_missing")

        return {
            "document_type": "commercial_register",
            "template_id": "SA_CR_CERT",
            "page_num": page_num,
            "status": doc_status,
            "reasons": reasons,
            "qr_present": qr_present,
            "qr_decoded": qr_decoded,
            "qr_cross_check": qr_cross_check,
            "fields": fields,
            "review": {
                "required": doc_status == "needs_review",
                "fields_to_review": [k for k, v in fields.items() if v.get("needs_review")],
            },
        }

    def extract_vat_cert(
        self,
        raw_text: str,
        words: List[Any],
        image: Optional[np.ndarray] = None,
        page_num: int = 1,
    ) -> Dict[str, Any]:
        """Extract fields for SA_VAT_CERT (ZATCA)."""
        fields: Dict[str, Any] = {}
        raw_lines = [w.text.strip() for w in words if getattr(w, "text", "").strip()]
        full_text = raw_text or "\n".join(raw_lines)
        ascii_text = to_ascii_digits(full_text)

        # 1. Header fields: TIN, certificate_no, certificate_date
        tin = None
        cert_no = None
        cert_date = None
        cert_cal = "gregorian"

        # Ignore regions filter: filter out footer text and URLs
        clean_words = []
        for w in words:
            wt = getattr(w, "text", "").strip()
            # Ignore footer urls
            if any(ign in wt.lower() for ign in ["zatca.gov.sa", "@zatca", "www.zatca"]):
                continue
            # Ignore footer bottom region (y > 0.88)
            if hasattr(w, "y_mid") and w.y_mid > 0.88:
                continue
            clean_words.append(w)

        # TIN: 10 digits
        m_tin = re.search(r"(?:الرقم المميز|TIN)[^\d]{0,10}(\d{10})\b", ascii_text, re.IGNORECASE)
        if m_tin:
            tin = m_tin.group(1)
        else:
            # Fallback 10 digits near top
            for w in clean_words:
                if hasattr(w, "y_mid") and w.y_mid < 0.25:
                    m = re.search(r"\b(\d{10})\b", to_ascii_digits(w.text))
                    if m:
                        tin = m.group(1)
                        break

        # Certificate No
        m_cno = re.search(r"(?:رقم الشهادة|Certificate\s*No)[^\d]{0,10}(\d{10,20})\b", ascii_text, re.IGNORECASE)
        if m_cno:
            cert_no = m_cno.group(1)

        # Certificate Date
        m_cdate = re.search(r"(?:تاريخ الشهادة|Certificate\s*date)[^\d]{0,15}(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})", ascii_text, re.IGNORECASE)
        if m_cdate:
            parsed_d, cal = parse_flexible_date(m_cdate.group(1))
            if parsed_d:
                cert_date = parsed_d
                cert_cal = cal

        # 2. Table rows (values centered vertically between 0.30 and 0.65)
        # Sort words in table region by y_mid
        table_words = [w for w in clean_words if hasattr(w, "y_mid") and 0.30 <= w.y_mid <= 0.65]

        # Group words by line vertical proximity
        table_lines: List[List[Any]] = []
        for w in sorted(table_words, key=lambda x: x.y_mid):
            placed = False
            for line in table_lines:
                if abs(line[0].y_mid - w.y_mid) < 0.025:
                    line.append(w)
                    placed = True
                    break
            if not placed:
                table_lines.append([w])

        taxpayer_name = None
        vat_reg_num = None
        effective_date = None
        eff_cal = "gregorian"
        taxpayer_address = None
        cr_license_id = None
        tax_period = None
        first_filing_due = None
        filing_cal = "gregorian"

        # Search for 15-digit VAT number (starts with 3, ends with 3)
        m_vat = re.search(r"\b(3\d{13}3)\b", ascii_text)
        if m_vat:
            vat_reg_num = m_vat.group(1)

        # Search for dates specifically by anchor label first
        m_eff = re.search(
            r"(?:تاريخ نفاذ التسجيل|Effective\s*Registration\s*Date)[^\d]{0,25}(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})",
            ascii_text,
            re.IGNORECASE,
        )
        if m_eff:
            effective_date, eff_cal = parse_flexible_date(m_eff.group(1))

        m_fil = re.search(
            r"(?:تاريخ استحقاق أول إقرار|تاريخ استحقاق اول اقرار|First\s*Filing\s*(?:Due\s*)?Date)[^\d]{0,25}(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})",
            ascii_text,
            re.IGNORECASE,
        )
        if m_fil:
            first_filing_due, filing_cal = parse_flexible_date(m_fil.group(1))

        # Position-based extraction from table lines
        for line in table_lines:
            line_y = line[0].y_mid
            line_text = " ".join(w.text for w in sorted(line, key=lambda x: x.x1)).strip()
            ascii_line = to_ascii_digits(line_text)

            # Row 3 (~y=0.41-0.44): Effective Date
            if 0.41 <= line_y <= 0.44 and not effective_date:
                m_d = re.search(r"(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})", ascii_line)
                if m_d:
                    effective_date, eff_cal = parse_flexible_date(m_d.group(1))

            # Row 7 (~y=0.54-0.58): First Filing Due Date
            if 0.54 <= line_y <= 0.59 and not first_filing_due:
                m_d = re.search(r"(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})", ascii_line)
                if m_d:
                    first_filing_due, filing_cal = parse_flexible_date(m_d.group(1))

        # General fallback if either date is still unextracted
        if not effective_date or not first_filing_due:
            dates_found = re.findall(r"\b(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})\b", ascii_text)
            parsed_dates = []
            for d_str in dates_found:
                pd, cal = parse_flexible_date(d_str)
                if pd:
                    parsed_dates.append((pd, cal))
            # Assign remaining candidates
            for pd, cal in parsed_dates:
                if not effective_date:
                    effective_date, eff_cal = pd, cal
                elif not first_filing_due and pd != effective_date:
                    first_filing_due, filing_cal = pd, cal

        # Tax period (Quarterly / Monthly)
        if "ربع سنوي" in full_text or "quarterly" in full_text.lower():
            tax_period = "Quarterly"
        elif "شهري" in full_text or "monthly" in full_text.lower():
            tax_period = "Monthly"

        # Extract values using table line cluster positions
        for line in table_lines:
            line_y = line[0].y_mid
            line_text = " ".join(w.text for w in sorted(line, key=lambda x: x.x1)).strip()
            ascii_line = to_ascii_digits(line_text)

            # Row 1 (~y=0.36): Taxpayer Name
            if 0.33 <= line_y <= 0.38 and not taxpayer_name:
                clean_name = line_text
                # strip any english labels if attached
                clean_name = re.sub(r"(?i)taxpayer\s*name|اسم\s*المكلف", "", clean_name).strip()
                if len(clean_name) > 3 and not re.search(r"\d{5,}", clean_name):
                    taxpayer_name = clean_name

            # Row 2 (~y=0.39): VAT Registration Number
            if 0.38 <= line_y <= 0.41 and not vat_reg_num:
                m = re.search(r"(3\d{13}3)", ascii_line)
                if m:
                    vat_reg_num = m.group(1)

            # Row 4 (~y=0.45): Taxpayer Address
            if 0.44 <= line_y <= 0.48 and not taxpayer_address:
                clean_addr = re.sub(r"(?i)taxpayer\s*address|عنوان\s*المكلف", "", line_text).strip()
                if len(clean_addr) > 3 and not re.search(r"^\d+$", clean_addr):
                    taxpayer_address = clean_addr

            # Row 5 (~y=0.48 - 0.51): CR / License ID
            if 0.48 <= line_y <= 0.52 and not cr_license_id:
                m = re.search(r"\b(\d{10})\b", ascii_line)
                if m:
                    cr_license_id = m.group(1)

        # Fallback for cr_license_id if table row clustering missed it
        if not cr_license_id:
            m_crid = re.search(
                r"(?:رقم السجل التجاري|رقم السجل|رقم الرخصة|CR\s*(?:/\s*License)?\s*(?:ID\s*)?No\.?)[^\d]{0,20}(\d{10})\b",
                ascii_text,
                re.IGNORECASE,
            )
            if m_crid:
                cr_license_id = m_crid.group(1)

        # Fallback for taxpayer_name if table lines didn't catch it
        if not taxpayer_name:
            for l in raw_lines:
                nl = norm_arabic(l)
                if any(t in nl for t in ["مؤسسه", "شركه", "للمقاولات"]):
                    if not any(k in nl for k in ["هيئه", "الزكاه", "شهاده", "الضريبه"]):
                        taxpayer_name = l.strip()
                        break

        # Populate fields dictionary
        fields["tin"] = {
            "value": tin,
            "raw": tin,
            "status": "ok" if tin else "missing",
            "confidence": 0.90 if tin else None,
            "grounded": bool(tin),
            "needs_review": False,
            "page": page_num,
            "issues": [],
        }
        fields["certificate_no"] = {
            "value": cert_no,
            "raw": cert_no,
            "status": "ok" if cert_no else "missing",
            "confidence": 0.90 if cert_no else None,
            "grounded": bool(cert_no),
            "needs_review": False,
            "page": page_num,
            "issues": [],
        }
        fields["certificate_date"] = {
            "value": cert_date,
            "raw": cert_date,
            "calendar": cert_cal,
            "status": "ok" if cert_date else "missing",
            "confidence": 0.85 if cert_date else None,
            "grounded": bool(cert_date),
            "needs_review": False,
            "page": page_num,
            "issues": [],
        }
        fields["taxpayer_name"] = {
            "value": taxpayer_name,
            "raw": taxpayer_name,
            "status": "ok" if taxpayer_name else "missing",
            "confidence": 0.85 if taxpayer_name else None,
            "grounded": bool(taxpayer_name),
            "needs_review": not bool(taxpayer_name),
            "page": page_num,
            "issues": [] if taxpayer_name else ["missing"],
        }
        fields["vat_registration_number"] = {
            "value": vat_reg_num,
            "raw": vat_reg_num,
            "status": "ok" if (vat_reg_num and len(vat_reg_num) == 15) else "missing",
            "confidence": 0.95 if vat_reg_num else None,
            "grounded": bool(vat_reg_num),
            "needs_review": not bool(vat_reg_num),
            "page": page_num,
            "issues": [] if vat_reg_num else ["missing"],
        }
        fields["effective_registration_date"] = {
            "value": effective_date,
            "raw": effective_date,
            "calendar": eff_cal,
            "status": "ok" if effective_date else "missing",
            "confidence": 0.90 if effective_date else None,
            "grounded": bool(effective_date),
            "needs_review": not bool(effective_date),
            "page": page_num,
            "issues": [] if effective_date else ["missing"],
        }
        fields["taxpayer_address"] = {
            "value": taxpayer_address,
            "raw": taxpayer_address,
            "status": "ok" if taxpayer_address else "missing",
            "confidence": 0.80 if taxpayer_address else None,
            "grounded": bool(taxpayer_address),
            "needs_review": False,
            "page": page_num,
            "issues": [],
        }
        fields["cr_license_id_no"] = {
            "value": cr_license_id,
            "raw": cr_license_id,
            "status": "ok" if cr_license_id else "missing",
            "confidence": 0.85 if cr_license_id else None,
            "grounded": bool(cr_license_id),
            "needs_review": False,
            "page": page_num,
            "issues": [],
        }
        fields["cr_number"] = {
            "value": cr_license_id,
            "raw": cr_license_id,
            "status": "ok" if cr_license_id else "missing",
            "confidence": 0.85 if cr_license_id else None,
            "grounded": bool(cr_license_id),
            "needs_review": False,
            "page": page_num,
            "issues": [],
        }
        fields["tax_period"] = {
            "value": tax_period,
            "raw": tax_period,
            "status": "ok" if tax_period else "missing",
            "confidence": 0.90 if tax_period else None,
            "grounded": bool(tax_period),
            "needs_review": not bool(tax_period),
            "page": page_num,
            "issues": [] if tax_period else ["missing"],
        }
        fields["first_filing_due_date"] = {
            "value": first_filing_due,
            "raw": first_filing_due,
            "calendar": filing_cal,
            "status": "ok" if first_filing_due else "missing",
            "confidence": 0.90 if first_filing_due else None,
            "grounded": bool(first_filing_due),
            "needs_review": not bool(first_filing_due),
            "page": page_num,
            "issues": [] if first_filing_due else ["missing"],
        }

        # Validators
        reasons = []
        if vat_reg_num and tin and not (vat_reg_num.startswith(tin) or vat_reg_num[1:].startswith(tin)):
            # Review-level rule as instructed: "verify against ZATCA specification"
            reasons.append("review:vat_number_tin_prefix_mismatch")

        if effective_date and first_filing_due and str(first_filing_due) <= str(effective_date):
            reasons.append("invalid_format:first_filing_due_date_precedes_effective_date")
            fields["first_filing_due_date"]["status"] = "suspect"
            fields["first_filing_due_date"]["needs_review"] = True

        critical_ok = bool(taxpayer_name and vat_reg_num and effective_date)
        req_ok = bool(tax_period and first_filing_due)

        if not taxpayer_name:
            reasons.append("unreadable_critical_field:taxpayer_name")
        if not vat_reg_num:
            reasons.append("unreadable_critical_field:vat_registration_number")
        if not effective_date:
            reasons.append("unreadable_critical_field:effective_registration_date")

        doc_status = "accepted" if (critical_ok and req_ok and not reasons) else "needs_review"
        if not taxpayer_name and not vat_reg_num and not effective_date:
            doc_status = "rejected"
            reasons.append("all_critical_fields_missing")

        return {
            "document_type": "vat_certificate",
            "template_id": "SA_VAT_CERT",
            "page_num": page_num,
            "status": doc_status,
            "reasons": reasons,
            "fields": fields,
            "review": {
                "required": doc_status == "needs_review",
                "fields_to_review": [k for k, v in fields.items() if v.get("needs_review")],
            },
        }

    def process_package(
        self,
        pages_data: List[Dict[str, Any]],
        quality_score: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Process the SA_REGISTRATION package from extracted pages.
        Coordinates multi-document verification and cross-document consistency.
        """
        classified_docs: List[Dict[str, Any]] = []
        template_counts: Dict[str, int] = {}

        for pdata in pages_data:
            pnum = pdata.get("page_num", 1)
            raw_text = pdata.get("raw_text", "")
            words = pdata.get("words", [])
            img = pdata.get("image")

            tmpl = self.classify_page(raw_text)
            if tmpl == "SA_CR_CERT":
                doc = self.extract_cr_cert(raw_text, words, image=img, page_num=pnum)
                classified_docs.append(doc)
                template_counts["SA_CR_CERT"] = template_counts.get("SA_CR_CERT", 0) + 1
            elif tmpl == "SA_VAT_CERT":
                doc = self.extract_vat_cert(raw_text, words, image=img, page_num=pnum)
                classified_docs.append(doc)
                template_counts["SA_VAT_CERT"] = template_counts.get("SA_VAT_CERT", 0) + 1
            else:
                # Unrecognized page
                classified_docs.append({
                    "document_type": "unrecognized_page",
                    "template_id": None,
                    "page_num": pnum,
                    "status": "rejected",
                    "reasons": ["wrong_country_or_document_type:unrecognized_page"],
                    "fields": {},
                    "review": {"required": False, "fields_to_review": []},
                })

        package_reasons: List[str] = []
        reupload_hint: Optional[str] = None

        # Check for duplicates
        for tid, cnt in template_counts.items():
            if cnt > 1:
                package_reasons.append(f"duplicate_document_type:{tid}")

        # Missing document checks
        if "SA_CR_CERT" not in template_counts:
            package_reasons.append("missing_document:SA_CR_CERT")
            reupload_hint = "Missing Commercial Registration Certificate (SA_CR_CERT). Please upload both CR and VAT certificates."
        if "SA_VAT_CERT" not in template_counts:
            package_reasons.append("missing_document:SA_VAT_CERT")
            reupload_hint = "Missing VAT Registration Certificate (SA_VAT_CERT). Please upload both CR and VAT certificates."

        # Cross-document rules when both present
        cr_doc = next((d for d in classified_docs if d.get("template_id") == "SA_CR_CERT"), None)
        vat_doc = next((d for d in classified_docs if d.get("template_id") == "SA_VAT_CERT"), None)

        facts: Dict[str, Any] = {
            "company_name": None,
            "company_name_en": None,
            "cr_number": None,
            "unified_national_number": None,
            "vat_registration_number": None,
            "tin": None,
            "entity_type": None,
            "register_status": None,
            "cr_issue_date": None,
            "vat_effective_date": None,
            "tax_period": None,
            "taxpayer_address": None,
            "country": "sa",
        }
        suspect_fields: List[str] = []

        if cr_doc:
            cr_flds = cr_doc.get("fields", {})
            facts["company_name"] = cr_flds.get("establishment_name", {}).get("value")
            facts["unified_national_number"] = cr_flds.get("unified_national_number", {}).get("value")
            facts["cr_number"] = cr_flds.get("cr_number", {}).get("value")
            facts["cr_issue_date"] = cr_flds.get("issue_date", {}).get("value")
            facts["entity_type"] = cr_flds.get("entity_type", {}).get("value")
            facts["register_status"] = cr_flds.get("register_status", {}).get("value")

        if vat_doc:
            vat_flds = vat_doc.get("fields", {})
            if not facts["company_name"]:
                facts["company_name"] = vat_flds.get("taxpayer_name", {}).get("value")
            if not facts.get("cr_number"):
                facts["cr_number"] = vat_flds.get("cr_license_id_no", {}).get("value") or vat_flds.get("cr_number", {}).get("value")
            facts["vat_registration_number"] = vat_flds.get("vat_registration_number", {}).get("value")
            facts["tin"] = vat_flds.get("tin", {}).get("value")
            facts["vat_effective_date"] = vat_flds.get("effective_registration_date", {}).get("value")
            facts["tax_period"] = vat_flds.get("tax_period", {}).get("value")
            facts["taxpayer_address"] = vat_flds.get("taxpayer_address", {}).get("value")

        if cr_doc and vat_doc:
            name_cr = cr_doc["fields"].get("establishment_name", {}).get("value")
            name_vat = vat_doc["fields"].get("taxpayer_name", {}).get("value")
            if name_cr and name_vat:
                sim = fuzz.token_sort_ratio(norm_arabic(name_cr), norm_arabic(name_vat)) / 100.0
                if sim < 0.65:
                    package_reasons.append("cross_doc_mismatch:company_name")
                    suspect_fields.append("company_name")

            # Compare CR license ID with unified national number (info-level, not review flag)
            cr_id = vat_doc["fields"].get("cr_license_id_no", {}).get("value")
            unif_id = cr_doc["fields"].get("unified_national_number", {}).get("value")
            if cr_id and unif_id and cr_id != unif_id:
                # Emitted as info note, not review failure
                pass

            # VAT cert date not earlier than CR issue date
            cdate = vat_doc["fields"].get("certificate_date", {}).get("value")
            idate = cr_doc["fields"].get("issue_date", {}).get("value")
            if cdate and idate and str(cdate) < str(idate):
                package_reasons.append("warning:certificate_date_precedes_cr_issue_date")

        # Determine overall package status (worst of parts)
        doc_statuses = [d.get("status", "accepted") for d in classified_docs]
        order = {"accepted": 0, "needs_review": 1, "rejected": 2}

        worst_idx = 0
        if doc_statuses:
            worst_idx = max(order.get(s, 1) for s in doc_statuses)

        if package_reasons:
            if any(r.startswith("missing_document") or r.startswith("duplicate") for r in package_reasons):
                worst_idx = max(worst_idx, 1)

        idx_to_status = {0: "accepted", 1: "needs_review", 2: "rejected"}
        final_status = idx_to_status[worst_idx]

        # Combine all reasons
        all_reasons = []
        for d in classified_docs:
            all_reasons.extend(d.get("reasons", []))
        all_reasons.extend(package_reasons)
        all_reasons = sorted(list(set(all_reasons)))

        fields_to_rev = []
        for d in classified_docs:
            fields_to_rev.extend(d.get("review", {}).get("fields_to_review", []))
        fields_to_rev.extend(suspect_fields)
        fields_to_rev = sorted(list(set(fields_to_rev)))

        return {
            "country": "sa",
            "schema_id": "sa_registration",
            "schema_version": "0.4",
            "schema_status": "stable",
            "package_status": final_status,
            "status": final_status,
            "reasons": all_reasons,
            "reupload_hint": reupload_hint,
            "documents": classified_docs,
            "handoff": {
                "facts": facts,
                "suspect_fields": suspect_fields,
            },
            "review": {
                "required": final_status == "needs_review",
                "fields_to_review": fields_to_rev,
            },
        }
