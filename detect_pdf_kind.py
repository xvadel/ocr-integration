#!/usr/bin/env python3
"""
detect_pdf_kind.py - PDF Type Detection Layer.

Detects whether a PDF is:
  - 'digital': Clean digital text layer present with valid Arabic words/anchors.
  - 'digital_garbled': Text layer exists but is reversed/corrupted (visual RTL encoding).
  - 'scanned': Pure image / scanned PDF with no usable text layer.

Can be imported as a module or executed via CLI:
  python detect_pdf_kind.py document.pdf
  python detect_pdf_kind.py pdfs/ --json
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

# Arabic normalization helpers
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
DATE_RE = re.compile(r"(?<!\d)(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})(?!\d)")

DEFAULT_ANCHORS = [
    "السجل التجاري", "سجل تجاري", "رقم القيد", "تاريخ القيد", "اسم الشركة",
    "الشكل القانوني", "رأس المال", "رقم البطاقة الضريبية", "المحافظة", "النشاط"
]


def norm_arabic(t: str) -> str:
    """Normalize Arabic text for anchor matching."""
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = t.translate(AR_DIGITS)
    # Remove diacritics and formatting controls
    t = re.sub(r"[\u064B-\u0652\u0640\u200e\u200f\u202a-\u202e]", "", t)
    t = re.sub(r"[إأآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def are_dates_valid(text: str) -> bool:
    """Checks whether date patterns in text conform to standard DD/MM/YYYY or YYYY-MM-DD."""
    norm_txt = unicodedata.normalize("NFKC", text).translate(AR_DIGITS)
    matches = DATE_RE.findall(norm_txt)
    if not matches:
        return True  # No dates to invalidate
    for a, b, c in matches:
        # DD/MM/YYYY
        if len(c) == 4 and 1900 <= int(c) <= 2100:
            if 1 <= int(a) <= 31 and 1 <= int(b) <= 12:
                continue
        # YYYY-MM-DD
        if len(a) == 4 and 1900 <= int(a) <= 2100:
            if 1 <= int(b) <= 12 and 1 <= int(c) <= 31:
                continue
        # Invalid / reversed date
        return False
    return True


def check_anchors(text: str, anchors: List[str]) -> Tuple[int, List[str]]:
    """Counts how many expected anchors appear in normalized text."""
    flat = norm_arabic(text)
    found = [a for a in anchors if norm_arabic(a) in flat]
    return len(found), found


def extract_direct_text(doc: pymupdf.Document) -> str:
    """Extracts raw text layer across all pages using PyMuPDF."""
    return "\n".join(page.get_text() for page in doc)


def extract_bidi_text(pdf_path: Path) -> str:
    """Extracts text using pdfplumber with python-bidi reordering."""
    try:
        import pdfplumber
        from bidi.algorithm import get_display

        with pdfplumber.open(str(pdf_path)) as d:
            raw = "\n".join((p.extract_text() or "") for p in d.pages)
        return "\n".join(unicodedata.normalize("NFKC", get_display(ln)) for ln in raw.splitlines())
    except Exception:
        return ""


def detect_pdf_kind(pdf_path: Union[str, Path], anchors: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyzes a PDF file and classifies its format into:
      - 'digital': PyMuPDF direct text layer is clean and valid.
      - 'digital_garbled': Text layer is reversed/corrupted; repairable via bidi.
      - 'scanned': Pure image / scanned document requiring full OCR.

    Returns:
      {
        "pdf_kind": str ("digital", "digital_garbled", "scanned"),
        "pages": int,
        "char_count": int,
        "has_text_layer": bool,
        "anchors_found": list[str],
        "recommended_method": str ("text_layer" or "ocr"),
        "recommended_engine": str ("pymupdf", "pdfplumber_bidi", "tesseract", "easyocr"),
        "reason": str
      }
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    anchors = anchors or DEFAULT_ANCHORS
    doc = pymupdf.open(str(pdf_path))
    pages = len(doc)
    raw_mupdf = extract_direct_text(doc)
    doc.close()

    char_count = len(raw_mupdf.strip())

    # 1. Scanned check
    if char_count < 40:
        return {
            "pdf_kind": "scanned",
            "pages": pages,
            "char_count": char_count,
            "has_text_layer": False,
            "anchors_found": [],
            "recommended_method": "ocr",
            "recommended_engine": "tesseract",
            "reason": f"No usable text layer found (only {char_count} chars). Pure scan.",
        }

    # 2. Check direct PyMuPDF text layer
    found_count, found_anchors = check_anchors(raw_mupdf, anchors)
    dates_ok = are_dates_valid(raw_mupdf)

    if found_count >= 2 and dates_ok:
        return {
            "pdf_kind": "digital",
            "pages": pages,
            "char_count": char_count,
            "has_text_layer": True,
            "anchors_found": found_anchors,
            "recommended_method": "text_layer",
            "recommended_engine": "pymupdf",
            "reason": f"Clean digital text layer with {found_count} anchors verified.",
        }

    # 3. Check if text is reversed / visual RTL (digital_garbled)
    bidi_text = extract_bidi_text(pdf_path)
    bidi_found_count, bidi_found_anchors = check_anchors(bidi_text, anchors)
    bidi_dates_ok = are_dates_valid(bidi_text)

    if bidi_found_count >= 2 and bidi_dates_ok:
        return {
            "pdf_kind": "digital_garbled",
            "pages": pages,
            "char_count": len(bidi_text.strip()),
            "has_text_layer": True,
            "anchors_found": bidi_found_anchors,
            "recommended_method": "text_layer",
            "recommended_engine": "pdfplumber_bidi",
            "reason": f"Visual Arabic ordering repaired via pdfplumber_bidi ({bidi_found_count} anchors found).",
        }

    # If text layer exists but garbled beyond direct bidi fix, fall back to high-res OCR
    return {
        "pdf_kind": "digital_garbled",
        "pages": pages,
        "char_count": char_count,
        "has_text_layer": True,
        "anchors_found": found_anchors or bidi_found_anchors,
        "recommended_method": "ocr",
        "recommended_engine": "tesseract",
        "reason": f"Text layer garbled/corrupted (found only {found_count} anchors). Routing to OCR.",
    }


def main():
    parser = argparse.ArgumentParser(description="Detect PDF kind: digital, digital_garbled, or scanned.")
    parser.add_argument("path", type=Path, help="Path to PDF file or directory of PDFs")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    target = args.path
    if target.is_file():
        pdf_files = [target]
    elif target.is_dir():
        pdf_files = sorted(target.rglob("*.pdf"))
    else:
        sys.exit(f"Invalid path: {target}")

    results = {}
    for p in pdf_files:
        try:
            info = detect_pdf_kind(p)
            rel_name = str(p.relative_to(target.parent if target.is_file() else target))
            results[rel_name] = info
        except Exception as e:
            results[str(p.name)] = {"error": str(e)}

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'File':<40} {'Kind':<16} {'Pages':<6} {'Method':<12} {'Engine':<16}")
        print("=" * 92)
        for name, info in results.items():
            if "error" in info:
                print(f"{name:<40} {'ERROR':<16} {info['error']}")
            else:
                print(f"{name:<40} {info['pdf_kind']:<16} {info['pages']:<6} {info['recommended_method']:<12} {info['recommended_engine']:<16}")
        print("=" * 92)


if __name__ == "__main__":
    main()
