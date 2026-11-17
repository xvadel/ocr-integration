"""
Ingestion module for PDFs and images (multi-page aware, 100% permissive licenses).
Uses pypdfium2 (BSD/Apache-2.0) and pdfplumber (MIT) + pure-Python visual-RTL reordering.
No PyMuPDF (AGPL) and no python-bidi (LGPL) imports.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageOps

from regocr.normalize import norm_arabic, reorder_visual_rtl, to_ascii_digits

DEFAULT_ANCHORS = [
    "السجل التجاري", "سجل تجاري", "رقم القيد", "تاريخ القيد", "اسم الشركة",
    "الشكل القانوني", "رأس المال", "رقم البطاقة الضريبية", "المحافظة", "النشاط",
    "الاسم التجاري", "الرقم الموحد", "الرقم القومي", "عنوان المركز", "المدير"
]

DATE_RE = re.compile(r"(?<!\d)(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})(?!\d)")


def compute_document_id(file_path: Union[str, Path]) -> str:
    """Compute a stable content-addressed document identifier (doc_<16-hex-sha256>)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"doc_{h.hexdigest()[:16]}"


def are_dates_valid(text: str) -> bool:
    """Checks whether date patterns in text conform to standard DD/MM/YYYY or YYYY-MM-DD."""
    norm_txt = to_ascii_digits(text)
    matches = DATE_RE.findall(norm_txt)
    if not matches:
        return True
    for a, b, c in matches:
        if len(c) == 4 and 1900 <= int(c) <= 2100:
            if 1 <= int(a) <= 31 and 1 <= int(b) <= 12:
                continue
        if len(a) == 4 and 1900 <= int(a) <= 2100:
            if 1 <= int(b) <= 12 and 1 <= int(c) <= 31:
                continue
        return False
    return True


def check_anchors(text: str, anchors: List[str]) -> Tuple[int, List[str]]:
    """Counts how many expected anchors appear in normalized text."""
    flat = norm_arabic(text)
    found = [a for a in anchors if norm_arabic(a) in flat]
    return len(found), found


def extract_page_text_pypdfium2(pdf_path: Union[str, Path], page_num: int) -> str:
    """Extract raw text layer from a single PDF page using pypdfium2."""
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_num >= len(doc):
            raise IndexError(f"Page {page_num} out of range (PDF has {len(doc)} pages)")
        page = doc.get_page(page_num)
        text_page = page.get_textpage()
        txt = text_page.get_text_range()
        text_page.close()
        page.close()
        return txt
    finally:
        doc.close()


def extract_text_layer_pypdfium2(pdf_path: Union[str, Path]) -> str:
    """Extract raw text from all pages of a PDF using pypdfium2."""
    doc = pdfium.PdfDocument(str(pdf_path))
    pages_text = []
    try:
        for i in range(len(doc)):
            page = doc.get_page(i)
            text_page = page.get_textpage()
            pages_text.append(text_page.get_text_range())
            text_page.close()
            page.close()
    finally:
        doc.close()
    return "\n".join(pages_text)


def extract_page_bidi_text_layer(pdf_path: Union[str, Path], page_num: int) -> str:
    """Extract text for one page using pdfplumber + pure-Python visual-RTL reordering (zero LGPL)."""
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as d:
            if page_num >= len(d.pages):
                return ""
            raw = d.pages[page_num].extract_text() or ""
        return "\n".join(reorder_visual_rtl(ln) for ln in raw.splitlines())
    except Exception:
        return ""


def extract_bidi_text_layer(pdf_path: Union[str, Path]) -> str:
    """Extract text for all pages using pdfplumber + pure-Python visual-RTL reordering (zero LGPL)."""
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as d:
            raw = "\n".join((p.extract_text() or "") for p in d.pages)
        return "\n".join(reorder_visual_rtl(ln) for ln in raw.splitlines())
    except Exception:
        return ""


def _classify_single_page(
    pdf_path: Path,
    page_num: int,
    raw_text: str,
    anchors: List[str],
) -> Dict[str, Any]:
    """Classify one PDF page independently as digital, digital_garbled, or scanned."""
    char_count = len(raw_text.strip())
    if char_count < 30:
        return {
            "page": page_num + 1,
            "pdf_kind": "scanned",
            "char_count": char_count,
            "has_text_layer": False,
            "anchors_found": [],
            "recommended_method": "ocr",
            "recommended_engine": "paddleocr",
            "text": "",
            "reason": f"Page {page_num + 1}: no usable text layer ({char_count} chars).",
        }

    # Check direct pypdfium2 text layer (normalized)
    norm_raw = unicodedata.normalize("NFKC", raw_text)
    found_count, found_anchors = check_anchors(norm_raw, anchors)
    dates_ok = are_dates_valid(norm_raw)
    if found_count >= 2 and dates_ok:
        return {
            "page": page_num + 1,
            "pdf_kind": "digital",
            "char_count": char_count,
            "has_text_layer": True,
            "anchors_found": found_anchors,
            "recommended_method": "text_layer",
            "recommended_engine": "pypdfium2",
            "text": norm_raw,
            "reason": f"Page {page_num + 1}: clean digital text layer ({found_count} anchors).",
        }

    # Check pure-Python visual-RTL reordered text via pdfplumber FIRST (preserves Arabic Presentation Form ligatures!)
    bidi_text = extract_page_bidi_text_layer(pdf_path, page_num)
    bidi_count, bidi_anchors = check_anchors(bidi_text, anchors)
    bidi_dates_ok = are_dates_valid(bidi_text)
    if bidi_count >= 2 and bidi_dates_ok:
        return {
            "page": page_num + 1,
            "pdf_kind": "digital_garbled",
            "char_count": len(bidi_text.strip()),
            "has_text_layer": True,
            "anchors_found": bidi_anchors,
            "recommended_method": "text_layer",
            "recommended_engine": "pdfplumber_rtl",
            "text": unicodedata.normalize("NFKC", bidi_text),
            "reason": f"Page {page_num + 1}: visual RTL repaired via pure-Python reorder ({bidi_count} anchors).",
        }

    # Fallback: visual-RTL reordered text directly on pypdfium2 text (e.g., ReportLab PDFs)
    rtl_direct = "\n".join(reorder_visual_rtl(ln) for ln in raw_text.splitlines())
    rtl_count, rtl_anchors = check_anchors(rtl_direct, anchors)
    rtl_dates_ok = are_dates_valid(rtl_direct)
    if rtl_count >= 2 and rtl_dates_ok:
        return {
            "page": page_num + 1,
            "pdf_kind": "digital_garbled",
            "char_count": len(rtl_direct.strip()),
            "has_text_layer": True,
            "anchors_found": rtl_anchors,
            "recommended_method": "text_layer",
            "recommended_engine": "pypdfium2_rtl",
            "text": unicodedata.normalize("NFKC", rtl_direct),
            "reason": f"Page {page_num + 1}: visual RTL repaired via pypdfium2_rtl ({rtl_count} anchors).",
        }

    # Also check if a secondary page has real Arabic/Latin text (even with only 1 anchor, e.g. continuation page)
    arabic_chars = len(re.findall(r"[\u0621-\u064A]", norm_raw))
    if found_count >= 1 and dates_ok and arabic_chars >= 15:
        return {
            "page": page_num + 1,
            "pdf_kind": "digital",
            "char_count": char_count,
            "has_text_layer": True,
            "anchors_found": found_anchors,
            "recommended_method": "text_layer",
            "recommended_engine": "pypdfium2",
            "text": norm_raw,
            "reason": f"Page {page_num + 1}: continuation digital text layer ({found_count} anchor, {arabic_chars} Arabic chars).",
        }

    arabic_bidi = len(re.findall(r"[\u0621-\u064A]", bidi_text))
    if bidi_count >= 1 and bidi_dates_ok and arabic_bidi >= 15:
        return {
            "page": page_num + 1,
            "pdf_kind": "digital_garbled",
            "char_count": len(bidi_text.strip()),
            "has_text_layer": True,
            "anchors_found": bidi_anchors,
            "recommended_method": "text_layer",
            "recommended_engine": "pdfplumber_rtl",
            "text": unicodedata.normalize("NFKC", bidi_text),
            "reason": f"Page {page_num + 1}: continuation visual RTL repaired ({bidi_count} anchor).",
        }

    return {
        "page": page_num + 1,
        "pdf_kind": "digital_garbled",
        "char_count": char_count,
        "has_text_layer": True,
        "anchors_found": found_anchors or bidi_anchors,
        "recommended_method": "ocr",
        "recommended_engine": "paddleocr",
        "text": "",
        "reason": f"Page {page_num + 1}: text layer garbled ({found_count} anchors). Routing to OCR.",
    }


def detect_pdf_pages(
    pdf_path: Union[str, Path],
    anchors: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Inspect and route EVERY page of a PDF independently (text_layer vs ocr)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    anchors = anchors or DEFAULT_ANCHORS
    doc = pdfium.PdfDocument(str(pdf_path))
    raw_pages: List[str] = []
    try:
        for i in range(len(doc)):
            page = doc.get_page(i)
            tp = page.get_textpage()
            raw_pages.append(tp.get_text_range())
            tp.close()
            page.close()
    finally:
        doc.close()

    return [
        _classify_single_page(pdf_path, idx, txt, anchors)
        for idx, txt in enumerate(raw_pages)
    ]


def detect_pdf_kind(pdf_path: Union[str, Path], anchors: Optional[List[str]] = None) -> Dict[str, Any]:
    """Document-level summary derived from per-page PDF classification."""
    pages_info = detect_pdf_pages(pdf_path, anchors=anchors)
    num_pages = len(pages_info)
    kinds = {p["pdf_kind"] for p in pages_info}
    methods = {p["recommended_method"] for p in pages_info}

    if len(kinds) == 1:
        doc_kind = next(iter(kinds))
    else:
        doc_kind = "mixed"

    if len(methods) == 1:
        doc_method = next(iter(methods))
    else:
        doc_method = "mixed"

    first = pages_info[0] if pages_info else {}
    return {
        "pdf_kind": doc_kind,
        "pages": num_pages,
        "char_count": sum(p["char_count"] for p in pages_info),
        "has_text_layer": any(p["has_text_layer"] for p in pages_info),
        "anchors_found": sorted(list({a for p in pages_info for a in p["anchors_found"]})),
        "recommended_method": doc_method,
        "recommended_engine": first.get("recommended_engine", "paddleocr"),
        "pages_info": pages_info,
        "reason": "; ".join(p["reason"] for p in pages_info),
    }


def render_pdf_page(pdf_path: Union[str, Path], page_num: int = 0, dpi: int = 300) -> np.ndarray:
    """Renders a PDF page to a BGR numpy array using pypdfium2 at specified DPI."""
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_num >= len(doc):
            raise IndexError(f"Page {page_num} out of range (PDF has {len(doc)} pages)")
        page = doc.get_page(page_num)
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        page.close()
    finally:
        doc.close()
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Loads an image file, applies EXIF orientation fix, and returns a BGR cv2 image."""
    pil_img = Image.open(image_path)
    pil_img = ImageOps.exif_transpose(pil_img)
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
