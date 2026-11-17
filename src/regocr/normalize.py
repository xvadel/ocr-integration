"""Text normalization utilities for Arabic/English OCR output (100% stdlib, zero LGPL)."""

from __future__ import annotations

import datetime
import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Digit conversion tables
# ---------------------------------------------------------------------------

_ARABIC_INDIC_CHARS = frozenset("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")

_DIGIT_TABLE = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

# ---------------------------------------------------------------------------
# Bidi control characters to strip
# ---------------------------------------------------------------------------

_BIDI_CHARS = frozenset(
    "\u200f"  # RLM
    "\u200e"  # LRM
    "\u202a"  # LRE
    "\u202b"  # RLE
    "\u202c"  # PDF
    "\u202d"  # LRO
    "\u202e"  # RLO
    "\u2066"  # LRI
    "\u2067"  # RLI
    "\u2068"  # FSI
    "\u2069"  # PDI
)

_BIDI_RE = re.compile("[" + "".join(_BIDI_CHARS) + "]")

# ---------------------------------------------------------------------------
# Arabic normalization maps
# ---------------------------------------------------------------------------

_ALEF_VARIANTS = str.maketrans({
    "\u0622": "\u0627",  # آ -> ا
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0671": "\u0627",  # ٱ -> ا
})

_TA_MARBUTA = str.maketrans({"\u0629": "\u0647"})  # ة -> ه
_ALEF_MAQSURA = str.maketrans({"\u0649": "\u064a"})  # ى -> ي

_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed\u0640]"
)
_FORMATTING_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff\u00ad]")

# Mirror brackets when reversing visual RTL lines
_MIRROR_MAP = str.maketrans("()[]{}<>", ")(][}{><")

# ---------------------------------------------------------------------------
# Date patterns
# ---------------------------------------------------------------------------

_DATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})(?!\d)"), "dmy"),
    (re.compile(r"(?<!\d)(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)"), "ymd"),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_arabic_indic_digits(text: str) -> bool:
    """Return True if text contains Arabic-Indic or Persian digits."""
    return any(ch in _ARABIC_INDIC_CHARS for ch in (text or ""))


def to_ascii_digits(text: str) -> str:
    """Convert Arabic-Indic (٠-٩) and Persian (۰-۹) digits to ASCII 0-9."""
    if not text:
        return ""
    return str(text).translate(_DIGIT_TABLE)


def strip_bidi(text: str) -> str:
    """Remove Unicode bidi control characters."""
    if not text:
        return ""
    return _BIDI_RE.sub("", str(text))


def norm_arabic(text: str) -> str:
    """Normalize Arabic text for fuzzy matching (NOT for display output)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = strip_bidi(text)
    text = _DIACRITICS_RE.sub("", text)
    text = _FORMATTING_RE.sub("", text)
    text = text.translate(_ALEF_VARIANTS)
    text = text.translate(_TA_MARBUTA)
    text = text.translate(_ALEF_MAQSURA)
    return text


def reorder_visual_rtl(line: str) -> str:
    """Pure-Python permissive replacement for python-bidi's get_display on visual RTL PDF lines.

    Converts a single visual-order line (left-to-right characters with reversed Arabic)
    into logical RTL reading order while preserving LTR orientation of Latin words,
    Western numbers, Western dates/tax-IDs, and individual Arabic-Indic numeric tokens.

    IMPORTANT: Reverse BEFORE NFKC so single-codepoint Arabic Presentation Form lam-alef
    ligatures (e.g. U+FEFB 'ﻻ') decompose to 'لا' in logical order rather than reversed 'ال'.
    """
    if not line:
        return ""
    # Reverse raw line first (preserves lam-alef ligatures as single codepoints), mirror brackets
    rev = line[::-1].translate(_MIRROR_MAP)
    # Now normalize Presentation Forms-A/B to canonical Arabic
    rev = unicodedata.normalize("NFKC", rev)

    # Restore LTR order for Western/Latin tokens (including dates/hyphenated Western IDs/commas)
    rev = re.sub(
        r"[A-Za-z0-9]+(?:[/\-.,:٬][A-Za-z0-9]+)*",
        lambda m: m.group(0)[::-1],
        rev,
    )
    # Restore LTR digit order inside Arabic-Indic runs (keep hyphen group reversal intact!)
    rev = re.sub(
        r"[٠-٩۰-۹]+(?:[,٬.][٠-٩۰-۹]+)*",
        lambda m: m.group(0)[::-1],
        rev,
    )
    return rev


def parse_iso_date(raw_date: str) -> Optional[str]:
    """Parse a date string in common formats and return strict calendar-validated ISO-8601 (YYYY-MM-DD)."""
    if not raw_date:
        return None
    cleaned = strip_bidi(to_ascii_digits(str(raw_date).strip()))

    for pattern, fmt in _DATE_PATTERNS:
        m = pattern.search(cleaned)
        if m:
            if fmt == "dmy":
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))

            if 1900 <= year <= 2100:
                try:
                    dt = datetime.date(year, month, day)
                    return dt.isoformat()
                except ValueError:
                    # Try swapped day/month only if day > 12 isn't valid (e.g. visual reversed YYYY/MM/DD)
                    continue

    # Handle visual-reversed Arabic-Indic date YYYY/MM/DD or DD/MM/YYYY already handled above;
    # check reversed separator order if day>31 wasn't matched.
    return None


def parse_tax_id(raw_val: str, reverse_indic_groups: bool = True) -> Optional[str]:
    """Extract and normalize an Egyptian 9-digit Tax ID (XXX-XXX-XXX).

    Under the Unicode Bidirectional Algorithm, hyphenated Arabic-Indic triplets
    (e.g. ١٦٠-٩٤٦-٦٧٩) are rendered visually right-to-left as ٦٧٩-٩٤٦-١٦٠.
    When OCR reads the visual image left-to-right and yields Arabic-Indic digits,
    reversing the 3 hyphen-delimited groups restores the logical order.
    """
    if not raw_val:
        return None
    raw_str = strip_bidi(str(raw_val))
    is_indic = has_arabic_indic_digits(raw_str)
    ascii_str = to_ascii_digits(raw_str)

    # 1. Explicit 3-3-3 pattern separated by -, space, dot, or Arabic zero misread as separator
    m = re.search(r"(?<!\d)(\d{3})\s*[-–—.\s]\s*(\d{3})\s*[-–—.\s]\s*(\d{3})(?!\d)", ascii_str)
    if m:
        g1, g2, g3 = m.group(1), m.group(2), m.group(3)
        if is_indic and reverse_indic_groups:
            return f"{g3}-{g2}-{g1}"
        return f"{g1}-{g2}-{g3}"

    # 2. Fallback: exact 9 digits (e.g. EasyOCR read '-' as '0' in 10 digits or dropped separator)
    digits_only = re.sub(r"\D", "", ascii_str)
    if len(digits_only) == 9:
        g1, g2, g3 = digits_only[0:3], digits_only[3:6], digits_only[6:9]
        if is_indic and reverse_indic_groups:
            return f"{g3}-{g2}-{g1}"
        return f"{g1}-{g2}-{g3}"

    # 3. Special OCR artifact: 10 digits where a '-' between 3-digit groups became '0' (e.g. 6790946160 or 9460160-679)
    # Specifically handle pattern like 3 digits + '0' + 3 digits + '-' + 3 digits (seen in EasyOCR Indic)
    m_art = re.search(r"(\d{3})[0٠](\d{3})[-–—](\d{3})", ascii_str)
    if m_art:
        # e.g. '٩٤٦٠١٦٠-٦٧٩' where visual was ٦٧٩-٩٤٦-١٦٠ reordered by EasyOCR
        g1, g2, g3 = m_art.group(1), m_art.group(2), m_art.group(3)
        if is_indic and reverse_indic_groups:
            return f"{g2}-{g1}-{g3}"
        return f"{g1}-{g2}-{g3}"

    return None


def parse_numeric_capital(raw_val: str) -> Optional[float]:
    """Extract a numeric value from a capital string.

    Examples:
        "1,000,000 جنيه مصري" -> 1000000.0
        "٥٠٠٬٠٠٠"              -> 500000.0
        "٠١,٠٠٠,٠٠٠"           -> 1000000.0
    """
    if not raw_val:
        return None
    text = strip_bidi(to_ascii_digits(str(raw_val)))
    # Remove thousands separators: ASCII comma, Arabic thousands sep ٬, Arabic comma ،, spaces
    text = text.replace("\u066c", "").replace("\u060c", "").replace(",", "").replace(" ", "")
    # Remove trailing stray dot
    text = text.strip(".")

    # Find number (prefer longest digit sequence with optional 1-2 decimal places)
    matches = re.findall(r"\d+(?:\.\d{1,2})?", text)
    if not matches:
        return None

    # Pick the longest numeric match (avoids single stray digit artifacts before real amount)
    best = max(matches, key=lambda s: len(s.replace(".", "")))
    try:
        val = float(best)
        if val <= 0:
            return None
        return val
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    """Full normalization pipeline for matching purposes."""
    return norm_arabic(to_ascii_digits(strip_bidi(text)))
