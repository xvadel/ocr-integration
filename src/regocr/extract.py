"""
Layout-aware field extraction module using fields.yaml configuration,
2D spatial RTL bounding-box matching on OCRWord boxes, fuzzy anchor matching (rapidfuzz),
and deterministic per-field validation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rapidfuzz import fuzz

from regocr.models import FieldCandidate, FieldResult
from regocr.normalize import (
    has_arabic_indic_digits,
    norm_arabic,
    parse_iso_date,
    parse_numeric_capital,
    parse_tax_id,
    strip_bidi,
    to_ascii_digits,
)
from regocr.ocr_engines import OCRResult, OCRWord
from regocr.validate import compute_field_confidence, validate_field_value

HERE = Path(__file__).resolve().parent.parent.parent
FIELDS_CONFIG_PATH = HERE / "fields.yaml"

DOC_LABELS = [
    "اسم الشركة", "الاسم التجاري", "اسم المنشأة", "الشكل القانوني", "الشكل الفانوني", "الكيان القانوني",
    "رقم القيد", "رقم الفيد", "رقم السجل", "السجل التجاري", "الرقم الموحد", "الرقم القومي", "مستوى القيد",
    "تاريخ القيد", "تاريخ الفيد", "تاريخ الإصدار", "تاريخ آخر تجديد", "تاريخ اخر تجديد",
    "النشاط", "الغرض", "المحافظة", "العنوان", "المقر", "عنوان المركز", "رأس المال",
    "المدير", "المالك", "المدير المفوض", "رقم البطاقة الضريبية", "البطاقة الضريبية",
    "تاريخ انتهاء", "جهة الإصدار", "مكتب السجل", "مكتب سجل", "ختم", "عينة اختبار", "نموذج وهمي",
    "جهاز تنمية التجارة",
]

NOISE_TOKENS = [
    "نموذج وهمي", "للاختبار فقط", "مستندا رسميا", "عينة اختبار", "بيانات عشوائية",
    "ختم تجريبي", "ختم", "خنم", "ختة", "عينه للاختبار", "عينة للاختبار",
]


def _union_bboxes(bboxes: List[List[float]]) -> Optional[List[float]]:
    valid = [b for b in bboxes if b and len(b) == 4]
    if not valid:
        return None
    return [
        round(min(b[0] for b in valid), 4),
        round(min(b[1] for b in valid), 4),
        round(max(b[2] for b in valid), 4),
        round(max(b[3] for b in valid), 4),
    ]


def _is_noise_word(text: str) -> bool:
    nt = norm_arabic(text).strip()
    if not nt:
        return True
    for n in NOISE_TOKENS:
        if norm_arabic(n) in nt and len(nt) <= len(norm_arabic(n)) + 4:
            return True
    return nt in ("عين", "عينة", "للاختبار", "اختبار", "c", "o", "|", "-")


class FieldExtractor:
    """Extracts structured fields from OCR output or text layers based on fields.yaml."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or FIELDS_CONFIG_PATH
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.fields_def = self.config.get("fields", [])
        self.fields_by_name = {f["name"]: f for f in self.fields_def}

    def extract_from_ocr(
        self,
        ocr_result: OCRResult,
        quality_score: float = 1.0,
    ) -> Tuple[Dict[str, FieldResult], Dict[str, Optional[List[float]]]]:
        """Extract all fields from an OCRResult (or text_layer).

        Returns:
          (fields_dict, crop_boxes_dict)
          where `crop_boxes_dict[field_name]` is the best normalized bbox [x1,y1,x2,y2]
          to crop for a secondary numeric/verification OCR pass (even if value was missed!).
        """
        raw_text = ocr_result.text or ""
        words = ocr_result.words or []
        engine_name = ocr_result.engine
        is_text_layer = engine_name in ("text_layer", "pypdfium2", "pypdfium2_rtl", "pdfplumber_rtl")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # Pre-cluster right-column label rows when OCR word boxes are available
        label_clusters = self._build_label_clusters(words) if words else []

        results: Dict[str, FieldResult] = {}
        crop_boxes: Dict[str, Optional[List[float]]] = {}

        for fdef in self.fields_def:
            name = fdef["name"]
            ftype = fdef.get("type", "text")
            anchors = fdef.get("anchors", [])
            enum_values = fdef.get("enum_values", {})

            val, raw_val, bbox, crop_bbox, eng_conf = self._extract_field(
                name=name,
                ftype=ftype,
                anchors=anchors,
                enum_values=enum_values,
                raw_text=raw_text,
                lines=lines,
                words=words,
                label_clusters=label_clusters,
                engine_name=engine_name,
            )

            crop_boxes[name] = crop_bbox or bbox

            # Grounding & deterministic validation
            grounded = self._check_grounding(val, raw_val, raw_text) if val is not None else False
            validated: Optional[bool] = None
            val_issues: List[str] = []

            if val is not None:
                validated, val_issues = validate_field_value(name, val, raw_val, enum_values)
            elif fdef.get("required", False):
                validated = False
                val_issues = ["not_found"]

            # Status & calibrated confidence
            issues: List[str] = list(val_issues)
            if val is None:
                status = "missing"
                conf = None
                needs_review = bool(fdef.get("required", False))
            else:
                if not grounded:
                    issues.append("not_grounded")
                conf = compute_field_confidence(
                    engine_confidence=eng_conf,
                    engines_agree=None,
                    validated=validated,
                    grounded=grounded,
                    quality_score=quality_score,
                    is_text_layer=is_text_layer,
                )
                if validated is False or not grounded:
                    status = "suspect"
                    needs_review = True
                else:
                    status = "ok"
                    needs_review = False

            # Enum code / Capital unit
            code = None
            unit = None
            if ftype == "enum" and val and enum_values:
                for k, v in enum_values.items():
                    if val == v or norm_arabic(str(val)) == norm_arabic(v):
                        code = k
                        break
            if name == "capital" and val is not None:
                unit = "EGP"

            candidates: List[FieldCandidate] = []
            if val is not None:
                candidates.append(
                    FieldCandidate(
                        value=str(val),
                        raw=raw_val,
                        engine=engine_name,
                        confidence=float(eng_conf if eng_conf is not None else (conf or 0.5)),
                        bbox=bbox,
                        page=ocr_result.page,
                        variant=ocr_result.variant,
                    )
                )

            results[name] = FieldResult(
                value=val,
                raw=raw_val,
                confidence=conf,
                engines_agree=None,
                validated=validated,
                needs_review=needs_review,
                candidates=candidates,
                page=ocr_result.page,
                bbox=bbox,
                status=status,
                grounded=grounded,
                issues=sorted(list(set(issues))),
                code=code,
                unit=unit,
            )

        return results, crop_boxes

    # ------------------------------------------------------------------
    # 2D Spatial RTL Table Geometry Helpers
    # ------------------------------------------------------------------

    def _build_label_clusters(self, words: List[OCRWord]) -> List[Dict[str, Any]]:
        """Group words in the right-hand label column (x1 >= 0.58) into row label clusters.

        Using x1 >= 0.58 (rather than x_mid >= 0.50) ensures right-aligned value tokens
        (e.g. '597472', 'شركة', '100,000', 'الجيزة') that end near x2=0.605 are NEVER
        swallowed into the label column.
        """
        right_words = [w for w in words if w.x1 >= 0.58 and not _is_noise_word(w.text)]
        right_words.sort(key=lambda w: w.y_mid)

        clusters: List[List[OCRWord]] = []
        for w in right_words:
            placed = False
            for cl in clusters:
                cl_y = sum(x.y_mid for x in cl) / len(cl)
                cl_h = max(x.height for x in cl)
                if abs(w.y_mid - cl_y) <= max(0.025, 0.85 * max(cl_h, w.height)):
                    cl.append(w)
                    placed = True
                    break
            if not placed:
                clusters.append([w])

        out = []
        for cl in clusters:
            cl_sorted = sorted(cl, key=lambda x: (round(x.y_mid, 2), -x.x_mid))
            text = " ".join(x.text for x in cl_sorted)
            bbox = _union_bboxes([x.bbox for x in cl_sorted])
            if bbox:
                out.append({"text": text, "norm": norm_arabic(text), "bbox": bbox, "words": cl_sorted})
        return out

    def _match_anchor_cluster(
        self,
        anchors: List[str],
        label_clusters: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find best matching right-column label cluster for a field's anchors."""
        if not label_clusters:
            return None

        best_cl = None
        best_score = 0.0
        for cl in label_clusters:
            cl_norm = cl["norm"]
            if not cl_norm:
                continue
            for anchor in anchors:
                an = norm_arabic(anchor)
                if not an:
                    continue
                if an == cl_norm:
                    return cl
                if an in cl_norm or (len(cl_norm) >= 4 and cl_norm in an):
                    score = 95.0 + min(4.0, len(an) / 10.0)
                else:
                    score = float(fuzz.ratio(an, cl_norm))
                    if len(an) >= 6:
                        score = max(score, float(fuzz.partial_ratio(an, cl_norm)) * 0.92)
                if score > best_score and score >= 80.0:
                    best_score = score
                    best_cl = cl
        return best_cl

    def _find_spatial_row_value(
        self,
        anchors: List[str],
        words: List[OCRWord],
        label_clusters: List[Dict[str, Any]],
        numeric_mode: bool = False,
    ) -> Tuple[Optional[str], Optional[List[float]], Optional[List[float]], Optional[float]]:
        """Use 2D (x, y) geometry to find the value words to the LEFT of the matched label anchor.

        Returns:
          (joined_text, value_bbox, cell_crop_bbox, mean_word_confidence)
        """
        cl = self._match_anchor_cluster(anchors, label_clusters)
        if not cl:
            return None, None, None, None

        ax1, ay1, ax2, ay2 = cl["bbox"]
        ay_mid = 0.5 * (ay1 + ay2)
        ah = max(0.020, ay2 - ay1)
        y_tol = max(0.026, 0.85 * ah)

        # Expected value cell box to the left of the label (for cropped re-OCR fallback!)
        cell_x2 = min(0.66, max(0.615, ax1 - 0.008))
        cell_bbox = [
            0.08,
            round(max(0.0, ay_mid - max(0.022, 0.75 * ah)), 4),
            round(cell_x2, 4),
            round(min(1.0, ay_mid + max(0.022, 0.75 * ah)), 4),
        ]

        # Collect value words in the same horizontal band to the LEFT of the anchor
        val_words = [
            w for w in words
            if w.x1 < ax1 - 0.012
            and w.x2 <= ax1 + 0.025
            and abs(w.y_mid - ay_mid) <= y_tol
            and not _is_noise_word(w.text)
        ]

        if not val_words:
            return None, None, cell_bbox, None

        # Sort words: for pure numeric rows, LTR (ascending x_mid) keeps numbers in natural order;
        # for Arabic text rows, RTL (descending x_mid) joins right-to-left words cleanly.
        if numeric_mode:
            val_words.sort(key=lambda w: w.x_mid)
        else:
            val_words.sort(key=lambda w: -w.x_mid)

        joined = " ".join(w.text for w in val_words).strip()
        val_bbox = _union_bboxes([w.bbox for w in val_words])
        mean_conf = round(sum(w.confidence for w in val_words) / len(val_words), 4)
        return joined, val_bbox, cell_bbox, mean_conf

    # ------------------------------------------------------------------
    # Per-field dispatch & extractors
    # ------------------------------------------------------------------

    def _extract_field(
        self,
        name: str,
        ftype: str,
        anchors: List[str],
        enum_values: Dict[str, str],
        raw_text: str,
        lines: List[str],
        words: List[OCRWord],
        label_clusters: List[Dict[str, Any]],
        engine_name: str,
    ) -> Tuple[Optional[Any], Optional[str], Optional[List[float]], Optional[List[float]], Optional[float]]:
        """Returns (normalized_value, raw_value, value_bbox, crop_bbox, engine_confidence)."""
        if name == "company_name":
            return self._extract_company_name(anchors, lines, words, label_clusters, raw_text)
        elif name == "legal_form":
            return self._extract_legal_form(anchors, enum_values, words, label_clusters, raw_text)
        elif name == "register_number":
            return self._extract_register_number(anchors, lines, words, label_clusters, raw_text)
        elif name == "unified_register_number":
            return self._extract_unified_number(anchors, lines, words, label_clusters)
        elif name == "national_entity_number":
            return self._extract_national_entity(anchors, lines, words, label_clusters)
        elif name == "registration_level":
            return self._extract_registration_level(anchors, lines, words, label_clusters)
        elif name in ("issue_date", "expiry_date"):
            return self._extract_date(name, anchors, words, label_clusters, raw_text)
        elif name == "capital":
            return self._extract_capital(anchors, words, label_clusters, raw_text)
        elif name == "tax_id":
            return self._extract_tax_id(anchors, words, label_clusters, raw_text, engine_name=engine_name)
        elif name == "governorate":
            return self._extract_governorate(anchors, enum_values, words, label_clusters, raw_text)
        elif name == "manager":
            return self._extract_manager(anchors, lines, words, label_clusters)
        elif name == "activity":
            return self._extract_activity(anchors, lines, words, label_clusters)
        elif name == "address":
            return self._extract_address(anchors, lines, words, label_clusters)
        elif name == "office":
            return self._extract_office(anchors, lines, words, label_clusters)
        return self._extract_generic(anchors, lines, words, label_clusters)

    def _find_value_near_anchor_lines(self, anchors: List[str], lines: List[str]) -> Optional[str]:
        """Line-based fallback for text_layer PDFs (when no OCR word boxes exist)."""
        delimiters = " :-\t|/،,؛"
        for i, line in enumerate(lines):
            for anchor in anchors:
                if anchor in line or fuzz.partial_ratio(norm_arabic(anchor), norm_arabic(line)) > 88:
                    parts = line.split(anchor, 1)
                    if len(parts) > 1:
                        rem = parts[1].strip(delimiters)
                        if rem and len(rem) >= 2 and not any(norm_arabic(lbl) in norm_arabic(rem) for lbl in DOC_LABELS if lbl != anchor):
                            if rem not in ("المنشأة", "المنشاه", "التجاري", "الصلاحية"):
                                return rem
                    if len(parts) > 0:
                        pre = parts[0].strip(delimiters)
                        if pre and len(pre) >= 2 and not any(norm_arabic(lbl) in norm_arabic(pre) for lbl in DOC_LABELS if lbl != anchor):
                            if not any(hdr in pre for hdr in ["نموذج وهمي", "سجل تجاري"]):
                                if pre not in ("المنشأة", "المنشاه", "التجاري", "الصلاحية"):
                                    return pre
                    for j in range(i + 1, min(len(lines), i + 3)):
                        cand = lines[j].strip(delimiters)
                        if cand and cand not in ("التجاري", "المنشأة", "المنشاه", "الصلاحية"):
                            if any(norm_arabic(lbl) in norm_arabic(cand) for lbl in DOC_LABELS):
                                break
                            return cand
                    if i > 0:
                        prev = lines[i - 1].strip(delimiters)
                        if prev and not any(norm_arabic(lbl) in norm_arabic(prev) for lbl in DOC_LABELS):
                            return prev
        return None

    def _extract_company_name(self, anchors, lines, words, label_clusters, raw_text):
        # 1. Spatial row match first
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(anchors, words, label_clusters)
        if sp_txt and len(sp_txt) >= 5:
            clean = re.sub(r"^(?:اسم الشركة / المنشأة|اسم الشركة|الاسم التجاري)\s*[:\-]?\s*", "", sp_txt).strip()
            if not any(norm_arabic(clean) == norm_arabic(lf) for lf in ["شركة مساهمة مصرية", "شركة ذات مسئولية محدودة", "شركة تضامن", "شركة توصية بسيطة"]):
                return clean, sp_txt, sp_bbox, cell_bbox, sp_conf

        # 2. Prefix scan in words (to preserve bbox & confidence)
        for w in words:
            clean = w.text.strip()
            if any(norm_arabic(clean) == norm_arabic(lf) for lf in ["شركة مساهمة مصرية", "شركة ذات مسئولية محدودة", "شركة تضامن", "شركة توصية بسيطة"]):
                continue
            if (clean.startswith("شركة ") or clean.startswith("مؤسسة ")) and len(clean) >= 8:
                if not any(stop in clean for stop in ["وهمي", "للاختبار"]):
                    return clean, clean, w.bbox, cell_bbox or w.bbox, w.confidence

        # 3. Line-based fallback (text_layer)
        for ln in lines:
            clean = ln.strip()
            if any(norm_arabic(clean) == norm_arabic(lf) for lf in ["شركة مساهمة مصرية", "شركة ذات مسئولية محدودة", "شركة تضامن", "شركة توصية بسيطة"]):
                continue
            if (clean.startswith("شركة ") or clean.startswith("مؤسسة ")) and len(clean) >= 8:
                if not any(stop in clean for stop in ["وهمي", "للاختبار"]):
                    for a in anchors:
                        clean = clean.split(a)[0].strip()
                    clean = clean.strip(" :-\t|/،,؛")
                    return clean, clean, None, cell_bbox, 0.96

        raw = self._find_value_near_anchor_lines(anchors, lines)
        if raw and raw not in ("المنشأة", "المنشاه", "الشركة", "الشركه"):
            return raw, raw, None, cell_bbox, 0.90
        return None, None, None, cell_bbox, None

    def _extract_legal_form(self, anchors, enum_values, words, label_clusters, raw_text):
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(anchors, words, label_clusters)
        if sp_txt:
            for code, name in enum_values.items():
                if norm_arabic(name) in norm_arabic(sp_txt) or fuzz.ratio(norm_arabic(name), norm_arabic(sp_txt)) >= 82:
                    return name, sp_txt, sp_bbox, cell_bbox, sp_conf

        for w in words:
            for code, name in enum_values.items():
                if norm_arabic(name) in norm_arabic(w.text):
                    return name, w.text, w.bbox, cell_bbox or w.bbox, w.confidence

        for code, name in enum_values.items():
            if norm_arabic(name) in norm_arabic(raw_text):
                return name, name, None, cell_bbox, 0.96
        return None, None, None, cell_bbox, None

    def _extract_register_number(self, anchors, lines, words, label_clusters, raw_text):
        # 1. Spatial row match (guarantees we don't grab unified_register_number or tax_id!)
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(
            anchors, words, label_clusters, numeric_mode=True
        )
        if sp_txt:
            clean = re.sub(r"[0-9٠-٩]{1,4}[/\-.][0-9٠-٩]{1,2}[/\-.][0-9٠-٩]{1,4}", "", sp_txt)
            clean = re.sub(r"[0-9٠-٩]{3}[-\s][0-9٠-٩]{3}[-\s][0-9٠-٩]{3}", "", clean)
            m = re.search(r"([0-9٠-٩]{3,10})", clean)
            if m:
                raw_d = m.group(1)
                return to_ascii_digits(raw_d), raw_d, sp_bbox, sp_bbox or cell_bbox, sp_conf

        # 2. Line-based near anchor (text_layer or OCR fallback)
        for i, ln in enumerate(lines):
            if any(a in ln for a in ["رقم القيد", "رقم الفيد", "رقم السجل"]):
                for offset in [0, -1, 1]:
                    idx = i + offset
                    if 0 <= idx < len(lines):
                        cand = lines[idx].strip()
                        if "/" in cand or "-" in cand:
                            continue
                        m = re.search(r"\b([0-9٠-٩]{4,8})\b", cand)
                        if m:
                            raw_d = m.group(1)
                            return to_ascii_digits(raw_d), raw_d, None, cell_bbox, 0.95

        # 3. Isolated 5-7 digit word fallback
        for w in words:
            if "/" not in w.text and "-" not in w.text and "," not in w.text and "٬" not in w.text:
                m = re.fullmatch(r"\s*([0-9٠-٩]{5,7})\s*", w.text)
                if m:
                    raw_d = m.group(1)
                    return to_ascii_digits(raw_d), raw_d, w.bbox, w.bbox, w.confidence

        for token in re.findall(r"(?<![/\-.\d٠-٩])([0-9٠-٩]{5,7})(?![/\-.\d٠-٩])", raw_text):
            return to_ascii_digits(token), token, None, cell_bbox, 0.85

        return None, None, None, cell_bbox, None

    def _extract_unified_number(self, anchors, lines, words, label_clusters):
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(
            anchors, words, label_clusters, numeric_mode=True
        )
        if sp_txt and re.search(r"[0-9٠-٩]{4,}", sp_txt):
            return to_ascii_digits(sp_txt), sp_txt, sp_bbox, sp_bbox or cell_bbox, sp_conf

        for i, ln in enumerate(lines):
            if any(a in ln for a in anchors):
                for offset in [-1, 1]:
                    idx = i + offset
                    if 0 <= idx < len(lines) and re.search(r"[0-9٠-٩]{4,}", lines[idx]):
                        c = lines[idx].strip()
                        return to_ascii_digits(c), c, None, cell_bbox, 0.90
        return None, None, None, cell_bbox, None

    def _extract_national_entity(self, anchors, lines, words, label_clusters):
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(
            anchors, words, label_clusters, numeric_mode=True
        )
        if sp_txt:
            m = re.search(r"([0-9٠-٩]{10,16})", sp_txt)
            if m:
                return to_ascii_digits(m.group(1)), m.group(1), sp_bbox, sp_bbox or cell_bbox, sp_conf

        for i, ln in enumerate(lines):
            if any(a in ln for a in anchors):
                for offset in [-1, 1]:
                    idx = i + offset
                    if 0 <= idx < len(lines):
                        m = re.search(r"\b([0-9٠-٩]{10,16})\b", lines[idx])
                        if m:
                            return to_ascii_digits(m.group(1)), m.group(1), None, cell_bbox, 0.90
        return None, None, None, cell_bbox, None

    def _extract_registration_level(self, anchors, lines, words, label_clusters):
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(anchors, words, label_clusters)
        if sp_txt and sp_txt.strip() in ("فرع", "رئيسي", "مركز رئيسي"):
            return sp_txt.strip(), sp_txt.strip(), sp_bbox, cell_bbox, sp_conf
        for i, ln in enumerate(lines):
            if any(a in ln for a in anchors):
                for offset in [-1, 1]:
                    idx = i + offset
                    if 0 <= idx < len(lines) and lines[idx].strip() in ("فرع", "رئيسي", "مركز رئيسي"):
                        c = lines[idx].strip()
                        return c, c, None, cell_bbox, 0.90
        return None, None, None, cell_bbox, None

    def _extract_date(self, field_name, anchors, words, label_clusters, raw_text):
        # 1. Spatial row match first!
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(
            anchors, words, label_clusters, numeric_mode=True
        )
        if sp_txt:
            iso = parse_iso_date(sp_txt)
            if iso:
                return iso, sp_txt, sp_bbox, sp_bbox or cell_bbox, sp_conf

        # 2. Line-anchor match (critical for multi-page PDFs where expiry_date is the sole date on Page 2!)
        for ln in (raw_text or "").splitlines():
            if any(norm_arabic(a) in norm_arabic(ln) for a in anchors):
                iso = parse_iso_date(ln)
                if iso:
                    return iso, ln.strip(), None, cell_bbox, 0.96

        # 3. Scan word-level dates (preserves bbox & confidence)
        date_pat = r"(?<![0-9٠-٩])([0-9٠-٩]{1,4}[/\-.][0-9٠-٩]{1,2}[/\-.][0-9٠-٩]{1,4})(?![0-9٠-٩])"
        word_dates: List[Tuple[str, str, Optional[List[float]], float]] = []
        for w in words:
            for m in re.findall(date_pat, w.text):
                iso = parse_iso_date(m)
                if iso:
                    word_dates.append((m, iso, w.bbox, w.confidence))

        if word_dates:
            # Deduplicate by ISO date, sort chronologically
            seen: Dict[str, Tuple[str, str, Optional[List[float]], float]] = {}
            for item in word_dates:
                if item[1] not in seen or item[3] > seen[item[1]][3]:
                    seen[item[1]] = item
            sorted_wd = sorted(seen.values(), key=lambda x: x[1])
            if field_name == "issue_date":
                raw_d, iso_d, wb, wc = sorted_wd[0]
                return iso_d, raw_d, wb, wb or cell_bbox, wc
            elif len(sorted_wd) >= 2:
                raw_d, iso_d, wb, wc = sorted_wd[-1]
                return iso_d, raw_d, wb, wb or cell_bbox, wc

        # 3. Raw text fallback (text_layer)
        all_matches = re.findall(date_pat, raw_text)
        parsed = []
        for d_str in all_matches:
            iso = parse_iso_date(d_str)
            if iso:
                parsed.append((d_str, iso))
        if parsed:
            sorted_dates = sorted(list(dict.fromkeys(parsed)), key=lambda x: x[1])
            if field_name == "issue_date":
                return sorted_dates[0][1], sorted_dates[0][0], None, cell_bbox, 0.95
            elif len(sorted_dates) >= 2:
                return sorted_dates[-1][1], sorted_dates[-1][0], None, cell_bbox, 0.95

        return None, None, None, cell_bbox, None

    def _extract_capital(self, anchors, words, label_clusters, raw_text):
        # 1. Spatial row match on "رأس المال"
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(
            anchors, words, label_clusters, numeric_mode=True
        )

        # Compute a tight numeric crop box for the capital amount!
        # In Arabic RTL ("١٬٠٠٠٬٠٠٠ جنيه مصري"), currency words sit on the LEFT and
        # the numeric amount sits immediately to their RIGHT (near x ~ 0.45..0.61).
        numeric_crop_bbox = cell_bbox
        cl = self._match_anchor_cluster(anchors, label_clusters)
        if cl and words:
            ax1, ay1, ax2, ay2 = cl["bbox"]
            ay_mid = 0.5 * (ay1 + ay2)
            ah = max(0.020, ay2 - ay1)
            row_words = [
                w for w in words
                if w.x1 < ax1 - 0.012 and w.x2 <= ax1 + 0.025 and abs(w.y_mid - ay_mid) <= max(0.026, 0.85 * ah)
            ]
            curr_words = [w for w in row_words if any(cw in w.text for cw in ("جنيه", "مصري", "صر"))]
            num_words = [w for w in row_words if re.search(r"[0-9٠-٩]{2,}", w.text)]
            if num_words:
                numeric_crop_bbox = _union_bboxes([w.bbox for w in num_words])
            elif curr_words:
                curr_x2 = max(w.x2 for w in curr_words)
                curr_x1 = min(w.x1 for w in curr_words)
                right_limit = min(0.625, ax1 - 0.010)
                if curr_x2 < 0.52:
                    # Currency box sits on left, missing number sits directly to its right!
                    numeric_crop_bbox = [
                        round(max(0.08, curr_x2 - 0.005), 4),
                        round(max(0.0, ay_mid - max(0.020, 0.70 * ah)), 4),
                        round(max(curr_x2 + 0.08, right_limit), 4),
                        round(min(1.0, ay_mid + max(0.020, 0.70 * ah)), 4),
                    ]
                else:
                    # Single box spanned both number + currency (e.g. "100,000 جنيه صر") -> crop whole row value!
                    numeric_crop_bbox = [
                        round(max(0.08, curr_x1 - 0.01), 4),
                        round(max(0.0, ay_mid - max(0.020, 0.70 * ah)), 4),
                        round(max(curr_x2 + 0.01, right_limit), 4),
                        round(min(1.0, ay_mid + max(0.020, 0.70 * ah)), 4),
                    ]

        if sp_txt:
            val = parse_numeric_capital(sp_txt)
            if val is not None and val >= 1000:
                return val, sp_txt, sp_bbox, numeric_crop_bbox or sp_bbox, sp_conf

        # 2. Regex on raw_text (handles text_layer PDFs & fallback)
        m = re.search(r"([0-9٠-٩][0-9٠-٩,٬،.\s]*\s*(?:جنيه مصري|جنيه|EGP|LE))", raw_text)
        raw = m.group(1).strip() if m else None
        if not raw:
            m2 = re.search(r"(?:جنيه مصري|جنيه)\s*([0-9٠-٩,٬،.]+)", raw_text)
            if m2:
                raw = m2.group(1).strip()
        if not raw:
            for token in re.findall(r"([0-9٠-٩]{1,3}(?:[,٬،][0-9٠-٩]{3})+)", raw_text):
                raw = token
                break

        if raw:
            val = parse_numeric_capital(raw)
            if val is not None and val >= 1000:
                return val, raw, sp_bbox, numeric_crop_bbox or sp_bbox, sp_conf if sp_conf is not None else 0.95

        # Even if missed, return numeric_crop_bbox so Priority 2's cropped recognizer can recover it!
        return None, None, None, numeric_crop_bbox, None

    def _extract_tax_id(self, anchors, words, label_clusters, raw_text, engine_name: str = ""):
        # Only visual-order sources (pypdfium2 direct on digital_visual PDFs, or EasyOCR LTR)
        # output Arabic-Indic tax_id triplets in reversed visual order (g3-g2-g1).
        # PaddleOCR Arabic recognizer & pdfplumber_rtl/pypdfium2_rtl ALREADY output logical RTL order!
        reverse_indic = engine_name in ("pypdfium2", "easyocr")

        # 1. Spatial row match
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(
            anchors, words, label_clusters, numeric_mode=True
        )
        if sp_txt:
            val = parse_tax_id(sp_txt, reverse_indic_groups=reverse_indic)
            if val:
                return val, sp_txt, sp_bbox, sp_bbox or cell_bbox, sp_conf

        # 2. Scan words for 3-3-3 pattern
        tax_re = r"([0-9٠-٩]{3}[-\s–—][0-9٠-٩]{3}[-\s–—][0-9٠-٩]{3})"
        for w in words:
            m = re.search(tax_re, w.text)
            if m:
                raw_m = m.group(1)
                val = parse_tax_id(raw_m, reverse_indic_groups=reverse_indic)
                if val:
                    return val, raw_m, w.bbox, w.bbox, w.confidence

        # 3. Raw text scan (text_layer PDFs)
        m = re.search(tax_re, raw_text)
        if m:
            raw_m = m.group(1)
            val = parse_tax_id(raw_m, reverse_indic_groups=reverse_indic)
            if val:
                return val, raw_m, None, cell_bbox, 0.96

        return None, None, None, cell_bbox, None

    def _extract_governorate(self, anchors, enum_values, words, label_clusters, raw_text):
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(anchors, words, label_clusters)
        if sp_txt:
            for code, name in enum_values.items():
                if norm_arabic(name) in norm_arabic(sp_txt):
                    return name, sp_txt, sp_bbox, cell_bbox, sp_conf

        for w in words:
            for code, name in enum_values.items():
                if norm_arabic(w.text).strip() == norm_arabic(name):
                    return name, w.text, w.bbox, cell_bbox or w.bbox, w.confidence

        for code, name in enum_values.items():
            if name in raw_text or norm_arabic(name) in norm_arabic(raw_text):
                return name, name, None, cell_bbox, 0.94
        return None, None, None, cell_bbox, None

    def _extract_manager(self, anchors, lines, words, label_clusters):
        stop_words = [
            "جنيه", "جنيه مصري", "رأس المال", "رأس مال", "العنوان", "النشاط",
            "ختم", "خنم", "نموذج", "سجل تجاري",
        ]
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(anchors, words, label_clusters)
        if sp_txt:
            clean = re.sub(r"^(?:المدير / المالك|المدير المفوض بالتوقيع|المدير|المالك)\s*[:\-]?\s*", "", sp_txt).strip()
            if not any(sw in clean for sw in stop_words) and len(clean.split()) >= 2 and not re.search(r"\d{3,}", clean):
                return clean, sp_txt, sp_bbox, cell_bbox, sp_conf

        raw = self._find_value_near_anchor_lines(anchors, lines)
        if raw:
            clean = re.sub(r"^(?:المدير / المالك|المدير المفوض بالتوقيع|المدير|المالك)\s*[:\-]?\s*", "", raw).strip()
            if not any(sw in clean for sw in stop_words) and len(clean.split()) >= 2 and not re.search(r"\d{3,}", clean):
                return clean, raw, None, cell_bbox, 0.92
        return None, None, None, cell_bbox, None

    def _extract_activity(self, anchors, lines, words, label_clusters):
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(anchors, words, label_clusters)
        if sp_txt and len(sp_txt) >= 4:
            clean = re.sub(r"^(?:النشاط|الغرض)\s*[:\-]?\s*", "", sp_txt).strip()
            return clean, sp_txt, sp_bbox, cell_bbox, sp_conf

        raw = self._find_value_near_anchor_lines(anchors, lines)
        if raw:
            clean = re.sub(r"^(?:النشاط|الغرض)\s*[:\-]?\s*", "", raw).strip()
            return clean, raw, None, cell_bbox, 0.92
        return None, None, None, cell_bbox, None

    def _extract_address(self, anchors, lines, words, label_clusters):
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(anchors, words, label_clusters)
        if sp_txt and len(sp_txt) >= 5:
            clean = re.sub(r"^(?:عنوان المركز الرئيسي|العنوان|المقر)\s*[:\-]?\s*", "", sp_txt).strip()
            return to_ascii_digits(clean), sp_txt, sp_bbox, cell_bbox, sp_conf

        raw = self._find_value_near_anchor_lines(anchors, lines)
        if raw:
            clean = re.sub(r"^(?:عنوان المركز الرئيسي|العنوان|المقر)\s*[:\-]?\s*", "", raw).strip()
            return to_ascii_digits(clean), raw, None, cell_bbox, 0.90
        return None, None, None, cell_bbox, None

    def _extract_office(self, anchors, lines, words, label_clusters):
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(anchors, words, label_clusters)
        if sp_txt and len(sp_txt) >= 6:
            clean = re.sub(r"(?:ختم\s*تجريبي|ختم|تجريبي|ريبي|يبي)\b", " ", sp_txt)
            clean = re.sub(r"\s+", " ", clean).strip(" :-\t|/،,؛")
            if clean:
                return clean, sp_txt, sp_bbox, cell_bbox, sp_conf

        for w in words:
            if ("مكتب سجل" in w.text or "مكتب السجل" in w.text) and len(w.text) >= 12:
                return w.text.strip(), w.text.strip(), w.bbox, cell_bbox or w.bbox, w.confidence

        for ln in lines:
            clean = ln.strip()
            if "مكتب السجل التجاري" in clean or "مكتب سجل تجاري" in clean:
                clean = re.sub(r"\s*جهة الإصدار\s*[:\-]?\s*", " ", clean).strip(" :-\t|/،,؛")
                clean = re.sub(r"\s+", " ", clean)
                if len(clean) >= 10:
                    return clean, clean, None, cell_bbox, 0.92

        raw = self._find_value_near_anchor_lines(anchors, lines)
        if raw:
            return raw, raw, None, cell_bbox, 0.88
        return None, None, None, cell_bbox, None

    def _extract_generic(self, anchors, lines, words, label_clusters):
        sp_txt, sp_bbox, cell_bbox, sp_conf = self._find_spatial_row_value(anchors, words, label_clusters)
        if sp_txt:
            return sp_txt, sp_txt, sp_bbox, cell_bbox, sp_conf
        raw = self._find_value_near_anchor_lines(anchors, lines)
        if raw:
            return raw, raw, None, cell_bbox, 0.90
        return None, None, None, cell_bbox, None

    def _check_grounding(self, extracted_value: Any, raw_value: Optional[str], raw_text: str) -> bool:
        if not raw_text:
            return False
        if raw_value and raw_value in raw_text:
            return True
        if raw_value:
            norm_raw = norm_arabic(raw_value)
            if norm_raw and norm_raw in norm_arabic(raw_text):
                return True
        if extracted_value is not None:
            ext_digits = re.sub(r"\D", "", to_ascii_digits(str(extracted_value)))
            raw_digits = re.sub(r"\D", "", to_ascii_digits(raw_text))
            if ext_digits and len(ext_digits) >= 3 and ext_digits in raw_digits:
                return True
            # Check ISO date formatted as DD/MM/YYYY in text
            if isinstance(extracted_value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", extracted_value):
                y, m, d = extracted_value.split("-")
                ascii_raw = to_ascii_digits(raw_text)
                for p in (f"{d}/{m}/{y}", f"{int(d)}/{int(m)}/{y}", f"{y}/{m}/{d}", f"{y}-{m}-{d}", f"{d}-{m}-{y}"):
                    if p in ascii_raw:
                        return True
            # Check reversed Arabic-Indic tax_id triplet in raw text
            if isinstance(extracted_value, str) and re.fullmatch(r"\d{3}-\d{3}-\d{3}", extracted_value):
                g1, g2, g3 = extracted_value.split("-")
                rev_digits = f"{g3}{g2}{g1}"
                if rev_digits in raw_digits:
                    return True
        return False
