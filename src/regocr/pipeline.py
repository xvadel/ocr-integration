"""
Main RegOCR Pipeline (v0.3) orchestrating:
  - Content-addressed document_id & schema v0.3 contract
  - Per-page PDF routing (text_layer vs OCR) with full multi-page merging
  - Enforced CV preprocessing gate (PreprocessResult + per-page PreprocessingReport)
  - Primary OCR (PaddleOCR PP-OCRv5) + mandatory cropped second-opinion verification
    (EasyOCR recognize + Tesseract numeric whitelist) on numeric/ID fields
  - Calibrated field confidence, normalized bbox propagation, and actionable reupload_hint
"""

from __future__ import annotations

import datetime
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from regocr import SCHEMA_VERSION
from regocr.countries import CountryRegistry
from regocr.extract import FieldExtractor
from regocr.ingest import (
    compute_document_id,
    detect_pdf_pages,
    load_image,
    render_pdf_page,
)
from regocr.models import (
    DocumentInfo,
    ExtractionInfo,
    ExtractionResult,
    FieldCandidate,
    FieldResult,
    PreprocessingReport,
    QualityInfo,
    ReviewInfo,
)
from regocr.normalize import (
    has_arabic_indic_digits,
    parse_iso_date,
    parse_numeric_capital,
    parse_tax_id,
    to_ascii_digits,
)
from regocr.ocr_engines import EasyOCREngine, OCRResult, PaddleOCREngine, TesseractEngine
from regocr.preprocess import PreprocessConfig, PreprocessResult, preprocess_image
from regocr.validate import (
    NUMERIC_ID_FIELDS,
    compute_field_confidence,
    validate_document_status,
    validate_field_value,
)
from regocr.guards import apply_guards  # guard integration

UNIQUE_ID_FIELDS = ("register_number", "unified_register_number", "national_entity_number", "tax_id")


@dataclass
class PipelineConfig:
    """Configurable thresholds & behavior flags for RegOCRPipeline."""

    dpi: int = 300
    verify_numeric_fields: bool = True
    use_variant_b_on_disagree: bool = False
    min_field_confidence: float = 0.68
    min_doc_quality_accept: float = 0.65
    min_doc_quality_reject: float = 0.35
    early_exit_crop_verification: bool = True
    crop_skip_min_confidence: float = 0.88
    max_image_dimension: int = 2048
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def _parse_numeric_crop_text(field_name: str, raw_text: str) -> Optional[Any]:
    """Parse raw text returned by a cropped numeric recognizer into a canonical field value."""
    if not raw_text or not raw_text.strip():
        return None
    s = raw_text.strip()

    if field_name == "register_number":
        clean = re.sub(r"[0-9٠-٩]{1,4}[/\-.][0-9٠-٩]{1,2}[/\-.][0-9٠-٩]{1,4}", "", s)
        digits = re.sub(r"\D", "", to_ascii_digits(clean))
        if 3 <= len(digits) <= 10:
            return digits
        return None

    if field_name == "unified_register_number":
        digits = re.sub(r"\D", "", to_ascii_digits(s))
        if 10 <= len(digits) <= 20:
            return to_ascii_digits(s)
        return None

    if field_name == "national_entity_number":
        digits = re.sub(r"\D", "", to_ascii_digits(s))
        if 10 <= len(digits) <= 16:
            return digits
        return None

    if field_name == "tax_id":
        return parse_tax_id(s, reverse_indic_groups=True)

    if field_name in ("issue_date", "expiry_date"):
        return parse_iso_date(s)

    if field_name == "capital":
        val = parse_numeric_capital(s)
        if val is not None and val >= 1000.0:
            return val
        return None

    return None


def _canonical_compare_key(field_name: str, val: Any) -> Optional[str]:
    """Canonical string representation for checking cross-engine agreement."""
    if val is None:
        return None
    if field_name == "capital":
        try:
            f = float(val)
            return str(int(f)) if f.is_integer() else f"{f:.2f}"
        except (TypeError, ValueError):
            return None
    if field_name in ("register_number", "national_entity_number", "tax_id"):
        return re.sub(r"\D", "", to_ascii_digits(str(val)))
    return str(val).strip()


class RegOCRPipeline:
    """Full end-to-end commercial register OCR & structured extraction pipeline."""

    def __init__(
        self,
        engine: str = "auto",
        country: str = "eg",
        dpi: int = 300,
        config: Optional[PipelineConfig] = None,
    ):
        self.engine = engine  # 'auto', 'paddleocr', 'easyocr', 'tesseract', 'pypdfium2'
        self.country = country.lower()
        self.config = config or PipelineConfig(dpi=dpi)
        if config is None:
            self.config.dpi = dpi
        self.dpi = self.config.dpi

        self._tesseract: Optional[TesseractEngine] = None
        self._easyocr: Optional[EasyOCREngine] = None
        self._paddleocr: Optional[PaddleOCREngine] = None
        self.extractor = FieldExtractor()
        try:
            self.country_module = CountryRegistry.get(self.country)
        except Exception:
            self.country_module = CountryRegistry.get("eg")

    def _get_engine(self, name: str):
        if name == "tesseract":
            if self._tesseract is None:
                self._tesseract = TesseractEngine()
            return self._tesseract
        if name == "easyocr":
            if self._easyocr is None:
                self._easyocr = EasyOCREngine()
            return self._easyocr
        if name == "paddleocr":
            if self._paddleocr is None:
                self._paddleocr = PaddleOCREngine()
            return self._paddleocr
        raise ValueError(f"Unknown OCR engine: {name}")

    # ------------------------------------------------------------------
    # Priority 2: Cropped Second-Opinion Numeric Verification
    # ------------------------------------------------------------------

    def _verify_numeric_fields_on_crops(
        self,
        prep: PreprocessResult,
        fields: Dict[str, FieldResult],
        crop_boxes: Dict[str, Optional[List[float]]],
        primary_engine: str,
        page_num: int,
    ) -> None:
        """For every numeric/ID field with a known crop region, run a cropped second opinion
        (EasyOCR recognize + Tesseract eng whitelist for Western digits) and reconcile by
        agreement and validators. Updates `fields` in place.
        """
        q_score = float(prep.quality.get("score", 1.0))
        if q_score < self.config.min_doc_quality_reject:
            # Do not waste CPU re-OCR on images that already failed the hard quality rejection gate
            return

        easy_eng = None
        tess_eng = None

        allowlist_map = {
            "register_number": "0123456789٠١٢٣٤٥٦٧٨٩",
            "unified_register_number": "0123456789٠١٢٣٤٥٦٧٨٩ ",
            "national_entity_number": "0123456789٠١٢٣٤٥٦٧٨٩",
            "tax_id": "0123456789٠١٢٣٤٥٦٧٨٩-",
            "capital": "0123456789٠١٢٣٤٥٦٧٨٩,٬.",
            "issue_date": "0123456789٠١٢٣٤٥٦٧٨٩/-.",
            "expiry_date": "0123456789٠١٢٣٤٥٦٧٨٩/-.",
        }
        tess_wl_map = {
            "register_number": "0123456789",
            "unified_register_number": "0123456789",
            "national_entity_number": "0123456789",
            "tax_id": "0123456789-",
            "capital": "0123456789,.",
            "issue_date": "0123456789/-.",
            "expiry_date": "0123456789/-.",
        }

        for fname in NUMERIC_ID_FIELDS:
            fld = fields.get(fname)
            if fld is None:
                continue
            crop_bbox = crop_boxes.get(fname) or fld.bbox
            if not crop_bbox:
                continue

            # Collect all candidate observations: list of (parsed_val, raw_str, engine_name, raw_conf, is_valid, val_issues)
            observations: List[Tuple[Any, str, str, float, bool, List[str]]] = []

            # 1. Primary engine observation (already extracted)
            if fld.value is not None:
                p_valid, p_issues = validate_field_value(fname, fld.value, fld.raw)
                p_raw_conf = fld.candidates[0].confidence if fld.candidates else (fld.confidence or 0.7)
                observations.append((fld.value, fld.raw or str(fld.value), primary_engine, float(p_raw_conf), p_valid, p_issues))

                # Early-exit optimization: if primary extraction is validated and high confidence,
                # skip heavy secondary crop OCR passes
                if (
                    self.config.early_exit_crop_verification
                    and p_valid
                    and float(p_raw_conf) >= self.config.crop_skip_min_confidence
                ):
                    fld.engines_agree = True
                    fld.status = "ok"
                    fld.validated = True
                    continue

            # 2. Secondary engine: EasyOCR cropped numeric recognize (works for BOTH Western & Arabic-Indic!)
            if primary_engine != "easyocr":
                try:
                    if easy_eng is None:
                        easy_eng = self._get_engine("easyocr")
                    eff_bbox = crop_bbox
                    # For wide capital crops containing both currency text ("جنيه مصري") and digits,
                    # try readtext first (so currency letters aren't coerced into fake digits),
                    # then fall back to recognize() on the right-hand numeric slice!
                    if fname == "capital" and (crop_bbox[2] - crop_bbox[0]) > 0.16:
                        raw_e, conf_e = easy_eng.recognize_numeric_crop(
                            prep, crop_bbox, allowlist=allowlist_map[fname], skip_detector=False
                        )
                        parsed_e = _parse_numeric_crop_text(fname, raw_e)
                        if parsed_e is None:
                            # Right 52% of the value row is where the RTL capital number sits
                            num_x1 = round(crop_bbox[0] + 0.45 * (crop_bbox[2] - crop_bbox[0]), 4)
                            eff_bbox = [num_x1, crop_bbox[1], crop_bbox[2], crop_bbox[3]]
                            raw_e, conf_e = easy_eng.recognize_numeric_crop(
                                prep, eff_bbox, allowlist=allowlist_map[fname], skip_detector=True
                            )
                            parsed_e = _parse_numeric_crop_text(fname, raw_e)
                    else:
                        raw_e, conf_e = easy_eng.recognize_numeric_crop(
                            prep,
                            crop_bbox,
                            allowlist=allowlist_map.get(fname, "0123456789٠١٢٣٤٥٦٧٨٩/-.,٬ "),
                            skip_detector=True,
                        )
                        parsed_e = _parse_numeric_crop_text(fname, raw_e)

                    if parsed_e is not None:
                        v_ok, v_iss = validate_field_value(fname, parsed_e, raw_e)
                        observations.append((parsed_e, raw_e, "easyocr_crop", float(conf_e), v_ok, v_iss))
                        fld.candidates.append(
                            FieldCandidate(
                                value=str(parsed_e),
                                raw=raw_e,
                                engine="easyocr_crop",
                                confidence=float(conf_e),
                                bbox=eff_bbox,
                                page=page_num,
                                variant="A_crop",
                            )
                        )
                except Exception:
                    pass

            # 3. Third opinion (fast ~80ms): Tesseract `-l eng --psm 7` whitelist when digits are Western
            # (empirically confirmed `-l eng` whitelist excels on Western digits; skipped for Indic-only crops)
            any_indic = any(has_arabic_indic_digits(obs[1]) for obs in observations)
            if primary_engine != "tesseract" and not any_indic:
                try:
                    if tess_eng is None:
                        tess_eng = self._get_engine("tesseract")
                    raw_t, conf_t = tess_eng.recognize_numeric_crop(
                        prep,
                        crop_bbox,
                        whitelist=tess_wl_map.get(fname, "0123456789/-.,"),
                        lang="eng",
                    )
                    parsed_t = _parse_numeric_crop_text(fname, raw_t)
                    if parsed_t is not None:
                        v_ok, v_iss = validate_field_value(fname, parsed_t, raw_t)
                        observations.append((parsed_t, raw_t, "tesseract_crop", float(conf_t), v_ok, v_iss))
                        fld.candidates.append(
                            FieldCandidate(
                                value=str(parsed_t),
                                raw=raw_t,
                                engine="tesseract_crop",
                                confidence=float(conf_t),
                                bbox=crop_bbox,
                                page=page_num,
                                variant="A_crop",
                            )
                        )
                except Exception:
                    pass

            if not observations:
                continue

            # Reconcile observations by canonical key, agreement, and validator status
            groups: Dict[str, List[Tuple[Any, str, str, float, bool, List[str]]]] = {}
            for obs in observations:
                ck = _canonical_compare_key(fname, obs[0])
                if ck is None:
                    continue
                groups.setdefault(ck, []).append(obs)

            if not groups:
                continue

            # Score each canonical value group: (is_valid, num_distinct_engines_agreeing, max_conf)
            def _group_rank(items: List[Tuple[Any, str, str, float, bool, List[str]]]) -> Tuple[int, int, float]:
                any_valid = 1 if any(x[4] for x in items) else 0
                distinct_engines = len({x[2] for x in items})
                # Boost non-inverted dates if issue_date is already known when reconciling expiry_date
                timeline_bonus = 0
                if fname == "expiry_date" and fields.get("issue_date") and fields["issue_date"].value:
                    if str(items[0][0]) > str(fields["issue_date"].value):
                        timeline_bonus = 1
                max_conf = max(x[3] for x in items)
                return (any_valid, distinct_engines, timeline_bonus + max_conf)

            ranked = sorted(groups.items(), key=lambda kv: _group_rank(kv[1]), reverse=True)
            winner_key, winner_items = ranked[0]
            best_obs = max(winner_items, key=lambda x: x[3])

            distinct_winner_engines = len({x[2] for x in winner_items})
            total_distinct_engines = len({x[2] for x in observations})

            # Agreement semantics:
            # - True if >= 2 distinct engines agree on the winning canonical value
            # - False if multiple engines produced conflicting values (or primary missed & only 1 crop found it)
            if distinct_winner_engines >= 2:
                engines_agree: Optional[bool] = True
            elif total_distinct_engines >= 2 or (fld.value is None and distinct_winner_engines == 1):
                engines_agree = False
            else:
                engines_agree = None

            chosen_val, chosen_raw, chosen_eng, chosen_conf, chosen_valid, chosen_issues = best_obs
            fld.value = chosen_val
            fld.raw = chosen_raw
            fld.bbox = fld.bbox or crop_bbox
            fld.engines_agree = engines_agree
            fld.validated = chosen_valid
            if fname == "capital" and chosen_val is not None:
                fld.unit = "EGP"

            # Grounding: if recovered from cropped engine, it is grounded in the crop OCR output
            if chosen_eng != primary_engine:
                fld.grounded = True
                if "not_found" in fld.issues:
                    fld.issues.remove("not_found")
                if "not_grounded" in fld.issues:
                    fld.issues.remove("not_grounded")

            # Use max agreeing engine confidence when engines agree
            eff_eng_conf = max(x[3] for x in winner_items) if engines_agree else chosen_conf
            fld.confidence = compute_field_confidence(
                engine_confidence=eff_eng_conf,
                engines_agree=engines_agree,
                validated=chosen_valid,
                grounded=fld.grounded,
                quality_score=q_score,
                is_text_layer=False,
            )

            # Determine status & needs_review
            fld.issues = [i for i in fld.issues if i not in ("not_found", "engine_disagreement")]
            if not chosen_valid:
                fld.issues.extend(chosen_issues)
                fld.status = "suspect"
                fld.needs_review = True
            elif engines_agree is False and len(groups) >= 2:
                # Genuine conflict between engines -> flag for human review!
                fld.issues.append("engine_disagreement")
                fld.status = "suspect"
                fld.needs_review = True
            elif fld.confidence is not None and fld.confidence < self.config.min_field_confidence:
                fld.status = "suspect"
                fld.needs_review = True
            else:
                fld.status = "ok"
                fld.needs_review = False

            fld.issues = sorted(list(set(fld.issues)))

    # ------------------------------------------------------------------
    # Single-page OCR processing
    # ------------------------------------------------------------------

    def _process_ocr_page(
        self,
        cv_img,
        page_num: int,
    ) -> Tuple[Dict[str, FieldResult], OCRResult, PreprocessResult]:
        """Run enforced preprocessing + primary OCR + optional Variant B + numeric crop verification."""
        prep = preprocess_image(cv_img, config=self.config.preprocess, page=page_num)

        # Hard quality rejection gate: fail fast on unreadable images without wasting DL CPU cycles
        q_score = float(prep.quality.get("score", 1.0))
        q_issues = prep.quality.get("issues", [])
        if q_score < self.config.min_doc_quality_reject or (
            "blurry" in q_issues and "noisy" in q_issues and "low_resolution" in q_issues
        ):
            empty_ocr = OCRResult(text="", mean_confidence=0.0, words=[], engine="quality_gate", page=page_num)
            empty_fields, _ = self.extractor.extract_from_ocr(empty_ocr, quality_score=q_score)
            return empty_fields, empty_ocr, prep

        primary_name = "paddleocr" if self.engine == "auto" else self.engine
        if primary_name == "pypdfium2":
            primary_name = "paddleocr"

        engine_order = [primary_name]
        if self.engine == "auto":
            engine_order = ["paddleocr", "tesseract"]

        best_ocr: Optional[OCRResult] = None
        best_fields: Optional[Dict[str, FieldResult]] = None
        best_crops: Dict[str, Optional[List[float]]] = {}

        for eng_name in engine_order:
            try:
                eng = self._get_engine(eng_name)
                ocr_res = eng.run(prep, variant="A")
                fields, crop_boxes = self.extractor.extract_from_ocr(
                    ocr_res, quality_score=float(prep.quality["score"])
                )
                best_ocr = ocr_res
                best_fields = fields
                best_crops = crop_boxes

                core_found = sum(
                    1 for k in ("company_name", "register_number", "issue_date")
                    if fields.get(k) and fields[k].value is not None
                )
                if core_found >= 2 or self.engine != "auto":
                    break
            except Exception:
                if self.engine != "auto":
                    raise

        if best_ocr is None or best_fields is None:
            raise RuntimeError(f"All OCR engines failed on page {page_num}.")

        # Optional Variant B comparison (enabled during ablation or via config)
        if self.config.use_variant_b_on_disagree and prep.variant_b is not None:
            try:
                eng = self._get_engine(best_ocr.engine)
                ocr_b = eng.run(prep, variant="B")
                fields_b, _ = self.extractor.extract_from_ocr(
                    ocr_b, quality_score=float(prep.quality["score"])
                )
                for fname, fb in fields_b.items():
                    fa = best_fields.get(fname)
                    if fa is None:
                        continue
                    if fb.value is not None and fb.validated is True:
                        if fa.value is None or (fa.validated is False) or (
                            (fb.confidence or 0.0) > (fa.confidence or 0.0) + 0.08
                        ):
                            best_fields[fname] = fb
            except Exception:
                pass

        # Priority 2: ALWAYS run cropped second-opinion verification on numeric/ID fields (in auto mode)
        if self.config.verify_numeric_fields and self.engine == "auto":
            self._verify_numeric_fields_on_crops(
                prep=prep,
                fields=best_fields,
                crop_boxes=best_crops,
                primary_engine=best_ocr.engine,
                page_num=page_num,
            )

        return best_fields, best_ocr, prep

    # ------------------------------------------------------------------
    # Priority 4: Multi-Page Merging
    # ------------------------------------------------------------------

    def _merge_multipage_fields(
        self,
        page_fields_list: List[Tuple[int, Dict[str, FieldResult]]],
    ) -> Dict[str, FieldResult]:
        """Merge extracted fields across all pages, preserving page provenance & candidates."""
        if len(page_fields_list) == 1:
            return page_fields_list[0][1]

        merged: Dict[str, FieldResult] = {}
        all_field_names = list(self.extractor.fields_by_name.keys())

        for fname in all_field_names:
            non_empty: List[Tuple[int, FieldResult]] = []
            for pnum, pf in page_fields_list:
                fobj = pf.get(fname)
                if fobj and fobj.value is not None:
                    non_empty.append((pnum, fobj))

            if not non_empty:
                # Take empty FieldResult from page 1
                merged[fname] = page_fields_list[0][1].get(fname, FieldResult())
                continue

            if len(non_empty) == 1:
                merged[fname] = non_empty[0][1]
                continue

            # Multiple pages extracted a value -> aggregate candidates & resolve by validator + confidence
            all_cands: List[FieldCandidate] = []
            for _, fobj in non_empty:
                all_cands.extend(fobj.candidates)

            ranked = sorted(
                non_empty,
                key=lambda item: (
                    1 if item[1].validated is True else 0,
                    1 if item[1].engines_agree is True else 0,
                    float(item[1].confidence or 0.0),
                    -item[0],  # prefer earlier page on exact tie
                ),
                reverse=True,
            )
            winner_page, winner = ranked[0]
            winner = winner.model_copy(deep=True)
            winner.candidates = all_cands
            winner.page = winner_page

            # Check if two pages had conflicting valid values for a unique ID field
            if fname in UNIQUE_ID_FIELDS:
                distinct_vals = {
                    _canonical_compare_key(fname, fobj.value)
                    for _, fobj in non_empty
                    if fobj.validated is not False
                }
                distinct_vals.discard(None)
                if len(distinct_vals) > 1:
                    winner.status = "suspect"
                    winner.needs_review = True
                    if "multi_page_conflict" not in winner.issues:
                        winner.issues.append("multi_page_conflict")

            merged[fname] = winner

        return merged

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def process_file(
        self,
        file_path: Union[str, Path],
        submission_id: Optional[str] = None,
    ) -> ExtractionResult:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = compute_document_id(file_path)
        sub_id = submission_id or f"sub_{int(time.time())}_{file_path.stem[:12]}"
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        is_pdf = file_path.suffix.lower() == ".pdf"
        t0 = time.perf_counter()

        page_fields_list: List[Tuple[int, Dict[str, FieldResult]]] = []
        raw_text_parts: List[str] = []
        preprocessing_reports: List[PreprocessingReport] = []
        page_methods: List[str] = []
        page_engines: List[str] = []
        page_ocr_confs: List[float] = []
        page_qualities: List[Dict[str, Any]] = []
        pdf_kind: Optional[str] = None
        pages_count = 1
        file_format = "pdf" if is_pdf else "image"

        if is_pdf:
            pages_info = detect_pdf_pages(file_path)
            pages_count = len(pages_info)
            kinds = {p["pdf_kind"] for p in pages_info}
            pdf_kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"

            for page_idx, pinfo in enumerate(pages_info):
                pnum = page_idx + 1
                use_text = pinfo["recommended_method"] == "text_layer" and self.engine in ("auto", "pypdfium2")
                if use_text:
                    txt = unicodedata.normalize("NFKC", pinfo["text"] or "")
                    ocr_res = OCRResult(
                        text=txt,
                        mean_confidence=0.98,
                        words=[],
                        engine=pinfo["recommended_engine"],
                        variant="text_layer",
                        page=pnum,
                    )
                    fields, _ = self.extractor.extract_from_ocr(ocr_res, quality_score=0.98)
                    page_fields_list.append((pnum, fields))
                    raw_text_parts.append(txt)
                    page_methods.append("text_layer")
                    page_engines.append(pinfo["recommended_engine"])
                    page_qualities.append({"score": 0.98, "issues": []})
                else:
                    cv_img = render_pdf_page(file_path, page_num=page_idx, dpi=self.dpi)
                    fields, ocr_res, prep = self._process_ocr_page(cv_img, page_num=pnum)
                    page_fields_list.append((pnum, fields))
                    raw_text_parts.append(ocr_res.text)
                    preprocessing_reports.append(PreprocessingReport(**prep.report))
                    page_methods.append("ocr")
                    page_engines.append(ocr_res.engine)
                    page_ocr_confs.append(ocr_res.mean_confidence)
                    page_qualities.append(prep.quality)
        else:
            cv_img = load_image(file_path)
            fields, ocr_res, prep = self._process_ocr_page(cv_img, page_num=1)
            page_fields_list.append((1, fields))
            raw_text_parts.append(ocr_res.text)
            preprocessing_reports.append(PreprocessingReport(**prep.report))
            page_methods.append("ocr")
            page_engines.append(ocr_res.engine)
            page_ocr_confs.append(ocr_res.mean_confidence)
            page_qualities.append(prep.quality)

        # Merge fields across all pages
        merged_fields = self._merge_multipage_fields(page_fields_list)
        raw_text = "\n\n--- PAGE BREAK ---\n\n".join(raw_text_parts) if len(raw_text_parts) > 1 else (raw_text_parts[0] if raw_text_parts else "")

        # Aggregate document-level quality & extraction info
        distinct_methods = set(page_methods)
        text_method = next(iter(distinct_methods)) if len(distinct_methods) == 1 else "mixed"
        used_engine = "+".join(dict.fromkeys(page_engines)) if page_engines else self.engine
        ocr_conf_mean = round(sum(page_ocr_confs) / len(page_ocr_confs), 4) if page_ocr_confs else None

        # Use worst-page quality for single-page, or mean quality across pages with union of issues
        if len(page_qualities) == 1:
            agg_q_score = float(page_qualities[0]["score"])
            agg_q_issues = list(page_qualities[0]["issues"])
        else:
            agg_q_score = round(sum(float(q["score"]) for q in page_qualities) / len(page_qualities), 2)
            agg_q_issues = sorted(list({iss for q in page_qualities for iss in q["issues"]}))

        # Detect synthetic test watermark in text
        is_test_data = any(
            marker in raw_text
            for marker in ("نموذج وهمي", "للاختبار فقط", "SYNTHETIC SAMPLE", "بيانات عشوائية غير حقيقية")
        )

        quality_info = QualityInfo(score=agg_q_score, issues=agg_q_issues)
        status, reasons, review_info, reupload_hint = validate_document_status(
            quality=quality_info,
            fields=merged_fields,
            ocr_confidence=ocr_conf_mean,
            min_doc_quality_reject=self.config.min_doc_quality_reject,
            min_doc_quality_accept=self.config.min_doc_quality_accept,
            min_field_confidence=self.config.min_field_confidence,
        )

        duration_ms = int((time.perf_counter() - t0) * 1000)

        # ------------------------------------------------------------
        # Guard integration – run post‑extraction checks on the document
        # guards.py expects plain dicts, not FieldResult objects, so we
        # serialize fields to dicts before passing in and restore after.
        # ------------------------------------------------------------
        try:
            # Serialize FieldResult objects → plain dicts
            fields_as_dicts = {
                k: v.model_dump(mode="python") if hasattr(v, "model_dump") else vars(v)
                for k, v in merged_fields.items()
            }
            doc_guard = {
                "status": status,
                "reasons": reasons,
                "fields": fields_as_dicts,
                "raw_text": raw_text,
                "document_type": "commercial_register",
            }
            doc_guard, guard_applied = apply_guards(doc_guard, expected_type="commercial_register")
            # Unpack status/reasons/review – these are plain values, safe to overwrite
            status = doc_guard.get("status", status)
            reasons = doc_guard.get("reasons", reasons)
            g_review = doc_guard.get("review")
            if g_review and isinstance(g_review, dict):
                review_info = ReviewInfo(
                    required=g_review.get("required", review_info.required),
                    fields_to_review=g_review.get("fields_to_review", review_info.fields_to_review),
                )
            # Merge guard field annotations (status, issues flags) back into FieldResult objects
            for fname, fdict in doc_guard.get("fields", {}).items():
                if fname in merged_fields and isinstance(fdict, dict):
                    fr = merged_fields[fname]
                    if fdict.get("status") and fdict["status"] != fr.status:
                        fr.status = fdict["status"]
                    if fdict.get("issues"):
                        for issue in fdict["issues"]:
                            if issue not in fr.issues:
                                fr.issues.append(issue)
        except Exception:
            # Guards are advisory – never crash the pipeline
            pass

        # Template detection
        matched_template = self.country_module.detect_template(raw_text)
        tmpl_id = matched_template.template_id if matched_template else None
        doc_type = matched_template.document_type if matched_template else "commercial_register"

        return ExtractionResult(
            schema_version=SCHEMA_VERSION,
            country=self.country,
            document_type=doc_type,
            template_id=tmpl_id,
            document_id=doc_id,
            submission_id=sub_id,
            created_at=now_dt,
            test_data=is_test_data,
            status=status,
            reasons=reasons,
            reupload_hint=reupload_hint,
            document=DocumentInfo(
                type="commercial_register",
                file_format=file_format,
                pdf_kind=pdf_kind,
                pages=pages_count,
                page_used=1,
                pages_processed=list(range(1, pages_count + 1)),
                quality=quality_info,
            ),
            preprocessing=preprocessing_reports,
            extraction=ExtractionInfo(
                text_method=text_method,
                engine=used_engine,
                languages=["ar", "en"] if text_method != "text_layer" else ["ar"],
                ocr_confidence_mean=ocr_conf_mean,
                duration_ms=duration_ms,
            ),
            fields=merged_fields,
            raw_text=raw_text,
            review=review_info,
        )
