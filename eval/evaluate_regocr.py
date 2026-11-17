"""
Comprehensive evaluation harness for RegOCR.

Features:
  - Evaluates all 7 document categories (Digital PDF, Digital Visual PDF, Scanned PDF,
    Clean Image, Scan Image (Indic), Phone Photo, Bad Image) against ground truth.
  - Supports dev (samples 01-08) vs held-out (samples 09-12) splits so thresholds
    are never tuned on held-out samples.
  - Reports per-field exact-match accuracy (all 12 fields), per-category accuracy,
    needs_review / rejected rates, error catch rate, digit confusion matrix,
    and confidence calibration (reliability table + ECE).
  - Supports --out, --split, --limit, --engine, --save-failures, and preprocessing ablation flags.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from regocr.normalize import norm_arabic, parse_iso_date, parse_numeric_capital, to_ascii_digits
from regocr.pipeline import RegOCRPipeline

_WORKER_PIPELINE: Optional[RegOCRPipeline] = None


def _init_worker(engine: str, pipeline_kwargs: Dict[str, Any]) -> None:
    global _WORKER_PIPELINE
    _WORKER_PIPELINE = RegOCRPipeline(engine=engine, **pipeline_kwargs)


def _process_one_worker(task: Tuple[str, str]) -> Tuple[str, str, Dict[str, Any]]:
    cat_name, file_str = task
    assert _WORKER_PIPELINE is not None
    doc = _WORKER_PIPELINE.process_file(Path(file_str))
    return cat_name, file_str, doc.model_dump(mode="json")

ALL_FIELDS = [
    "company_name",
    "legal_form",
    "register_number",
    "issue_date",
    "expiry_date",
    "activity",
    "governorate",
    "address",
    "capital",
    "manager",
    "tax_id",
    "office",
]

NUMERIC_FIELDS = ["register_number", "issue_date", "expiry_date", "tax_id", "capital"]
NUMERIC_ID_FIELDS = ["register_number", "tax_id"]
DEV_IDS = {1, 2, 3, 4, 5, 6, 7, 8}
HELDOUT_IDS = {9, 10, 11, 12}


def get_sample_id(filename: str) -> Optional[int]:
    m = re.search(r"sample_(\d+)", filename)
    return int(m.group(1)) if m else None


def get_gt_record(gt_data: dict, filename: str) -> Optional[dict]:
    idx = get_sample_id(filename)
    if idx is None:
        return None
    for k, v in gt_data.items():
        if re.search(rf"sample_{idx:02d}", k):
            return v
    return None


def normalize_for_exact_match(field_name: str, val: Any) -> Optional[str]:
    """Normalize a value (predicted or ground truth) to canonical comparable string."""
    if val is None:
        return None

    if field_name == "register_number":
        digits = re.sub(r"\D", "", to_ascii_digits(str(val)))
        return digits if digits else None

    if field_name in ("issue_date", "expiry_date"):
        iso = parse_iso_date(str(val))
        return iso

    if field_name == "tax_id":
        digits = re.sub(r"\D", "", to_ascii_digits(str(val)))
        return digits if digits else None

    if field_name == "capital":
        if isinstance(val, (int, float)):
            return str(int(val)) if float(val).is_integer() else str(float(val))
        num = parse_numeric_capital(str(val))
        if num is None:
            return None
        return str(int(num)) if float(num).is_integer() else str(float(num))

    # Text fields: canonical Arabic normalization + digit conversion + collapsed spaces
    s = norm_arabic(to_ascii_digits(str(val)))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def exact_match(field_name: str, ext_val: Any, gt_val: Any) -> bool:
    if gt_val is None:
        return ext_val is None
    if ext_val is None:
        return False
    return normalize_for_exact_match(field_name, ext_val) == normalize_for_exact_match(field_name, gt_val)


def update_digit_confusion(
    confusion: Dict[str, Counter],
    gt_str: Optional[str],
    pred_str: Optional[str],
) -> None:
    """Align digit sequences and record character-level digit confusions."""
    if not gt_str:
        return
    gt_d = re.sub(r"\D", "", gt_str)
    pred_d = re.sub(r"\D", "", pred_str or "")
    if not gt_d:
        return

    if not pred_d:
        for ch in gt_d:
            confusion[ch]["<MISSING>"] += 1
        return

    sm = difflib.SequenceMatcher(None, gt_d, pred_d)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ch = gt_d[i1 + k]
                confusion[ch][ch] += 1
        elif tag == "replace":
            g_seg = gt_d[i1:i2]
            p_seg = pred_d[j1:j2]
            m = min(len(g_seg), len(p_seg))
            for k in range(m):
                confusion[g_seg[k]][p_seg[k]] += 1
            for k in range(m, len(g_seg)):
                confusion[g_seg[k]]["<DEL>"] += 1
            for k in range(m, len(p_seg)):
                confusion["<INS>"][p_seg[k]] += 1
        elif tag == "delete":
            for ch in gt_d[i1:i2]:
                confusion[ch]["<DEL>"] += 1
        elif tag == "insert":
            for ch in pred_d[j1:j2]:
                confusion["<INS>"][ch] += 1


def bucket_for_conf(conf: Optional[float]) -> str:
    if conf is None:
        return "none"
    if conf < 0.50:
        return "[0.00, 0.50)"
    if conf < 0.70:
        return "[0.50, 0.70)"
    if conf < 0.85:
        return "[0.70, 0.85)"
    if conf < 0.95:
        return "[0.85, 0.95)"
    return "[0.95, 1.00]"


def build_targets(split: str = "all", limit: Optional[int] = None) -> List[Tuple[str, List[Path], dict]]:
    gt_images = json.loads(Path("images/ground_truth.json").read_text(encoding="utf-8"))
    gt_pdfs = json.loads(Path("ocr_pdfs/pdfs/ground_truth.json").read_text(encoding="utf-8"))

    raw_targets = [
        ("Digital PDF", sorted(Path("ocr_pdfs/pdfs/digital").glob("*.pdf")), gt_pdfs),
        ("Digital Visual PDF", sorted(Path("ocr_pdfs/pdfs/digital_visual").glob("*.pdf")), gt_pdfs),
        ("Scanned PDF", sorted(Path("ocr_pdfs/pdfs/scanned").glob("*.pdf")), gt_pdfs),
        ("Clean Image", [Path(f"images/sample_{i:02d}_clean.jpg") for i in (1, 5, 9)], gt_images),
        ("Scan Image (Indic)", [Path(f"images/sample_{i:02d}_scan.jpg") for i in (2, 6, 10)], gt_images),
        ("Phone Photo", [Path(f"images/sample_{i:02d}_phone.jpg") for i in (3, 7, 11)], gt_images),
        ("Bad Image (Low-Res/Blur)", [Path(f"images/sample_{i:02d}_bad.jpg") for i in (4, 8, 12)], gt_images),
    ]

    filtered = []
    for cat_name, files, gt_src in raw_targets:
        kept = []
        for f in files:
            if not f.exists():
                continue
            sid = get_sample_id(f.name)
            if split == "dev" and sid not in DEV_IDS:
                continue
            if split == "heldout" and sid not in HELDOUT_IDS:
                continue
            kept.append(f)
        if limit is not None:
            kept = kept[:limit]
        if kept:
            filtered.append((cat_name, kept, gt_src))
    return filtered


def run_eval(
    out_path: Path,
    split: str = "all",
    limit: Optional[int] = None,
    engine: str = "auto",
    workers: int = 1,
    save_failures_dir: Optional[Path] = None,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kwargs = dict(pipeline_kwargs or {})
    targets = build_targets(split=split, limit=limit)

    t_start = time.perf_counter()

    # Pre-execute all documents (either sequentially or in parallel worker pool)
    flat_tasks: List[Tuple[str, str]] = [
        (cat_name, str(f))
        for cat_name, file_list, _ in targets
        for f in file_list
    ]
    results_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if workers > 1 and len(flat_tasks) > 1:
        print(f"[Eval] Running {len(flat_tasks)} documents with {workers} parallel workers...", flush=True)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(engine, kwargs),
        ) as ex:
            future_to_task = {ex.submit(_process_one_worker, t): t for t in flat_tasks}
            done_cnt = 0
            for fut in concurrent.futures.as_completed(future_to_task):
                cat_name, file_str, doc_dict = fut.result()
                results_map[(cat_name, file_str)] = doc_dict
                done_cnt += 1
                print(
                    f"  [{done_cnt:>2}/{len(flat_tasks)}] {cat_name:<24} {Path(file_str).name:<24} "
                    f"status={doc_dict['status']:<12} {doc_dict['extraction']['duration_ms']}ms",
                    flush=True,
                )
    else:
        pipeline = RegOCRPipeline(engine=engine, **kwargs)
        for idx_t, (cat_name, file_str) in enumerate(flat_tasks, 1):
            doc_dict = pipeline.process_file(Path(file_str)).model_dump(mode="json")
            results_map[(cat_name, file_str)] = doc_dict
            print(
                f"  [{idx_t:>2}/{len(flat_tasks)}] {cat_name:<24} {Path(file_str).name:<24} "
                f"status={doc_dict['status']:<12} {doc_dict['extraction']['duration_ms']}ms",
                flush=True,
            )

    category_reports: Dict[str, Any] = {}
    per_field_totals: Dict[str, Dict[str, int]] = {
        f: {"correct": 0, "total": 0, "wrong": 0, "wrong_caught_field": 0, "wrong_caught_doc": 0}
        for f in ALL_FIELDS
    }
    passing_field_totals: Dict[str, Dict[str, int]] = {
        f: {"correct": 0, "total": 0, "wrong": 0, "wrong_caught_field": 0, "wrong_caught_doc": 0}
        for f in ALL_FIELDS
    }
    split_totals: Dict[str, Dict[str, Dict[str, int]]] = {
        "dev": {f: {"correct": 0, "total": 0} for f in ALL_FIELDS},
        "heldout": {f: {"correct": 0, "total": 0} for f in ALL_FIELDS},
    }

    digit_confusion: Dict[str, Counter] = defaultdict(Counter)
    conf_buckets: Dict[str, Dict[str, Any]] = {
        b: {"count": 0, "correct": 0, "conf_sum": 0.0}
        for b in ["[0.00, 0.50)", "[0.50, 0.70)", "[0.70, 0.85)", "[0.85, 0.95)", "[0.95, 1.00]", "none"]
    }

    failures: List[Dict[str, Any]] = []
    doc_records: List[Dict[str, Any]] = []

    if save_failures_dir:
        save_failures_dir.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    synthetic_docs = 0
    real_docs = 0
    total_status_counts: Counter = Counter()

    print(
        f"\n{'Category':<26} {'N':<4} {'Acc%':<6} {'Rev%':<6} {'Rej%':<6} "
        f"{'Reg#':<6} {'Date':<6} {'Comp':<6} {'TaxID':<6} {'Cap':<6} {'All12':<6} {'AvgMs':<7}"
    )
    print("=" * 108)

    for cat_name, file_list, gt_source in targets:
        cat_status: Counter = Counter()
        cat_field_ok: Counter = Counter()
        cat_durations: List[int] = []
        cat_qscores: List[float] = []
        cat_wrong = 0
        cat_wrong_caught_field = 0
        cat_wrong_caught_doc = 0
        cat_total_fields = 0

        for f in file_list:
            total_docs += 1
            sid = get_sample_id(f.name)
            doc_split = "dev" if sid in DEV_IDS else ("heldout" if sid in HELDOUT_IDS else "other")

            gt_rec = get_gt_record(gt_source, f.name)
            gt_fields = gt_rec["fields"] if gt_rec else {}

            doc = results_map[(cat_name, str(f))]
            if doc.get("test_data", True):
                synthetic_docs += 1
            else:
                real_docs += 1

            t_ms = doc["extraction"]["duration_ms"]
            q_score = doc["document"]["quality"]["score"]
            cat_durations.append(t_ms)
            cat_qscores.append(q_score)
            cat_status[doc["status"]] += 1
            total_status_counts[doc["status"]] += 1

            doc_flagged = doc["status"] in ("needs_review", "rejected")
            quality_passed = q_score >= 0.35 and "image_quality_too_low" not in doc.get("reasons", [])

            doc_field_summary: Dict[str, Any] = {}

            for fname in ALL_FIELDS:
                gt_val = gt_fields.get(fname)
                fobj = doc["fields"].get(fname) or {}
                ext_val = fobj.get("value")
                fconf = fobj.get("confidence")
                f_needs_rev = bool(fobj.get("needs_review")) if fobj else True
                f_in_doc_rev = fname in (doc.get("review", {}).get("fields_to_review") or [])

                is_ok = exact_match(fname, ext_val, gt_val)

                cat_total_fields += 1
                per_field_totals[fname]["total"] += 1
                if doc_split in split_totals:
                    split_totals[doc_split][fname]["total"] += 1

                if quality_passed:
                    passing_field_totals[fname]["total"] += 1

                if is_ok:
                    cat_field_ok[fname] += 1
                    per_field_totals[fname]["correct"] += 1
                    if doc_split in split_totals:
                        split_totals[doc_split][fname]["correct"] += 1
                    if quality_passed:
                        passing_field_totals[fname]["correct"] += 1
                else:
                    cat_wrong += 1
                    per_field_totals[fname]["wrong"] += 1
                    caught_field = f_needs_rev or f_in_doc_rev
                    caught_doc = caught_field or doc_flagged
                    if caught_field:
                        cat_wrong_caught_field += 1
                        per_field_totals[fname]["wrong_caught_field"] += 1
                    if caught_doc:
                        cat_wrong_caught_doc += 1
                        per_field_totals[fname]["wrong_caught_doc"] += 1
                    if quality_passed:
                        passing_field_totals[fname]["wrong"] += 1
                        if caught_field:
                            passing_field_totals[fname]["wrong_caught_field"] += 1
                        if caught_doc:
                            passing_field_totals[fname]["wrong_caught_doc"] += 1

                    failures.append(
                        {
                            "category": cat_name,
                            "file": str(f),
                            "split": doc_split,
                            "field": fname,
                            "expected": gt_val,
                            "expected_norm": normalize_for_exact_match(fname, gt_val),
                            "got": ext_val,
                            "got_raw": fobj.get("raw"),
                            "got_norm": normalize_for_exact_match(fname, ext_val),
                            "confidence": fconf,
                            "field_status": fobj.get("status", "missing"),
                            "field_needs_review": f_needs_rev,
                            "doc_status": doc["status"],
                            "quality_score": q_score,
                            "engine": doc["extraction"].get("engine"),
                            "bbox": fobj.get("bbox"),
                        }
                    )

                if fname in NUMERIC_FIELDS and quality_passed:
                    update_digit_confusion(
                        digit_confusion,
                        normalize_for_exact_match(fname, gt_val),
                        normalize_for_exact_match(fname, ext_val),
                    )

                b = bucket_for_conf(fconf)
                conf_buckets[b]["count"] += 1
                if is_ok:
                    conf_buckets[b]["correct"] += 1
                if fconf is not None:
                    conf_buckets[b]["conf_sum"] += float(fconf)

                doc_field_summary[fname] = {
                    "ok": is_ok,
                    "expected": normalize_for_exact_match(fname, gt_val),
                    "got": normalize_for_exact_match(fname, ext_val),
                    "conf": fconf,
                    "needs_review": f_needs_rev,
                    "engines_agree": fobj.get("engines_agree"),
                    "validated": fobj.get("validated"),
                }

            doc_records.append(
                {
                    "category": cat_name,
                    "file": str(f),
                    "split": doc_split,
                    "status": doc["status"],
                    "reasons": doc.get("reasons", []),
                    "engine": doc["extraction"].get("engine"),
                    "text_method": doc["extraction"].get("text_method"),
                    "quality_score": q_score,
                    "duration_ms": t_ms,
                    "fields": doc_field_summary,
                }
            )

        n = len(file_list)
        avg_ms = sum(cat_durations) / n
        avg_q = sum(cat_qscores) / n
        acc_pct = (cat_status["accepted"] / n) * 100
        rev_pct = (cat_status["needs_review"] / n) * 100
        rej_pct = (cat_status["rejected"] / n) * 100

        field_acc_map = {fn: round((cat_field_ok[fn] / n) * 100, 1) for fn in ALL_FIELDS}
        all12_acc = round((sum(cat_field_ok.values()) / cat_total_fields) * 100, 1) if cat_total_fields else 0.0

        category_reports[cat_name] = {
            "count": n,
            "synthetic_count": n,  # all current samples are synthetic
            "real_count": 0,
            "avg_time_ms": round(avg_ms),
            "avg_quality_score": round(avg_q, 2),
            "accepted_pct": round(acc_pct, 1),
            "needs_review_pct": round(rev_pct, 1),
            "rejected_pct": round(rej_pct, 1),
            "all_12_fields_acc": all12_acc,
            "field_accuracy": field_acc_map,
            "wrong_fields": cat_wrong,
            "wrong_caught_by_field_review_pct": round((cat_wrong_caught_field / cat_wrong) * 100, 1) if cat_wrong else 100.0,
            "wrong_caught_by_doc_review_or_reject_pct": round((cat_wrong_caught_doc / cat_wrong) * 100, 1) if cat_wrong else 100.0,
        }

        print(
            f"{cat_name:<26} {n:<4} {acc_pct:<5.0f}% {rev_pct:<5.0f}% {rej_pct:<5.0f}% "
            f"{field_acc_map['register_number']:<5.0f}% {field_acc_map['issue_date']:<5.0f}% "
            f"{field_acc_map['company_name']:<5.0f}% {field_acc_map['tax_id']:<5.0f}% "
            f"{field_acc_map['capital']:<5.0f}% {all12_acc:<5.0f}% {avg_ms:<7.0f}"
        )

    print("=" * 108)

    # Per-field summary table
    per_field_summary: Dict[str, Any] = {}
    print("\n=== PER-FIELD EXACT-MATCH ACCURACY (ALL DOCS vs QUALITY-PASSING DOCS) ===")
    print(f"{'Field':<18} {'All Acc%':<10} {'Pass Acc%':<10} {'Dev Acc%':<10} {'HeldOut%':<10} {'Wrong':<7} {'Caught(Fld)':<12} {'Caught(Doc)'}")
    print("-" * 92)
    for fn in ALL_FIELDS:
        t = per_field_totals[fn]
        pt = passing_field_totals[fn]
        dt = split_totals["dev"][fn]
        ht = split_totals["heldout"][fn]
        all_p = round((t["correct"] / t["total"]) * 100, 1) if t["total"] else 0.0
        pass_p = round((pt["correct"] / pt["total"]) * 100, 1) if pt["total"] else 0.0
        dev_p = round((dt["correct"] / dt["total"]) * 100, 1) if dt["total"] else 0.0
        ho_p = round((ht["correct"] / ht["total"]) * 100, 1) if ht["total"] else 0.0
        cf_p = round((t["wrong_caught_field"] / t["wrong"]) * 100, 1) if t["wrong"] else 100.0
        cd_p = round((t["wrong_caught_doc"] / t["wrong"]) * 100, 1) if t["wrong"] else 100.0
        per_field_summary[fn] = {
            "all_exact_match_pct": all_p,
            "quality_passing_exact_match_pct": pass_p,
            "dev_exact_match_pct": dev_p,
            "heldout_exact_match_pct": ho_p,
            "correct": t["correct"],
            "total": t["total"],
            "passing_correct": pt["correct"],
            "passing_total": pt["total"],
            "wrong": t["wrong"],
            "wrong_caught_field_pct": cf_p,
            "wrong_caught_doc_pct": cd_p,
            "passing_wrong": pt["wrong"],
            "passing_wrong_caught_field_pct": round((pt["wrong_caught_field"] / pt["wrong"]) * 100, 1) if pt["wrong"] else 100.0,
            "passing_wrong_caught_doc_pct": round((pt["wrong_caught_doc"] / pt["wrong"]) * 100, 1) if pt["wrong"] else 100.0,
        }
        print(f"{fn:<18} {all_p:<9.1f}% {pass_p:<9.1f}% {dev_p:<9.1f}% {ho_p:<9.1f}% {t['wrong']:<7} {cf_p:<11.1f}% {cd_p:.1f}%")

    # Reliability / Calibration table
    reliability: Dict[str, Any] = {}
    ece_num = 0.0
    ece_den = 0
    print("\n=== CONFIDENCE CALIBRATION (RELIABILITY TABLE) ===")
    print(f"{'Bucket':<16} {'Count':<8} {'Avg Conf':<10} {'Actual Acc%':<12} {'Gap'}")
    print("-" * 58)
    for b, stats in conf_buckets.items():
        cnt = stats["count"]
        acc = (stats["correct"] / cnt) if cnt else 0.0
        avg_c = (stats["conf_sum"] / cnt) if (cnt and b != "none") else None
        gap = abs(acc - avg_c) if avg_c is not None else None
        if avg_c is not None and cnt > 0:
            ece_num += cnt * gap
            ece_den += cnt
        reliability[b] = {
            "count": cnt,
            "correct": stats["correct"],
            "avg_confidence": round(avg_c, 4) if avg_c is not None else None,
            "actual_accuracy_pct": round(acc * 100, 1),
            "calibration_gap": round(gap, 4) if gap is not None else None,
        }
        avg_c_str = f"{avg_c:.3f}" if avg_c is not None else "N/A"
        gap_str = f"{gap:.3f}" if gap is not None else "N/A"
        print(f"{b:<16} {cnt:<8} {avg_c_str:<10} {acc*100:<11.1f}% {gap_str}")
    ece = round(ece_num / ece_den, 4) if ece_den else None
    print(f"Expected Calibration Error (ECE): {ece}")

    # Digit confusion matrix (non-identity confusions)
    digit_conf_dict = {k: dict(v) for k, v in sorted(digit_confusion.items())}
    digit_errors = []
    total_digits = 0
    correct_digits = 0
    for true_ch, preds in digit_confusion.items():
        for pred_ch, cnt in preds.items():
            if true_ch != "<INS>":
                total_digits += cnt
            if true_ch == pred_ch:
                correct_digits += cnt
            else:
                digit_errors.append((cnt, true_ch, pred_ch))
    digit_errors.sort(reverse=True)
    digit_acc = round((correct_digits / total_digits) * 100, 2) if total_digits else 0.0

    print(f"\n=== DIGIT CONFUSION SUMMARY (Quality-Passing Docs, {correct_digits}/{total_digits} = {digit_acc}%) ===")
    for cnt, tc, pc in digit_errors[:15]:
        print(f"  True '{tc}' -> Pred '{pc}': {cnt}")

    elapsed_s = round(time.perf_counter() - t_start, 1)
    print(f"\nTotal docs: {total_docs} (synthetic={synthetic_docs}, real={real_docs}) | Elapsed: {elapsed_s}s")
    print(f"Doc status counts: {dict(total_status_counts)}")

    # Save failure details if requested
    if save_failures_dir:
        (save_failures_dir / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Save full report JSON
    full_report = {
        "meta": {
            "split": split,
            "limit": limit,
            "engine": engine,
            "total_docs": total_docs,
            "synthetic_docs": synthetic_docs,
            "real_docs": real_docs,
            "dev_sample_ids": sorted(DEV_IDS),
            "heldout_sample_ids": sorted(HELDOUT_IDS),
            "elapsed_seconds": elapsed_s,
            "status_counts": dict(total_status_counts),
            "needs_review_rate_pct": round((total_status_counts["needs_review"] / total_docs) * 100, 1) if total_docs else 0.0,
            "rejected_rate_pct": round((total_status_counts["rejected"] / total_docs) * 100, 1) if total_docs else 0.0,
            "accepted_rate_pct": round((total_status_counts["accepted"] / total_docs) * 100, 1) if total_docs else 0.0,
        },
        "categories": category_reports,
        "per_field": per_field_summary,
        "calibration": {
            "ece": ece,
            "buckets": reliability,
        },
        "digit_confusion": {
            "total_digits": total_digits,
            "correct_digits": correct_digits,
            "digit_accuracy_pct": digit_acc,
            "top_errors": [{"true": tc, "pred": pc, "count": cnt} for cnt, tc, pc in digit_errors[:25]],
            "matrix": digit_conf_dict,
        },
        "failures": failures,
        "documents": doc_records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to: {out_path}")
    return full_report


def main():
    parser = argparse.ArgumentParser(description="Run full RegOCR evaluation harness.")
    parser.add_argument("--out", type=Path, default=Path("eval/eval_report.json"), help="Output JSON path")
    parser.add_argument("--split", choices=["all", "dev", "heldout"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of files per category")
    parser.add_argument("--engine", default="auto", choices=["auto", "paddleocr", "easyocr", "tesseract"])
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--save-failures", type=Path, default=None, help="Directory to dump failure records")
    args = parser.parse_args()

    run_eval(
        out_path=args.out,
        split=args.split,
        limit=args.limit,
        engine=args.engine,
        workers=args.workers,
        save_failures_dir=args.save_failures,
    )


if __name__ == "__main__":
    main()
