#!/usr/bin/env python3
"""
evaluate_accuracy.py - Comprehensive Accuracy Benchmark & Metrics Generator.

Evaluates pipeline extraction accuracy, routing logic, and latencies across:
  - Digital PDFs (PyMuPDF direct extraction)
  - Visual RTL Digital PDFs (pdfplumber_bidi)
  - Scanned PDFs
  - Clean Images
  - Scanned Images (Arabic-Indic digits)
  - Phone Photos (Shadows, perspective skew)
  - Heavily Degraded Images (Quality Gate rejections)

Generates exact metrics JSON and prints summary table.
"""

import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline import RegistrationPipeline, norm_arabic, to_ascii_digits, parse_iso_date, parse_numeric_capital
from validation_output import business_rules, Draft202012Validator, HERE

SCHEMA = json.loads((HERE / "registration_schema.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)


def get_gt_record(gt_data: dict, filename: str) -> Optional[dict]:
    """Finds matching ground truth record by sample number."""
    m = re.search(r"sample_(\d+)", filename)
    if not m:
        return None
    idx = int(m.group(1))
    for k, v in gt_data.items():
        if re.search(rf"sample_{idx:02d}", k):
            return v
    return None


def match_field(field_name: str, ext_val: Any, gt_val: Any) -> bool:
    """Checks if extracted value matches ground truth value."""
    if gt_val is None:
        return ext_val is None
    if ext_val is None:
        return False

    if field_name == "register_number":
        return to_ascii_digits(str(ext_val)).strip() == to_ascii_digits(str(gt_val)).strip()

    if field_name in ("issue_date", "expiry_date"):
        gt_iso = parse_iso_date(str(gt_val))
        return ext_val == gt_iso

    if field_name == "tax_id":
        ext_digits = re.sub(r"\D", "", to_ascii_digits(str(ext_val)))
        gt_digits = re.sub(r"\D", "", to_ascii_digits(str(gt_val)))
        return ext_digits == gt_digits and len(ext_digits) >= 9

    if field_name == "capital":
        gt_num = parse_numeric_capital(str(gt_val))
        return ext_val == gt_num

    if field_name == "company_name":
        ext_norm = norm_arabic(str(ext_val))
        gt_norm = norm_arabic(str(gt_val))
        # Word overlap check (handles partial prefixes)
        ext_words = set(ext_norm.split())
        gt_words = set(gt_norm.split())
        if not gt_words:
            return False
        overlap = len(ext_words & gt_words) / len(gt_words)
        return overlap >= 0.6 or ext_norm in gt_norm or gt_norm in ext_norm

    # Generic string match
    return norm_arabic(str(ext_val)) in norm_arabic(str(gt_val)) or norm_arabic(str(gt_val)) in norm_arabic(str(ext_val))


def evaluate_all():
    pipeline = RegistrationPipeline(engine="auto")

    # Load ground truths
    gt_images = json.loads(Path("images/ground_truth.json").read_text(encoding="utf-8"))
    gt_pdfs = json.loads(Path("ocr_pdfs/pdfs/ground_truth.json").read_text(encoding="utf-8"))

    targets = [
        # 1. Digital PDFs
        ("Digital PDF", sorted(Path("ocr_pdfs/pdfs/digital").glob("*.pdf")), gt_pdfs),
        # 2. Digital Visual PDFs
        ("Digital Visual PDF", sorted(Path("ocr_pdfs/pdfs/digital_visual").glob("*.pdf")), gt_pdfs),
        # 3. Scanned PDFs
        ("Scanned PDF", sorted(Path("ocr_pdfs/pdfs/scanned").glob("*.pdf")), gt_pdfs),
        # 4. Images by variant
        ("Clean Image", [Path(f"images/sample_{i:02d}_clean.jpg") for i in (1, 5, 9)], gt_images),
        ("Scan Image (Indic)", [Path(f"images/sample_{i:02d}_scan.jpg") for i in (2, 6, 10)], gt_images),
        ("Phone Photo", [Path(f"images/sample_{i:02d}_phone.jpg") for i in (3, 7, 11)], gt_images),
        ("Bad Image (Low-Res/Blur)", [Path(f"images/sample_{i:02d}_bad.jpg") for i in (4, 8, 12)], gt_images),
    ]

    report = {}

    print(f"\n{'Category':<24} {'Count':<6} {'Schema OK':<10} {'Accepted':<10} {'NeedsRev':<10} {'Rejected':<10} {'Reg# Acc':<10} {'Date Acc':<10} {'Company Acc':<12} {'Avg Time':<10}")
    print("=" * 115)

    for cat_name, file_list, gt_source in targets:
        existing_files = [f for f in file_list if f.exists()]
        if not existing_files:
            continue

        cat_stats = {
            "count": len(existing_files),
            "schema_valid": 0,
            "status_counts": defaultdict(int),
            "field_matches": defaultdict(int),
            "total_durations": 0,
            "quality_scores": [],
        }

        for f in existing_files:
            gt_rec = get_gt_record(gt_source, f.name)
            gt_fields = gt_rec["fields"] if gt_rec else {}

            doc = pipeline.process_file(f)
            t_ms = doc["extraction"]["duration_ms"]
            cat_stats["total_durations"] += t_ms
            cat_stats["quality_scores"].append(doc["document"]["quality"]["score"])

            # Schema validation
            schema_ok = len(list(VALIDATOR.iter_errors(doc))) == 0 and len(business_rules(doc)) == 0
            if schema_ok:
                cat_stats["schema_valid"] += 1

            cat_stats["status_counts"][doc["status"]] += 1

            # Field checks
            ext_fields = doc["fields"]
            for field in ["company_name", "register_number", "issue_date", "expiry_date", "tax_id", "capital"]:
                ext_val = ext_fields.get(field, {}).get("value")
                gt_val = gt_fields.get(field)
                if match_field(field, ext_val, gt_val):
                    cat_stats["field_matches"][field] += 1

        n = cat_stats["count"]
        avg_t = cat_stats["total_durations"] / n
        avg_q = sum(cat_stats["quality_scores"]) / n

        reg_acc = (cat_stats["field_matches"]["register_number"] / n) * 100
        date_acc = (cat_stats["field_matches"]["issue_date"] / n) * 100
        comp_acc = (cat_stats["field_matches"]["company_name"] / n) * 100

        acc_pct = (cat_stats["status_counts"]["accepted"] / n) * 100
        rev_pct = (cat_stats["status_counts"]["needs_review"] / n) * 100
        rej_pct = (cat_stats["status_counts"]["rejected"] / n) * 100
        sch_pct = (cat_stats["schema_valid"] / n) * 100

        report[cat_name] = {
            "count": n,
            "avg_time_ms": round(avg_t),
            "avg_quality_score": round(avg_q, 2),
            "schema_valid_pct": round(sch_pct, 1),
            "accepted_pct": round(acc_pct, 1),
            "needs_review_pct": round(rev_pct, 1),
            "rejected_pct": round(rej_pct, 1),
            "register_number_acc": round(reg_acc, 1),
            "issue_date_acc": round(date_acc, 1),
            "company_name_acc": round(comp_acc, 1),
            "tax_id_acc": round((cat_stats["field_matches"]["tax_id"] / n) * 100, 1),
            "capital_acc": round((cat_stats["field_matches"]["capital"] / n) * 100, 1),
        }

        print(f"{cat_name:<24} {n:<6} {sch_pct:<10.0f}% {acc_pct:<10.0f}% {rev_pct:<10.0f}% {rej_pct:<10.0f}% {reg_acc:<10.0f}% {date_acc:<10.0f}% {comp_acc:<12.0f}% {avg_t:<10.0f}ms")

    print("=" * 115)
    Path("evaluation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    evaluate_all()
