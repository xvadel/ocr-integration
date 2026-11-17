"""
CV Preprocessing Ablation Runner (Priority 5.3 & 5.4).

Runs on the DEV split ONLY (samples 01-08, never held-out 09-12) across OCR image variants
(clean, scan/Indic, phone, bad) while reusing loaded OCR engine weights in memory.
Measures per-step impact on exact-match accuracy (all 12 fields + numeric IDs) and latency.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from evaluate_regocr import ALL_FIELDS, NUMERIC_FIELDS, exact_match, get_gt_record
from regocr.pipeline import PipelineConfig, RegOCRPipeline
from regocr.preprocess import PreprocessConfig

DEV_OCR_SAMPLES = [
    ("Scan Image (Indic)", Path("images/sample_02_scan.jpg")),
    ("Phone Photo", Path("images/sample_03_phone.jpg")),
]


def run_ablation() -> Dict[str, Any]:
    gt_images = json.loads((ROOT / "images/ground_truth.json").read_text(encoding="utf-8"))

    # Create ONE pipeline so PaddleOCR & EasyOCR weights stay warm in RAM across all arms
    pipe = RegOCRPipeline(engine="auto")

    arms: List[Tuple[str, PreprocessConfig, bool]] = [
        ("full_default (Variant A)", PreprocessConfig(generate_variant_b=False), False),
        ("no_deskew", PreprocessConfig(deskew=False, generate_variant_b=False), False),
        ("no_stamp_suppression", PreprocessConfig(stamp_suppression=False, generate_variant_b=False), False),
        ("no_illumination_norm", PreprocessConfig(illumination_norm=False, generate_variant_b=False), False),
        ("no_clahe", PreprocessConfig(clahe=False, generate_variant_b=False), False),
        ("no_denoise", PreprocessConfig(denoise=False, generate_variant_b=False), False),
        ("no_conditional_sharpen", PreprocessConfig(conditional_sharpen=False, generate_variant_b=False), False),
        ("with_variant_b (A+B)", PreprocessConfig(generate_variant_b=True), True),
    ]

    results: Dict[str, Any] = {}
    print(f"\n{'Ablation Arm':<26} {'Pass All12%':<12} {'Pass Num%':<11} {'Office%':<9} {'Cap%':<8} {'AvgMs':<8}")
    print("=" * 78)

    for arm_name, pcfg, use_vb in arms:
        pipe.config.preprocess = pcfg
        pipe.config.use_variant_b_on_disagree = use_vb

        all12_ok = 0
        all12_tot = 0
        num_ok = 0
        num_tot = 0
        office_ok = 0
        office_tot = 0
        cap_ok = 0
        cap_tot = 0
        durations: List[int] = []

        for cat_name, rel_path in DEV_OCR_SAMPLES:
            f = ROOT / rel_path
            gt_rec = get_gt_record(gt_images, f.name)
            gt_fields = gt_rec["fields"] if gt_rec else {}
            doc = pipe.process_file(f)
            durations.append(doc.extraction.duration_ms)

            # Evaluate accuracy on quality-passing dev images (clean, scan, phone = 6 images)
            if cat_name == "Bad Image":
                continue

            for fn in ALL_FIELDS:
                ext_val = doc.fields[fn].value if fn in doc.fields else None
                ok = exact_match(fn, ext_val, gt_fields.get(fn))
                all12_tot += 1
                if ok:
                    all12_ok += 1
                if fn in NUMERIC_FIELDS:
                    num_tot += 1
                    if ok:
                        num_ok += 1
                if fn == "office":
                    office_tot += 1
                    if ok:
                        office_ok += 1
                if fn == "capital":
                    cap_tot += 1
                    if ok:
                        cap_ok += 1

        all12_pct = round((all12_ok / all12_tot) * 100, 1) if all12_tot else 0.0
        num_pct = round((num_ok / num_tot) * 100, 1) if num_tot else 0.0
        off_pct = round((office_ok / office_tot) * 100, 1) if office_tot else 0.0
        cap_pct = round((cap_ok / cap_tot) * 100, 1) if cap_tot else 0.0
        avg_ms = round(sum(durations) / len(durations))

        results[arm_name] = {
            "pass_all12_acc_pct": all12_pct,
            "pass_numeric_acc_pct": num_pct,
            "pass_office_acc_pct": off_pct,
            "pass_capital_acc_pct": cap_pct,
            "avg_ms": avg_ms,
        }
        print(f"{arm_name:<26} {all12_pct:<11.1f}% {num_pct:<10.1f}% {off_pct:<8.1f}% {cap_pct:<7.1f}% {avg_ms:<8}")

    out_path = ROOT / "eval/ablation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved ablation results to: {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_ablation()
