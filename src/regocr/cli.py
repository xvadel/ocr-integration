"""
Command line interface for RegOCR.
Usage:
  python -m regocr path/to/document.pdf
  python -m regocr path/to/image.jpg --engine paddleocr --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import psutil

from regocr.pipeline import RegOCRPipeline


def main():
    parser = argparse.ArgumentParser(
        description="RegOCR: Multi-Country Commercial Register OCR & Structured Extraction"
    )
    parser.add_argument("path", type=Path, help="Input file (image or PDF) or directory")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON file path")
    parser.add_argument(
        "--country",
        choices=["eg", "sa", "ae"],
        default=None,
        help="Country selector: eg (Egypt), sa (Saudi Arabia), ae (UAE). Defaults to 'eg'.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile execution time (wall/CPU) and peak memory to stderr.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "paddleocr", "easyocr", "tesseract", "pypdfium2"],
        default="auto",
        help="OCR engine selection (default: auto)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Rendering DPI for PDF scans")

    args = parser.parse_args()

    # Backward compatibility with deprecation warning when --country is omitted
    country = args.country
    if country is None:
        warnings.warn(
            "Defaulting to country 'eg' is deprecated and will require explicit '--country' in v1.0.",
            DeprecationWarning,
            stacklevel=1,
        )
        country = "eg"

    pipeline = RegOCRPipeline(engine=args.engine, country=country, dpi=args.dpi)

    target = args.path
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(
            list(target.glob("*.jpg"))
            + list(target.glob("*.jpeg"))
            + list(target.glob("*.png"))
            + list(target.glob("*.pdf"))
        )
    else:
        sys.exit(f"Target path does not exist: {target}")

    results = {}
    print(f"\n[RegOCR] Processing {len(files)} file(s) for country='{country}', engine='{args.engine}'...\n")

    process = psutil.Process(os.getpid()) if args.profile else None

    # For Saudi country with multiple files in directory, process as single package
    if country == "sa" and len(files) > 1 and target.is_dir():
        try:
            res = pipeline.process_files(files)
            doc_dict = res.model_dump(mode="json", exclude_none=True)
            results["package_sa"] = doc_dict
            status = res.status.upper()
            facts = res.handoff.get("facts", {}) if res.handoff else {}
            comp = facts.get("company_name") or "N/A"
            unif = facts.get("unified_national_number") or "N/A"
            cr = facts.get("cr_number") or "N/A"
            vat = facts.get("vat_registration_number") or "N/A"
            print(
                f"[{status:<12}] Saudi Package ({len(files)} files) | "
                f"UNN: {str(unif):<10} | CR: {str(cr):<10} | VAT: {str(vat):<15} | Company: {str(comp)[:25]}"
            )
            if res.reasons:
                print(f"    Reasons: {', '.join(res.reasons)}")
            if res.reupload_hint:
                print(f"    Hint: {res.reupload_hint}")
        except Exception as e:
            print(f"[ERROR       ] Saudi Package | Error: {e}")
    else:
        for f in files:
            try:
                if args.profile:
                    cpu_t0 = time.process_time()
                    wall_t0 = time.perf_counter()
                    rss_start = process.memory_info().rss / (1024 * 1024)

                res = pipeline.process_file(f)

                if args.profile:
                    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
                    cpu_ms = (time.process_time() - cpu_t0) * 1000.0
                    rss_end = process.memory_info().rss / (1024 * 1024)
                    sys.stderr.write(
                        f"[PROFILE] {f.name}: Wall={wall_ms:.1f}ms, CPU={cpu_ms:.1f}ms, RSS={rss_end:.1f}MB (delta={rss_end - rss_start:+.1f}MB)\n"
                    )

                doc_dict = res.model_dump(mode="json", exclude_none=True)
                results[f.name] = doc_dict

                status = res.status.upper()
                score = res.document.quality.score
                if res.fields:
                    reg_fld = res.fields.get("register_number")
                    comp_fld = res.fields.get("company_name")
                    reg_num = reg_fld.value if reg_fld and reg_fld.value is not None else "N/A"
                    comp = comp_fld.value if comp_fld and comp_fld.value is not None else "N/A"
                elif res.handoff and "facts" in res.handoff:
                    facts = res.handoff["facts"]
                    comp = facts.get("company_name") or "N/A"
                    reg_num = facts.get("cr_number") or facts.get("unified_national_number") or facts.get("vat_registration_number") or "N/A"
                else:
                    reg_num, comp = "N/A", "N/A"
                rev = res.review.required

                print(
                    f"[{status:<12}] {f.name:<28} | Score: {score:.2f} | "
                    f"Reg#: {str(reg_num):<8} | Company: {str(comp)[:25]} | Review: {rev}"
                )
                if res.reasons:
                    print(f"    Reasons: {', '.join(res.reasons)}")
                if res.reupload_hint:
                    print(f"    Hint: {res.reupload_hint}")
            except Exception as e:
                print(f"[ERROR       ] {f.name:<28} | Error: {e}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[RegOCR] Saved structured JSON to: {args.out}")
    elif len(files) == 1:
        first_doc = list(results.values())[0] if results else {}
        print("\n=== STRUCTURED JSON OUTPUT ===")
        print(json.dumps(first_doc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
