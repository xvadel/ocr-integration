"""
Command line interface for RegOCR.
Usage:
  python -m regocr path/to/document.pdf
  python -m regocr path/to/image.jpg --engine paddleocr --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from regocr.pipeline import RegOCRPipeline


def main():
    parser = argparse.ArgumentParser(
        description="RegOCR: Egyptian Commercial Register OCR & Structured Extraction"
    )
    parser.add_argument("path", type=Path, help="Input file (image or PDF) or directory")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON file path")
    parser.add_argument(
        "--engine",
        choices=["auto", "paddleocr", "easyocr", "tesseract", "pypdfium2"],
        default="auto",
        help="OCR engine selection (default: auto)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Rendering DPI for PDF scans")

    args = parser.parse_args()

    pipeline = RegOCRPipeline(engine=args.engine, dpi=args.dpi)

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
    print(f"\n[RegOCR] Processing {len(files)} file(s) with engine='{args.engine}'...\n")

    for f in files:
        try:
            res = pipeline.process_file(f)
            doc_dict = res.model_dump(mode="json")
            results[f.name] = doc_dict

            status = res.status.upper()
            score = res.document.quality.score
            reg_fld = res.fields.get("register_number")
            comp_fld = res.fields.get("company_name")
            reg_num = reg_fld.value if reg_fld and reg_fld.value is not None else "N/A"
            comp = comp_fld.value if comp_fld and comp_fld.value is not None else "N/A"
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
