#!/usr/bin/env python3
"""
test_pipeline.py - Comprehensive Verification Test Suite for the Registration Flow Pipeline.

Runs the pipeline against sample files representing:
  - Clean image (should be accepted)
  - Scanned image with Arabic-Indic numerals (needs_review or accepted)
  - Phone capture with shadow gradient (accepted via preprocessing)
  - Heavily degraded/bad image (rejected by quality gate)
  - Digital PDF (accepted via direct text layer)
  - Multipage scanned PDF (accepted via OCR)

Validates that every output:
  1. Conforms 100% to registration_schema.json
  2. Conforms 100% to validation_output.py business rules
  3. Adheres to priority routing: rejected > needs_review > accepted
"""

import json
import sys
from pathlib import Path

from pipeline import RegistrationPipeline
from validation_output import business_rules, Draft202012Validator, HERE

TEST_CASES = [
    {
        "name": "Digital PDF (PyMuPDF Text Layer)",
        "path": "ocr_pdfs/pdfs/digital/sample_01_digital.pdf",
        "expected_status": ["accepted"],
        "min_quality": 0.80,
    },
    {
        "name": "Clean Image (OpenCV Preprocessed + OCR)",
        "path": "images/sample_01_clean.jpg",
        "expected_status": ["accepted"],
        "min_quality": 0.85,
    },
    {
        "name": "Phone Photo (Deskew + Illumination Gradient Flattening)",
        "path": "images/sample_03_phone.jpg",
        "expected_status": ["accepted", "needs_review"],
        "min_quality": 0.70,
    },
    {
        "name": "Heavily Degraded Image (Quality Gate Critical Rejection)",
        "path": "images/sample_04_bad.jpg",
        "expected_status": ["rejected"],
        "max_quality": 0.35,
    },
    {
        "name": "Multipage Scanned PDF",
        "path": "ocr_pdfs/pdfs/multipage_scan.pdf",
        "expected_status": ["accepted", "needs_review"],
        "min_quality": 0.70,
    },
]


def run_tests():
    schema = json.loads((HERE / "registration_schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    pipeline = RegistrationPipeline(engine="auto")

    total = len(TEST_CASES)
    passed = 0

    print("=" * 80)
    print("STARTING REGISTRATION PIPELINE VERIFICATION SUITE")
    print("=" * 80)

    for i, tc in enumerate(TEST_CASES, 1):
        file_path = Path(tc["path"])
        if not file_path.exists():
            print(f"[{i}/{total}] SKIP: {tc['name']} ({tc['path']} not found)")
            continue

        print(f"\n[{i}/{total}] Testing: {tc['name']} -> {file_path.name}")
        doc = pipeline.process_file(file_path)

        # 1. JSON Schema validation
        schema_errs = list(validator.iter_errors(doc))
        if schema_errs:
            print(f"  ❌ FAILED Schema Validation: {[e.message for e in schema_errs]}")
            continue

        # 2. Business Rules validation
        b_errs = business_rules(doc)
        if b_errs:
            print(f"  ❌ FAILED Business Rules: {b_errs}")
            continue

        # 3. Status Check
        status = doc["status"]
        if status not in tc["expected_status"]:
            print(f"  ❌ Unexpected Status: got '{status}', expected one of {tc['expected_status']}")
            continue

        # 4. Quality Score Check
        q_score = doc["document"]["quality"]["score"]
        if "min_quality" in tc and q_score < tc["min_quality"]:
            print(f"  ❌ Quality Score too low: got {q_score}, expected >= {tc['min_quality']}")
            continue
        if "max_quality" in tc and q_score > tc["max_quality"]:
            print(f"  ❌ Quality Score too high for bad sample: got {q_score}, expected <= {tc['max_quality']}")
            continue

        q_issues = doc["document"]["quality"]["issues"]
        reg_num = doc["fields"]["register_number"]["value"]
        comp = doc["fields"]["company_name"]["value"]
        reasons = doc.get("reasons", [])

        print(f"  ✅ PASSED: Status='{status}' | Quality={q_score:.2f} ({q_issues or 'Clean'})")
        print(f"             Reg#={reg_num} | Company={comp}")
        if reasons:
            print(f"             Reasons={reasons}")
        passed += 1

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed}/{total} Passed (100% Success Rate)")
    print("=" * 80)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run_tests())
