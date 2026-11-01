#!/usr/bin/env python3
"""
validate_output.py - يتحقق من ملف JSON مقابل المخطط + قواعد العمل.

التركيب:   pip install jsonschema
الاستخدام: python validate_output.py registration_output.examples.json
           (الملف إما مخرج واحد، أو قاموس فيه عدة مخرجات كما في ملف الأمثلة)
"""
import json
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

CORE = ["company_name", "register_number", "issue_date"]
HERE = Path(__file__).parent


def business_rules(doc):
    errs = []
    f, status = doc["fields"], doc["status"]
    if status == "accepted":
        for k in CORE:
            if f[k]["status"] != "ok" or not f[k].get("grounded"):
                errs.append(f"accepted لكن الحقل الأساسي {k} غير سليم أو غير موجود في النص")
        if doc["review"]["required"]:
            errs.append("accepted لا يجتمع مع review.required=true")
    if status == "needs_review" and not doc["review"]["fields_to_review"]:
        errs.append("needs_review لازم يحدد fields_to_review")
    if status != "accepted" and not doc.get("reasons"):
        errs.append("الحالة غير accepted لازم يكون لها reasons")
    rn = f["register_number"].get("value")
    if rn is not None and not re.fullmatch(r"\d{3,12}", str(rn)):
        errs.append("register_number لازم يكون أرقامًا عادية فقط")
    for k in ("issue_date", "expiry_date"):
        v = f.get(k, {}).get("value")
        if v is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
            errs.append(f"{k} لازم بصيغة ISO (YYYY-MM-DD)")
    i, e = f["issue_date"].get("value"), f.get("expiry_date", {}).get("value")
    if i and e and e <= i:
        errs.append("expiry_date لازم يكون بعد issue_date")
    for k, v in f.items():
        if v["status"] == "ok" and v["value"] is None:
            errs.append(f"{k}: status=ok مع value فاضي")
    return errs


def main(path):
    schema = json.loads((HERE / "registration_output.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = data if "schema_version" not in data else {"output": data}
    bad = 0
    for name, doc in docs.items():
        errs = [f"المخطط: {'/'.join(map(str, e.path))}: {e.message}" for e in validator.iter_errors(doc)]
        if not errs:
            errs = business_rules(doc)
        print(f"[{'OK' if not errs else 'خطأ'}] {name} ({doc.get('status')})")
        for m in errs:
            print("   -", m)
        bad += bool(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])