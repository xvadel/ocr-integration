#!/usr/bin/env python3
"""
show_result.py - يعرض ناتج pipeline.py بشكل مقروء ويقارن أكثر من ملف نتيجة.

الاستخدام:
    python show_result.py saudi_result.json
    python show_result.py gen_result.json saudi_result.json          # تفصيل + جدول مقارنة
    python show_result.py saudi_result.json --raw                      # النص الخام كاملًا
    python show_result.py saudi_result.json --txt summary.txt          # حفظ UTF-8 (أفضل لقراءة العربي)

يتحمّل اختلاف أسماء الحقول بين النسخ (يستخدم .get ولا ينهار لو مفتاح ناقص).
"""
import argparse
import json
import sys
from pathlib import Path


def get(d, *path, default=None):
    for k in path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def summarize(path, show_raw):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    L = []
    status = r.get("status")
    reasons = get(r, "review", "reasons") or r.get("reasons") or []
    fields = r.get("fields") or {}
    core = [k for k in fields if k in ("company_name", "registration_number", "register_number", "issue_date")]
    raw = get(r, "extraction", "raw_text") or r.get("raw_text") or ""
    L.append(f"=== {Path(path).name} ===")
    L.append(f"status          : {status}")
    L.append(f"reasons         : {reasons}")
    L.append(f"document        : {get(r, 'document', 'type')} | pdf_kind={get(r, 'document', 'pdf_kind')} | pages={get(r, 'document', 'pages')}")
    L.append(f"quality         : {get(r, 'document', 'quality')}")
    L.append(f"extraction      : method={get(r, 'extraction', 'method') or get(r, 'extraction', 'text_method')} "
             f"engine={get(r, 'extraction', 'engine')} conf={get(r, 'extraction', 'ocr_confidence') or get(r, 'extraction', 'ocr_confidence_mean')}")
    L.append(f"fields_to_review: {get(r, 'review', 'fields_to_review')}")
    L.append("")
    L.append(f"{'field':22}{'status':18}{'grounded':10}{'conf':7}value")
    for k, f in fields.items():
        if not isinstance(f, dict):
            continue
        L.append(f"{k:22}{str(f.get('status')):18}{str(f.get('grounded')):10}{str(f.get('confidence')):7}{f.get('value')}")
    L.append("")
    L.append(f"raw_text ({len(raw)} حرفًا):")
    L.append(raw if show_raw else (raw[:900] + (" ..." if len(raw) > 900 else "")))
    ok_core = sum(1 for k in core if isinstance(fields.get(k), dict) and fields[k].get("status") == "ok")
    row = {"file": Path(path).name, "status": status, "core_ok": f"{ok_core}/{len(core)}",
           "fields_ok": f"{sum(1 for f in fields.values() if isinstance(f, dict) and f.get('status') == 'ok')}/{len(fields)}",
           "raw_chars": len(raw)}
    return "\n".join(L), row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--txt", default=None, help="احفظ المخرجات في ملف UTF-8")
    a = ap.parse_args()
    out, rows = [], []
    for f in a.files:
        text, row = summarize(f, a.raw)
        out.append(text)
        rows.append(row)
    if len(rows) > 1:
        out.append("\n=== مقارنة ===")
        out.append(f"{'file':34}{'status':15}{'core_ok':9}{'fields_ok':11}raw_chars")
        for r in rows:
            out.append(f"{r['file']:34}{str(r['status']):15}{r['core_ok']:9}{r['fields_ok']:11}{r['raw_chars']}")
    s = "\n\n".join(out)
    if a.txt:
        Path(a.txt).write_text(s, encoding="utf-8")
        print("تم الحفظ في", a.txt)
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        print(s)


if __name__ == "__main__":
    main()