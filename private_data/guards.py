#!/usr/bin/env python3
"""
guards.py - حواجز "ما بعد القرار" لمنع القبول الخاطئ (False Accept).

تُطبَّق على ناتج pipeline (schema 0.3) قبل الحفظ أو الإرسال، ولا تُحسّن الحالة أبدًا:
تخفضها فقط (accepted → needs_review → rejected).

الحواجز:
  G1 نوع المستند: لو النص الخام يدل على شهادة ضريبية أو غيرها وليس سجلًا تجاريًا ← rejected
  G2 منطقة الحقل: قيمة حقل أساسي مصدرها الفوتر أو الهيدر (bbox) ← suspect
  G3 اتفاق ضعيف: engines_agree=true لكن أقل من قراءتين بثقة ≥ 0.5 ← engines_agree=false
  G4 الاكتمال: أكثر من نصف الحقول مفقودة ← needs_review
  G5 تاريخ منتهٍ: expiry_date قبل اليوم ← needs_review

الاستخدام:
    from guards import apply_guards
    doc, applied = apply_guards(doc, expected_type="commercial_register")
"""
import copy
import re
import unicodedata
from datetime import date

CORE = ("company_name", "register_number", "issue_date")
ORDER = {"accepted": 0, "needs_review": 1, "rejected": 2}


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[\u064B-\u0652\u0640]", "", t)
    t = re.sub(r"[إأآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", t)


_DOC_PATTERNS = {
    "commercial_register": [r"السجل التجاري", r"commercial (registration|register)", r"وزارة التجارة",
                            r"ministry of commerce", r"رقم القيد", r"الكيان القانوني",
                            r"trade licen[cs]e", r"رخصة تجارية"],
    "tax_certificate": [r"هيئة الزكاة", r"zakat,? tax and customs", r"ضريبة القيمة المضافة", r"\bvat\b",
                        r"\btin\b", r"certificate no", r"الرقم المميز", r"الشهادة الضريبية", r"tax registration"],
}
PATTERNS = {k: [re.compile(norm(p), re.I) for p in v] for k, v in _DOC_PATTERNS.items()}


def classify_document(raw_text: str):
    """(النوع، الدرجات). النوع unknown لو مفيش فارق واضح."""
    t = norm(raw_text or "")
    scores = {k: sum(1 for p in ps if p.search(t)) for k, ps in PATTERNS.items()}
    best = max(scores, key=scores.get)
    others = max(v for k, v in scores.items() if k != best)
    return (best if scores[best] >= 2 and scores[best] > others else "unknown"), scores


def _worse(cur, new):
    return new if ORDER[new] > ORDER[cur] else cur


def apply_guards(doc, expected_type="commercial_register", today=None, strict_completeness=0.5):
    doc = copy.deepcopy(doc)
    today = today or date.today()
    applied, status = [], doc.get("status", "accepted")
    fields = doc.get("fields", {})
    review = doc.setdefault("review", {"required": False, "fields_to_review": []})

    def flag(code, new_status, field=None):
        nonlocal status
        applied.append(code if not field else f"{field}:{code}")
        status = _worse(status, new_status)
        if field:
            f = fields.get(field)
            if f and f.get("status") == "ok":
                f["status"] = "suspect"
            if f is not None:
                f.setdefault("issues", []).append(code)
            if field not in review.setdefault("fields_to_review", []):
                review["fields_to_review"].append(field)

    # G1: نوع المستند
    kind, scores = classify_document(doc.get("raw_text"))
    if kind not in (expected_type, "unknown"):
        flag(f"wrong_document_type:{kind}", "rejected")
        doc["reupload_hint"] = f"wrong_document_type:{kind} (المطلوب: {expected_type})"
    elif kind == "unknown":
        flag("document_type_uncertain", "needs_review")
    if kind != "unknown":
        doc.setdefault("document", {})["detected_type"] = kind

    # G2: منطقة الحقل
    for k in CORE + ("expiry_date",):
        f = fields.get(k) or {}
        bb = f.get("bbox")
        if f.get("value") is not None and bb and (bb[1] > 0.92 or bb[3] < 0.04):
            flag("suspicious_region", "needs_review", k)

    # G3: اتفاق ضعيف
    for k, f in fields.items():
        if f.get("engines_agree") is True and f.get("value") is not None:
            strong = [c for c in f.get("candidates", []) if c.get("value") == f["value"] and (c.get("confidence") or 0) >= 0.5]
            if len(strong) < 2:
                f["engines_agree"] = False
                flag("weak_agreement", "needs_review", k)

    # G4: الاكتمال
    if fields:
        missing = [k for k, f in fields.items() if f.get("value") is None]
        if len(missing) / len(fields) > strict_completeness:
            flag(f"low_completeness({len(missing)}/{len(fields)})", "needs_review")

    # G5: تاريخ منتهٍ
    ex = (fields.get("expiry_date") or {}).get("value")
    if ex and ex < today.isoformat():
        flag("expired_or_stale_date", "needs_review", "expiry_date")

    if applied:
        doc["status"] = status
        doc["reasons"] = list(dict.fromkeys(doc.get("reasons", []) + applied))
        review["required"] = status == "needs_review"
    return doc, applied
