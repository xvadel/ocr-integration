"""
guards.py - Post-decision guards to prevent false accepts.

Applied to pipeline output before saving/sending. Only downgrades status:
  accepted → needs_review → rejected

Guards:
  G1 Document type: raw_text signals wrong doc type → rejected
  G2 Field region:  critical field bbox in footer/header zone → suspect
  G3 Weak agreement: engines_agree=True but fewer than 2 high-conf readings → engines_agree=False
  G4 Completeness:  >50% fields missing → needs_review
  G5 Expiry date:   expiry_date in the past → needs_review

Usage:
    from regocr.guards import apply_guards
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
    "commercial_register": [
        r"السجل التجاري",
        r"commercial (registration|register)",
        r"وزارة التجارة",
        r"ministry of commerce",
        r"رقم القيد",
        r"الكيان القانوني",
        r"trade licen[cs]e",
        r"رخصة تجارية",
    ],
    "tax_certificate": [
        r"هيئة الزكاة",
        r"zakat,? tax and customs",
        r"ضريبة القيمة المضافة",
        r"\bvat\b",
        r"\btin\b",
        r"certificate no",
        r"الرقم المميز",
        r"الشهادة الضريبية",
        r"tax registration",
    ],
}
PATTERNS = {k: [re.compile(norm(p), re.I) for p in v] for k, v in _DOC_PATTERNS.items()}


def classify_document(raw_text: str):
    """Return (detected_type, scores). Type is 'unknown' when no clear winner."""
    t = norm(raw_text or "")
    scores = {k: sum(1 for p in ps if p.search(t)) for k, ps in PATTERNS.items()}
    best = max(scores, key=scores.get)
    others = max(v for k, v in scores.items() if k != best)
    return (best if scores[best] >= 2 and scores[best] > others else "unknown"), scores


def _worse(cur, new):
    return new if ORDER[new] > ORDER[cur] else cur


def apply_guards(doc, expected_type="commercial_register", today=None, strict_completeness=0.5):
    """
    Run all guards on a plain-dict document representation.

    Parameters
    ----------
    doc : dict
        Must have keys: status, reasons, fields (plain dicts), raw_text.
    expected_type : str
        The document type the user selected.
    today : date, optional
        Override for expiry check (useful in tests).
    strict_completeness : float
        Fraction of missing fields that triggers G4 (default 0.5).

    Returns
    -------
    (doc_copy, applied_codes)
    """
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

    # G1: Document type check
    kind, scores = classify_document(doc.get("raw_text") or "")
    if kind not in (expected_type, "unknown"):
        flag(f"wrong_document_type:{kind}", "rejected")
        doc["reupload_hint"] = f"wrong_document_type:{kind} (expected: {expected_type})"
    elif kind == "unknown":
        flag("document_type_uncertain", "needs_review")
    if kind != "unknown":
        doc.setdefault("document", {})["detected_type"] = kind

    # G2: Field region — bbox in footer (y1>0.92) or header (y2<0.04)
    for k in CORE + ("expiry_date",):
        f = fields.get(k) or {}
        bb = f.get("bbox")
        if f.get("value") is not None and bb and (bb[1] > 0.92 or bb[3] < 0.04):
            flag("suspicious_region", "needs_review", k)

    # G3: Weak agreement — engines_agree=True but <2 candidates with conf>=0.5
    for k, f in fields.items():
        if f.get("engines_agree") is True and f.get("value") is not None:
            strong = [
                c for c in f.get("candidates", [])
                if c.get("value") == f["value"] and (c.get("confidence") or 0) >= 0.5
            ]
            if len(strong) < 2:
                f["engines_agree"] = False
                flag("weak_agreement", "needs_review", k)

    # G4: Completeness — >strict_completeness fraction of fields missing
    if fields:
        missing = [k for k, f in fields.items() if f.get("value") is None]
        if len(missing) / len(fields) > strict_completeness:
            flag(f"low_completeness({len(missing)}/{len(fields)})", "needs_review")

    # G5: Expiry date in the past
    ex = (fields.get("expiry_date") or {}).get("value")
    if ex and ex < today.isoformat():
        flag("expired_or_stale_date", "needs_review", "expiry_date")

    if applied:
        doc["status"] = status
        doc["reasons"] = list(dict.fromkeys(doc.get("reasons", []) + applied))
        review["required"] = status == "needs_review"
    return doc, applied
