#!/usr/bin/env python3
"""
pdf_scan_compare.py - جرّب طرق قراءة ملفات PDF وقارن دقتها بالقيم الصحيحة.

الطرق:
  pymupdf      استخراج النص المباشر من PDF (طبقة النص) - سريع ومجاني
  pdfplumber   استخراج نص مباشر بمكتبة تانية (للمقارنة)
  pdfplumber_bidi  نفس pdfplumber + إعادة ترتيب الاتجاه (bidi) لإصلاح العربي المعكوس
  tesseract    تحويل كل صفحة لصورة ثم OCR بـ Tesseract
  easyocr      تحويل كل صفحة لصورة ثم OCR بـ EasyOCR
  auto         المسار المقترح: يجرّب النص المباشر، ولو الملف ممسوح أو النص مشوّه
               (كلمات الحقول مش ظاهرة، أو تواريخ بأرقام معكوسة) يرجع لـ OCR تلقائيًا

التركيب:
    pip install pymupdf pdfplumber pytesseract easyocr pillow
    (و Tesseract نفسه + ara.traineddata كما في ocr_compare.py)

الاستخدام:
    python pdf_scan_compare.py pdfs
    python pdf_scan_compare.py pdfs --methods pymupdf tesseract auto
    python pdf_scan_compare.py pdfs --dpi 250 --psm 4 --save-text
    python pdf_scan_compare.py file.pdf --methods auto        # ملف واحد (بدون تقييم)

مجلد pdfs لازم يكون ناتج make_pdfs.py (فيه scanned/ و digital/ و digital_visual/ و ground_truth.json)
عشان يحسب الدقة. الناتج: results.csv + ملخص في الشاشة (+ النصوص المستخرجة مع --save-text).
"""

import argparse
import csv
import json
import re
import statistics as st
import sys
import time
import unicodedata
from pathlib import Path

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from PIL import Image

ALL_METHODS = ["pymupdf", "pdfplumber", "pdfplumber_bidi", "tesseract", "easyocr", "auto"]
DEFAULT_ANCHORS = ["رقم القيد", "تاريخ", "السجل", "الشكل القانوني", "الشركه"]
NUM = ["register_number", "issue_date", "expiry_date", "tax_id"]
TXT = ["company_name", "legal_form", "activity", "governorate", "address", "manager", "office"]


# ----------------------------------------------------------------- تطبيع وتقييم
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)  # يحوّل الحروف المتشكلة (ﻣ ﻟ) لحروفها الأصلية
    t = t.translate(AR_DIGITS)
    t = re.sub(r"[\u064B-\u0652\u0640\u200e\u200f\u202a-\u202e]", "", t)
    t = re.sub(r"[إأآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return t


def digits(t):
    return re.sub(r"\D", "", norm(t))


def toks(t):
    return [w for w in re.split(r"[\s\-/،,:.؛|()]+", norm(t)) if w]


def flat(t: str) -> str:
    """نص مسطّح للمقارنة: تطبيع + حذف علامات الترقيم + مسافة واحدة."""
    t = re.sub(r"[^\w\s]", " ", norm(t))
    return re.sub(r"\s+", " ", t).strip()


def score(fields: dict, text: str):
    """(أرقام، نص بالكلمات، نص بالترتيب الكامل) كنسب 0-1.
    - أرقام: رقم القيد والتواريخ والبطاقة الضريبية (تطابق الأرقام فقط)
    - نص بالكلمات: نسبة كلمات القيمة الموجودة في الناتج (بدون اعتبار الترتيب)
    - نص بالترتيب: القيمة كاملة وبنفس ترتيب الكلمات (يكشف العربي المعكوس)"""
    nd, tt, ft = digits(text), set(toks(text)), flat(text)
    num = [1.0 if digits(fields[k]) and digits(fields[k]) in nd else 0.0 for k in NUM]
    words, exact = [], []
    for k in TXT:
        v = toks(fields[k])
        words.append(sum(w in tt for w in v) / len(v) if v else 0.0)
        exact.append(1.0 if flat(fields[k]) and flat(fields[k]) in ft else 0.0)
    return st.mean(num), st.mean(words), st.mean(exact)


# ----------------------------------------------------------------- كشف حالة النص
DATE_RE = re.compile(r"(?<!\d)(\d{1,4})/(\d{1,2})/(\d{1,4})(?!\d)")


def dates_valid(text: str):
    """كل تاريخ بصيغة x/y/z لازم يكون يوم/شهر/سنة معقولة. التواريخ المعكوسة (سنة أولًا) تفشل."""
    for a, b, c in DATE_RE.findall(norm(text)):
        if not (len(a) <= 2 and len(b) <= 2 and len(c) == 4
                and 1 <= int(a) <= 31 and 1 <= int(b) <= 12 and 1950 <= int(c) <= 2100):
            return False
    return True


def text_layer_status(raw: str, anchors):
    """يرجّع (نثق في طبقة النص؟، السبب). الفحوص:
    1) فيه نص أصلًا؟  2) كلمات الحقول المتوقعة ظاهرة (يعني العربي سليم)؟
    3) التواريخ منطقية (يعني الأرقام مش معكوسة)؟"""
    if len(raw.strip()) < 40:
        return False, "no_text_layer (ممسوح)"
    ft = flat(raw)
    found = sum(1 for w in anchors if flat(w) in ft)
    if found < 2:
        return False, f"anchors_missing ({found} من {len(anchors)}: نص مشوّه أو معكوس)"
    if not dates_valid(raw):
        return False, "dates_invalid (أرقام معكوسة؟)"
    return True, "ok"


# ----------------------------------------------------------------- الطرق
def render_pages(pdf: Path, dpi: int):
    doc = pymupdf.open(str(pdf))
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()


def m_pymupdf(pdf, **_):
    doc = pymupdf.open(str(pdf))
    return "\n".join(p.get_text() for p in doc)


def m_pdfplumber(pdf, **_):
    import pdfplumber
    with pdfplumber.open(str(pdf)) as d:
        return "\n".join((p.extract_text() or "") for p in d.pages)


def m_pdfplumber_bidi(pdf, **_):
    from bidi.algorithm import get_display
    raw = m_pdfplumber(pdf)
    # الترتيب مهم: bidi الأول ثم NFKC. لو عكسناهم، حرفا (لا / لأ / لإ) بيتبدلوا (ال / األ / اإل)
    return "\n".join(unicodedata.normalize("NFKC", get_display(ln)) for ln in raw.splitlines())


def m_tesseract(pdf, dpi, psm, tess_lang, **_):
    import pytesseract
    cfg = f"--psm {psm}"
    return "\n".join(pytesseract.image_to_string(im, lang=tess_lang, config=cfg)
                     for im in render_pages(pdf, dpi))


_reader = None


def m_easyocr(pdf, dpi, gpu, **_):
    global _reader
    import easyocr
    import numpy as np
    if _reader is None:
        print("تحميل نموذج EasyOCR...")
        _reader = easyocr.Reader(["ar", "en"], gpu=gpu)
    out = []
    for im in render_pages(pdf, dpi):
        out += [r[1] for r in _reader.readtext(np.array(im), detail=1, paragraph=False)]
    return "\n".join(out)


def m_auto(pdf, ocr_method, anchors, **kw):
    """يجرّب طرق استخراج النص بالترتيب، ويقبل أول نتيجة تعدّي فحص الصلاحية.
    لو مفيش نتيجة صالحة (ممسوح أو نص مشوّه) يرجع لـ OCR."""
    reasons = []
    for name in ("pymupdf", "pdfplumber_bidi"):
        try:
            raw = METHODS[name](pdf)
        except ImportError:
            continue
        ok, reason = text_layer_status(raw, anchors)
        if ok:
            return raw, f"text_layer:{name}"
        reasons.append(f"{name}: {reason}")
        if reason.startswith("no_text_layer"):
            break  # مفيش نص أصلًا، مفيش داعي نجرّب طريقة تانية
    return METHODS[ocr_method](pdf, **kw), f"fallback_ocr:{ocr_method} ← " + " | ".join(reasons)


METHODS = {"pymupdf": m_pymupdf, "pdfplumber": m_pdfplumber,
           "pdfplumber_bidi": m_pdfplumber_bidi, "tesseract": m_tesseract, "easyocr": m_easyocr}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Test PDF reading methods.")
    ap.add_argument("path", type=Path, help="مجلد PDFs (من make_pdfs.py) أو ملف PDF واحد")
    ap.add_argument("--methods", nargs="+", default=["pymupdf", "tesseract", "auto"],
                    choices=ALL_METHODS)
    ap.add_argument("--dpi", type=int, default=200, help="دقة تحويل الصفحة لصورة (200-300)")
    ap.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")
    ap.add_argument("--tess-lang", default="ara+eng")
    ap.add_argument("--tesseract-cmd", default=None)
    ap.add_argument("--auto-ocr", default="tesseract", choices=["tesseract", "easyocr"],
                    help="طريقة OCR الاحتياطية داخل auto")
    ap.add_argument("--anchors", nargs="+", default=DEFAULT_ANCHORS,
                    help="كلمات متوقعة في السجل، تُستخدم في فحص سلامة النص داخل auto")
    ap.add_argument("--gt", type=Path, default=None, help="مسار ground_truth.json لو مش في المجلد")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("pdf_results"))
    ap.add_argument("--save-text", action="store_true", help="احفظ النص المستخرج لكل ملف وطريقة")
    a = ap.parse_args()

    if a.tesseract_cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = a.tesseract_cmd

    if a.path.is_file():
        pdfs, root = [a.path], a.path.parent
    else:
        pdfs, root = sorted(a.path.rglob("*.pdf")), a.path
    if not pdfs:
        sys.exit("مفيش ملفات PDF.")

    gt_file = a.gt or next((c for c in (root / "ground_truth.json", root.parent / "ground_truth.json")
                            if c.exists()), root / "ground_truth.json")
    gt = json.loads(gt_file.read_text(encoding="utf-8")) if gt_file.exists() else {}
    if not gt:
        print("تنبيه: ground_truth.json مش موجود، فمفيش حساب دقة (الوقت وعدد الحروف فقط).")

    a.out.mkdir(parents=True, exist_ok=True)
    if a.save_text:
        (a.out / "text").mkdir(exist_ok=True)

    kw = dict(dpi=a.dpi, psm=a.psm, tess_lang=a.tess_lang, gpu=a.gpu, ocr_method=a.auto_ocr,
              anchors=a.anchors)
    rows = []
    for pdf in pdfs:
        kind = pdf.parent.name if pdf.parent != root else "other"
        m = re.search(r"sample_(\d+)", pdf.name)
        fields = None
        if m and gt:
            key = next((k for k in gt if k.startswith(f"sample_{m.group(1)}_")), None)
            fields = gt[key]["fields"] if key else None
        try:
            pages = len(pymupdf.open(str(pdf)))
        except Exception:
            pages = 0
        for meth in a.methods:
            t0 = time.perf_counter()
            note = ""
            try:
                res = m_auto(pdf, **kw) if meth == "auto" else METHODS[meth](pdf, **kw)
                text, note = res if isinstance(res, tuple) else (res, "")
            except Exception as e:
                text, note = "", f"ERROR: {e}"
            dt = time.perf_counter() - t0
            num, txt, ex = score(fields, text) if fields else (None, None, None)
            rows.append({"pdf": pdf.name, "kind": kind, "pages": pages, "method": meth,
                         "time_s": round(dt, 2), "chars": len(text.strip()),
                         "numbers_acc": None if num is None else round(100 * num),
                         "text_words_acc": None if txt is None else round(100 * txt),
                         "text_exact_acc": None if ex is None else round(100 * ex),
                         "note": note})
            if a.save_text:
                (a.out / "text" / f"{pdf.stem}__{meth}.txt").write_text(text, encoding="utf-8")
            print(f"{pdf.name:30} {meth:15} {dt:6.2f}s  chars={len(text.strip()):5}  "
                  f"num={rows[-1]['numbers_acc']}  words={rows[-1]['text_words_acc']}  "
                  f"exact={rows[-1]['text_exact_acc']}  {note}")

    with open(a.out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== ملخص: متوسط لكل (نوع الـ PDF × الطريقة) ===")
    print(f"{'النوع':16}{'الطريقة':17}{'n':>3}{'زمن(ث)':>9}{'أرقام%':>9}{'كلمات%':>8}{'ترتيب%':>8}")
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["method"]), []).append(r)
    for (kind, meth), rs in sorted(groups.items()):
        def avg(k):
            v = [r[k] for r in rs if r[k] is not None]
            return round(st.mean(v)) if v else "-"
        print(f"{kind:16}{meth:17}{len(rs):>3}{round(st.mean(r['time_s'] for r in rs), 2):>9}"
              f"{avg('numbers_acc'):>9}{avg('text_words_acc'):>8}{avg('text_exact_acc'):>8}")
    print(f"\nالتفاصيل: {a.out / 'results.csv'}")


if __name__ == "__main__":
    main()