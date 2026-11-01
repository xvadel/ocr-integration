#!/usr/bin/env python3
"""
make_pdfs.py - يحوّل صور السجلات التجريبية (من make_samples.py) إلى ملفات PDF لاختبار طرق قراءة الـ PDF.

بينتج 3 أنواع PDF (كلها نماذج وهمية للاختبار):
  scanned/         PDF ممسوح: صورة فقط بدون أي نص (لازم OCR)
  digital/         PDF رقمي سليم: فيه طبقة نص عربي صحيحة (استخراج مباشر)
  digital_visual/  PDF رقمي بنص "بصري": الحروف متشكّلة ومعكوسة الترتيب، زي الملفات
                   اللي بيطلع منها العربي مقلوب أو مفكك عند الاستخراج
  multipage_scan.pdf  4 صفحات ممسوحة في ملف واحد (لاختبار الملفات متعددة الصفحات)

التركيب:
    pip install pymupdf arabic-reshaper python-bidi

الاستخدام:
    python make_pdfs.py images --out pdfs
    (مجلد images لازم فيه الصور و ground_truth.json من make_samples.py)
"""

import argparse
import html
import json
import re
import shutil
import tempfile
from pathlib import Path

try:
    import pymupdf
except ImportError:  # نسخ PyMuPDF القديمة
    import fitz as pymupdf

import arabic_reshaper
from bidi.algorithm import get_display

FONTS = ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans.ttf",
         "/System/Library/Fonts/Supplemental/Arial.ttf"]
BOLDS = ["C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]

ROWS = [
    ("اسم الشركة / المنشأة", "company_name"), ("الشكل القانوني", "legal_form"),
    ("رقم القيد بالسجل التجاري", "register_number"), ("تاريخ القيد", "issue_date"),
    ("النشاط", "activity"), ("المحافظة", "governorate"), ("العنوان", "address"),
    ("رأس المال", "capital"), ("المدير / المالك", "manager"),
    ("رقم البطاقة الضريبية", "tax_id"), ("تاريخ انتهاء الصلاحية", "expiry_date"),
    ("جهة الإصدار", "office"),
]
TITLE = "نموذج وهمي - سجل تجاري"
SUB = "للاختبار فقط - ليس مستندًا رسميًا"
FOOT = "عينة اختبار OCR - بيانات عشوائية غير حقيقية"


def pick(cands, what):
    for c in cands:
        if Path(c).exists():
            return c
    raise SystemExit(f"مالقيتش خط عربي ({what}). عدّل قائمة الخطوط في أول الملف.")


def shape(t):
    return get_display(arabic_reshaper.reshape(t))


# ------------------------------------------------------------- 1) PDF ممسوح
def image_pdf(images, out):
    doc = pymupdf.open()
    for p in images:
        pix = pymupdf.Pixmap(str(p))
        page = doc.new_page(width=595, height=595 * pix.height / pix.width)
        page.insert_image(page.rect, filename=str(p))
    doc.save(str(out), deflate=True)
    doc.close()


# ------------------------------------------------------------- 2) PDF رقمي سليم
def digital_pdf(fields, out, font_path, bold_path):
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(font_path, tmp / "f.ttf")
    shutil.copy(bold_path, tmp / "fb.ttf")
    css = ("@font-face{font-family:ar;src:url(f.ttf)}"
           "@font-face{font-family:ar;font-weight:bold;src:url(fb.ttf)}"
           "*{font-family:ar} h1{text-align:center;font-size:20px;margin:0}"
           "p{text-align:center;font-size:11px;margin:4px 0 14px 0}"
           "table{border-collapse:collapse;width:507px;table-layout:fixed}"
           "td{border:1px solid #444;padding:8px;font-size:12px}"
           ".l{background:#eef0f5;font-weight:bold}")
    rows = "".join(
        f"<tr><td class='l' width='150' style='width:150px'>{label}</td>"
        f"<td width='325' style='width:325px'>{html.escape(fields[key])}</td></tr>"
        for label, key in ROWS)
    body = (f"<h1>{TITLE}</h1><p>{SUB}</p><table dir='rtl'>{rows}</table>"
            f"<p style='margin-top:14px'>{FOOT}</p>")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_htmlbox(pymupdf.Rect(40, 40, 555, 800),
                        f"<div dir='rtl'>{body}</div>", css=css,
                        archive=pymupdf.Archive(str(tmp)))
    doc.save(str(out))
    doc.close()
    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- 3) PDF رقمي بنص بصري (مشكلة شائعة)
def visual_pdf(fields, out, font_path, bold_path):
    doc = pymupdf.open()
    page = doc.new_page()
    f = pymupdf.Font(fontfile=font_path)
    fb = pymupdf.Font(fontfile=bold_path)
    tw = pymupdf.TextWriter(page.rect)

    def put(text, right_x, y, font, size):
        s = shape(text)  # حروف متشكلة + ترتيب بصري (معكوس منطقيًا)
        w = font.text_length(s, fontsize=size)
        tw.append((right_x - w, y), s, font=font, fontsize=size)

    def center(text, y, font, size):
        s = shape(text)
        tw.append(((595 - font.text_length(s, fontsize=size)) / 2, y), s, font=font, fontsize=size)

    center(TITLE, 70, fb, 20)
    center(SUB, 92, f, 11)
    left, right, split, top, rh = 50, 545, 340, 110, 50
    page.draw_rect(pymupdf.Rect(left, top, right, top + rh * len(ROWS)), width=1)
    for i, (label, key) in enumerate(ROWS):
        y = top + i * rh
        if i:
            page.draw_line((left, y), (right, y), width=0.6)
        put(label, right - 8, y + 30, fb, 11)
        put(fields[key], split - 8, y + 30, f, 12)
    page.draw_line((split, top), (split, top + rh * len(ROWS)), width=1)
    center(FOOT, top + rh * len(ROWS) + 30, f, 10)
    tw.write_text(page)
    doc.save(str(out))
    doc.close()


def main():
    ap = argparse.ArgumentParser(description="Make test PDFs from the sample register images.")
    ap.add_argument("images", type=Path, help="مجلد الصور (فيه ground_truth.json)")
    ap.add_argument("--out", type=Path, default=Path("pdfs"))
    a = ap.parse_args()

    gt_path = a.images / "ground_truth.json"
    if not gt_path.exists():
        raise SystemExit("ground_truth.json مش موجود في مجلد الصور. شغّل make_samples.py الأول.")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    font_path, bold_path = pick(FONTS, "عادي"), pick(BOLDS, "عريض")

    for sub in ("scanned", "digital", "digital_visual"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)

    scans = []
    for fname, info in sorted(gt.items()):
        n = re.search(r"sample_(\d+)", fname).group(1)
        img = a.images / fname
        image_pdf([img], a.out / "scanned" / (Path(fname).stem + ".pdf"))
        digital_pdf(info["fields"], a.out / "digital" / f"sample_{n}_digital.pdf", font_path, bold_path)
        visual_pdf(info["fields"], a.out / "digital_visual" / f"sample_{n}_visual.pdf", font_path, bold_path)
        scans.append(img)
        print("تم:", fname)

    image_pdf(scans[:4], a.out / "multipage_scan.pdf")
    shutil.copy(gt_path, a.out / "ground_truth.json")
    print(f"\nالـ PDFs في: {a.out}")


if __name__ == "__main__":
    main()
    