#!/usr/bin/env python3
"""
ocr_compare.py - شغّل Tesseract و EasyOCR على مجلد صور وقارن النتايج جنب بعض.

التركيب (مرة واحدة):
    pip install pytesseract easyocr pillow
    # و Tesseract نفسه + حزمة العربي:
    #   Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-ara
    #   macOS:         brew install tesseract tesseract-lang
    #   Windows:       نزّل المثبّت من مستودع UB-Mannheim واختر Arabic وقت التثبيت
    #                  (ولو tesseract مش في الـ PATH استخدم --tesseract-cmd)

الاستخدام:
    python ocr_compare.py ./images
    python ocr_compare.py ./images --out results --tess-lang ara+eng --easy-langs ar en
    python ocr_compare.py ./images --gpu        # لو عندك GPU

الناتج (داخل مجلد --out):
    results.csv   -> فتحه في Excel (الترميز UTF-8 مع BOM عشان العربي يظهر صح)
    report.html   -> مقارنة جنب بعض مع الصورة الأصلية
"""

import argparse
import csv
import html
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: Path) -> Image.Image:
    """يفتح الصورة ويصحّح اتجاهها حسب بيانات EXIF (صور الموبايل)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def run_tesseract(img: Image.Image, lang: str):
    import pytesseract

    start = time.perf_counter()
    text = pytesseract.image_to_string(img, lang=lang)
    data = pytesseract.image_to_data(
        img, lang=lang, output_type=pytesseract.Output.DICT
    )
    elapsed = time.perf_counter() - start

    confs = [
        float(c) for c, t in zip(data["conf"], data["text"])
        if str(c) != "-1" and t.strip()
    ]
    avg_conf = statistics.mean(confs) if confs else 0.0  # من 0 إلى 100
    return text.strip(), elapsed, avg_conf


def run_easyocr(reader, img: Image.Image):
    import numpy as np

    start = time.perf_counter()
    # detail=1 يرجّع (box, text, confidence)، و paragraph=False عشان نحتفظ بالثقة
    results = reader.readtext(np.array(img), detail=1, paragraph=False)
    elapsed = time.perf_counter() - start

    lines = [r[1] for r in results]
    confs = [float(r[2]) for r in results]
    avg_conf = statistics.mean(confs) * 100 if confs else 0.0  # نفس مقياس Tesseract
    return "\n".join(lines).strip(), elapsed, avg_conf


def write_html(rows, out_path: Path, image_dir: Path):
    parts = [
        "<!DOCTYPE html><html lang='ar' dir='rtl'><head><meta charset='utf-8'>",
        "<title>OCR comparison</title>",
        "<style>",
        "body{font-family:Tahoma,Arial,sans-serif;margin:24px;background:#f6f7f9}",
        ".card{background:#fff;border-radius:10px;padding:16px;margin-bottom:24px;"
        "box-shadow:0 1px 4px rgba(0,0,0,.12)}",
        ".grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}",
        "img{max-width:100%;border:1px solid #ddd;border-radius:6px}",
        "pre{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;"
        "padding:10px;border-radius:6px;font-family:inherit;min-height:80px}",
        ".meta{color:#555;font-size:13px;margin-bottom:6px}",
        "h2{margin-top:0}",
        "</style></head><body>",
        "<h1>مقارنة Tesseract و EasyOCR</h1>",
    ]
    for r in rows:
        img_rel = (image_dir / r["image"]).as_posix()
        parts.append(f"<div class='card'><h2>{html.escape(r['image'])}</h2><div class='grid'>")
        parts.append(
            f"<div><div class='meta'>الصورة الأصلية</div>"
            f"<img src='{html.escape(img_rel)}' alt=''></div>"
        )
        parts.append(
            f"<div><div class='meta'>Tesseract — {r['tess_time_s']}s — "
            f"ثقة {r['tess_conf']}%</div><pre>{html.escape(r['tess_text'])}</pre></div>"
        )
        parts.append(
            f"<div><div class='meta'>EasyOCR — {r['easy_time_s']}s — "
            f"ثقة {r['easy_conf']}%</div><pre>{html.escape(r['easy_text'])}</pre></div>"
        )
        parts.append("</div></div>")
    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Compare Tesseract and EasyOCR on a folder of images.")
    p.add_argument("folder", type=Path, help="مجلد الصور")
    p.add_argument("--out", type=Path, default=Path("ocr_results"), help="مجلد الناتج")
    p.add_argument("--tess-lang", default="ara+eng", help="لغات Tesseract (الافتراضي ara+eng)")
    p.add_argument("--easy-langs", nargs="+", default=["ar", "en"], help="لغات EasyOCR")
    p.add_argument("--tesseract-cmd", default=None, help="مسار tesseract لو مش في الـ PATH")
    p.add_argument("--gpu", action="store_true", help="استخدم GPU مع EasyOCR")
    args = p.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"المجلد مش موجود: {args.folder}")

    images = sorted(f for f in args.folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit("مفيش صور في المجلد.")

    if args.tesseract_cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    import easyocr
    print("تحميل نموذج EasyOCR (أول مرة هينزّل الملفات)...")
    reader = easyocr.Reader(args.easy_langs, gpu=args.gpu)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name}")
        img = load_image(path)

        try:
            t_text, t_time, t_conf = run_tesseract(img, args.tess_lang)
        except Exception as e:
            t_text, t_time, t_conf = f"[خطأ Tesseract: {e}]", 0.0, 0.0

        try:
            e_text, e_time, e_conf = run_easyocr(reader, img)
        except Exception as e:
            e_text, e_time, e_conf = f"[خطأ EasyOCR: {e}]", 0.0, 0.0

        rows.append({
            "image": path.name,
            "tess_text": t_text,
            "tess_time_s": round(t_time, 2),
            "tess_conf": round(t_conf, 1),
            "easy_text": e_text,
            "easy_time_s": round(e_time, 2),
            "easy_conf": round(e_conf, 1),
        })

    # CSV (utf-8-sig عشان Excel يعرض العربي صح)
    csv_path = args.out / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # HTML: مسار الصور نسبي من مجلد الناتج
    html_path = args.out / "report.html"
    rel_dir = Path(
        __import__("os").path.relpath(args.folder.resolve(), args.out.resolve())
    )
    write_html(rows, html_path, rel_dir)

    # ملخص سريع
    def avg(key):
        return round(statistics.mean(r[key] for r in rows), 2)

    print("\n=== الملخص ===")
    print(f"Tesseract: متوسط الزمن {avg('tess_time_s')}s | متوسط الثقة {avg('tess_conf')}%")
    print(f"EasyOCR  : متوسط الزمن {avg('easy_time_s')}s | متوسط الثقة {avg('easy_conf')}%")
    print(f"\nالناتج: {csv_path}  و  {html_path}")
    print("تنبيه: درجة الثقة مش دقة حقيقية. قارن النص بالنص الصحيح بنفسك.")


if __name__ == "__main__":
    main()