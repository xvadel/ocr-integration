#!/usr/bin/env python3
"""
make_samples.py - يولّد صور "سجل تجاري" تجريبية (نموذج وهمي) لاختبار الـ OCR.

مهم: الصور دي مش مستند رسمي ومش نسخة من أي نموذج حكومي.
  - كل البيانات وهمية (عشوائية).
  - مكتوب على كل صورة "نموذج وهمي" + علامة مائية "عينة للاختبار" + ختم تجريبي.
  - مفيش شعار الدولة ولا أي ختم رسمي.
الهدف: تقيس قدرة الأدوات على قراءة العربي والأرقام (عادية وهندية) وتحمّل جودة الصور.
بعد كده لازم تجرّب على سجلات حقيقية (مع تغطية البيانات الحساسة) لأن شكل المستند الحقيقي مختلف.

التركيب:
    pip install pillow numpy arabic-reshaper python-bidi

الاستخدام:
    python make_samples.py                      # 12 صورة في مجلد samples
    python make_samples.py --count 30 --out images
    python make_samples.py --font "C:\\Windows\\Fonts\\tahoma.ttf"

الناتج:
    <out>/sample_01_clean.jpg ...   الصور
    <out>/ground_truth.json         القيم الصحيحة لكل صورة (للمقارنة مع ناتج الـ OCR)
"""

import argparse
import json
import random
from pathlib import Path

import arabic_reshaper
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

W, H = 1240, 1754  # A4 تقريبًا بدقة 150 dpi

# ---------------------------------------------------------------- الخطوط
FONT_CANDIDATES = [
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]
BOLD_CANDIDATES = [
    "C:/Windows/Fonts/tahomabd.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def find_font(cands, override=None):
    if override:
        return override
    for c in cands:
        if Path(c).exists():
            return c
    raise SystemExit("مالقيتش خط عربي. مرّر مسار خط بـ --font (مثلًا tahoma.ttf أو arial.ttf).")


# لو Pillow عنده libraqm بيشكّل العربي ويرتّب الاتجاه بنفسه، فمنكررش الشغل.
# لو مش موجود بنستخدم arabic-reshaper + python-bidi.
HAS_RAQM = features.check("raqm")


def shape(text: str) -> str:
    """تشكيل الحروف العربية + ترتيب الاتجاه (RTL) - للحالة اللي مفيهاش raqm."""
    return get_display(arabic_reshaper.reshape(text))


def tlen(d, text, font):
    if HAS_RAQM:
        return d.textlength(text, font=font, direction="rtl", language="ar")
    return d.textlength(shape(text), font=font)


def dtext(d, xy, text, font, fill):
    if HAS_RAQM:
        d.text(xy, text, font=font, fill=fill, direction="rtl", language="ar")
    else:
        d.text(xy, shape(text), font=font, fill=fill)


# ---------------------------------------------------------------- البيانات الوهمية
WORDS = ["النيل", "الدلتا", "الأهرام", "السلام", "الفجر", "النور", "الشروق", "الوادي",
         "المستقبل", "الأمل", "الصفوة", "التقدم", "الريادة", "المحروسة", "الواحة"]
BUSINESS = [
    ("للتوريدات الهندسية", "توريدات هندسية واستيراد وتصدير"),
    ("للتجارة والتوزيع", "تجارة وتوزيع السلع الاستهلاكية"),
    ("للمقاولات العامة", "مقاولات عامة وأعمال البناء"),
    ("لتكنولوجيا المعلومات", "تطوير البرمجيات وخدمات تكنولوجيا المعلومات"),
    ("للصناعات الغذائية", "تصنيع وتعبئة المواد الغذائية"),
    ("للخدمات اللوجستية", "النقل والتخزين والخدمات اللوجستية"),
    ("للأدوات المكتبية", "تجارة الأدوات المكتبية والقرطاسية"),
    ("للتسويق والدعاية", "خدمات التسويق والدعاية والإعلان"),
    ("للمستلزمات الطبية", "تجارة المستلزمات الطبية"),
    ("للأعمال الزراعية", "استصلاح واستزراع الأراضي وتجارة المحاصيل"),
]
LEGAL_FORMS = ["شركة ذات مسئولية محدودة", "شركة مساهمة مصرية", "منشأة فردية", "شركة تضامن"]
PLACES = [("الدقهلية", "المنصورة"), ("القاهرة", "مدينة نصر"), ("الجيزة", "الدقي"),
          ("الإسكندرية", "سموحة"), ("الغربية", "طنطا"), ("الشرقية", "الزقازيق"),
          ("دمياط", "دمياط الجديدة"), ("القليوبية", "بنها")]
STREETS = ["الجمهورية", "النصر", "الثورة", "التحرير", "الجلاء", "الحرية"]
FIRST = ["أحمد", "محمد", "محمود", "مصطفى", "خالد", "عمر", "علي", "حسن", "إبراهيم", "يوسف"]
LAST = ["السيد", "منصور", "فؤاد", "الشافعي", "رمضان", "سليمان", "عيسى", "الجندي", "حجازي", "النجار"]
CAPITALS = [50000, 100000, 250000, 500000, 1000000, 5000000]

AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def make_record(rng: random.Random, arabic_digits: bool) -> dict:
    biz, activity = rng.choice(BUSINESS)
    form = rng.choice(LEGAL_FORMS)
    kind = "مؤسسة" if form == "منشأة فردية" else "شركة"
    gov, city = rng.choice(PLACES)
    year = rng.randint(2014, 2024)
    issue = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{year}"
    expiry = f"{issue[:6]}{year + 5}"
    tax = "-".join(f"{rng.randint(100, 999)}" for _ in range(3))

    rec = {
        "company_name": f"{kind} {rng.choice(WORDS)} {biz}",
        "legal_form": form,
        "register_number": str(rng.randint(10000, 999999)),
        "issue_date": issue,
        "activity": activity,
        "governorate": gov,
        "address": f"{rng.randint(1, 120)} شارع {rng.choice(STREETS)}، {city}",
        "capital": f"{rng.choice(CAPITALS):,} جنيه مصري",
        "manager": f"{rng.choice(FIRST)} {rng.choice(FIRST)} {rng.choice(LAST)}",
        "tax_id": tax,
        "expiry_date": expiry,
        "office": f"مكتب السجل التجاري - {city}",
    }
    if arabic_digits:
        rec = {k: v.translate(AR_DIGITS).replace(",", "٬") for k, v in rec.items()}
    return rec


ROWS = [
    ("اسم الشركة / المنشأة", "company_name"),
    ("الشكل القانوني", "legal_form"),
    ("رقم القيد بالسجل التجاري", "register_number"),
    ("تاريخ القيد", "issue_date"),
    ("النشاط", "activity"),
    ("المحافظة", "governorate"),
    ("العنوان", "address"),
    ("رأس المال", "capital"),
    ("المدير / المالك", "manager"),
    ("رقم البطاقة الضريبية", "tax_id"),
    ("تاريخ انتهاء الصلاحية", "expiry_date"),
    ("جهة الإصدار", "office"),
]


# ---------------------------------------------------------------- الرسم
def draw_rtl(draw, text, right_x, y, font, fill=(20, 20, 20)):
    w = tlen(draw, text, font)
    dtext(draw, (right_x - w, y), text, font, fill)


def fit_font(draw, text, path, size, max_w, min_size=22):
    while size > min_size:
        f = ImageFont.truetype(path, size)
        if tlen(draw, text, f) <= max_w:
            return f
        size -= 2
    return ImageFont.truetype(path, min_size)


def render_page(rec: dict, font_path: str, bold_path: str) -> Image.Image:
    img = Image.new("RGB", (W, H), (252, 252, 250))
    d = ImageDraw.Draw(img)

    title_f = ImageFont.truetype(bold_path, 54)
    sub_f = ImageFont.truetype(font_path, 30)

    # العنوان (واضح إنه نموذج وهمي)
    for text, f, y in [("نموذج وهمي - سجل تجاري", title_f, 90),
                       ("للاختبار فقط - ليس مستندًا رسميًا", sub_f, 170)]:
        dtext(d, ((W - tlen(d, text, f)) / 2, y), text, f, (40, 40, 40))

    right, left = W - 90, 90
    label_w = 380
    split = right - label_w
    top, row_h = 260, 112

    d.rectangle([left, top, right, top + row_h * len(ROWS)], outline=(60, 60, 60), width=3)
    for i, (label, key) in enumerate(ROWS):
        y = top + i * row_h
        if i:
            d.line([left, y, right, y], fill=(120, 120, 120), width=2)
        d.rectangle([split, y, right, y + row_h], fill=(238, 240, 244))
        lf = fit_font(d, label, bold_path, 32, label_w - 36, min_size=20)
        draw_rtl(d, label, right - 18, y + 36, lf, fill=(30, 30, 30))
        vf = fit_font(d, rec[key], font_path, 36, split - left - 40)
        draw_rtl(d, rec[key], split - 20, y + 34, vf)
    d.line([split, top, split, top + row_h * len(ROWS)], fill=(60, 60, 60), width=3)

    foot_f = ImageFont.truetype(font_path, 26)
    ft = "عينة اختبار OCR - بيانات عشوائية غير حقيقية"
    dtext(d, ((W - tlen(d, ft, foot_f)) / 2, H - 110), ft, foot_f, (90, 90, 90))

    # علامة مائية مائلة
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    wm_f = ImageFont.truetype(bold_path, 150)
    wt = "عينة للاختبار"
    dtext(ld, ((W - tlen(ld, wt, wm_f)) / 2, H / 2 - 80), wt, wm_f, (190, 190, 190, 85))
    layer = layer.rotate(28, resample=Image.BICUBIC, center=(W / 2, H / 2))
    img = Image.alpha_composite(img.convert("RGBA"), layer)

    # ختم تجريبي أزرق (بيعمل تداخل زي الأختام الحقيقية)
    stamp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    cx, cy, r = 300, top + row_h * len(ROWS) - 40, 130
    sd.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(30, 60, 190, 150), width=6)
    sd.ellipse([cx - r + 16, cy - r + 16, cx + r - 16, cy + r - 16], outline=(30, 60, 190, 150), width=3)
    st_f = ImageFont.truetype(bold_path, 34)
    stt = "ختم تجريبي"
    dtext(sd, (cx - tlen(sd, stt, st_f) / 2, cy - 22), stt, st_f, (30, 60, 190, 160))
    stamp = stamp.rotate(-15, resample=Image.BICUBIC, center=(cx, cy))
    img = Image.alpha_composite(img, stamp)
    return img.convert("RGB")


# ---------------------------------------------------------------- تدهور الجودة
VARIANTS = {
    #            زاوية, blur, ضجيج, تباين, تصغير, جودة JPEG, تدرج إضاءة
    "clean": dict(rot=0.0, blur=0.0, noise=0, contrast=1.0, scale=1.0, q=95, grad=False, gray=False),
    "scan": dict(rot=1.5, blur=0.7, noise=9, contrast=0.88, scale=0.9, q=70, grad=False, gray=True),
    "phone": dict(rot=3.0, blur=0.9, noise=6, contrast=0.95, scale=0.75, q=50, grad=True, gray=False),
    "bad": dict(rot=4.0, blur=1.5, noise=18, contrast=0.7, scale=0.55, q=30, grad=True, gray=True),
}


def degrade(img: Image.Image, v: dict, rng: random.Random, nprng) -> Image.Image:
    if v["rot"]:
        ang = rng.uniform(-v["rot"], v["rot"])
        img = img.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=(228, 228, 228))
    if v["scale"] != 1.0:
        img = img.resize((int(img.width * v["scale"]), int(img.height * v["scale"])), Image.LANCZOS)
    if v["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(v["blur"]))

    arr = np.asarray(img).astype(np.float32)
    arr = (arr - 128) * v["contrast"] + 128
    if v["grad"]:
        ramp = np.linspace(rng.uniform(0.7, 0.9), 1.05, arr.shape[1], dtype=np.float32)
        if rng.random() < 0.5:
            ramp = ramp[::-1]
        arr *= ramp[None, :, None]
    if v["noise"]:
        arr += nprng.normal(0, v["noise"], arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.convert("L").convert("RGB") if v["gray"] else img


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generate mock commercial-register images for OCR testing.")
    ap.add_argument("--count", type=int, default=12, help="عدد الصور (الافتراضي 12)")
    ap.add_argument("--out", type=Path, default=Path("samples"), help="مجلد الناتج")
    ap.add_argument("--seed", type=int, default=7, help="رقم عشوائي ثابت لتكرار نفس النتيجة")
    ap.add_argument("--font", default=None, help="مسار خط عربي (اختياري)")
    ap.add_argument("--bold-font", default=None, help="مسار خط عريض (اختياري)")
    args = ap.parse_args()

    font_path = find_font(FONT_CANDIDATES, args.font)
    bold_path = find_font(BOLD_CANDIDATES, args.bold_font) if (args.bold_font or not args.font) else args.font
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    names = list(VARIANTS)
    truth = {}
    for i in range(1, args.count + 1):
        variant = names[(i - 1) % len(names)]
        arabic_digits = (i % 2 == 0)  # كل صورة زوجية بأرقام هندية
        rec = make_record(rng, arabic_digits)
        page = render_page(rec, font_path, bold_path)
        page = degrade(page, VARIANTS[variant], rng, nprng)

        fname = f"sample_{i:02d}_{variant}.jpg"
        page.save(args.out / fname, quality=VARIANTS[variant]["q"])
        truth[fname] = {"variant": variant,
                        "digits": "arabic-indic" if arabic_digits else "western",
                        "fields": rec}
        print("تم:", fname)

    (args.out / "ground_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nالصور والـ ground_truth.json في: {args.out}")


if __name__ == "__main__":
    main()