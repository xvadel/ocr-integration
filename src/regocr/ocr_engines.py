"""
OCR engines module providing unified interfaces for:
  - PaddleOCR (PP-OCRv5 via PaddleX / PaddleOCR 3.x)
  - EasyOCR (ar, en) — full-page AND fast cropped numeric recognizer
  - Tesseract 5 (via pytesseract, LSTM, ara+eng / eng numeric whitelist)

HARD INVARIANT (Priority 5.1):
  Every OCR method accepts ONLY a `PreprocessResult` created by `regocr.preprocess.preprocess_image`.
  Passing a raw numpy array or PIL Image raises TypeError.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from regocr.preprocess import PreprocessResult, crop_field_image

HERE = Path(__file__).resolve().parent.parent.parent

# Cache directories for DL OCR models locally in project (offline-safe)
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(HERE / ".paddlex"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Tesseract default path on Windows
if sys.platform == "win32":
    default_win_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_win_tess):
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = default_win_tess
        except ImportError:
            pass


def _require_preprocessed(prep: Any, variant: str = "A") -> np.ndarray:
    """Enforce that every OCR call receives a PreprocessResult from the preprocessing gate."""
    if not isinstance(prep, PreprocessResult):
        raise TypeError(
            f"OCR engines only accept PreprocessResult from preprocess_image(), got {type(prep).__name__}"
        )
    if variant.upper() == "B":
        if prep.variant_b is None:
            raise ValueError("Requested Variant B, but PreprocessResult.variant_b is None.")
        return prep.variant_b
    return prep.variant_a


class OCRWord:
    def __init__(self, text: str, confidence: float, bbox: List[float]):
        self.text = text
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.bbox = [max(0.0, min(1.0, float(b))) for b in bbox]  # [x1, y1, x2, y2] normalized [0, 1]

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def x_mid(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def y_mid(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def height(self) -> float:
        return max(0.001, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": [round(b, 4) for b in self.bbox],
        }


class OCRResult:
    def __init__(
        self,
        text: str,
        mean_confidence: float,
        words: List[OCRWord],
        engine: str,
        variant: str = "A",
        page: int = 1,
    ):
        self.text = text
        self.mean_confidence = max(0.0, min(1.0, float(mean_confidence)))
        self.words = words
        self.engine = engine
        self.variant = variant
        self.page = page


class TesseractEngine:
    """Tesseract 5 OCR engine (Apache-2.0)."""

    def __init__(self, lang: str = "ara+eng", psm: int = 4):
        self.lang = lang
        self.psm = psm

    def run(self, prep: PreprocessResult, variant: str = "A") -> OCRResult:
        import pytesseract

        image = _require_preprocessed(prep, variant)
        h, w = image.shape[:2]
        config = f"--psm {self.psm}"

        raw_text = pytesseract.image_to_string(image, lang=self.lang, config=config)
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=config, output_type=pytesseract.Output.DICT
        )

        words: List[OCRWord] = []
        conf_scores: List[float] = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word_text = str(data["text"][i]).strip()
            conf_val = str(data["conf"][i])
            if not word_text or conf_val == "-1":
                continue

            try:
                score = max(0.0, min(1.0, float(conf_val) / 100.0))
                if score > 0:
                    conf_scores.append(score)
            except (ValueError, TypeError):
                score = 0.5

            x = data["left"][i]
            y = data["top"][i]
            bw = data["width"][i]
            bh = data["height"][i]

            norm_bbox = [x / w, y / h, (x + bw) / w, (y + bh) / h]
            words.append(OCRWord(text=word_text, confidence=score, bbox=norm_bbox))

        mean_conf = round(sum(conf_scores) / len(conf_scores), 4) if conf_scores else 0.0
        return OCRResult(
            text=unicodedata.normalize("NFKC", raw_text),
            mean_confidence=mean_conf,
            words=words,
            engine="tesseract",
            variant=variant,
            page=prep.page,
        )

    def recognize_numeric_crop(
        self,
        prep: PreprocessResult,
        bbox: List[float],
        whitelist: str = "0123456789/-.,",
        lang: str = "eng",
    ) -> Tuple[str, float]:
        """Single-line numeric re-OCR on a cropped, 2x upscaled field region.

        Empirically verified on Windows Tesseract 5 LSTM: `-l eng --psm 7` with ASCII digit
        whitelist works accurately for Western digits, whereas `-l ara` whitelist fails on
        Arabic-Indic digits (handled by EasyOCR.recognize_numeric_crop instead).
        """
        import pytesseract

        _require_preprocessed(prep, "A")
        crop = crop_field_image(prep, bbox, pad_x=0.020, pad_y=0.010, upscale=2.0)
        # Add clean white border to prevent edge clipping in --psm 7
        crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=250)
        config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
        data = pytesseract.image_to_data(
            crop, lang=lang, config=config, output_type=pytesseract.Output.DICT
        )

        tokens: List[str] = []
        scores: List[float] = []
        for txt, conf in zip(data["text"], data["conf"]):
            t = str(txt).strip()
            if not t or str(conf) == "-1":
                continue
            tokens.append(t)
            try:
                s = max(0.0, min(1.0, float(conf) / 100.0))
                if s > 0:
                    scores.append(s)
            except (ValueError, TypeError):
                pass

        text = " ".join(tokens).strip()
        mean_c = round(sum(scores) / len(scores), 4) if scores else 0.0
        return unicodedata.normalize("NFKC", text), mean_c


class EasyOCREngine:
    """EasyOCR engine (Apache-2.0)."""

    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = languages or ["ar", "en"]
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            model_dir = str(HERE / ".easyocr")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=False,
                model_storage_directory=model_dir,
                user_network_directory=model_dir,
            )
        return self._reader

    def run(self, prep: PreprocessResult, variant: str = "A") -> OCRResult:
        image = _require_preprocessed(prep, variant)
        reader = self._get_reader()
        h, w = image.shape[:2]

        results = reader.readtext(image, detail=1, paragraph=False)

        lines: List[str] = []
        words: List[OCRWord] = []
        conf_scores: List[float] = []

        for bbox, text, conf in results:
            clean_text = str(text).strip()
            if not clean_text:
                continue

            lines.append(clean_text)
            try:
                score = max(0.0, min(1.0, float(conf)))
                conf_scores.append(score)
            except (ValueError, TypeError):
                score = 0.5

            pts = np.array(bbox)
            norm_bbox = [
                float(np.min(pts[:, 0])) / w,
                float(np.min(pts[:, 1])) / h,
                float(np.max(pts[:, 0])) / w,
                float(np.max(pts[:, 1])) / h,
            ]
            words.append(OCRWord(text=clean_text, confidence=score, bbox=norm_bbox))

        full_text = "\n".join(lines)
        mean_conf = round(sum(conf_scores) / len(conf_scores), 4) if conf_scores else 0.0

        return OCRResult(
            text=unicodedata.normalize("NFKC", full_text),
            mean_confidence=mean_conf,
            words=words,
            engine="easyocr",
            variant=variant,
            page=prep.page,
        )

    def recognize_numeric_crop(
        self,
        prep: PreprocessResult,
        bbox: List[float],
        allowlist: str = "0123456789٠١٢٣٤٥٦٧٨٩/-.,٬ ",
        skip_detector: bool = True,
    ) -> Tuple[str, float]:
        """Fast cropped numeric recognition (~60-150ms on CPU) supporting both Western & Arabic-Indic digits.

        Uses `reader.recognize()` directly when `skip_detector=True` so CRAFT does not miss
        dot-like Arabic-Indic zeros (٠) in numbers like ١٬٠٠٠٬٠٠٠.
        """
        _require_preprocessed(prep, "A")
        reader = self._get_reader()
        crop = crop_field_image(prep, bbox, pad_x=0.020, pad_y=0.010, upscale=2.0)

        if skip_detector:
            res = reader.recognize(crop, allowlist=allowlist, detail=1)
        else:
            res = reader.readtext(crop, allowlist=allowlist, detail=1, paragraph=False)

        if not res:
            return "", 0.0

        texts: List[str] = []
        confs: List[float] = []
        for item in res:
            if len(item) >= 3:
                t = str(item[1]).strip()
                if t:
                    texts.append(t)
                    try:
                        confs.append(max(0.0, min(1.0, float(item[2]))))
                    except (ValueError, TypeError):
                        pass

        out_text = " ".join(texts).strip()
        mean_c = round(sum(confs) / len(confs), 4) if confs else 0.0
        return unicodedata.normalize("NFKC", out_text), mean_c


class PaddleOCREngine:
    """PaddleOCR (PP-OCRv5) engine (Apache-2.0)."""

    def __init__(self, lang: str = "ar"):
        self.lang = lang
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            try:
                import torch  # Prevents Windows DLL loading order issue with albumentations/shm.dll
            except ImportError:
                pass
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                lang=self.lang,
                ocr_version="PP-OCRv5",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._ocr

    def run(self, prep: PreprocessResult, variant: str = "A") -> OCRResult:
        image = _require_preprocessed(prep, variant)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]
        ocr = self._get_ocr()
        prediction = ocr.predict(input=image)

        extracted_text: List[str] = []
        words: List[OCRWord] = []
        conf_scores: List[float] = []

        for page in prediction:
            payload = page.json
            if isinstance(payload, str):
                payload = json.loads(payload)

            data = payload.get("res", payload)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            polys = data.get("dt_polys", [])

            for idx, text in enumerate(texts):
                clean_t = str(text).strip()
                if not clean_t:
                    continue

                extracted_text.append(clean_t)
                score = 0.5
                if idx < len(scores):
                    try:
                        s = float(scores[idx])
                        if 0.0 <= s <= 1.0:
                            score = s
                            conf_scores.append(score)
                    except (ValueError, TypeError):
                        pass

                norm_bbox = [0.0, 0.0, 1.0, 1.0]
                if idx < len(polys) and polys[idx] is not None and len(polys[idx]) > 0:
                    pts = np.array(polys[idx])
                    norm_bbox = [
                        float(np.min(pts[:, 0])) / w,
                        float(np.min(pts[:, 1])) / h,
                        float(np.max(pts[:, 0])) / w,
                        float(np.max(pts[:, 1])) / h,
                    ]

                words.append(OCRWord(text=clean_t, confidence=score, bbox=norm_bbox))

        full_text = "\n".join(extracted_text)
        mean_conf = round(sum(conf_scores) / len(conf_scores), 4) if conf_scores else 0.0

        return OCRResult(
            text=unicodedata.normalize("NFKC", full_text),
            mean_confidence=mean_conf,
            words=words,
            engine="paddleocr",
            variant=variant,
            page=prep.page,
        )
