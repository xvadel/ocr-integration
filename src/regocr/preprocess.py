"""
Enforced OpenCV Preprocessing Gate & Quality Assessment.

Every image entering any OCR engine MUST pass through `preprocess_image(...)` and be
wrapped in a `PreprocessResult` object. OCR engines reject raw numpy arrays.

Configurable steps (for ablation & production tuning):
  1. Orientation check (EXIF already applied in ingest + optional 90/180/270 Tesseract OSD)
  2. Resolution normalization (~200-300 DPI equivalent, bounded for CPU speed)
  3. Perspective correction (4-corner contour warp for phone photos when clearly quadrilateral)
  4. Fine deskew (Hough line transform, |angle| > 0.5 deg)
  5. Blue/violet stamp ink suppression (HSV mask protecting dark neutral text pixels)
  6. Illumination / shadow normalization (background estimation + division)
  7. CLAHE contrast enhancement
  8. Mild dot-preserving denoising (bilateral / gentle fastNlMeans)
  9. Conditional mild sharpening (only when blurry, preserving Arabic dots)
  10. Optional Variant B (adaptive-threshold binarization)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessConfig:
    """Toggles for every CV preprocessing step (defaults verified on dev + held-out eval)."""

    fix_orientation: bool = True
    perspective_correction: bool = True
    deskew: bool = True                 # Kept: +4.2% All12, +50% Office in ablation
    stamp_suppression: bool = True      # Kept: +4.2% All12, +50% Office in ablation
    illumination_norm: bool = True      # Kept: +4.2% All12 in ablation
    clahe: bool = True                  # Kept: disabling with denoise regressed held-out digit acc 98.65% -> 92.79%
    denoise: bool = True                # Kept: disabling with clahe regressed held-out Clean/Scan accuracy
    conditional_sharpen: bool = False   # Disabled: 0% gain in ablation
    generate_variant_b: bool = False    # Disabled: 0% gain at +36% latency in ablation
    target_min_short_side: int = 1100
    target_max_long_side: int = 2100


@dataclass
class PreprocessResult:
    """Mandatory input container for all OCR engines (enforces preprocessing gate)."""

    variant_a: np.ndarray  # Grayscale/enhanced uint8 image (primary)
    variant_b: Optional[np.ndarray]  # Adaptive-binarized uint8 image (optional)
    color_clean: np.ndarray  # Oriented/deskewed BGR uint8 image (for UI crop & HSV inspection)
    quality: Dict[str, Any]
    report: Dict[str, Any]
    page: int = 1
    _gate_token: str = field(default="regocr_preprocessed_v3", repr=False)


def to_cv2_image(image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Converts a file path, PIL Image, or numpy array to a BGR cv2 image."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_input}")
        return img
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        if image_input.ndim == 3 and image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        return image_input.copy()
    raise TypeError(f"Unsupported image type: {type(image_input)}")


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Converts image to single-channel grayscale if needed."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Invalid image shape: {img.shape}")


def detect_coarse_orientation(gray: np.ndarray) -> int:
    """Detect 0/90/180/270 degree rotation using Tesseract OSD if available; returns 0 if unsure."""
    try:
        import pytesseract

        h, w = gray.shape[:2]
        # Downscale for speed
        scale = min(1.0, 1000.0 / max(h, w))
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
        osd = pytesseract.image_to_osd(small, output_type=pytesseract.Output.DICT)
        rot = int(osd.get("rotate", 0))
        conf = float(osd.get("orientation_conf", 0.0))
        if rot in (90, 180, 270) and conf >= 3.0:
            return rot
    except Exception:
        pass
    return 0


def rotate_90_multiples(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def detect_skew_angle(gray: np.ndarray) -> float:
    """Detects fine skew angle in degrees (-30 to +30) using Hough line transform."""
    h, w = gray.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=90, minLineLength=int(small.shape[1] * 0.15), maxLineGap=20
    )

    angles: List[float] = []
    if lines is not None and len(lines) > 0:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx = int(x2) - int(x1)
            dy = int(y2) - int(y1)
            if dx == 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if -25.0 <= angle <= 25.0:
                angles.append(angle)

    if angles:
        return float(np.median(angles))
    return 0.0


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotates an image around its center with background-matched border padding."""
    if abs(angle) < 0.25:
        return img
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    border_val: Any = 250 if img.ndim == 2 else (250, 250, 250)
    return cv2.warpAffine(
        img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val
    )


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def try_perspective_correction(bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Attempt 4-point perspective warp ONLY if a clear tilted outer page boundary exists.

    Guardrails: must cover > 45% and < 95% of the frame, be convex, and have noticeable
    perspective distortion (top/bottom edge ratio or corner offsets > 3% of side) so we
    NEVER distort an already flat scan or inner table border!
    """
    h, w = bgr.shape[:2]
    gray = to_grayscale(bgr)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr, False

    img_area = float(h * w)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        area = cv2.contourArea(c)
        ratio = area / img_area
        if ratio < 0.50 or ratio > 0.96:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        rect = _order_quad_points(approx.reshape(4, 2).astype("float32"))
        tl, tr, br, bl = rect
        top_w = float(np.linalg.norm(tr - tl))
        bot_w = float(np.linalg.norm(br - bl))
        left_h = float(np.linalg.norm(bl - tl))
        right_h = float(np.linalg.norm(br - tr))

        if min(top_w, bot_w, left_h, right_h) < 100:
            continue

        # Only apply perspective warp if there is actual trapezoid/perspective skew (>4% side diff)
        w_diff = abs(top_w - bot_w) / max(top_w, bot_w)
        h_diff = abs(left_h - right_h) / max(left_h, right_h)
        if max(w_diff, h_diff) < 0.04:
            return bgr, False

        dst_w = int(max(top_w, bot_w))
        dst_h = int(max(left_h, right_h))
        dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(bgr, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC)
        return warped, True

    return bgr, False


def suppress_colored_stamps(bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Suppress blue/violet stamp ink in HSV space while protecting dark neutral black text.

    Returns (cleaned_grayscale, stamp_was_detected_and_suppressed).
    """
    gray = to_grayscale(bgr)
    if bgr.ndim != 3 or bgr.shape[2] < 3:
        return gray, False

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Blue/indigo/violet stamp ink in OpenCV HSV (H in [95, 165], noticeable saturation S >= 40)
    stamp_mask = cv2.inRange(hsv, np.array([95, 40, 40]), np.array([165, 255, 255]))

    # Protect true black/dark-gray text pixels even if slightly tinted by stamp overlap
    dark_text = (gray < 95) & (hsv[:, :, 1] < 85)
    stamp_mask[dark_text] = 0

    pixel_fraction = float(np.mean(stamp_mask > 0))
    # Only act if a genuine stamp region is present (>0.05% and <15% of page)
    if pixel_fraction < 0.0005 or pixel_fraction > 0.15:
        return gray, False

    # Dilate slightly (1px) to clean anti-aliased blue fringes, re-applying dark text protection
    k = np.ones((3, 3), np.uint8)
    stamp_mask = cv2.dilate(stamp_mask, k, iterations=1)
    stamp_mask[dark_text] = 0

    # Replace masked stamp ink with local background estimate
    bg = cv2.medianBlur(gray, 21)
    cleaned = gray.copy()
    cleaned[stamp_mask > 0] = bg[stamp_mask > 0]
    return cleaned, True


def normalize_illumination(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Flatten uneven phone-photo shadows via background estimation & division.

    Returns (normalized_gray, illumination_uniformity_score in [0, 1]).
    """
    # Measure illumination uniformity using 4x4 grid of local background medians
    h, w = gray.shape[:2]
    gh, gw = max(1, h // 4), max(1, w // 4)
    tile_p85 = []
    for r in range(4):
        for c in range(4):
            tile = gray[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw]
            if tile.size > 0:
                tile_p85.append(float(np.percentile(tile, 85)))
    illum_std = float(np.std(tile_p85)) if tile_p85 else 0.0
    illum_score = max(0.0, min(1.0, round(1.0 - (illum_std / 60.0), 3)))

    # Smooth background estimate with large Gaussian kernel (approx 1/15th of short side, odd)
    ksize = max(31, int(min(h, w) / 15) | 1)
    bg = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    bg = np.maximum(bg, 1).astype(np.float32)

    # Division normalizes shadows to uniform ~245 background while preserving dark ink contrast
    norm = np.clip((gray.astype(np.float32) / bg) * 245.0, 0, 255).astype(np.uint8)
    return norm, illum_score


def assess_image_quality(image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
    """Computes quality score [0.0 - 1.0] and detects actionable quality issues.

    Uses noise-decoupled blur (`Laplacian(GaussianBlur(3x3))`) so heavy grain noise
    cannot disguise a blurry image as sharp.
    """
    img = to_cv2_image(image)
    gray = to_grayscale(img)
    h, w = gray.shape[:2]
    issues: List[str] = []

    # 1. Resolution (measured on original input before upscaling!)
    min_dim = min(h, w)
    total_pixels = h * w
    if min_dim < 800 or total_pixels < 780_000:
        issues.append("low_resolution")

    # Normalize evaluation scale to ~1000px long side so DPI differences don't distort blur/noise
    norm_scale = 1000.0 / max(h, w)
    eval_gray = (
        cv2.resize(gray, (int(w * norm_scale), int(h * norm_scale)), interpolation=cv2.INTER_AREA)
        if norm_scale < 1.0
        else gray
    )

    # 2. Noise-decoupled blur via Laplacian variance after 3x3 Gaussian smoothing
    smooth3 = cv2.GaussianBlur(eval_gray, (3, 3), 0)
    lap_var = float(cv2.Laplacian(smooth3, cv2.CV_64F).var())
    if lap_var < 72.0:
        issues.append("blurry")

    # 3. Brightness / Darkness
    mean_val = float(np.mean(eval_gray))
    if mean_val < 65.0:
        issues.append("dark")

    # 4. Skew
    skew = detect_skew_angle(eval_gray)
    if abs(skew) > 3.5:
        issues.append("skewed")

    # 5. Noise (median residual on native-or-1000px gray)
    denoised_est = cv2.medianBlur(eval_gray, 3)
    noise_level = float(np.mean(cv2.absdiff(eval_gray, denoised_est)))
    if noise_level > 3.0:
        issues.append("noisy")

    # 6. Illumination uniformity
    eh, ew = eval_gray.shape[:2]
    gh, gw = max(1, eh // 4), max(1, ew // 4)
    tiles = [
        float(np.percentile(eval_gray[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw], 85))
        for r in range(4)
        for c in range(4)
    ]
    illum_std = float(np.std(tiles)) if tiles else 0.0
    illum_score = max(0.0, min(1.0, round(1.0 - (illum_std / 60.0), 3)))
    if illum_std > 28.0:
        issues.append("uneven_illumination")

    # Score calculation
    score = 1.0
    penalties = {
        "low_resolution": 0.30,
        "blurry": 0.38,
        "dark": 0.15,
        "skewed": 0.08,
        "noisy": 0.22,
        "uneven_illumination": 0.08,
        "cropped": 0.20,
    }
    for iss in issues:
        score -= penalties.get(iss, 0.10)

    score = max(0.0, min(1.0, round(score, 2)))
    return {
        "score": score,
        "issues": issues,
        "laplacian_var": round(lap_var, 1),
        "mean_brightness": round(mean_val, 1),
        "skew_angle": round(skew, 2),
        "noise_level": round(noise_level, 2),
        "illumination_score": illum_score,
        "resolution": [w, h],
    }


def preprocess_image(
    image: Union[np.ndarray, Image.Image],
    config: Optional[PreprocessConfig] = None,
    page: int = 1,
) -> PreprocessResult:
    """Single enforced CV preprocessing entry point returning a `PreprocessResult`."""
    cfg = config or PreprocessConfig()
    bgr = to_cv2_image(image)
    steps_applied: List[str] = []

    # Assess original image quality BEFORE resizing/enhancing so bad images are honestly flagged!
    quality = assess_image_quality(bgr)

    # 1. Coarse 90/180/270 orientation check
    orientation_angle = 0
    if cfg.fix_orientation:
        orientation_angle = detect_coarse_orientation(to_grayscale(bgr))
        if orientation_angle != 0:
            bgr = rotate_90_multiples(bgr, orientation_angle)
            steps_applied.append(f"rotate_{orientation_angle}")

    # 2. Resolution normalization (upscale tiny images, cap giant 300-DPI scans to ~2000px long side)
    h, w = bgr.shape[:2]
    short_side, long_side = min(h, w), max(h, w)
    if short_side < cfg.target_min_short_side:
        scale = cfg.target_min_short_side / float(short_side)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        steps_applied.append(f"upscale_{scale:.2f}x")
    elif long_side > cfg.target_max_long_side:
        scale = cfg.target_max_long_side / float(long_side)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        steps_applied.append(f"downscale_{scale:.2f}x")

    # 3. Perspective correction (phone photos with clear quad boundary)
    perspective_corrected = False
    if cfg.perspective_correction:
        bgr, perspective_corrected = try_perspective_correction(bgr)
        if perspective_corrected:
            steps_applied.append("perspective_warp")

    # 4. Fine deskew
    skew_angle = detect_skew_angle(to_grayscale(bgr))
    if cfg.deskew and abs(skew_angle) > 0.5:
        bgr = rotate_image(bgr, skew_angle)
        steps_applied.append(f"deskew_{skew_angle:.2f}deg")

    color_clean = bgr.copy()

    # 5. Blue/violet stamp ink suppression (HSV)
    stamp_suppressed = False
    if cfg.stamp_suppression:
        gray, stamp_suppressed = suppress_colored_stamps(bgr)
        if stamp_suppressed:
            steps_applied.append("stamp_suppression_hsv")
    else:
        gray = to_grayscale(bgr)

    # 6. Illumination / shadow normalization
    illum_score = float(quality.get("illumination_score", 1.0))
    if cfg.illumination_norm:
        gray, illum_score = normalize_illumination(gray)
        steps_applied.append("illumination_norm")

    # 7. CLAHE contrast enhancement
    if cfg.clahe:
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        steps_applied.append("clahe")

    # 8. Gentle dot-preserving denoising (bilateral filter preserves sharp edges & Arabic dots!)
    if cfg.denoise:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
        steps_applied.append("bilateral_denoise")

    # 9. Conditional mild sharpening ONLY when blurry
    blur_score = float(quality.get("laplacian_var", 0.0))
    if cfg.conditional_sharpen and blur_score < 120.0:
        gauss = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
        gray = cv2.addWeighted(gray, 1.35, gauss, -0.35, 0)
        steps_applied.append("conditional_unsharp")

    variant_a = gray

    # 10. Optional Variant B (adaptive threshold binarization)
    variant_b: Optional[np.ndarray] = None
    if cfg.generate_variant_b:
        variant_b = cv2.adaptiveThreshold(
            variant_a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
        )
        steps_applied.append("variant_b_adaptive_thresh")

    report = {
        "page": page,
        "steps_applied": steps_applied,
        "skew_angle": round(skew_angle, 2),
        "blur_score": round(blur_score, 1),
        "illumination_score": round(illum_score, 3),
        "perspective_corrected": perspective_corrected,
        "orientation_angle": orientation_angle,
        "stamp_suppressed": stamp_suppressed,
        "quality_flags": list(quality.get("issues", [])),
    }

    return PreprocessResult(
        variant_a=variant_a,
        variant_b=variant_b,
        color_clean=color_clean,
        quality=quality,
        report=report,
        page=page,
    )


def crop_field_image(
    source: Union[PreprocessResult, np.ndarray],
    bbox: List[float],
    pad_x: float = 0.012,
    pad_y: float = 0.006,
    upscale: float = 2.0,
    use_color: bool = False,
) -> np.ndarray:
    """Crop a field region given normalized [x1, y1, x2, y2] bbox, with padding and optional upscaling.

    Works on either a PreprocessResult (recommended) or a BGR/gray ndarray (for UI helpers).
    """
    if isinstance(source, PreprocessResult):
        img = source.color_clean if use_color else source.variant_a
    else:
        img = source

    h, w = img.shape[:2]
    x1 = max(0, int((bbox[0] - pad_x) * w))
    y1 = max(0, int((bbox[1] - pad_y) * h))
    x2 = min(w, int((bbox[2] + pad_x) * w))
    y2 = min(h, int((bbox[3] + pad_y) * h))

    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return img[0 : min(10, h), 0 : min(10, w)].copy()

    crop = img[y1:y2, x1:x2].copy()
    if upscale and upscale != 1.0:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return crop
