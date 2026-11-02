#!/usr/bin/env python3
"""
preprocess.py - OpenCV Image Preprocessing Layer & Document Quality Gate.

Provides:
  - assess_image_quality: Computes document quality score and identifies issues
    ('blurry', 'low_resolution', 'skewed', 'dark', 'cropped', 'noisy').
  - preprocess_image: Deskewing, shadow/illumination correction, CLAHE contrast
    enhancement, and denoising for optimal OCR recognition.
"""

import math
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def to_cv2_image(image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Converts a file path, PIL Image, or numpy array to a BGR or Gray cv2 image."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_input}")
        return img
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Converts image to single-channel grayscale if needed."""
    if len(img.shape) == 2:
        return img
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Invalid image shape: {img.shape}")


def detect_skew_angle(gray: np.ndarray) -> float:
    """
    Detects skew angle in degrees using Hough line transform and bounding boxes.
    Returns angle in degrees (-45.0 to 45.0).
    """
    h, w = gray.shape[:2]
    # Downscale for speed if very large
    max_dim = 1200
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    # Edge detection
    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=int(small.shape[1] * 0.15), maxLineGap=20)

    angles = []
    if lines is not None and len(lines) > 0:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx = int(x2) - int(x1)
            dy = int(y2) - int(y1)
            if dx == 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            # Focus on near-horizontal lines (-30 to +30 degrees)
            if -30.0 <= angle <= 30.0:
                angles.append(angle)

    if angles:
        # Use median for robustness against outliers
        median_angle = float(np.median(angles))
        return median_angle

    # Fallback to contour minAreaRect
    thresh = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    pts = cv2.findNonZero(thresh)
    if pts is not None and len(pts) > 50:
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        if -30.0 <= angle <= 30.0:
            return float(angle)

    return 0.0


def rotate_image(img: np.ndarray, angle: float, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Rotates image by given angle in degrees, expanding canvas so no content is clipped."""
    if abs(angle) < 0.2:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos_val = abs(rot_mat[0, 0])
    sin_val = abs(rot_mat[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    # Adjust transformation matrix for translation
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]

    fill = bg_color[0] if len(img.shape) == 2 else bg_color
    rotated = cv2.warpAffine(
        img, rot_mat, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill
    )
    return rotated


def assess_image_quality(image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
    """
    Evaluates image quality and checks for degradation issues.

    Returns:
      {
        "score": float in [0.0, 1.0],
        "issues": list of issues from:
          ['blurry', 'low_resolution', 'skewed', 'dark', 'cropped', 'noisy'],
        "metrics": dict of raw measurement values
      }
    """
    cv_img = to_cv2_image(image_input)
    gray = to_grayscale(cv_img)
    h, w = gray.shape[:2]
    total_pixels = h * w

    issues: List[str] = []
    penalties = 0.0

    # 1. Resolution
    # Full page registration docs should be at least ~1000x700 (0.7 MP) for reliable Arabic OCR
    is_low_res = total_pixels < 900_000 or min(h, w) < 650
    if is_low_res:
        issues.append("low_resolution")
        res_ratio = max(0.2, total_pixels / 1_500_000.0)
        penalties += (1.0 - res_ratio) * 0.35

    # 2. Text Edge Sharpness & Blurriness
    # Uses Sobel edge gradient magnitude at 95th percentile, which directly measures text stroke sharpness
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobel_x, sobel_y)
    p95_edge = float(np.percentile(grad_mag, 95))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if p95_edge < 200.0 or (laplacian_var < 100.0 and p95_edge < 260.0):
        issues.append("blurry")
        blur_factor = max(0.0, (220.0 - p95_edge) / 220.0)
        penalties += 0.15 + blur_factor * 0.25

    # 3. Noise level
    blurred = cv2.medianBlur(gray, 3)
    noise_diff = cv2.absdiff(gray, blurred)
    noise_level = float(np.std(noise_diff))
    # Low contrast/noisy artifacts
    std_val = float(np.std(gray))
    if noise_level > 8.0 or (noise_level > 4.0 and std_val < 26.0):
        issues.append("noisy")
        penalties += 0.20

    # 4. Darkness / Illumination
    mean_val = float(np.mean(gray))
    if mean_val < 75.0 or (mean_val < 100.0 and std_val < 35.0):
        issues.append("dark")
        dark_factor = min(1.0, (85.0 - mean_val) / 60.0) if mean_val < 85.0 else 0.15
        penalties += dark_factor * 0.25

    # 5. Skew
    skew_angle = detect_skew_angle(gray)
    if abs(skew_angle) > 2.5:
        issues.append("skewed")
        skew_factor = min(1.0, (abs(skew_angle) - 2.5) / 10.0)
        penalties += 0.08 + skew_factor * 0.12

    # 6. Cropping check
    margin_y = max(2, int(h * 0.015))
    margin_x = max(2, int(w * 0.015))
    top_edge = gray[:margin_y, :]
    bot_edge = gray[-margin_y:, :]
    left_edge = gray[:, :margin_x]
    right_edge = gray[:, -margin_x:]
    edge_means = [np.mean(top_edge), np.mean(bot_edge), np.mean(left_edge), np.mean(right_edge)]
    min_edge = min(edge_means)
    if min_edge < 60.0 and (mean_val - min_edge) > 50.0:
        issues.append("cropped")
        penalties += 0.15

    raw_score = max(0.05, 1.0 - penalties)
    score = round(float(np.clip(raw_score, 0.0, 1.0)), 2)

    return {
        "score": score,
        "issues": sorted(list(set(issues))),
        "metrics": {
            "resolution": f"{w}x{h} ({total_pixels:,} px)",
            "edge_sharpness_p95": round(p95_edge, 2),
            "laplacian_var": round(laplacian_var, 2),
            "noise_level": round(noise_level, 2),
            "mean_brightness": round(mean_val, 2),
            "contrast_std": round(std_val, 2),
            "skew_angle": round(skew_angle, 2),
        }
    }


def remove_illumination_gradient(gray: np.ndarray) -> np.ndarray:
    """
    Normalizes uneven illumination/shadows (typical in phone scans)
    using morphological background division.
    """
    kernel_size = max(25, int(min(gray.shape) / 40))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Dilate to approximate background illumination
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    bg = cv2.medianBlur(bg, 21)
    # Divide original by background to flatten lighting
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm


def enhance_contrast_clahe(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def mild_sharpen(gray: np.ndarray) -> np.ndarray:
    """Applies mild unsharp mask to clarify character strokes."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    sharpened = cv2.addWeighted(gray, 1.4, blur, -0.4, 0)
    return sharpened


def denoise_image(gray: np.ndarray, noise_level: float) -> np.ndarray:
    """Applies edge-preserving bilateral or fastNlMeans denoising based on noise intensity."""
    if noise_level > 12.0:
        # Fast non-local means for heavy noise
        h_param = min(15, int(noise_level * 0.7))
        return cv2.fastNlMeansDenoising(gray, None, h=h_param, templateWindowSize=7, searchWindowSize=21)
    elif noise_level > 7.0:
        # Bilateral filter to preserve crisp text edges
        return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    return gray


def preprocess_image(
    image_input: Union[str, np.ndarray, Image.Image],
    deskew: bool = True,
    normalize_lighting: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
    sharpen: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Complete OpenCV preprocessing pipeline for OCR:
    1. Grayscale conversion
    2. Skew detection and correction
    3. Illumination flattening / shadow removal
    4. Contrast enhancement (CLAHE)
    5. Adaptive denoising
    6. Mild unsharp mask

    Returns:
      (preprocessed_grayscale_image, quality_assessment)
    """
    cv_img = to_cv2_image(image_input)
    quality = assess_image_quality(cv_img)
    gray = to_grayscale(cv_img)

    # 1. Deskew
    skew_angle = quality["metrics"]["skew_angle"]
    if deskew and abs(skew_angle) > 0.4:
        gray = rotate_image(gray, skew_angle, bg_color=(255, 255, 255))

    # 2. Lighting normalization if dark or noisy phone scan
    mean_val = quality["metrics"]["mean_brightness"]
    if normalize_lighting and ("dark" in quality["issues"] or mean_val < 180.0):
        gray = remove_illumination_gradient(gray)

    # 3. Denoising
    noise_level = quality["metrics"]["noise_level"]
    if denoise and noise_level > 7.0:
        gray = denoise_image(gray, noise_level)

    # 4. Contrast enhancement
    if enhance_contrast:
        gray = enhance_contrast_clahe(gray, clip_limit=2.0)

    # 5. Mild sharpening
    if sharpen and "blurry" not in quality["issues"]:
        gray = mild_sharpen(gray)

    return gray, quality


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path_or_dir>")
        sys.exit(1)

    target = Path(sys.argv[1])
    files = [target] if target.is_file() else sorted(target.glob("*.jpg")) + sorted(target.glob("*.png"))

    print(f"Assessing {len(files)} files...\n")
    for f in files:
        prep, q = preprocess_image(str(f))
        print(f"File: {f.name}")
        print(f"  Quality Score : {q['score']}")
        print(f"  Issues        : {q['issues'] or 'None (Clean)'}")
        print(f"  Metrics       : {json.dumps(q['metrics'])}")
        print("-" * 50)
