"""
Privacy-preserving QR detector for Saudi documents (ZATCA / Ministry of Commerce).
Never logs raw QR payload or personal identifiers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import cv2
import numpy as np


class SaudiQRDetector:
    """Detects QR codes and extracts metadata without logging raw payload."""

    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def detect_and_decode(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect QR code. Returns boolean presence and sanitized length/bounds only."""
        if image is None:
            return None
        try:
            val, pts, _ = self.detector.detectAndDecode(image)
            if val and len(val.strip()) > 0:
                return {
                    "detected": True,
                    "payload_length": len(val),
                    "has_zatca_tlv": val.startswith("\x01") or len(val) > 40,
                }
        except Exception:
            pass
        return None
