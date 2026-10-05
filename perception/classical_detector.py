"""
perception/classical_detector.py - Classical Centroid Detection & Feature Extraction.
Implements the baseline FSO beacon detection algorithm:
  - Adaptive/fixed intensity thresholding
  - Morphological filtering for speckle noise suppression
  - Connected component analysis & contour filtering
  - Sub-pixel intensity-weighted moment centroiding:
      cx = sum(I * x) / sum(I), cy = sum(I * y) / sum(I)
  - Geometric and photometric feature extraction for downstream discriminator
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from core.config import DetectorConfig


class ClassicalBeaconDetector:
    """Fast, deterministic classical computer vision detector for optical beacons."""

    def __init__(self, config: DetectorConfig):
        self.cfg = config
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect_candidates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extracts all potential beacon candidate blobs and their visual features.
        Returns:
            List of candidate dictionaries containing centroid, bounding box, features, and confidence.
        """
        # Thresholding
        if self.cfg.threshold_adaptive:
            # Noise-floor adaptive thresholding: background + 4.5 * noise_std
            bg_mean = float(np.mean(frame))
            bg_std = float(np.std(frame))
            adaptive_t = int(bg_mean + 4.5 * bg_std)
            thresh_val = max(35, min(self.cfg.threshold_min_val, adaptive_t))
        else:
            thresh_val = self.cfg.threshold_min_val

        _, binary_mask = cv2.threshold(frame, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Morphological opening to suppress isolated single-pixel shot noise
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self._morph_kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_blob_area_px or area > self.cfg.max_blob_area_px:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            # Circularity: 4 * pi * area / (perimeter^2) -> 1.0 for perfect circle
            circularity = 4.0 * math.pi * area / (perimeter ** 2)
            if circularity < self.cfg.min_circularity:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(1, h)

            # Sub-pixel centroid via intensity-weighted moments
            # Extract ROI from original grayscale frame
            x_min = max(0, x - 2)
            x_max = min(frame.shape[1], x + w + 2)
            y_min = max(0, y - 2)
            y_max = min(frame.shape[0], y + h + 2)
            
            roi = frame[y_min:y_max, x_min:x_max].astype(np.float32)
            bg_level = float(np.percentile(roi, 10))
            roi_signal = np.maximum(0.0, roi - bg_level)

            m = cv2.moments(roi_signal)
            if m["m00"] > 0:
                sub_cx = x_min + (m["m10"] / m["m00"])
                sub_cy = y_min + (m["m01"] / m["m00"])
            else:
                sub_cx = x + w / 2.0
                sub_cy = y + h / 2.0

            peak_val = float(np.max(roi))
            mean_val = float(np.mean(roi))
            std_val = float(np.std(roi))
            flux = float(np.sum(roi_signal))

            # Reject dim background clutter, stars, and weak decoy artifacts
            min_flux = getattr(self.cfg, "min_beacon_flux", 6000.0)
            if flux < min_flux:
                continue

            # Classical heuristic confidence score (circularity, intensity, flux, compactness)
            intensity_score = min(1.0, peak_val / 240.0)
            flux_score = min(1.0, flux / 12000.0)  # Optical beacon has high integrated energy
            shape_score = min(1.0, circularity)
            aspect_score = 1.0 - min(1.0, abs(1.0 - aspect_ratio))
            
            classical_conf = (
                0.25 * intensity_score +
                0.35 * flux_score +
                0.25 * shape_score +
                0.15 * aspect_score
            )

            feature_vector = np.array([
                area,
                perimeter,
                circularity,
                aspect_ratio,
                peak_val,
                mean_val,
                std_val,
                flux,
                intensity_score,
                shape_score,
                peak_val / max(1.0, bg_level)
            ], dtype=np.float32)

            candidates.append({
                "cx": sub_cx,
                "cy": sub_cy,
                "bbox": (x, y, w, h),
                "area": area,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "peak_val": peak_val,
                "flux": flux,
                "confidence": classical_conf,
                "feature_vector": feature_vector
            })

        # Sort candidates by confidence descending
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return candidates
