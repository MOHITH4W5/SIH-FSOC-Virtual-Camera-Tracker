"""
perception/hybrid_detector.py - Integrated Hybrid Perception Pipeline.
Unifies classical candidate proposal with neural discriminator verification.
Ensures zero-latency fallback to pure classical detection if AI is disabled.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from core.config import DetectorConfig
from perception.classical_detector import ClassicalBeaconDetector
from perception.discriminator import NeuralBeaconDiscriminator


class HybridBeaconDetector:
    """Orchestrates candidate detection, AI feature discrimination, and confidence evaluation."""

    def __init__(self, config: DetectorConfig, model_path: Optional[str] = None):
        self.cfg = config
        self.classical = ClassicalBeaconDetector(config)
        
        # Initialize discriminator if hybrid mode is active
        self.discriminator: Optional[NeuralBeaconDiscriminator] = None
        if config.mode in ["hybrid", "discriminator"]:
            try:
                self.discriminator = NeuralBeaconDiscriminator(model_path)
            except Exception as e:
                print(f"[HybridDetector] Warning: Could not initialize neural discriminator ({e}). Falling back to classical.")
                self.discriminator = None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes a sensor frame and returns verified candidate detections.
        Each candidate dict contains:
            cx, cy, bbox, area, circularity, classical_conf, ai_conf, fused_confidence
        """
        # Step 1: Classical candidate extraction
        raw_candidates = self.classical.detect_candidates(frame)
        if not raw_candidates:
            return []

        verified_candidates = []
        for cand in raw_candidates:
            classical_conf = cand["confidence"]
            ai_conf = 1.0

            if self.discriminator is not None and self.cfg.mode == "hybrid":
                # Neural verification
                ai_conf = self.discriminator.predict_confidence(cand["feature_vector"])
                
                # Fused confidence score
                # 60% weight to neural discriminator, 40% weight to classical photometric score
                fused_conf = 0.60 * ai_conf + 0.40 * classical_conf
            else:
                fused_conf = classical_conf

            # Filter out candidates failing the discriminator confidence threshold
            if fused_conf >= self.cfg.discriminator_confidence_threshold:
                cand["ai_conf"] = ai_conf
                cand["fused_confidence"] = fused_conf
                verified_candidates.append(cand)

        # Sort by fused confidence descending
        verified_candidates.sort(key=lambda c: c["fused_confidence"], reverse=True)
        return verified_candidates
