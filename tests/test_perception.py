"""
tests/test_perception.py - Unit tests for detection, neural discrimination, and gating.
"""

import math
import numpy as np
import pytest

from core.config import DetectorConfig
from perception.classical_detector import ClassicalBeaconDetector
from perception.discriminator import NeuralBeaconDiscriminator
from perception.gating import MahalanobisGater


def test_classical_detector_subpixel_accuracy():
    cfg = DetectorConfig()
    detector = ClassicalBeaconDetector(cfg)
    
    # Synthetic frame with a single bright Gaussian spot at (450.4, 280.6)
    frame = np.full((720, 1280), 12, dtype=np.uint8)
    true_x = 450.4
    true_y = 280.6
    sigma = 3.5
    
    y_idx, x_idx = np.ogrid[0:720, 0:1280]
    dist_sq = (x_idx - true_x)**2 + (y_idx - true_y)**2
    psf = np.exp(-dist_sq / (2.0 * (sigma**2)))
    frame = np.clip(frame.astype(np.float32) + psf * 240.0, 0, 255).astype(np.uint8)

    candidates = detector.detect_candidates(frame)
    assert len(candidates) >= 1
    best = candidates[0]
    
    # Sub-pixel accuracy should be within 0.3 pixels of ground truth
    assert abs(best["cx"] - true_x) < 0.35
    assert abs(best["cy"] - true_y) < 0.35
    assert best["flux"] > 10000.0


def test_neural_discriminator():
    disc = NeuralBeaconDiscriminator("models/beacon_discriminator.pt")
    
    # Typical authentic beacon feature vector (high circularity, high peak, high flux)
    beacon_feat = np.array([220.0, 52.0, 0.92, 1.02, 245.0, 140.0, 60.0, 24000.0, 0.96, 0.92, 18.0], dtype=np.float32)
    conf_beacon = disc.predict_confidence(beacon_feat)
    assert conf_beacon > 0.65

    # Stray noise / cosmic ray artifact feature vector (tiny area, low flux)
    noise_feat = np.array([4.0, 8.0, 0.45, 0.8, 140.0, 60.0, 20.0, 800.0, 0.55, 0.45, 4.0], dtype=np.float32)
    conf_noise = disc.predict_confidence(noise_feat)
    assert conf_noise < conf_beacon


def test_mahalanobis_gating():
    gater = MahalanobisGater(gate_threshold=9.21)
    
    pred = np.array([2.0, -1.0])
    cov = np.eye(2) * 0.1  # std dev = ~0.316
    
    # Point very close to prediction (residual = 0.05, 0.05)
    meas_close = np.array([2.05, -0.95])
    is_valid, d_sq = gater.is_within_gate(meas_close, pred, cov)
    assert is_valid
    assert d_sq < 2.0

    # Point far from prediction (residual = 2.0, 2.0 -> outside gate)
    meas_far = np.array([4.0, 1.0])
    is_valid_far, d_sq_far = gater.is_within_gate(meas_far, pred, cov)
    assert not is_valid_far
    assert d_sq_far > 9.21
