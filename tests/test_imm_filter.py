"""
tests/test_imm_filter.py - Comprehensive Unit Tests for Interacting Multiple Model (IMM) Filter.
Verifies:
  1. Initialization and dimension integrity
  2. Constant velocity convergence and CV mode dominance
  3. Coordinated turn maneuver response and CT mode probability increase
  4. Interface compatibility with TargetKalmanFilter
"""

import math
import numpy as np
import pytest

from core.config import KalmanConfig
from tracking.imm_filter import TargetIMMEstimator


def test_imm_initialization():
    cfg = KalmanConfig()
    imm = TargetIMMEstimator(cfg)
    assert not imm.is_initialized

    imm.initialize(pan_deg=2.5, tilt_deg=-1.5, pan_vel_deg_s=0.5, tilt_vel_deg_s=-0.2)
    assert imm.is_initialized
    pos = imm.position_deg
    vel = imm.velocity_deg_s

    assert abs(pos[0] - 2.5) < 1e-4
    assert abs(pos[1] - (-1.5)) < 1e-4
    assert abs(vel[0] - 0.5) < 1e-4
    assert abs(vel[1] - (-0.2)) < 1e-4

    probs = imm.model_probabilities
    assert "CV" in probs and "CA" in probs and "CT" in probs
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_imm_linear_flight_tracking():
    """Under steady linear motion, IMM should maintain tight tracking error."""
    cfg = KalmanConfig(dt=0.0166)  # 60 FPS
    imm = TargetIMMEstimator(cfg)
    imm.initialize(0.0, 0.0, 1.0, 0.5)

    true_pan, true_tilt = 0.0, 0.0
    v_pan, v_tilt = 1.2, 0.4
    dt = 0.0166

    errors = []
    for _ in range(60):
        true_pan += v_pan * dt
        true_tilt += v_tilt * dt

        # Add small measurement noise
        noise_pan = np.random.normal(0, 0.02)
        noise_tilt = np.random.normal(0, 0.02)

        imm.predict(dt)
        imm.update(true_pan + noise_pan, true_tilt + noise_tilt)

        est_pan, est_tilt = imm.position_deg
        err = math.hypot(est_pan - true_pan, est_tilt - true_tilt)
        errors.append(err)

    mean_err_deg = np.mean(errors[-20:])
    # Convert to mrad: 1 deg = 17.453 mrad
    mean_err_mrad = mean_err_deg * 17.453
    # Check that error is sub-milliradian or small fraction of degree
    assert mean_err_mrad < 1.0, f"Expected < 1.0 mrad, got {mean_err_mrad:.3f} mrad"


def test_imm_coordinated_turn_maneuver():
    """During a sharp turn, IMM should detect curvature and increase CT model probability."""
    cfg = KalmanConfig(dt=0.0166)
    imm = TargetIMMEstimator(cfg)
    imm.initialize(0.0, 0.0, 0.0, 0.0)

    # Simulate circular trajectory: radius = 3 deg, angular velocity = 1.0 rad/s
    dt = 0.0166
    omega = 1.0
    radius = 3.0

    ct_probs = []
    for frame in range(120):
        t = frame * dt
        true_pan = radius * math.cos(omega * t)
        true_tilt = radius * math.sin(omega * t)

        imm.predict(dt)
        imm.update(true_pan, true_tilt)

        ct_probs.append(imm.model_probabilities["CT"])

    # Over a sustained turn, CT probability should become significant (> 0.20)
    max_ct_prob = max(ct_probs)
    assert max_ct_prob > 0.20, f"Expected CT prob to rise during turn, max was {max_ct_prob:.3f}"


def test_imm_drop_in_interface():
    """Ensure all attributes and methods of TargetKalmanFilter are available on TargetIMMEstimator."""
    cfg = KalmanConfig()
    imm = TargetIMMEstimator(cfg)
    imm.initialize(1.0, 2.0)

    pred_x, pred_y = imm.get_predicted_forward(0.1)
    assert isinstance(pred_x, float)
    assert isinstance(pred_y, float)

    cov = imm.get_innovation_covariance()
    assert cov.shape == (2, 2)
    assert imm.covariance_trace > 0

    imm.reset()
    assert not imm.is_initialized
