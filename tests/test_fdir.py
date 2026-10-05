"""
tests/test_fdir.py - Unit Tests for Autonomous FDIR Supervisor.
Verifies:
  1. Nominal measurement pass-through
  2. Gross outlier rejection via Chi-Square NIS gating
  3. Automatic stuck-pixel identification, masking, and isolation
  4. FDIR telemetry health reporting
"""

import numpy as np
import pytest

from telemetry.fdir_manager import FDIRSupervisor, FDIRStatus


def test_fdir_nominal_passthrough():
    fdir = FDIRSupervisor(nis_gate_threshold=9.21)
    filter_pos = (2.0, 1.0)
    innov_cov = np.eye(2) * 0.05

    det = {
        "pixel_x": 640.0,
        "pixel_y": 360.0,
        "inertial_pan_deg": 2.02,
        "inertial_tilt_deg": 0.99
    }

    valid, sanitized, reason = fdir.inspect_measurement(det, filter_pos, innov_cov, gimbal_moving=False)
    assert valid is True
    assert sanitized is not None
    assert fdir.current_status == FDIRStatus.NOMINAL


def test_fdir_outlier_rejection():
    fdir = FDIRSupervisor(nis_gate_threshold=9.21)
    filter_pos = (2.0, 1.0)
    innov_cov = np.eye(2) * 0.01  # small covariance: standard dev ~ 0.1 deg

    # Measurement 5 degrees away (gross outlier / decoy flare)
    decoy_det = {
        "pixel_x": 200.0,
        "pixel_y": 100.0,
        "inertial_pan_deg": 7.0,
        "inertial_tilt_deg": 5.0
    }

    valid, sanitized, reason = fdir.inspect_measurement(decoy_det, filter_pos, innov_cov, gimbal_moving=False)
    assert valid is False
    assert sanitized is None
    assert fdir.current_status == FDIRStatus.OUTLIER_REJECTED
    assert fdir.outliers_rejected_count == 1


def test_fdir_stuck_pixel_isolation():
    fdir = FDIRSupervisor(nis_gate_threshold=9.21)
    filter_pos = (0.0, 0.0)
    innov_cov = np.eye(2) * 1.0

    # Simulate 13 consecutive frames where pixel is frozen at exactly (420.0, 310.0)
    # while gimbal is reported moving
    stuck_det = {
        "pixel_x": 420.0,
        "pixel_y": 310.0,
        "inertial_pan_deg": 0.05,
        "inertial_tilt_deg": 0.05
    }

    for i in range(11):
        fdir.inspect_measurement(stuck_det, filter_pos, innov_cov, gimbal_moving=True)

    # Frame 12 should trigger the stuck pixel isolation
    valid, sanitized, reason = fdir.inspect_measurement(stuck_det, filter_pos, innov_cov, gimbal_moving=True)
    assert valid is False
    assert fdir.current_status == FDIRStatus.STUCK_PIXEL_ISOLATED
    assert len(fdir.bad_pixels) >= 1
    assert (420.0, 310.0) in fdir.bad_pixels

    # Subsequent frame at the same pixel should be immediately rejected via mask
    valid2, _, _ = fdir.inspect_measurement(stuck_det, filter_pos, innov_cov, gimbal_moving=False)
    assert valid2 is False


def test_fdir_telemetry_reporting():
    fdir = FDIRSupervisor()
    telem = fdir.get_telemetry()
    assert "status" in telem
    assert "current_nis" in telem
    assert "bad_pixel_count" in telem
    assert "outliers_rejected" in telem
