"""
tests/test_closed_loop_control.py - Unit tests for PID controller, damping, and gimbal dynamics.
"""

import numpy as np
import pytest

from core.config import ControllerConfig, GimbalConfig
from control.pid_controller import DualAxisGimbalController, AxisPID
from control.confidence_aware import CovarianceGainDamper
from simulation.camera import VirtualCameraGimbal


def test_pid_anti_windup():
    pid = AxisPID(kp=5.0, ki=2.0, kd=0.5, integral_limit=2.0)
    
    # Constant massive error over 100 steps
    for _ in range(100):
        pid.compute(error=10.0, dt=0.016)

    # Integral must be clamped to integral_limit (2.0) rather than winding up to 32.0
    assert abs(pid.integral) <= 2.0


def test_covariance_gain_damping():
    damper = CovarianceGainDamper(alpha=0.01, min_gain_scale=0.25, enabled=True)
    
    # Low uncertainty covariance
    cov_low = np.diag([0.05, 0.05])
    scale_high = damper.compute_gain_scale(cov_low)
    assert scale_high > 0.95

    # High uncertainty covariance (e.g. severe turbulence or measurement loss)
    cov_high = np.diag([50.0, 50.0])
    scale_damped = damper.compute_gain_scale(cov_high)
    assert scale_damped < 0.60
    assert scale_damped >= 0.25


def test_gimbal_slew_rate_saturation():
    cfg = GimbalConfig(max_slew_rate_pan_deg_s=25.0, max_accel_deg_s2=50.0)
    gimbal = VirtualCameraGimbal(cfg)
    
    # Request massive slew rate (100 deg/s)
    gimbal.set_rate_command(pan_rate_deg_s=100.0, tilt_rate_deg_s=-80.0)
    
    # Slew rate command must be saturated to 25 deg/s
    assert abs(gimbal.target_pan_rate_deg_s) <= 25.0
    assert abs(gimbal.target_tilt_rate_deg_s) <= 20.0
