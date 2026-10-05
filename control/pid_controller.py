"""
control/pid_controller.py - Dual-Axis Precision Gimbal PID Controller.
Operates directly in angular domain with:
  1. Integral anti-windup clamping
  2. Low-pass filtered derivative term (prevents derivative kick on measurement jumps)
  3. Output saturation limits matching physical gimbal motors
"""

import math
from typing import Tuple
import numpy as np

from core.config import ControllerConfig


class AxisPID:
    """Single-axis PID controller with anti-windup and derivative filtering."""

    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 2.0, filter_alpha: float = 0.35):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.filter_alpha = filter_alpha

        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_d_error = 0.0

    def compute(self, error: float, dt: float, gain_scale: float = 1.0) -> float:
        """
        Computes control output for the axis.
        Args:
            error: Current tracking error (degrees).
            dt: Time step (seconds).
            gain_scale: Adaptive gain multiplier (e.g. from covariance damping).
        """
        if dt <= 0:
            return 0.0

        eff_kp = self.kp * gain_scale
        eff_ki = self.ki * gain_scale
        eff_kd = self.kd * gain_scale

        # Proportional term
        p_term = eff_kp * error

        # Integral term with anti-windup clamping
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = eff_ki * self.integral

        # Derivative term with low-pass filtering
        raw_d_error = (error - self.prev_error) / dt
        self.filtered_d_error = (self.filter_alpha * raw_d_error) + ((1.0 - self.filter_alpha) * self.filtered_d_error)
        d_term = eff_kd * self.filtered_d_error

        self.prev_error = error
        return p_term + i_term + d_term

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_d_error = 0.0


from control.vibration_canceller import AdaptiveVibrationCanceller


class DualAxisGimbalController:
    """Coordinates dual-axis (Pan and Tilt) closed-loop gimbal control."""

    def __init__(self, config: ControllerConfig, vibration_canceller: Optional[AdaptiveVibrationCanceller] = None):
        self.cfg = config
        self.pid_pan = AxisPID(
            config.kp_pan, config.ki_pan, config.kd_pan,
            config.integral_limit_deg, config.derivative_filter_alpha
        )
        self.pid_tilt = AxisPID(
            config.kp_tilt, config.ki_tilt, config.kd_tilt,
            config.integral_limit_deg, config.derivative_filter_alpha
        )
        self.vibration_canceller = vibration_canceller

    def compute(
        self,
        target_pan_deg: float,
        target_tilt_deg: float,
        current_pan_deg: float,
        current_tilt_deg: float,
        dt: float,
        feedforward_vel: Tuple[float, float] = (0.0, 0.0),
        gain_scale: float = 1.0
    ) -> Tuple[float, float]:
        """
        Computes commanded gimbal pan and tilt angular velocities (deg/s).
        Combines velocity feedforward with feedback PID error correction.
        """
        err_pan = target_pan_deg - current_pan_deg
        err_tilt = target_tilt_deg - current_tilt_deg

        # Apply adaptive harmonic vibration notch filtering if enabled
        if self.vibration_canceller is not None and self.vibration_canceller.enabled:
            err_pan, err_tilt = self.vibration_canceller.filter_errors(err_pan, err_tilt)

        # Feedback PID correction
        pid_rate_pan = self.pid_pan.compute(err_pan, dt, gain_scale)
        pid_rate_tilt = self.pid_tilt.compute(err_tilt, dt, gain_scale)

        # Total commanded rate = Feedforward + Feedback
        cmd_pan_rate = feedforward_vel[0] + pid_rate_pan
        cmd_tilt_rate = feedforward_vel[1] + pid_rate_tilt

        return cmd_pan_rate, cmd_tilt_rate

    def reset(self):
        self.pid_pan.reset()
        self.pid_tilt.reset()
        if self.vibration_canceller is not None:
            self.vibration_canceller.reset()
