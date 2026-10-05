"""
simulation/camera.py - Virtual 2-Axis Pan/Tilt Camera Gimbal Model.
Simulates realistic servo motor actuation with slew-rate limits, angular acceleration limits,
mechanical boundary stops, and platform micro-vibration perturbation.
"""

import math
from typing import Tuple
import numpy as np

from core.config import GimbalConfig


class VirtualCameraGimbal:
    """Simulates a 2-axis mechanical gimbal with realistic rate and acceleration dynamics."""

    def __init__(self, config: GimbalConfig):
        self.cfg = config
        
        # Current true mechanical gimbal attitude (degrees)
        self.pan_deg = 0.0
        self.tilt_deg = 0.0
        
        # Current mechanical rates (deg/s)
        self.pan_rate_deg_s = 0.0
        self.tilt_rate_deg_s = 0.0

        # Commanded rates (deg/s)
        self.target_pan_rate_deg_s = 0.0
        self.target_tilt_rate_deg_s = 0.0

        # Disturbance offset (e.g. from platform micro-vibrations)
        self.jitter_pan_deg = 0.0
        self.jitter_tilt_deg = 0.0

    def set_rate_command(self, pan_rate_deg_s: float, tilt_rate_deg_s: float):
        """Sets requested pan and tilt rate commands from the controller."""
        # Clamp to maximum slew rates
        self.target_pan_rate_deg_s = np.clip(
            pan_rate_deg_s,
            -self.cfg.max_slew_rate_pan_deg_s,
            self.cfg.max_slew_rate_pan_deg_s
        )
        self.target_tilt_rate_deg_s = np.clip(
            tilt_rate_deg_s,
            -self.cfg.max_slew_rate_tilt_deg_s,
            self.cfg.max_slew_rate_tilt_deg_s
        )

    def set_vibration_jitter(self, jitter_pan_deg: float, jitter_tilt_deg: float):
        """Applies instantaneous platform micro-vibration perturbation to the camera mount."""
        self.jitter_pan_deg = jitter_pan_deg
        self.jitter_tilt_deg = jitter_tilt_deg

    def step(self, dt: float) -> Tuple[float, float]:
        """
        Integrates gimbal kinematics over dt seconds taking into account acceleration limits.
        Returns:
            (effective_pan_deg, effective_tilt_deg) including platform vibration jitter.
        """
        if dt <= 0:
            return self.effective_pose()

        # Rate integration with acceleration limits (deg/s^2)
        max_dv = self.cfg.max_accel_deg_s2 * dt

        # Pan rate adjustment
        dv_pan = self.target_pan_rate_deg_s - self.pan_rate_deg_s
        self.pan_rate_deg_s += np.clip(dv_pan, -max_dv, max_dv)

        # Tilt rate adjustment
        dv_tilt = self.target_tilt_rate_deg_s - self.tilt_rate_deg_s
        self.tilt_rate_deg_s += np.clip(dv_tilt, -max_dv, max_dv)

        # Position integration
        self.pan_deg += self.pan_rate_deg_s * dt
        self.tilt_deg += self.tilt_rate_deg_s * dt

        # Enforce mechanical hard stops
        if self.pan_deg < self.cfg.pan_min_deg:
            self.pan_deg = self.cfg.pan_min_deg
            self.pan_rate_deg_s = max(0.0, self.pan_rate_deg_s)
        elif self.pan_deg > self.cfg.pan_max_deg:
            self.pan_deg = self.cfg.pan_max_deg
            self.pan_rate_deg_s = min(0.0, self.pan_rate_deg_s)

        if self.tilt_deg < self.cfg.tilt_min_deg:
            self.tilt_deg = self.cfg.tilt_min_deg
            self.tilt_rate_deg_s = max(0.0, self.tilt_rate_deg_s)
        elif self.tilt_deg > self.cfg.tilt_max_deg:
            self.tilt_deg = self.cfg.tilt_max_deg
            self.tilt_rate_deg_s = min(0.0, self.tilt_rate_deg_s)

        return self.effective_pose()

    def effective_pose(self) -> Tuple[float, float]:
        """Returns the optical line-of-sight pointing angle including micro-vibration."""
        return (self.pan_deg + self.jitter_pan_deg, self.tilt_deg + self.jitter_tilt_deg)

    def reset(self, pan_deg: float = 0.0, tilt_deg: float = 0.0):
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
        self.pan_rate_deg_s = 0.0
        self.tilt_rate_deg_s = 0.0
        self.target_pan_rate_deg_s = 0.0
        self.target_tilt_rate_deg_s = 0.0
        self.jitter_pan_deg = 0.0
        self.jitter_tilt_deg = 0.0
