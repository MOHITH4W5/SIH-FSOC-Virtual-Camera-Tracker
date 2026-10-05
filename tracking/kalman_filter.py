"""
tracking/kalman_filter.py - Discrete Linear Kalman Filter for FSOC Target State Estimation.
Tracks target inertial position and angular velocity in world space:
  x = [pan_deg, tilt_deg, pan_vel_deg_s, tilt_vel_deg_s]^T
Provides continuous prediction, covariance propagation, innovation computation,
and forward-projection for predictive feedforward control.
"""

import math
from typing import Tuple, Optional
import numpy as np

from core.config import KalmanConfig


class TargetKalmanFilter:
    """Discrete Kalman Filter tracking 2D angular state and velocity in world inertial frame."""

    def __init__(self, config: KalmanConfig):
        self.cfg = config
        self.dt = config.dt
        
        # State vector: [pan_deg, tilt_deg, pan_vel_deg_s, tilt_vel_deg_s]^T
        self.x = np.zeros((4, 1), dtype=np.float64)
        
        # State transition matrix F
        self.F = np.eye(4, dtype=np.float64)
        self._update_transition_matrix(self.dt)

        # Measurement matrix H (measures angular position directly: 2x4)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process noise covariance Q
        self.q_std = config.process_noise_accel_std
        self.Q = self._compute_process_noise(self.dt, self.q_std)

        # Measurement noise covariance R
        self.r_std = config.measurement_noise_std
        self.R = np.eye(2, dtype=np.float64) * (self.r_std ** 2)

        # State estimation error covariance P
        self.P = np.diag([
            config.initial_covariance_pos,
            config.initial_covariance_pos,
            config.initial_covariance_vel,
            config.initial_covariance_vel
        ]).astype(np.float64)

        # Innovation and innovation covariance from latest update
        self.last_innovation = np.zeros((2, 1), dtype=np.float64)
        self.last_innovation_cov = self.R.copy()

        self.is_initialized = False

    def _update_transition_matrix(self, dt: float):
        self.F[0, 2] = dt
        self.F[1, 3] = dt

    def _compute_process_noise(self, dt: float, q_std: float) -> np.ndarray:
        """Computes discrete white noise acceleration covariance matrix Q."""
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        var_q = q_std ** 2

        q11 = 0.25 * dt4 * var_q
        q13 = 0.50 * dt3 * var_q
        q33 = dt2 * var_q

        Q = np.array([
            [q11, 0.0, q13, 0.0],
            [0.0, q11, 0.0, q13],
            [q13, 0.0, q33, 0.0],
            [0.0, q13, 0.0, q33]
        ], dtype=np.float64)
        return Q

    def initialize(self, pan_deg: float, tilt_deg: float, pan_vel_deg_s: float = 0.0, tilt_vel_deg_s: float = 0.0):
        """Initializes the filter with the first verified target detection."""
        self.x[0, 0] = pan_deg
        self.x[1, 0] = tilt_deg
        self.x[2, 0] = pan_vel_deg_s
        self.x[3, 0] = tilt_vel_deg_s

        self.P = np.diag([1.0, 1.0, 15.0, 15.0]).astype(np.float64)
        self.is_initialized = True

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """Propagates state and covariance forward by dt seconds."""
        if not self.is_initialized:
            return self.x.copy()

        if dt is not None and abs(dt - self.dt) > 1e-5 and dt > 0:
            self.dt = dt
            self._update_transition_matrix(dt)
            self.Q = self._compute_process_noise(dt, self.q_std)

        # x(k|k-1) = F * x(k-1|k-1)
        self.x = self.F @ self.x
        # P(k|k-1) = F * P(k-1|k-1) * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, z_pan_deg: float, z_tilt_deg: float, r_override_std: Optional[float] = None) -> np.ndarray:
        """
        Updates state estimate with a new measurement z.
        Returns:
            Updated state vector x(k|k).
        """
        if not self.is_initialized:
            self.initialize(z_pan_deg, z_tilt_deg)
            return self.x.copy()

        z = np.array([[z_pan_deg], [z_tilt_deg]], dtype=np.float64)
        
        # Measurement noise covariance
        if r_override_std is not None and r_override_std > 0:
            R = np.eye(2, dtype=np.float64) * (r_override_std ** 2)
        else:
            R = self.R

        # Innovation: y = z - H * x_pred
        y = z - (self.H @ self.x)
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + R
        
        # Kalman Gain: K = P * H^T * S^(-1)
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        # State update: x = x + K * y
        self.x = self.x + K @ y

        # Joseph form covariance update for numerical stability: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        self.last_innovation = y
        self.last_innovation_cov = S

        return self.x.copy()

    def get_innovation_covariance(self) -> np.ndarray:
        """Returns the current 2x2 innovation covariance matrix S."""
        return self.H @ self.P @ self.H.T + self.R

    def get_predicted_forward(self, lookahead_sec: float) -> Tuple[float, float]:
        """Predicts position lookahead_sec into the future without modifying filter state."""
        pred_x = self.x[0, 0] + self.x[2, 0] * lookahead_sec
        pred_y = self.x[1, 0] + self.x[3, 0] * lookahead_sec
        return pred_x, pred_y

    @property
    def position_deg(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity_deg_s(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def covariance_trace(self) -> float:
        """Trace of positional covariance (scalar indicator of tracking uncertainty)."""
        return float(self.P[0, 0] + self.P[1, 1])

    def reset(self):
        self.x.fill(0.0)
        self.P = np.diag([
            self.cfg.initial_covariance_pos,
            self.cfg.initial_covariance_pos,
            self.cfg.initial_covariance_vel,
            self.cfg.initial_covariance_vel
        ]).astype(np.float64)
        self.is_initialized = False
