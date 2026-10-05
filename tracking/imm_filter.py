"""
tracking/imm_filter.py - Interacting Multiple Model (IMM) Estimator for FSOC Tracking.

Implements an adaptive 3-model regime:
  - Model 1 (CV): Constant Velocity model for linear cruise & steady drift
  - Model 2 (CA): Constant Acceleration model for speed transitions
  - Model 3 (CT): Coordinated Turn model for banking maneuvers and high-g evasive turns

Literature Grounding:
  - Blom & Bar-Shalom (1988), IEEE Trans. Autom. Control 33(8): "The Interacting Multiple Model
    Algorithm for Systems with Markovian Switching Coefficients"
  - Genovese (2001), SPIE Proc. Vol. 4365: "Optical Tracking with Interacting Multiple Model Filters"
"""

import math
from typing import Tuple, Optional, Dict, List
import numpy as np

from core.config import KalmanConfig


class CVModel:
    """Constant Velocity (CV) 4D State Filter: [x, y, vx, vy]^T."""
    def __init__(self, dt: float, q_std: float, r_std: float):
        self.dt = dt
        self.dim_x = 4
        self.dim_z = 2
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 15.0, 15.0]).astype(np.float64)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * (r_std ** 2)
        self.q_std = q_std
        self.F = np.eye(4, dtype=np.float64)
        self._set_F(dt)
        self.Q = self._calc_Q(dt, q_std)

    def _set_F(self, dt: float):
        self.F[0, 2] = dt
        self.F[1, 3] = dt

    def _calc_Q(self, dt: float, q_std: float) -> np.ndarray:
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        var = q_std ** 2
        q11 = 0.25 * dt4 * var
        q13 = 0.50 * dt3 * var
        q33 = dt2 * var
        return np.array([
            [q11, 0.0, q13, 0.0],
            [0.0, q11, 0.0, q13],
            [q13, 0.0, q33, 0.0],
            [0.0, q13, 0.0, q33]
        ], dtype=np.float64)

    def predict(self, dt: float):
        if abs(dt - self.dt) > 1e-5:
            self.dt = dt
            self._set_F(dt)
            self.Q = self._calc_Q(dt, self.q_std)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def evaluate_likelihood(self, z: np.ndarray, R_in: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calculates innovation, innovation covariance S, and Gaussian likelihood."""
        R = self.R if R_in is None else R_in
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        det_S = np.linalg.det(S)
        if det_S <= 1e-12:
            det_S = 1e-12
        inv_S = np.linalg.pinv(S)
        nis = float((y.T @ inv_S @ y).item())
        norm_const = 1.0 / (2.0 * math.pi * math.sqrt(det_S) + 1e-12)
        likelihood = norm_const * math.exp(-0.5 * min(nis, 80.0))
        return max(likelihood, 1e-15), y, S

    def update(self, y: np.ndarray, S: np.ndarray, R_in: Optional[np.ndarray] = None):
        R = self.R if R_in is None else R_in
        inv_S = np.linalg.pinv(S)
        K = self.P @ self.H.T @ inv_S
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T


class CAModel:
    """Constant Acceleration (CA) 6D State Filter: [x, y, vx, vy, ax, ay]^T."""
    def __init__(self, dt: float, q_std: float, r_std: float):
        self.dt = dt
        self.dim_x = 6
        self.dim_z = 2
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 15.0, 15.0, 20.0, 20.0]).astype(np.float64)
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.R = np.eye(2, dtype=np.float64) * (r_std ** 2)
        self.q_std = q_std * 1.5  # acceleration jerk process noise
        self.F = np.eye(6, dtype=np.float64)
        self._set_F(dt)
        self.Q = self._calc_Q(dt, self.q_std)

    def _set_F(self, dt: float):
        self.F[0, 2] = dt
        self.F[0, 4] = 0.5 * dt**2
        self.F[1, 3] = dt
        self.F[1, 5] = 0.5 * dt**2
        self.F[2, 4] = dt
        self.F[3, 5] = dt

    def _calc_Q(self, dt: float, q_std: float) -> np.ndarray:
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        dt5 = dt**5
        var = q_std ** 2
        Q = np.zeros((6, 6), dtype=np.float64)
        q_pos = 0.05 * dt5 * var
        q_pos_vel = 0.125 * dt4 * var
        q_pos_acc = (1.0 / 6.0) * dt3 * var
        q_vel = (1.0 / 3.0) * dt3 * var
        q_vel_acc = 0.5 * dt2 * var
        q_acc = dt * var

        for axis in (0, 1):
            p, v, a = axis, axis + 2, axis + 4
            Q[p, p] = q_pos
            Q[p, v] = Q[v, p] = q_pos_vel
            Q[p, a] = Q[a, p] = q_pos_acc
            Q[v, v] = q_vel
            Q[v, a] = Q[a, v] = q_vel_acc
            Q[a, a] = q_acc
        return Q

    def predict(self, dt: float):
        if abs(dt - self.dt) > 1e-5:
            self.dt = dt
            self._set_F(dt)
            self.Q = self._calc_Q(dt, self.q_std)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def evaluate_likelihood(self, z: np.ndarray, R_in: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        R = self.R if R_in is None else R_in
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        det_S = np.linalg.det(S)
        if det_S <= 1e-12:
            det_S = 1e-12
        inv_S = np.linalg.pinv(S)
        nis = float((y.T @ inv_S @ y).item())
        norm_const = 1.0 / (2.0 * math.pi * math.sqrt(det_S) + 1e-12)
        likelihood = norm_const * math.exp(-0.5 * min(nis, 80.0))
        return max(likelihood, 1e-15), y, S

    def update(self, y: np.ndarray, S: np.ndarray, R_in: Optional[np.ndarray] = None):
        R = self.R if R_in is None else R_in
        inv_S = np.linalg.pinv(S)
        K = self.P @ self.H.T @ inv_S
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T


class CTModel:
    """Coordinated Turn (CT) Model tracking angular curvature: [x, y, vx, vy, omega]^T."""
    def __init__(self, dt: float, q_std: float, r_std: float):
        self.dt = dt
        self.dim_x = 5
        self.dim_z = 2
        self.x = np.zeros((5, 1), dtype=np.float64)
        # state: [x, y, vx, vy, omega]^T, omega init near 0.2 rad/s turn rate
        self.x[4, 0] = 0.05
        self.P = np.diag([1.0, 1.0, 15.0, 15.0, 1.0]).astype(np.float64)
        self.H = np.zeros((2, 5), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.R = np.eye(2, dtype=np.float64) * (r_std ** 2)
        self.q_std = q_std
        self.q_omega = 0.15  # turn-rate process noise
        self.Q = self._calc_Q(dt)

    def _calc_Q(self, dt: float) -> np.ndarray:
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        var = self.q_std ** 2
        q11 = 0.25 * dt4 * var
        q13 = 0.50 * dt3 * var
        q33 = dt2 * var
        Q = np.zeros((5, 5), dtype=np.float64)
        Q[0, 0] = Q[1, 1] = q11
        Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = q13
        Q[2, 2] = Q[3, 3] = q33
        Q[4, 4] = dt * (self.q_omega ** 2)
        return Q

    def _build_F(self, omega: float, dt: float) -> np.ndarray:
        F = np.eye(5, dtype=np.float64)
        if abs(omega) < 1e-4:
            # Degenerate case approaches CV
            F[0, 2] = dt
            F[1, 3] = dt
        else:
            sin_w = math.sin(omega * dt)
            cos_w = math.cos(omega * dt)
            F[0, 2] = sin_w / omega
            F[0, 3] = -(1.0 - cos_w) / omega
            F[1, 2] = (1.0 - cos_w) / omega
            F[1, 3] = sin_w / omega
            F[2, 2] = cos_w
            F[2, 3] = -sin_w
            F[3, 2] = sin_w
            F[3, 3] = cos_w
        return F

    def predict(self, dt: float):
        self.dt = dt
        omega = float(self.x[4, 0])
        F = self._build_F(omega, dt)
        self.Q = self._calc_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def evaluate_likelihood(self, z: np.ndarray, R_in: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        R = self.R if R_in is None else R_in
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        det_S = np.linalg.det(S)
        if det_S <= 1e-12:
            det_S = 1e-12
        inv_S = np.linalg.pinv(S)
        nis = float((y.T @ inv_S @ y).item())
        norm_const = 1.0 / (2.0 * math.pi * math.sqrt(det_S) + 1e-12)
        likelihood = norm_const * math.exp(-0.5 * min(nis, 80.0))
        return max(likelihood, 1e-15), y, S

    def update(self, y: np.ndarray, S: np.ndarray, R_in: Optional[np.ndarray] = None):
        R = self.R if R_in is None else R_in
        inv_S = np.linalg.pinv(S)
        K = self.P @ self.H.T @ inv_S
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T


class TargetIMMEstimator:
    """
    3-Model Interacting Multiple Model (IMM) Estimator for FSOC Coarse Alignment.
    Seamless drop-in replacement or parallel upgrade for TargetKalmanFilter.
    """

    def __init__(self, config: KalmanConfig):
        self.cfg = config
        self.dt = config.dt

        # Base filters
        self.cv = CVModel(self.dt, config.process_noise_accel_std, config.measurement_noise_std)
        self.ca = CAModel(self.dt, config.process_noise_accel_std, config.measurement_noise_std)
        self.ct = CTModel(self.dt, config.process_noise_accel_std, config.measurement_noise_std)
        self.models = [self.cv, self.ca, self.ct]
        self.model_names = ["CV", "CA", "CT"]

        # Prior model probabilities: [mu_cv, mu_ca, mu_ct]
        self.mu = np.array([0.70, 0.15, 0.15], dtype=np.float64)

        # Markov Mode Transition Probability Matrix P_trans[i, j] = P(M_k = j | M_{k-1} = i)
        self.P_trans = np.array([
            [0.85, 0.08, 0.07],  # From CV
            [0.12, 0.78, 0.10],  # From CA
            [0.10, 0.08, 0.82]   # From CT
        ], dtype=np.float64)

        # Combined output state [x, y, vx, vy]^T and covariance (4x4)
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.diag([config.initial_covariance_pos, config.initial_covariance_pos,
                          config.initial_covariance_vel, config.initial_covariance_vel]).astype(np.float64)

        self.last_innovation = np.zeros((2, 1), dtype=np.float64)
        self.last_innovation_cov = np.eye(2, dtype=np.float64) * (config.measurement_noise_std ** 2)
        self.last_nis = 0.0

        self.is_initialized = False

    def initialize(self, pan_deg: float, tilt_deg: float, pan_vel_deg_s: float = 0.0, tilt_vel_deg_s: float = 0.0):
        """Initializes all 3 models with the first verified target detection."""
        # CV: 4D
        self.cv.x[0, 0] = pan_deg
        self.cv.x[1, 0] = tilt_deg
        self.cv.x[2, 0] = pan_vel_deg_s
        self.cv.x[3, 0] = tilt_vel_deg_s
        self.cv.P = np.diag([1.0, 1.0, 15.0, 15.0]).astype(np.float64)

        # CA: 6D
        self.ca.x[0, 0] = pan_deg
        self.ca.x[1, 0] = tilt_deg
        self.ca.x[2, 0] = pan_vel_deg_s
        self.ca.x[3, 0] = tilt_vel_deg_s
        self.ca.x[4, 0] = 0.0
        self.ca.x[5, 0] = 0.0
        self.ca.P = np.diag([1.0, 1.0, 15.0, 15.0, 20.0, 20.0]).astype(np.float64)

        # CT: 5D
        self.ct.x[0, 0] = pan_deg
        self.ct.x[1, 0] = tilt_deg
        self.ct.x[2, 0] = pan_vel_deg_s
        self.ct.x[3, 0] = tilt_vel_deg_s
        self.ct.x[4, 0] = 0.02
        self.ct.P = np.diag([1.0, 1.0, 15.0, 15.0, 1.0]).astype(np.float64)

        self.mu = np.array([0.70, 0.15, 0.15], dtype=np.float64)
        self.x = np.array([[pan_deg], [tilt_deg], [pan_vel_deg_s], [tilt_vel_deg_s]], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 15.0, 15.0]).astype(np.float64)
        self.is_initialized = True

    def _extract_common_4d(self, model) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts [x, y, vx, vy] state and 4x4 covariance from any model."""
        x4 = model.x[:4, :].copy()
        P4 = model.P[:4, :4].copy()
        return x4, P4

    def _inject_mixed_state(self, model_idx: int, x_mixed: np.ndarray, P_mixed: np.ndarray):
        """Injects mixed 4D state and covariance back into model while preserving higher states."""
        m = self.models[model_idx]
        m.x[:4, :] = x_mixed
        m.P[:4, :4] = P_mixed

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """Propagates IMM: Step 1 (Mixing) -> Step 2 (Parallel Prediction)."""
        if not self.is_initialized:
            return self.x.copy()

        eff_dt = self.dt if dt is None or dt <= 0 else dt

        # -------------------------------------------------------------
        # Step 1: Interaction / Mixing of Model States
        # -------------------------------------------------------------
        # Normalization factor: c_j = sum_i(p_{ij} * mu_i)
        c_bar = self.mu @ self.P_trans  # 1x3 vector

        # Mixing probabilities mu_{i|j} = (p_{ij} * mu_i) / c_j
        mu_mix = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            denom = c_bar[j] if c_bar[j] > 1e-12 else 1e-12
            for i in range(3):
                mu_mix[i, j] = (self.P_trans[i, j] * self.mu[i]) / denom

        # Compute mixed initial state and covariance for each model j
        common_states = [self._extract_common_4d(m) for m in self.models]

        for j in range(3):
            x_0j = np.zeros((4, 1), dtype=np.float64)
            for i in range(3):
                x_0j += mu_mix[i, j] * common_states[i][0]

            P_0j = np.zeros((4, 4), dtype=np.float64)
            for i in range(3):
                dx = common_states[i][0] - x_0j
                P_0j += mu_mix[i, j] * (common_states[i][1] + dx @ dx.T)

            self._inject_mixed_state(j, x_0j, P_0j)

        # -------------------------------------------------------------
        # Step 2: Parallel Mode Prediction
        # -------------------------------------------------------------
        for m in self.models:
            m.predict(eff_dt)

        # Combined predicted state for output
        self._update_combined_state()
        return self.x.copy()

    def update(self, z_pan_deg: float, z_tilt_deg: float, r_override_std: Optional[float] = None) -> np.ndarray:
        """
        Step 3 (Likelihood evaluation) -> Step 4 (Mode update) -> Step 5 (State Combination).
        """
        if not self.is_initialized:
            self.initialize(z_pan_deg, z_tilt_deg)
            return self.x.copy()

        z = np.array([[z_pan_deg], [z_tilt_deg]], dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * (r_override_std ** 2) if r_override_std else None

        # Mode likelihoods and individual updates
        likelihoods = np.zeros(3, dtype=np.float64)
        innovations = []
        covariances = []

        for idx, m in enumerate(self.models):
            L, y, S = m.evaluate_likelihood(z, R)
            likelihoods[idx] = L
            innovations.append(y)
            covariances.append(S)
            m.update(y, S, R)

        # -------------------------------------------------------------
        # Step 4: Model Probability Update
        # -------------------------------------------------------------
        # Normalization factor: c_bar = sum_i(P_{ij} * mu_i)
        c_bar = self.mu @ self.P_trans
        unnorm_mu = c_bar * likelihoods
        sum_mu = np.sum(unnorm_mu)

        if sum_mu > 1e-15:
            self.mu = unnorm_mu / sum_mu
        else:
            # Fallback uniform if numerical underflow
            self.mu = np.array([0.70, 0.15, 0.15], dtype=np.float64)

        # Enforce minimum probability floor to prevent model starvation
        prob_floor = 0.02
        self.mu = np.maximum(self.mu, prob_floor)
        self.mu /= np.sum(self.mu)

        # -------------------------------------------------------------
        # Step 5: State Combination
        # -------------------------------------------------------------
        self._update_combined_state()

        # Update dominant innovation & covariance for telemetry & FDIR
        dominant_idx = int(np.argmax(self.mu))
        self.last_innovation = innovations[dominant_idx]
        self.last_innovation_cov = covariances[dominant_idx]
        inv_S = np.linalg.pinv(self.last_innovation_cov)
        self.last_nis = float((self.last_innovation.T @ inv_S @ self.last_innovation).item())

        return self.x.copy()

    def _update_combined_state(self):
        """Computes probability-weighted 4D state and covariance."""
        x_comb = np.zeros((4, 1), dtype=np.float64)
        for i in range(3):
            x4, _ = self._extract_common_4d(self.models[i])
            x_comb += self.mu[i] * x4

        P_comb = np.zeros((4, 4), dtype=np.float64)
        for i in range(3):
            x4, P4 = self._extract_common_4d(self.models[i])
            dx = x4 - x_comb
            P_comb += self.mu[i] * (P4 + dx @ dx.T)

        self.x = x_comb
        self.P = P_comb

    def get_innovation_covariance(self) -> np.ndarray:
        return self.last_innovation_cov.copy()

    def get_predicted_forward(self, lookahead_sec: float) -> Tuple[float, float]:
        """Predicts position lookahead_sec into future using combined velocity and curvature."""
        # Use combined velocity + acceleration if CA or CT are prominent
        vx = self.x[2, 0]
        vy = self.x[3, 0]
        ax = self.ca.x[4, 0] * self.mu[1]
        ay = self.ca.x[5, 0] * self.mu[1]

        pred_x = self.x[0, 0] + vx * lookahead_sec + 0.5 * ax * (lookahead_sec ** 2)
        pred_y = self.x[1, 0] + vy * lookahead_sec + 0.5 * ay * (lookahead_sec ** 2)
        return float(pred_x), float(pred_y)

    @property
    def position_deg(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity_deg_s(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def covariance_trace(self) -> float:
        return float(self.P[0, 0] + self.P[1, 1])

    @property
    def model_probabilities(self) -> Dict[str, float]:
        """Returns dictionary of model probabilities for UI display and telemetry."""
        return {
            "CV": float(self.mu[0]),
            "CA": float(self.mu[1]),
            "CT": float(self.mu[2])
        }

    @property
    def dominant_model(self) -> str:
        idx = int(np.argmax(self.mu))
        return self.model_names[idx]

    def reset(self):
        self.x.fill(0.0)
        self.P = np.diag([
            self.cfg.initial_covariance_pos,
            self.cfg.initial_covariance_pos,
            self.cfg.initial_covariance_vel,
            self.cfg.initial_covariance_vel
        ]).astype(np.float64)
        self.cv = CVModel(self.dt, self.cfg.process_noise_accel_std, self.cfg.measurement_noise_std)
        self.ca = CAModel(self.dt, self.cfg.process_noise_accel_std, self.cfg.measurement_noise_std)
        self.ct = CTModel(self.dt, self.cfg.process_noise_accel_std, self.cfg.measurement_noise_std)
        self.models = [self.cv, self.ca, self.ct]
        self.mu = np.array([0.70, 0.15, 0.15], dtype=np.float64)
        self.is_initialized = False
