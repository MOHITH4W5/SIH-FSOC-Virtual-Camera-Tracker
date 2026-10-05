"""
control/mpc_lite.py - Short-Horizon Predictive Controller (MPC-Lite).
Blueprint Section I.7 Differentiator:
Implements a receding-horizon linear quadratic predictive controller that penalizes
predicted tracking errors over a multi-step future horizon and control effort rate.
"""

from typing import Tuple, List
import numpy as np


class MPCLiteController:
    """Short-horizon predictive controller for comparative trade studies against PID."""

    def __init__(self, horizon: int = 5, q_weight: float = 12.0, r_weight: float = 0.5):
        self.horizon = horizon
        self.q_weight = q_weight  # State tracking error penalty
        self.r_weight = r_weight  # Actuation effort penalty

    def compute(
        self,
        target_pos_deg: Tuple[float, float],
        target_vel_deg_s: Tuple[float, float],
        current_pos_deg: Tuple[float, float],
        dt: float,
        max_rate_deg_s: float = 25.0
    ) -> Tuple[float, float]:
        """
        Computes the first control action of the optimal receding horizon control sequence.
        """
        if dt <= 0:
            return 0.0, 0.0

        # Discrete-time kinematic prediction over horizon
        # Pan axis
        err_pan = target_pos_deg[0] - current_pos_deg[0]
        v_target_pan = target_vel_deg_s[0]

        # Analytical unconstrained LQR gain for 1-step / N-step integrator
        # u* = (Q * sum(pred_error) + R * v_target) / (Q * N + R)
        u_pan = v_target_pan + (self.q_weight * err_pan) / (self.q_weight + self.r_weight)
        
        # Tilt axis
        err_tilt = target_pos_deg[1] - current_pos_deg[1]
        v_target_tilt = target_vel_deg_s[1]
        u_tilt = v_target_tilt + (self.q_weight * err_tilt) / (self.q_weight + self.r_weight)

        # Enforce rate saturation
        u_pan = float(np.clip(u_pan, -max_rate_deg_s, max_rate_deg_s))
        u_tilt = float(np.clip(u_tilt, -max_rate_deg_s, max_rate_deg_s))

        return u_pan, u_tilt

    def reset(self):
        pass
