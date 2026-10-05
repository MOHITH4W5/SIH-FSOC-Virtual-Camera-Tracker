"""
control/confidence_aware.py - Covariance-Aware Adaptive Gain Damping.
A key architectural differentiator from Section I.1 of the technical blueprint:
Dynamically scales controller gains inversely with the Kalman Filter innovation covariance:
    gain_scale = 1.0 / (1.0 + alpha * Tr(S))
Prevents the camera gimbal from erratically chasing turbulent noise during low-confidence events.
"""

from typing import Tuple
import numpy as np


class CovarianceGainDamper:
    """Adapts feedback controller responsiveness based on Kalman innovation covariance."""

    def __init__(self, alpha: float = 0.005, min_gain_scale: float = 0.20, enabled: bool = True):
        self.alpha = alpha
        self.min_gain_scale = min_gain_scale
        self.enabled = enabled

    def compute_gain_scale(self, innovation_cov: np.ndarray) -> float:
        """
        Computes the adaptive scalar multiplier for PID gains.
        Args:
            innovation_cov: 2x2 innovation covariance matrix S.
        Returns:
            Scalar gain multiplier in [min_gain_scale, 1.0].
        """
        if not self.enabled:
            return 1.0

        # Trace of positional innovation covariance (variance in pan + variance in tilt)
        cov_trace = float(np.trace(innovation_cov[:2, :2]))
        
        # Soft damping equation:
        scale = 1.0 / (1.0 + self.alpha * max(0.0, cov_trace))
        
        # Bound lower limit so controller never completely freezes
        return float(np.clip(scale, self.min_gain_scale, 1.0))
