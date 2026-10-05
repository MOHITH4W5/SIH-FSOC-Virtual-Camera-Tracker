"""
perception/gating.py - Mahalanobis Distance Gating & Validation Region.
Rejects out-of-gate clutter and false alarms by testing candidate detections
against the Kalman filter's predicted measurement and innovation covariance.
"""

import math
from typing import Tuple, Optional, Dict, Any
import numpy as np


class MahalanobisGater:
    """Computes Mahalanobis distance gates and validation ellipsoids."""

    def __init__(self, gate_threshold: float = 9.21):
        # Default chi-squared threshold for 2 degrees of freedom (p=0.01 / 99% confidence level is 9.21)
        self.gamma = gate_threshold

    def compute_distance(
        self,
        measurement: np.ndarray,
        predicted: np.ndarray,
        innovation_cov: np.ndarray
    ) -> float:
        """
        Computes the squared Mahalanobis distance d_M^2:
            d_M^2 = (z - z_hat)^T * S^(-1) * (z - z_hat)
        """
        diff = (measurement - predicted).reshape(-1, 1)
        try:
            inv_s = np.linalg.inv(innovation_cov)
            d_sq = float((diff.T @ inv_s @ diff).item())
            return max(0.0, d_sq)
        except np.linalg.LinAlgError:
            # Singular matrix fallback: Euclidean distance
            return float(np.sum(diff ** 2))

    def is_within_gate(
        self,
        measurement: np.ndarray,
        predicted: np.ndarray,
        innovation_cov: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Checks whether the measurement lies strictly within the acceptance gate.
        Returns:
            (is_accepted, d_M_squared)
        """
        d_sq = self.compute_distance(measurement, predicted, innovation_cov)
        return (d_sq <= self.gamma), d_sq

    def get_gate_ellipse_parameters(
        self,
        center: Tuple[float, float],
        innovation_cov: np.ndarray
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Extracts ellipse geometric parameters for UI overlay rendering.
        Returns:
            ((center_x, center_y), (axis_length_a, axis_length_b), angle_deg)
        """
        # Eigenvalues and eigenvectors of the 2x2 covariance matrix
        eigvals, eigvecs = np.linalg.eigh(innovation_cov[:2, :2])
        # Sort descending
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Scale by chi-squared gate factor sqrt(gamma)
        scale = math.sqrt(max(1.0, self.gamma))
        axis_a = 2.0 * scale * math.sqrt(max(0.1, eigvals[0]))
        axis_b = 2.0 * scale * math.sqrt(max(0.1, eigvals[1]))

        # Angle of primary eigenvector in degrees
        angle_rad = math.atan2(eigvecs[1, 0], eigvecs[0, 0])
        angle_deg = math.degrees(angle_rad)

        return (center, (axis_a, axis_b), angle_deg)
