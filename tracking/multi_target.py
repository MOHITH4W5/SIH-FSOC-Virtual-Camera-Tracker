"""
tracking/multi_target.py - Multi-Target Data Association & Prioritization.
Handles multi-target crossing events and assigns candidates using:
  1. Mahalanobis distance to existing active track prediction
  2. Boresight distance minimization
  3. Maximum fused confidence prioritization
Prevents track-swapping and decoy distraction.
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np

from perception.gating import MahalanobisGater


class MultiTargetAssociator:
    """Associates candidate detections with active tracks and enforces prioritization rules."""

    def __init__(self, gater: MahalanobisGater, selection_rule: str = "gated_nearest"):
        self.gater = gater
        self.selection_rule = selection_rule

    def select_best_candidate(
        self,
        candidates: List[Dict[str, Any]],
        predicted_pos_deg: np.ndarray,
        innovation_cov: np.ndarray,
        center_pixel: Tuple[float, float] = (640.0, 360.0),
        is_tracking: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Selects the single best candidate measurement for track update from a list of detections.
        
        Args:
            candidates: List of candidate detection dictionaries.
            predicted_pos_deg: 2D array [pred_pan_deg, pred_tilt_deg].
            innovation_cov: 2x2 innovation covariance matrix S.
            center_pixel: (cx, cy) of sensor center.
            is_tracking: True if in active TRACK state (enforces Mahalanobis gating).
                         False if in SEARCH/CANDIDATE/REACQUIRE state (accepts best candidate in FOV).
            
        Returns:
            Selected candidate dict or None if no candidate satisfies criteria.
        """
        if not candidates:
            return None

        # During unacquired search/reacquisition, prioritize genuine high-flux beacons
        if not is_tracking:
            sorted_by_conf = sorted(
                candidates,
                key=lambda c: c.get("flux", 0.0) * c.get("fused_confidence", 0.0),
                reverse=True
            )
            return sorted_by_conf[0]

        # Filter candidates by Mahalanobis gate during active tracking
        gated_candidates = []
        for cand in candidates:
            meas_pos = np.array([cand["inertial_pan_deg"], cand["inertial_tilt_deg"]], dtype=np.float64)
            d_sq = self.gater.compute_distance(meas_pos, predicted_pos_deg, innovation_cov)
            
            # Record association metrics
            cand["mahalanobis_dist_sq"] = d_sq
            dist_to_center = math.hypot(cand["cx"] - center_pixel[0], cand["cy"] - center_pixel[1])
            cand["boresight_dist_px"] = dist_to_center

            # Accept if inside standard gate or relaxed gate
            effective_gamma = self.gater.gamma * 2.5
            if d_sq <= effective_gamma:
                gated_candidates.append(cand)

        # If candidates exist inside gate, select according to active prioritization rule
        if gated_candidates:
            if self.selection_rule == "gated_nearest":
                # Nearest in Mahalanobis distance to Kalman prediction (most consistent with kinematics)
                gated_candidates.sort(key=lambda c: c["mahalanobis_dist_sq"])
                return gated_candidates[0]
            elif self.selection_rule == "closest_boresight":
                # Closest to optical axis
                gated_candidates.sort(key=lambda c: c["boresight_dist_px"])
                return gated_candidates[0]
            else:
                # Highest fused confidence
                gated_candidates.sort(key=lambda c: c.get("fused_confidence", 0.0), reverse=True)
                return gated_candidates[0]

        # Proximity fallback: If no candidate passed strict gate, but a confident candidate is within 1.5 deg
        for cand in sorted(candidates, key=lambda c: c.get("fused_confidence", 0.0), reverse=True):
            ang_sep = math.hypot(cand["inertial_pan_deg"] - predicted_pos_deg[0],
                                 cand["inertial_tilt_deg"] - predicted_pos_deg[1])
            if ang_sep < 1.5 and cand.get("fused_confidence", 0.0) >= 0.40:
                return cand

        return None
