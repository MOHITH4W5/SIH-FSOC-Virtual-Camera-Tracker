"""
tracking/state_machine.py - 5-State Autonomous Track Manager.
Implements the full acquisition and tracking lifecycle:
  1. SEARCH: Archimedean spiral scan over field of uncertainty
  2. CANDIDATE: M-of-N detection persistence confirmation
  3. TRACK: Active closed-loop tracking with innovation gating
  4. COAST: Dead-reckoning extrapolation on KF prediction during occlusions
  5. REACQUIRE: Velocity-biased directional scan along last known trajectory
"""

from enum import Enum
import math
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np

from core.config import StateMachineConfig
from tracking.kalman_filter import TargetKalmanFilter


class TrackState(str, Enum):
    SEARCH = "SEARCH"
    CANDIDATE = "CANDIDATE"
    TRACK = "TRACK"
    COAST = "COAST"
    REACQUIRE = "REACQUIRE"


class TrackStateMachine:
    """Manages the operational states and transitions of the virtual tracking system."""

    def __init__(self, config: StateMachineConfig, kalman: Any):
        self.cfg = config
        self.kf = kalman
        
        self.state: TrackState = TrackState.SEARCH
        
        # State counters and timers
        self.candidate_hits = 0
        self.candidate_window = 0
        self.coast_counter = 0
        self.search_time = 0.0
        self.reacquire_time = 0.0

        # Memory for velocity-biased reacquisition
        self.last_locked_pos = (0.0, 0.0)
        self.last_locked_vel = (0.0, 0.0)

        # Cumulative statistics
        self.total_frames = 0
        self.frames_in_track = 0
        self.lock_count = 0
        self.loss_count = 0

    def set_estimator(self, estimator: Any):
        """Allows dynamic switching between standard Kalman Filter and IMM Estimator."""
        self.kf = estimator

    def step(self, detection: Optional[Dict[str, Any]], dt: float) -> Tuple[TrackState, Tuple[float, float]]:
        """
        Updates state transitions and returns the recommended camera pointing target (pan_deg, tilt_deg).
        
        Args:
            detection: Verified candidate detection dict or None if no valid candidate found.
            dt: Frame time delta.
            
        Returns:
            (current_state, (target_pan_deg, target_tilt_deg))
        """
        self.total_frames += 1
        has_detection = (detection is not None)

        # =========================================================================
        # State Transitions
        # =========================================================================
        if self.state == TrackState.SEARCH:
            self.search_time += dt
            if has_detection:
                self.candidate_hits = 1
                self.candidate_window = 1
                self.state = TrackState.CANDIDATE
                # Initialize KF with candidate position
                meas_pan = detection["inertial_pan_deg"]
                meas_tilt = detection["inertial_tilt_deg"]
                self.kf.initialize(meas_pan, meas_tilt)

        elif self.state == TrackState.CANDIDATE:
            self.candidate_window += 1
            if has_detection:
                self.candidate_hits += 1
                # Update KF with candidate measurement
                self.kf.update(detection["inertial_pan_deg"], detection["inertial_tilt_deg"])

            # Check if M-of-N confirmation threshold met
            if self.candidate_hits >= self.cfg.candidate_confirm_frames:
                self.state = TrackState.TRACK
                self.lock_count += 1
                self.coast_counter = 0
                self.search_time = 0.0
                self.reacquire_time = 0.0
            elif self.candidate_window >= self.cfg.candidate_window_frames:
                # Failed confirmation: revert to SEARCH
                self.state = TrackState.SEARCH
                self.search_time = 0.0

        elif self.state == TrackState.TRACK:
            self.frames_in_track += 1
            if has_detection:
                # Update filter with confirmed measurement
                self.kf.update(detection["inertial_pan_deg"], detection["inertial_tilt_deg"])
                self.last_locked_pos = self.kf.position_deg
                self.last_locked_vel = self.kf.velocity_deg_s
                self.coast_counter = 0
            else:
                # Measurement dropped: switch to COAST
                self.state = TrackState.COAST
                self.coast_counter = 1
                self.loss_count += 1

        elif self.state == TrackState.COAST:
            if has_detection:
                # Target reappeared: restore lock immediately
                self.kf.update(detection["inertial_pan_deg"], detection["inertial_tilt_deg"])
                self.state = TrackState.TRACK
                self.frames_in_track += 1
                self.coast_counter = 0
            else:
                self.coast_counter += 1
                if self.coast_counter > self.cfg.coast_max_frames:
                    # Coast timeout expired: switch to REACQUIRE
                    self.state = TrackState.REACQUIRE
                    self.reacquire_time = 0.0

        elif self.state == TrackState.REACQUIRE:
            self.reacquire_time += dt
            if has_detection:
                # Found target during reacquisition scan: confirm candidate
                self.candidate_hits = 1
                self.candidate_window = 1
                self.state = TrackState.CANDIDATE
                self.kf.initialize(detection["inertial_pan_deg"], detection["inertial_tilt_deg"])
            elif self.reacquire_time > 4.0:
                # Reacquisition timed out: fall back to global SEARCH
                self.state = TrackState.SEARCH
                self.search_time = 0.0

        # =========================================================================
        # Compute Commanded Pointing Reference based on State
        # =========================================================================
        if self.state in [TrackState.TRACK, TrackState.COAST, TrackState.CANDIDATE]:
            # Command gimbal directly to KF predicted target inertial position
            pred_pan, pred_tilt = self.kf.get_predicted_forward(lookahead_sec=dt)
            return self.state, (pred_pan, pred_tilt)

        elif self.state == TrackState.REACQUIRE:
            # Velocity-Biased Spiral Scan:
            # Expand search outward centered along the projected trajectory corridor
            vel_x, vel_y = self.last_locked_vel
            base_x, base_y = self.last_locked_pos

            # Projected target anchor
            anchor_x = base_x + vel_x * (self.reacquire_time * self.cfg.velocity_bias_weight)
            anchor_y = base_y + vel_y * (self.reacquire_time * self.cfg.velocity_bias_weight)

            # Archimedean expansion
            omega = self.cfg.search_spiral_omega * 1.5
            radius = self.cfg.search_spiral_growth * self.reacquire_time * 1.2
            theta = omega * self.reacquire_time

            cmd_pan = anchor_x + radius * math.cos(theta)
            cmd_tilt = anchor_y + radius * math.sin(theta)
            return self.state, (cmd_pan, cmd_tilt)

        else:  # TrackState.SEARCH
            # Global Archimedean spiral search
            omega = self.cfg.search_spiral_omega
            radius = min(12.0, self.cfg.search_spiral_growth * self.search_time)
            theta = omega * self.search_time

            cmd_pan = radius * math.cos(theta)
            cmd_tilt = (radius * 0.6) * math.sin(theta)
            return self.state, (cmd_pan, cmd_tilt)

    @property
    def lock_retention_rate(self) -> float:
        """Percentage of frames spent in active TRACK state."""
        return (self.frames_in_track / max(1, self.total_frames)) * 100.0

    def reset(self):
        self.state = TrackState.SEARCH
        self.candidate_hits = 0
        self.candidate_window = 0
        self.coast_counter = 0
        self.search_time = 0.0
        self.reacquire_time = 0.0
        self.last_locked_pos = (0.0, 0.0)
        self.last_locked_vel = (0.0, 0.0)
        self.total_frames = 0
        self.frames_in_track = 0
        self.lock_count = 0
        self.loss_count = 0
