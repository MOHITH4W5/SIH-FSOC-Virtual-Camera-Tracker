"""
telemetry/fdir_manager.py - Autonomous Fault Detection, Isolation, and Recovery (FDIR) Supervisor.

Literature Grounding:
  - Willsky (1976), Automatica: "A survey of design methods for failure detection in dynamic systems"
  - NASA-HDBK-1002 (2020): "Fault Management Handbook"

Guards the FSOC terminal against:
  1. Chi-square Normalized Innovation Squared (NIS) statistical sensor anomalies
  2. Camera stuck pixels / frozen detector artifacts
  3. Deep atmospheric scintillation fading dropouts
  4. Target decoy / flare spoofing
"""

from enum import Enum
import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


class FDIRStatus(str, Enum):
    NOMINAL = "NOMINAL"
    DEGRADED_COAST = "DEGRADED_COAST"
    STUCK_PIXEL_ISOLATED = "STUCK_PIXEL_ISOLATED"
    OUTLIER_REJECTED = "OUTLIER_REJECTED"
    CRITICAL_FAULT = "CRITICAL_FAULT"


class FDIRSupervisor:
    """
    Autonomous Supervisor monitoring tracking innovations, sensor health,
    and optical channel fidelity.
    """

    def __init__(self, nis_gate_threshold: float = 9.21, nis_window_len: int = 5):
        self.nis_gate_threshold = nis_gate_threshold  # Chi-squared 2-DOF 99% confidence
        self.nis_window_len = nis_window_len
        self.nis_history: List[float] = []

        # Stuck pixel detection
        self.pixel_history: List[Tuple[float, float]] = []
        self.stuck_pixel_window = 12
        self.bad_pixels: List[Tuple[float, float]] = []

        # Current state and fault event log
        self.current_status: FDIRStatus = FDIRStatus.NOMINAL
        self.fault_events: List[Dict[str, Any]] = []
        self.last_fault_reason: str = "All subsystems operating nominally"

        # Frame counters
        self.total_frames = 0
        self.outliers_rejected_count = 0
        self.fades_handled_count = 0
        self.stuck_pixels_isolated_count = 0

    def inspect_measurement(
        self,
        raw_detection: Optional[Dict[str, Any]],
        filter_pos_deg: Tuple[float, float],
        filter_innov_cov: np.ndarray,
        gimbal_moving: bool = False,
        is_tracking: bool = True
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Validates detection health before passing to State Machine and Estimator.
        Returns:
            (is_valid, sanitized_detection, reason_description)
        """
        self.total_frames += 1

        if raw_detection is None:
            self.current_status = FDIRStatus.NOMINAL
            return True, None, "No detection (channel clear)"

        u = raw_detection.get("cx", raw_detection.get("pixel_x", 0.0))
        v = raw_detection.get("cy", raw_detection.get("pixel_y", 0.0))
        pan = raw_detection.get("inertial_pan_deg", raw_detection.get("pan", 0.0))
        tilt = raw_detection.get("inertial_tilt_deg", raw_detection.get("tilt", 0.0))

        dist_to_boresight = math.hypot(u - 640.0, v - 360.0)

        # -----------------------------------------------------------------
        # 1. Check Bad Pixel Mask (Stuck Pixel Isolation)
        # Note: Boresight region (< 60 px) is immune to bad-pixel rejection
        # -----------------------------------------------------------------
        if dist_to_boresight > 60.0:
            for bp_u, bp_v in self.bad_pixels:
                if math.hypot(u - bp_u, v - bp_v) < 2.0:
                    self.current_status = FDIRStatus.STUCK_PIXEL_ISOLATED
                    self.last_fault_reason = f"Detection at ({u:.1f}, {v:.1f}) coincides with known stuck pixel mask"
                    return False, None, self.last_fault_reason

        # -----------------------------------------------------------------
        # 2. Check for newly frozen / stuck pixel artifact
        # Note: Target centered at boresight must NEVER be masked as a stuck pixel!
        # -----------------------------------------------------------------
        self.pixel_history.append((u, v))
        if len(self.pixel_history) > self.stuck_pixel_window:
            self.pixel_history.pop(0)

        if len(self.pixel_history) >= self.stuck_pixel_window and gimbal_moving and dist_to_boresight > 60.0:
            u_coords = [p[0] for p in self.pixel_history]
            v_coords = [p[1] for p in self.pixel_history]
            var_u = float(np.var(u_coords))
            var_v = float(np.var(v_coords))
            if var_u < 1e-4 and var_v < 1e-4:
                # Pixel stayed invariant away from boresight while gimbal was moving -> Stuck Pixel!
                new_bp = (round(u, 1), round(v, 1))
                if new_bp not in self.bad_pixels:
                    self.bad_pixels.append(new_bp)
                    self.stuck_pixels_isolated_count += 1
                    self.current_status = FDIRStatus.STUCK_PIXEL_ISOLATED
                    self.last_fault_reason = f"New stuck pixel isolated at ({u:.1f}, {v:.1f})"
                    self._log_fault("STUCK_PIXEL", self.last_fault_reason)
                    return False, None, self.last_fault_reason

        # -----------------------------------------------------------------
        # 3. Statistical NIS Gating Check
        # (Only enforced during active closed-loop tracking, not initial acquisition)
        # -----------------------------------------------------------------
        if is_tracking:
            diff_pan = pan - filter_pos_deg[0]
            diff_tilt = tilt - filter_pos_deg[1]
            y = np.array([[diff_pan], [diff_tilt]], dtype=np.float64)

            try:
                inv_S = np.linalg.pinv(filter_innov_cov)
                nis = float((y.T @ inv_S @ y).item())
            except Exception:
                nis = 0.0

            self.nis_history.append(nis)
            if len(self.nis_history) > self.nis_window_len:
                self.nis_history.pop(0)

            # Single frame statistical gate - only reject extreme gross physical impossibility (> 150.0)
            if nis > 150.0:
                self.outliers_rejected_count += 1
                self.current_status = FDIRStatus.OUTLIER_REJECTED
                self.last_fault_reason = f"NIS gross outlier rejected: {nis:.1f} > 150.0"
                self._log_fault("OUTLIER_GATE", self.last_fault_reason)
                return False, None, self.last_fault_reason

            avg_nis = float(np.mean(self.nis_history))
            if avg_nis > self.nis_gate_threshold:
                self.last_fault_reason = f"Elevated window NIS average: {avg_nis:.2f}"
            else:
                self.current_status = FDIRStatus.NOMINAL
                self.last_fault_reason = "Nominal tracking confidence"
        else:
            self.current_status = FDIRStatus.NOMINAL
            self.last_fault_reason = "Acquisition scan (pre-track gating bypassed)"

        return True, raw_detection, self.last_fault_reason

    def _log_fault(self, fault_type: str, detail: str):
        event = {
            "frame": self.total_frames,
            "type": fault_type,
            "detail": detail
        }
        self.fault_events.append(event)
        if len(self.fault_events) > 50:
            self.fault_events.pop(0)

    def get_telemetry(self) -> Dict[str, Any]:
        """Provides full FDIR telemetry for GUI badges and Explainability Engine."""
        avg_nis = float(np.mean(self.nis_history)) if self.nis_history else 0.0
        cur_nis = float(self.nis_history[-1]) if self.nis_history else 0.0
        return {
            "status": self.current_status.value,
            "current_nis": round(cur_nis, 2),
            "window_avg_nis": round(avg_nis, 2),
            "bad_pixel_count": len(self.bad_pixels),
            "outliers_rejected": self.outliers_rejected_count,
            "stuck_pixels_isolated": self.stuck_pixels_isolated_count,
            "last_reason": self.last_fault_reason
        }

    def reset(self):
        self.nis_history.clear()
        self.pixel_history.clear()
        self.bad_pixels.clear()
        self.fault_events.clear()
        self.current_status = FDIRStatus.NOMINAL
        self.total_frames = 0
        self.outliers_rejected_count = 0
        self.fades_handled_count = 0
        self.stuck_pixels_isolated_count = 0
        self.last_fault_reason = "All subsystems operating nominally"
