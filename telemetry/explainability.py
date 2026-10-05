"""
telemetry/explainability.py - Causal Explainability & Live Decision-Audit Engine.

Literature Grounding:
  - Samek et al. (2019): "Explainable AI: Interpreting, Explaining and Visualizing Deep Learning"
  - NASA SP-2016-6105 Rev 2: "NASA Systems Engineering Handbook (V&V Auditing)"

Transforms raw black-box mathematical state vectors into human-interpretable,
defensible causal reasoning traces for ISRO / SAC jury evaluation.
"""

from dataclasses import dataclass, asdict
import json
from typing import Dict, Any, List, Optional
import time


@dataclass
class DecisionTrace:
    frame: int
    timestamp: float
    track_state: str
    dominant_filter_model: str
    imm_probabilities: Dict[str, float]
    fdir_status: str
    nis_metric: float
    active_notch_hz: float
    gain_scale: float
    pointing_error_mrad: float
    causal_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CausalExplainabilityEngine:
    """
    Engine that generates continuous mathematical causal audit trails
    and real-time human-readable rationale strings.
    """

    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.history: List[DecisionTrace] = []
        self.current_trace: Optional[DecisionTrace] = None
        self.last_explanation: str = "System initialized and awaiting optical beacon acquisition."

    def generate_trace(
        self,
        frame: int,
        track_state: str,
        dominant_model: str,
        imm_probs: Dict[str, float],
        fdir_status: str,
        nis_metric: float,
        vibration_diag: Dict[str, Any],
        gain_scale: float,
        pointing_error_mrad: float,
        detection_found: bool,
        coast_frames: int = 0
    ) -> DecisionTrace:
        """
        Synthesizes subsystem states into a single unified causal trace.
        """
        active_notch = vibration_diag.get("dominant_frequency_hz", 0.0)
        atten_db = vibration_diag.get("estimated_attenuation_db", 0.0)

        # -------------------------------------------------------------
        # Causal Logic Synthesizer
        # -------------------------------------------------------------
        reasons = []

        # 1. State Machine Context
        if track_state == "SEARCH":
            reasons.append("Archimedean spiral scanning field of uncertainty for optical beacon.")
        elif track_state == "CANDIDATE":
            reasons.append("Evaluating M-of-N persistence to reject false-positive optical clutter.")
        elif track_state == "COAST":
            reasons.append(f"Target occluded/faded; dead-reckoning on IMM prediction (frame {coast_frames}).")
        elif track_state == "REACQUIRE":
            reasons.append("Directional velocity-biased search along last confirmed target vector.")
        elif track_state == "TRACK":
            # 2. Tracking Filter Causal Reason
            ct_p = imm_probs.get("CT", 0.0)
            ca_p = imm_probs.get("CA", 0.0)
            cv_p = imm_probs.get("CV", 0.0)

            if ct_p > 0.40:
                reasons.append(f"High-g banking maneuver detected (P_turn={ct_p*100:.0f}%); coordinated turn feedforward active.")
            elif ca_p > 0.40:
                reasons.append(f"Target acceleration detected (P_acc={ca_p*100:.0f}%); 6-DOF jerk compensation engaged.")
            else:
                reasons.append(f"Steady linear cruise track (P_cv={cv_p*100:.0f}%); nominal Kalman smoothing.")

        # 3. FDIR Supervisor Reason
        if fdir_status == "STUCK_PIXEL_ISOLATED":
            reasons.append("FDIR: Quarantined invariant hot-pixel to prevent centroid latching.")
        elif fdir_status == "OUTLIER_REJECTED":
            reasons.append(f"FDIR: Gated statistical anomaly (NIS={nis_metric:.1f} > Chi2 threshold).")
        elif fdir_status != "NOMINAL":
            reasons.append(f"FDIR Alert: State={fdir_status}.")

        # 4. Vibration Cancellation Reason
        if vibration_diag.get("enabled", False) and active_notch > 0.8:
            reasons.append(f"Notch: Platform jitter at {active_notch:.1f} Hz suppressed (-{atten_db:.1f} dB).")

        # 5. Controller Covariance Damping
        if gain_scale < 0.85:
            reasons.append(f"Control: Gain damped to {gain_scale*100:.0f}% due to atmospheric scintillation jitter.")

        if not reasons:
            reasons.append("Nominal closed-loop coarse alignment active.")

        full_explanation = " | ".join(reasons)
        self.last_explanation = full_explanation

        trace = DecisionTrace(
            frame=frame,
            timestamp=time.time(),
            track_state=track_state,
            dominant_filter_model=dominant_model,
            imm_probabilities=imm_probs,
            fdir_status=fdir_status,
            nis_metric=round(nis_metric, 2),
            active_notch_hz=round(active_notch, 2),
            gain_scale=round(gain_scale, 3),
            pointing_error_mrad=round(pointing_error_mrad, 3),
            causal_explanation=full_explanation
        )

        self.current_trace = trace
        self.history.append(trace)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return trace

    def export_audit_log_json(self, filepath: str):
        """Exports full recorded decision trail for ISRO jury validation."""
        records = [t.to_dict() for t in self.history]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)

    def get_latest_summary_string(self) -> str:
        return self.last_explanation

    def reset(self):
        self.history.clear()
        self.current_trace = None
        self.last_explanation = "System reset and awaiting optical beacon."
