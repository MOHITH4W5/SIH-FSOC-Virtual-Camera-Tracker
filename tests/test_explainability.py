"""
tests/test_explainability.py - Unit Tests for Causal Explainability Engine.
Verifies:
  1. Synthesis of multi-subsystem states into human-readable causal text
  2. IMM turn detection rationale trigger
  3. FDIR anomaly explanation trigger
  4. JSON audit export
"""

import os
import json
import pytest

from telemetry.explainability import CausalExplainabilityEngine, DecisionTrace


def test_explainability_trace_generation():
    engine = CausalExplainabilityEngine()
    trace = engine.generate_trace(
        frame=42,
        track_state="TRACK",
        dominant_model="CT",
        imm_probs={"CV": 0.10, "CA": 0.15, "CT": 0.75},
        fdir_status="NOMINAL",
        nis_metric=2.3,
        vibration_diag={"enabled": True, "dominant_frequency_hz": 9.5, "estimated_attenuation_db": 18.5},
        gain_scale=0.95,
        pointing_error_mrad=0.45,
        detection_found=True
    )

    assert isinstance(trace, DecisionTrace)
    assert trace.frame == 42
    assert "banking maneuver detected" in trace.causal_explanation
    assert "9.5 Hz" in trace.causal_explanation
    assert trace.pointing_error_mrad == 0.45


def test_explainability_fdir_alert_rationale():
    engine = CausalExplainabilityEngine()
    trace = engine.generate_trace(
        frame=105,
        track_state="TRACK",
        dominant_model="CV",
        imm_probs={"CV": 0.85, "CA": 0.10, "CT": 0.05},
        fdir_status="STUCK_PIXEL_ISOLATED",
        nis_metric=1.5,
        vibration_diag={"enabled": False, "dominant_frequency_hz": 0.0, "estimated_attenuation_db": 0.0},
        gain_scale=1.0,
        pointing_error_mrad=0.8,
        detection_found=True
    )

    assert "FDIR: Quarantined invariant hot-pixel" in trace.causal_explanation


def test_explainability_json_export(tmp_path):
    engine = CausalExplainabilityEngine()
    for f in range(5):
        engine.generate_trace(
            frame=f,
            track_state="TRACK",
            dominant_model="CV",
            imm_probs={"CV": 0.8, "CA": 0.1, "CT": 0.1},
            fdir_status="NOMINAL",
            nis_metric=1.2,
            vibration_diag={},
            gain_scale=1.0,
            pointing_error_mrad=0.3,
            detection_found=True
        )

    export_path = str(tmp_path / "audit_trace.json")
    engine.export_audit_log_json(export_path)

    assert os.path.exists(export_path)
    with open(export_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert len(data) == 5
    assert data[0]["frame"] == 0
