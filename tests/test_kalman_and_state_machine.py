"""
tests/test_kalman_and_state_machine.py - Tests for Kalman filter and 5-state track manager.
"""

import numpy as np
import pytest

from core.config import KalmanConfig, StateMachineConfig
from tracking.kalman_filter import TargetKalmanFilter
from tracking.state_machine import TrackStateMachine, TrackState


def test_kalman_filter_convergence():
    cfg = KalmanConfig(dt=0.016, process_noise_accel_std=5.0, measurement_noise_std=0.05)
    kf = TargetKalmanFilter(cfg)
    
    # Initialize at (1.0, -1.0)
    kf.initialize(1.0, -1.0)
    
    # Target moving at constant velocity: pan_rate = 2.0 deg/s, tilt_rate = 1.0 deg/s
    true_pan = 1.0
    true_tilt = -1.0
    v_pan = 2.0
    v_tilt = 1.0
    
    for _ in range(50):
        true_pan += v_pan * cfg.dt
        true_tilt += v_tilt * cfg.dt
        
        kf.predict(cfg.dt)
        # Add small measurement noise
        z_pan = true_pan + np.random.normal(0, 0.02)
        z_tilt = true_tilt + np.random.normal(0, 0.02)
        kf.update(z_pan, z_tilt)

    est_pan, est_tilt = kf.position_deg
    est_vx, est_vy = kf.velocity_deg_s

    # Should accurately estimate position and velocity
    assert abs(est_pan - true_pan) < 0.08
    assert abs(est_tilt - true_tilt) < 0.08
    assert abs(est_vx - v_pan) < 0.35
    assert abs(est_vy - v_tilt) < 0.35
    assert kf.covariance_trace < 0.1


def test_state_machine_lifecycle():
    sm_cfg = StateMachineConfig(candidate_confirm_frames=3, coast_max_frames=5)
    kf_cfg = KalmanConfig()
    kf = TargetKalmanFilter(kf_cfg)
    sm = TrackStateMachine(sm_cfg, kf)

    # 1. Starts in SEARCH
    assert sm.state == TrackState.SEARCH

    # 2. First candidate detection -> CANDIDATE
    det = {"inertial_pan_deg": 2.0, "inertial_tilt_deg": 1.0}
    state, cmd = sm.step(det, 0.016)
    assert state == TrackState.CANDIDATE

    # 3. Two more consecutive detections -> TRACK (3 of 3)
    sm.step(det, 0.016)
    state, cmd = sm.step(det, 0.016)
    assert state == TrackState.TRACK

    # 4. Target loss -> COAST
    state, cmd = sm.step(None, 0.016)
    assert state == TrackState.COAST

    # 5. 5 consecutive missed frames -> REACQUIRE
    for _ in range(5):
        state, cmd = sm.step(None, 0.016)
    assert state == TrackState.REACQUIRE
