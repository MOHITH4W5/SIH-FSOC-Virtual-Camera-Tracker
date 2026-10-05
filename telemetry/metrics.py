"""
telemetry/metrics.py - Real-Time Performance & Tracking Quality Metrics Engine.
Computes and updates running performance statistics required by official SIH guidelines:
  1. Frames per second (Instantaneous & Rolling)
  2. Latency breakdown per pipeline stage (ms)
  3. Acquisition time (first lock) and Re-acquisition time (post-loss)
  4. Tracking error (Instantaneous, Average, RMSE, Max in both pixels and mrad)
  5. Lock retention rate (% of frames in stable lock)
"""

import time
import math
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np


class PerformanceMetrics:
    """Calculates and maintains real-time tracking statistics and benchmark metrics."""

    def __init__(self, window_size: int = 120):
        self.window_size = window_size

        # Frame timing
        self.frame_latencies = deque(maxlen=window_size)
        self.stage_latencies: Dict[str, deque] = {
            "simulation": deque(maxlen=window_size),
            "detection": deque(maxlen=window_size),
            "estimation": deque(maxlen=window_size),
            "control": deque(maxlen=window_size),
            "overlay": deque(maxlen=window_size)
        }

        # Tracking error histories (when locked)
        self.tracking_errors_px = deque(maxlen=window_size)
        self.tracking_errors_mrad = deque(maxlen=window_size)
        
        # Cumulative error stats
        self.all_errors_px: List[float] = []
        self.all_errors_mrad: List[float] = []

        # Lock timing
        self.sim_start_time = 0.0
        self.first_lock_time: Optional[float] = None
        self.current_loss_start_time: Optional[float] = None
        self.reacquisition_times: List[float] = []

        # State frame counters
        self.total_frames = 0
        self.locked_frames = 0
        self.coast_frames = 0
        self.search_frames = 0

        self.last_state = "SEARCH"

    @property
    def frame_count(self) -> int:
        return self.total_frames

    def record_frame_start(self, sim_time: float):
        if self.total_frames == 0:
            self.sim_start_time = sim_time

    def record_stage_latency(self, stage: str, latency_ms: float):
        if stage in self.stage_latencies:
            self.stage_latencies[stage].append(latency_ms)

    def update_frame(
        self,
        frame_latency_ms: float,
        current_state: str,
        sim_time: float,
        true_px: Optional[float],
        true_py: Optional[float],
        boresight_px: Tuple[float, float],
        error_mrad: Optional[float] = None
    ):
        """Updates rolling statistics at the conclusion of each frame."""
        self.total_frames += 1
        self.frame_latencies.append(frame_latency_ms)

        # State tracking
        if current_state == "TRACK":
            self.locked_frames += 1
            if self.first_lock_time is None:
                self.first_lock_time = sim_time - self.sim_start_time

            # Check if this frame recovered from a previous loss
            if self.current_loss_start_time is not None:
                reacq_time = sim_time - self.current_loss_start_time
                self.reacquisition_times.append(reacq_time)
                self.current_loss_start_time = None

        elif current_state == "COAST":
            self.coast_frames += 1
            if self.last_state == "TRACK" and self.current_loss_start_time is None:
                self.current_loss_start_time = sim_time

        elif current_state in ["SEARCH", "REACQUIRE"]:
            self.search_frames += 1
            if self.last_state == "TRACK" and self.current_loss_start_time is None:
                self.current_loss_start_time = sim_time

        self.last_state = current_state

        # Compute tracking error relative to sensor boresight
        if true_px is not None and true_py is not None and current_state == "TRACK":
            cx, cy = boresight_px
            err_px = math.hypot(true_px - cx, true_py - cy)
            self.tracking_errors_px.append(err_px)
            self.all_errors_px.append(err_px)

            if error_mrad is not None:
                self.tracking_errors_mrad.append(error_mrad)
                self.all_errors_mrad.append(error_mrad)

    def get_summary(self) -> Dict[str, Any]:
        """Returns instantaneous and cumulative summary dictionary."""
        # FPS
        avg_latency = float(np.mean(self.frame_latencies)) if self.frame_latencies else 16.6
        rolling_fps = 1000.0 / max(0.1, avg_latency)

        # Errors
        cur_err_px = float(self.tracking_errors_px[-1]) if self.tracking_errors_px else 0.0
        avg_err_px = float(np.mean(self.tracking_errors_px)) if self.tracking_errors_px else 0.0
        rmse_err_px = float(np.sqrt(np.mean(np.square(self.tracking_errors_px)))) if self.tracking_errors_px else 0.0
        max_err_px = float(np.max(self.all_errors_px)) if self.all_errors_px else 0.0

        cur_err_mrad = float(self.tracking_errors_mrad[-1]) if self.tracking_errors_mrad else 0.0
        avg_err_mrad = float(np.mean(self.tracking_errors_mrad)) if self.tracking_errors_mrad else 0.0
        rmse_err_mrad = float(np.sqrt(np.mean(np.square(self.tracking_errors_mrad)))) if self.tracking_errors_mrad else 0.0

        # Lock retention
        lock_rate = (self.locked_frames / max(1, self.total_frames)) * 100.0
        avg_reacq = float(np.mean(self.reacquisition_times)) if self.reacquisition_times else 0.0

        stage_avg = {
            k: (float(np.mean(v)) if v else 0.0)
            for k, v in self.stage_latencies.items()
        }

        return {
            "fps": rolling_fps,
            "frame_latency_ms": avg_latency,
            "acquisition_time_sec": self.first_lock_time if self.first_lock_time is not None else 0.0,
            "avg_reacquisition_time_sec": avg_reacq,
            "lock_retention_rate_pct": lock_rate,
            "current_error_px": cur_err_px,
            "avg_error_px": avg_err_px,
            "rmse_error_px": rmse_err_px,
            "max_error_px": max_err_px,
            "current_error_mrad": cur_err_mrad,
            "avg_error_mrad": avg_err_mrad,
            "rmse_error_mrad": rmse_err_mrad,
            "total_frames": self.total_frames,
            "locked_frames": self.locked_frames,
            "stage_latencies_ms": stage_avg
        }
