"""
benchmarks/benchmark_suite.py - Reproducible Automated Benchmark Runner.
Executes standard evaluation scenarios:
  1. Baseline Closed-Loop Tracking (Calm + Occlusion test)
  2. Atmospheric Turbulence Severity Sweep (Cn^2 proxy 0.0 to 0.8)
  3. Platform Micro-Vibration Sweep
Exports telemetry logs, generates figures, and writes the performance report.
"""

import os
import time
import math
from typing import Dict, Any, List
import numpy as np

from core.config import AppConfig
from simulation.environment import VirtualEnvironment
from perception.hybrid_detector import HybridBeaconDetector
from perception.gating import MahalanobisGater
from tracking.kalman_filter import TargetKalmanFilter
from tracking.imm_filter import TargetIMMEstimator
from tracking.state_machine import TrackStateMachine, TrackState
from tracking.multi_target import MultiTargetAssociator
from control.pid_controller import DualAxisGimbalController
from control.vibration_canceller import AdaptiveVibrationCanceller
from control.confidence_aware import CovarianceGainDamper
from telemetry.fdir_manager import FDIRSupervisor
from telemetry.metrics import PerformanceMetrics
from telemetry.logger import TelemetryLogger
from benchmarks.report_generator import BenchmarkReportGenerator


class BenchmarkSuite:
    """Automated benchmark harness executing reproducible test scenarios."""

    def __init__(self, output_dir: str = "reports/benchmark_run"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.reporter = BenchmarkReportGenerator(output_dir)

    def run_single_scenario(
        self,
        config: AppConfig,
        duration_sec: float = 6.0,
        occlusion_window: Optional[Tuple[float, float]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs one closed-loop scenario and returns (summary_metrics, time_series_dict).
        """
        np.random.seed(42)
        env = VirtualEnvironment(config)
        detector = HybridBeaconDetector(config.detector)
        gater = MahalanobisGater(config.detector.mahalanobis_gate_threshold)
        kalman = TargetIMMEstimator(config.kalman)
        state_machine = TrackStateMachine(config.state_machine, kalman)
        associator = MultiTargetAssociator(gater)
        fdir = FDIRSupervisor(nis_gate_threshold=config.detector.mahalanobis_gate_threshold)
        vibration_canceller = AdaptiveVibrationCanceller(fs_hz=config.target_fps)
        controller = DualAxisGimbalController(config.controller, vibration_canceller=vibration_canceller)
        damper = CovarianceGainDamper(config.controller.covariance_damping_alpha,
                                      enabled=config.controller.covariance_damping_enabled)
        metrics = PerformanceMetrics()
        logger = TelemetryLogger(self.output_dir)

        dt = 1.0 / config.target_fps
        total_frames = int(duration_sec * config.target_fps)

        time_series = {
            "time_sec": [],
            "error_px": [],
            "error_mrad": [],
            "state": [],
            "pan_err_deg": [],
            "tilt_err_deg": []
        }

        for frame_idx in range(total_frames):
            t_frame_start = time.perf_counter()
            sim_time = frame_idx * dt
            metrics.record_frame_start(sim_time)

            # Check scripted occlusion
            if occlusion_window is not None:
                if occlusion_window[0] <= sim_time <= occlusion_window[1]:
                    env.disturbance.occlusion_active = True
                else:
                    env.disturbance.occlusion_active = False

            # 1. Simulation Step
            t0 = time.perf_counter()
            sensor_frame, targets_gt, sim_telem = env.step(dt)
            t_sim_ms = (time.perf_counter() - t0) * 1000.0
            metrics.record_stage_latency("simulation", t_sim_ms)

            # 2. Perception Step
            t0 = time.perf_counter()
            raw_candidates = detector.detect(sensor_frame)
            # Add inertial angles to candidates for gating
            for cand in raw_candidates:
                pan_off, tilt_off = env.geom.pixel_to_angle_deg(cand["cx"], cand["cy"])
                cand["inertial_pan_deg"] = sim_telem["cam_pan_deg"] + pan_off
                cand["inertial_tilt_deg"] = sim_telem["cam_tilt_deg"] + tilt_off

            # Data association & gating
            pred_inertial = np.array(kalman.position_deg, dtype=np.float64)
            innov_cov = kalman.get_innovation_covariance()
            is_active_track = (state_machine.state in [TrackState.TRACK, TrackState.COAST])
            selected_cand = associator.select_best_candidate(
                raw_candidates, pred_inertial, innov_cov,
                center_pixel=(env.geom.cx, env.geom.cy),
                is_tracking=is_active_track
            )

            # Autonomous FDIR Validation & Isolation
            is_valid, fdir_cand, fdir_reason = fdir.inspect_measurement(
                selected_cand,
                kalman.position_deg,
                innov_cov,
                gimbal_moving=True,
                is_tracking=is_active_track
            )
            verified_cand = fdir_cand if is_valid else None
            t_det_ms = (time.perf_counter() - t0) * 1000.0
            metrics.record_stage_latency("detection", t_det_ms)

            # 3. State Estimation & Track Management
            t0 = time.perf_counter()
            kalman.predict(dt)
            track_state, cmd_pos_deg = state_machine.step(verified_cand, dt)
            t_est_ms = (time.perf_counter() - t0) * 1000.0
            metrics.record_stage_latency("estimation", t_est_ms)

            # 4. Control & Gimbal Actuation
            t0 = time.perf_counter()
            gain_scale = damper.compute_gain_scale(innov_cov)
            feedforward_vel = kalman.velocity_deg_s if track_state in [TrackState.TRACK, TrackState.COAST] else (0.0, 0.0)

            cmd_pan_rate, cmd_tilt_rate = controller.compute(
                target_pan_deg=cmd_pos_deg[0],
                target_tilt_deg=cmd_pos_deg[1],
                current_pan_deg=sim_telem["cam_pan_deg"],
                current_tilt_deg=sim_telem["cam_tilt_deg"],
                dt=dt,
                feedforward_vel=feedforward_vel,
                gain_scale=gain_scale
            )
            env.gimbal.set_rate_command(cmd_pan_rate, cmd_tilt_rate)
            t_ctrl_ms = (time.perf_counter() - t0) * 1000.0
            metrics.record_stage_latency("control", t_ctrl_ms)

            # 5. Metrics & Telemetry Recording
            primary_tgt = targets_gt[0] if targets_gt else None
            true_px = primary_tgt["pixel_x"] if primary_tgt else None
            true_py = primary_tgt["pixel_y"] if primary_tgt else None

            err_mrad = None
            err_px = None
            if true_px is not None and true_py is not None:
                err_px = math.hypot(true_px - env.geom.cx, true_py - env.geom.cy)
                err_mrad = env.geom.angular_distance_mrad(true_px, true_py, env.geom.cx, env.geom.cy)

            t_frame_end = time.perf_counter()
            frame_ms = (t_frame_end - t_frame_start) * 1000.0

            metrics.update_frame(
                frame_latency_ms=frame_ms,
                current_state=track_state.value,
                sim_time=sim_time,
                true_px=true_px,
                true_py=true_py,
                boresight_px=(env.geom.cx, env.geom.cy),
                error_mrad=err_mrad
            )

            # Collect time series
            if err_px is not None and not sim_telem["is_occluded"]:
                time_series["time_sec"].append(sim_time)
                time_series["error_px"].append(err_px)
                time_series["error_mrad"].append(err_mrad)
                time_series["state"].append(track_state.value)
                time_series["pan_err_deg"].append(cmd_pos_deg[0] - sim_telem["cam_pan_deg"])
                time_series["tilt_err_deg"].append(cmd_pos_deg[1] - sim_telem["cam_tilt_deg"])

            logger.log_frame({
                "time_sec": sim_time,
                "state": track_state.value,
                "error_px": err_px if err_px is not None else -1.0,
                "error_mrad": err_mrad if err_mrad is not None else -1.0,
                "cam_pan_deg": sim_telem["cam_pan_deg"],
                "cam_tilt_deg": sim_telem["cam_tilt_deg"],
                "target_pan_deg": primary_tgt["world_pan_deg"] if primary_tgt else 0.0,
                "target_tilt_deg": primary_tgt["world_tilt_deg"] if primary_tgt else 0.0,
                "is_occluded": sim_telem["is_occluded"]
            })

        logger.export_csv("benchmark_telemetry.csv")
        logger.export_json("benchmark_telemetry.json")

        summary = metrics.get_summary()
        return summary, time_series

    def run_full_suite(self) -> str:
        """Runs the complete SIH evaluation suite and compiles the final report."""
        print("=================================================================")
        print("SIH26169 — Executing Full Automated Evaluation Benchmark Suite")
        print("=================================================================")

        suite_results: Dict[str, Any] = {}

        # 1. Baseline Scenario Run (Calm + Occlusion)
        print("\n[1/3] Running Scenario: Baseline Tracking with Occlusion Recovery...")
        base_cfg = AppConfig()
        base_cfg.disturbance.turbulence_strength = 0.20
        base_summary, base_ts = self.run_single_scenario(base_cfg, duration_sec=7.0, occlusion_window=(2.5, 3.8))
        suite_results["baseline_summary"] = base_summary
        suite_results["time_series"] = base_ts
        print(f"      Completed! FPS: {base_summary['fps']:.1f}, RMSE Error: {base_summary['rmse_error_px']:.2f} px ({base_summary['rmse_error_mrad']:.3f} mrad)")

        # 2. Turbulence Severity Sweep (Robustness Curve)
        print("\n[2/3] Running Scenario: Atmospheric Turbulence Severity Sweep (Cn^2 Proxy)...")
        turb_sweep_results = []
        for turb_val in [0.0, 0.2, 0.4, 0.6, 0.8]:
            cfg = AppConfig()
            cfg.disturbance.turbulence_strength = turb_val
            # Short test per point
            summary, _ = self.run_single_scenario(cfg, duration_sec=4.0, occlusion_window=None)
            turb_sweep_results.append({
                "turbulence_strength": turb_val,
                "rmse_error_px": summary["rmse_error_px"],
                "rmse_error_mrad": summary["rmse_error_mrad"],
                "lock_retention_rate_pct": summary["lock_retention_rate_pct"]
            })
            print(f"      Cn^2 Proxy = {turb_val:.1f} -> RMSE Error: {summary['rmse_error_px']:.2f} px, Lock: {summary['lock_retention_rate_pct']:.1f}%")
        suite_results["turbulence_sweep"] = turb_sweep_results

        # 3. Generate Evaluation Plots and Report
        print("\n[3/3] Generating Figures and Markdown Performance Report...")
        plots = self.reporter.generate_plots(suite_results)
        report_path = self.reporter.generate_markdown_report(suite_results, plots)

        print("=================================================================")
        print(f"Evaluation Complete! Report generated at:\n  -> {report_path}")
        print("=================================================================")
        return report_path
