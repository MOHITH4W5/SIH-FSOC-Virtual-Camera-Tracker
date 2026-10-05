"""
main.py - Unified Entry Point for SIH26169 FSOC Virtual Camera Tracker.
Supports:
  1. Interactive Mission Control Desktop GUI (--gui)
  2. Headless Automated Benchmark Suite (--benchmark)
  3. Continuous Headless Simulation Logging (--headless)
  4. Camera-Agnostic Physical Optical Feed Mode (--camera-input)
"""

import sys
import os
import argparse
import time
import cv2

from core.config import AppConfig
from benchmarks.benchmark_suite import BenchmarkSuite
from ui.gui_app import launch_gui


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIH26169 — AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals"
    )
    parser.add_argument("--gui", action="store_true", default=True,
                        help="Launch interactive desktop mission control GUI (default)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Execute full reproducible automated benchmark suite and generate evaluation report")
    parser.add_argument("--headless", action="store_true",
                        help="Run in non-graphical headless simulation mode")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration in seconds for headless simulation (default: 10.0)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom JSON configuration file")
    parser.add_argument("--output-dir", type=str, default="reports/latest_run",
                        help="Directory to store evaluation reports, figures, and telemetry logs")
    parser.add_argument("--webcam", type=int, nargs="?", const=0, default=None,
                        help="Run on live optical webcam feed (optional device index, default: 0)")
    parser.add_argument("--video", type=str, default=None,
                        help="Run on recorded optical beacon video file")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load configuration
    if args.config and os.path.exists(args.config):
        print(f"Loading custom configuration from: {args.config}")
        cfg = AppConfig.load_json(args.config)
    else:
        cfg = AppConfig()

    # Determine input feed source
    source_type = "simulation"
    video_source = None
    if args.webcam is not None:
        source_type = "webcam"
        video_source = args.webcam
    elif args.video is not None:
        source_type = "video"
        video_source = args.video

    # 1. Benchmark Mode
    if args.benchmark:
        print("Starting Automated Benchmark Suite Mode...")
        suite = BenchmarkSuite(output_dir=args.output_dir)
        report_file = suite.run_full_suite()
        print(f"\nBenchmark suite completed successfully!\nReport available at: {report_file}")
        sys.exit(0)

    # 2. Headless Mode
    elif args.headless:
        if source_type in ["webcam", "video"]:
            print(f"Running Headless Processing on {source_type} ({video_source})...")
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print(f"Error: Unable to open {source_type} stream: {video_source}")
                sys.exit(1)

            from core.geometry import CameraGeometry
            from perception.hybrid_detector import HybridBeaconDetector
            from tracking.kalman_filter import TargetKalmanFilter
            from tracking.state_machine import TrackStateMachine
            from tracking.multi_target import MultiTargetAssociator
            from perception.gating import MahalanobisGater
            import numpy as np

            geom = CameraGeometry(cfg.optical.image_width, cfg.optical.image_height,
                                  cfg.optical.fov_horizontal_deg, cfg.optical.fov_vertical_deg)
            detector = HybridBeaconDetector(cfg.detector)
            kalman = TargetKalmanFilter(cfg.kalman)
            sm = TrackStateMachine(cfg.state_machine, kalman)
            gater = MahalanobisGater(cfg.detector.mahalanobis_gate_threshold)
            associator = MultiTargetAssociator(gater)

            frame_idx = 0
            t_start = time.perf_counter()
            dt = 1.0 / cfg.target_fps
            lock_frames = 0
            cam_pan = 0.0
            cam_tilt = 0.0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_idx += 1
                if frame.shape[1] != cfg.optical.image_width or frame.shape[0] != cfg.optical.image_height:
                    frame = cv2.resize(frame, (cfg.optical.image_width, cfg.optical.image_height))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                candidates = detector.detect(gray)
                for cand in candidates:
                    pan_off, tilt_off = geom.pixel_to_angle_deg(cand["cx"], cand["cy"])
                    cand["inertial_pan_deg"] = cam_pan + pan_off
                    cand["inertial_tilt_deg"] = cam_tilt + tilt_off

                pred_inertial = np.array(kalman.position_deg, dtype=np.float64)
                is_active_track = (sm.state.value in ["TRACK", "COAST"])
                selected = associator.select_best_candidate(
                    candidates, pred_inertial, kalman.get_innovation_covariance(),
                    center_pixel=(geom.cx, geom.cy),
                    is_tracking=is_active_track
                )
                kalman.predict(dt)
                state, _ = sm.step(selected, dt)
                if state.value == "TRACK":
                    lock_frames += 1

            cap.release()
            total_time = time.perf_counter() - t_start
            fps = frame_idx / max(1e-3, total_time)
            lock_rate = (lock_frames / max(1, frame_idx)) * 100.0
            print(f"Processed {frame_idx} frames in {total_time:.2f}s ({fps:.1f} FPS). Lock Retention: {lock_rate:.1f}%")
            sys.exit(0)

        print(f"Running Headless Simulation Mode for {args.duration:.1f} seconds...")
        suite = BenchmarkSuite(output_dir=args.output_dir)
        summary, _ = suite.run_single_scenario(cfg, duration_sec=args.duration)
        print("\n--- Headless Simulation Completed ---")
        print(f"  • Mean FPS            : {summary['fps']:.1f}")
        print(f"  • Frame Latency       : {summary['frame_latency_ms']:.2f} ms")
        print(f"  • Acquisition Time    : {summary['acquisition_time_sec']:.3f} s")
        print(f"  • Lock Retention Rate : {summary['lock_retention_rate_pct']:.1f} %")
        print(f"  • RMSE Tracking Error : {summary['rmse_error_px']:.2f} px ({summary['rmse_error_mrad']:.3f} mrad)")
        sys.exit(0)

    # 3. Interactive Desktop GUI Mode (Default)
    else:
        print(f"Launching SIH26169 Mission Control Desktop GUI (Source: {source_type})...")
        launch_gui(cfg, source_type=source_type, video_source=video_source)


if __name__ == "__main__":
    main()
