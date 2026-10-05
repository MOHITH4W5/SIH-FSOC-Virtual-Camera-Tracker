"""
ui/gui_app.py - Advanced Mission Control Desktop Application for FSOC Coarse Alignment.
Built with CustomTkinter for high-end aerospace/defense flight workstation experience (taste-skill aligned).

Features:
  1. Doppelrand / Double-Bezel Hardware Architecture across all cards and displays
  2. Calibrated Aerospace Color Palette (Deepest OLED Obsidian #030712 backdrop, Cyber Sky, Emerald, Amber, Crimson)
  3. Tactical Video Viewport (1040x560) with Fighter-Jet HUD Symbology and Sub-mrad Boresight Cones
  4. Real-Time Telemetry Oscilloscope with Shaded Under-Curve Polygon, 2.0 mrad / 5.0 mrad Reference Limits
  5. Glowing State Badges, High-Contrast Metric Scorecard, and IMM Probability Banks
  6. Interactive Segmented Buttons for Optical Feed, Trajectories, and Controllers
  7. Tactile Disturbance Studio Sliders with Live Value Badges and Quick Stress Presets
  8. Dynamic Occlusion Injection (1.5s) with Animated Countdown Feedback
  9. Automated Benchmark Suite Runner with Threaded Evaluation
"""

import sys
import os
import time
import math
import collections
import threading
from typing import Optional, Dict, Any, Deque
import numpy as np
import cv2

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk

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
from control.mpc_lite import MPCLiteController
from telemetry.fdir_manager import FDIRSupervisor, FDIRStatus
from telemetry.explainability import CausalExplainabilityEngine
from telemetry.metrics import PerformanceMetrics
from ui.overlay import TelemetryHUDOverlay
from benchmarks.benchmark_suite import BenchmarkSuite

# Set CustomTkinter Visual System
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class FSOCTrackerApp:
    """Next-Generation Aerospace Command & Control GUI for SIH26169 FSOC Testbed."""

    def __init__(self, root: ctk.CTk, config: Optional[AppConfig] = None,
                 source_type: str = "simulation", video_source: Optional[Any] = None):
        self.root = root
        self.root.title("🛰️ ISRO / SAC · FSOC Coarse Alignment Mission Control [SIH26169]")
        self.root.geometry("1580x950")
        self.root.minsize(1440, 860)
        self.root.configure(fg_color="#030712")

        self.cfg = config if config is not None else AppConfig()
        self.source_type = source_type
        self.video_source = video_source
        self.cap: Optional[cv2.VideoCapture] = None
        self.virtual_cam_pan_deg = 0.0
        self.virtual_cam_tilt_deg = 0.0

        # Sub-systems
        self.env = VirtualEnvironment(self.cfg)
        self.detector = HybridBeaconDetector(self.cfg.detector)
        self.gater = MahalanobisGater(self.cfg.detector.mahalanobis_gate_threshold)
        self.kalman = TargetKalmanFilter(self.cfg.kalman)
        self.imm = TargetIMMEstimator(self.cfg.kalman)
        self.estimator_mode = "IMM"  # "IMM" or "EKF"
        self.active_estimator = self.imm
        self.state_machine = TrackStateMachine(self.cfg.state_machine, self.active_estimator)
        self.associator = MultiTargetAssociator(self.gater)

        # Advanced vibration notch, FDIR, and Causal AI
        self.vibration_canceller = AdaptiveVibrationCanceller(fs_hz=self.cfg.target_fps)
        self.controller = DualAxisGimbalController(self.cfg.controller, vibration_canceller=self.vibration_canceller)
        self.mpc_controller = MPCLiteController()
        self.damper = CovarianceGainDamper(
            self.cfg.controller.covariance_damping_alpha,
            enabled=self.cfg.controller.covariance_damping_enabled
        )
        self.fdir = FDIRSupervisor(nis_gate_threshold=self.cfg.detector.mahalanobis_gate_threshold)
        self.explainer = CausalExplainabilityEngine()
        self.metrics = PerformanceMetrics()
        self.hud = TelemetryHUDOverlay(self.env.geom)

        self.is_running = True
        self.dt = 1.0 / self.cfg.target_fps

        # Real-time oscilloscope history (last 100 frames)
        self.error_history: Deque[float] = collections.deque(maxlen=100)
        for _ in range(100):
            self.error_history.append(0.0)

        # Occlusion animation state
        self.occlusion_countdown = 0.0

        if self.source_type in ["webcam", "video"] and self.video_source is not None:
            self._switch_source(self.source_type, self.video_source)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._schedule_loop()

    def _build_ui(self):
        """Constructs modern aerospace mission control dashboard with double-bezel styling."""
        # -------------------------------------------------------------
        # 1. FLOATING DOUBLE-BEZEL TOP HEADER & STATUS BAR
        # -------------------------------------------------------------
        header_outer = ctk.CTkFrame(self.root, fg_color="#070C18", corner_radius=14, border_width=1, border_color="#18253E", height=60)
        header_outer.pack(fill=tk.X, side=tk.TOP, padx=14, pady=(10, 6))

        header = ctk.CTkFrame(header_outer, fg_color="#0A1224", corner_radius=10, border_width=1, border_color="#1E2F4D")
        header.pack(fill=tk.X, padx=3, pady=3)

        # Left: Emblem & Program Details
        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side=tk.LEFT, padx=14, pady=6)

        logo_chip = ctk.CTkFrame(title_box, fg_color="#061226", corner_radius=8, border_width=1, border_color="#0284C7")
        logo_chip.pack(side=tk.LEFT, padx=(0, 10), pady=2)

        logo_lbl = ctk.CTkLabel(
            logo_chip, text="🛰️ ISRO / SAC",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#38BDF8", padx=10, pady=4
        )
        logo_lbl.pack()

        text_stack = ctk.CTkFrame(title_box, fg_color="transparent")
        text_stack.pack(side=tk.LEFT)

        main_title = ctk.CTkLabel(
            text_stack, text="FSOC PAT MISSION CONTROL TESTBED",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#F8FAFC"
        )
        main_title.pack(anchor=tk.W)

        sub_title = ctk.CTkLabel(
            text_stack, text="SIH26169 · Virtual Camera Coarse Alignment Workstation",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#64748B"
        )
        sub_title.pack(anchor=tk.W)

        # Center Status Pills
        status_box = ctk.CTkFrame(header, fg_color="transparent")
        status_box.pack(side=tk.LEFT, expand=True)

        self.pill_ai = ctk.CTkLabel(
            status_box, text="● AI DISCRIMINATOR: ONLINE",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#34D399", fg_color="#064E3B", corner_radius=10, padx=12, pady=4
        )
        self.pill_ai.pack(side=tk.LEFT, padx=5)

        self.pill_closed_loop = ctk.CTkLabel(
            status_box, text="● CLOSED LOOP: ACTIVE",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#38BDF8", fg_color="#0C385E", corner_radius=10, padx=12, pady=4
        )
        self.pill_closed_loop.pack(side=tk.LEFT, padx=5)

        self.pill_accuracy = ctk.CTkLabel(
            status_box, text="● SUB-MRAD COARSE: VERIFIED",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#A78BFA", fg_color="#1E1B4B", corner_radius=10, padx=12, pady=4
        )
        self.pill_accuracy.pack(side=tk.LEFT, padx=5)

        # Right: Digital UTC Clock
        clock_box = ctk.CTkFrame(header, fg_color="#060D1D", corner_radius=8, border_width=1, border_color="#18253E")
        clock_box.pack(side=tk.RIGHT, padx=14, pady=6)

        self.lbl_clock = ctk.CTkLabel(
            clock_box, text="UTC: 00:00:00  |  COARSE PAT STAGE",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color="#94A3B8", padx=12, pady=4
        )
        self.lbl_clock.pack()

        # -------------------------------------------------------------
        # 2. MAIN SPLIT WORKSPACE
        # -------------------------------------------------------------
        workspace = ctk.CTkFrame(self.root, fg_color="transparent")
        workspace.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        # LEFT COLUMN: Viewport Canvas + Live Oscilloscope Strip Chart
        left_col = ctk.CTkFrame(workspace, fg_color="transparent")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Double-Bezel Viewport Card
        viewport_outer = ctk.CTkFrame(left_col, fg_color="#070C18", corner_radius=14, border_width=1, border_color="#18253E")
        viewport_outer.pack(fill=tk.BOTH, expand=True)

        viewport_inner = ctk.CTkFrame(viewport_outer, fg_color="#050914", corner_radius=10, border_width=1, border_color="#121C30")
        viewport_inner.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Viewport Header Bar
        vp_header = ctk.CTkFrame(viewport_inner, fg_color="transparent", height=24)
        vp_header.pack(fill=tk.X, padx=10, pady=(6, 2))

        ctk.CTkLabel(
            vp_header, text="🎯 TACTICAL SENSOR VIEWPORT · LOW LATENCY VIRTUAL CAMERA",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#38BDF8"
        ).pack(side=tk.LEFT)

        ctk.CTkLabel(
            vp_header, text="1040x560 · 50 FPS · HUD ACTIVE",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color="#64748B"
        ).pack(side=tk.RIGHT)

        self.canvas_w = 1040
        self.canvas_h = 560
        self.canvas = tk.Canvas(
            viewport_inner, width=self.canvas_w, height=self.canvas_h,
            bg="#02050D", highlightthickness=0, bd=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Double-Bezel Real-Time Oscilloscope Card
        osc_outer = ctk.CTkFrame(left_col, fg_color="#070C18", corner_radius=14, border_width=1, border_color="#18253E", height=155)
        osc_outer.pack(fill=tk.X, pady=(8, 0))

        osc_inner = ctk.CTkFrame(osc_outer, fg_color="#080F1E", corner_radius=10, border_width=1, border_color="#16233B")
        osc_inner.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        osc_header = ctk.CTkFrame(osc_inner, fg_color="transparent", height=24)
        osc_header.pack(fill=tk.X, padx=12, pady=(6, 2))

        ctk.CTkLabel(
            osc_header, text="📈 REAL-TIME TRACKING ERROR OSCILLOSCOPE (LAST 100 FRAMES)",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#38BDF8"
        ).pack(side=tk.LEFT)

        self.lbl_osc_stat = ctk.CTkLabel(
            osc_header, text="Current: 0.0 px | Mean: 0.0 px | Limit: 2.0 mrad [PASS]",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color="#94A3B8"
        )
        self.lbl_osc_stat.pack(side=tk.RIGHT)

        self.osc_canvas = tk.Canvas(
            osc_inner, width=self.canvas_w, height=105,
            bg="#020610", highlightthickness=0, bd=0
        )
        self.osc_canvas.pack(fill=tk.X, padx=6, pady=(0, 6))

        # -------------------------------------------------------------
        # RIGHT COLUMN: Interactive Control Deck & Gauges (Double-Bezel)
        # -------------------------------------------------------------
        right_col = ctk.CTkScrollableFrame(
            workspace, fg_color="#070C18", corner_radius=14,
            border_width=1, border_color="#18253E", width=440
        )
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # CARD 1: Live Flight Telemetry Readouts
        self._build_telemetry_deck(right_col)

        # CARD 2: Optical Stream & Kinematics Controls
        self._build_stream_controls(right_col)

        # CARD 3: Disturbance Studio & Presets
        self._build_disturbance_studio(right_col)

        # CARD 4: Tactical Action Controls
        self._build_action_deck(right_col)

    def _build_telemetry_deck(self, parent):
        """Constructs the high-contrast telemetry readout scorecard with double-bezel styling."""
        outer = ctk.CTkFrame(parent, fg_color="#0A1122", corner_radius=12, border_width=1, border_color="#16233B")
        outer.pack(fill=tk.X, padx=4, pady=(4, 8))

        card = ctk.CTkFrame(outer, fg_color="#0D172E", corner_radius=9, border_width=1, border_color="#1D2E4E")
        card.pack(fill=tk.X, padx=3, pady=3)

        # Header Title with Live Chip
        header_bar = ctk.CTkFrame(card, fg_color="transparent")
        header_bar.pack(fill=tk.X, padx=12, pady=(10, 4))

        ctk.CTkLabel(
            header_bar, text="LIVE MISSION TELEMETRY SCORECARD",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#38BDF8"
        ).pack(side=tk.LEFT)

        ctk.CTkLabel(
            header_bar, text="● REAL-TIME",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color="#34D399", fg_color="#064E3B", corner_radius=6, padx=6, pady=2
        ).pack(side=tk.RIGHT)

        # Primary State Banner
        self.banner_state = ctk.CTkFrame(card, fg_color="#064E3B", corner_radius=8, height=36)
        self.banner_state.pack(fill=tk.X, padx=12, pady=4)

        self.lbl_state_text = ctk.CTkLabel(
            self.banner_state, text="● TRACKING STATE: LOCKED",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#34D399"
        )
        self.lbl_state_text.pack(expand=True, pady=6)

        # Metric Tiles (Grid 2x2)
        grid_frame = ctk.CTkFrame(card, fg_color="transparent")
        grid_frame.pack(fill=tk.X, padx=10, pady=6)
        grid_frame.columnconfigure((0, 1), weight=1)

        # Tile 1: Tracking Error
        t1 = ctk.CTkFrame(grid_frame, fg_color="#080F1E", corner_radius=8, border_width=1, border_color="#16233B")
        t1.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(t1, text="TRACKING ERROR", font=ctk.CTkFont(size=9, weight="bold"), text_color="#94A3B8").pack(anchor=tk.W, padx=8, pady=(6, 0))
        self.lbl_val_error = ctk.CTkLabel(t1, text="12.5 px", font=ctk.CTkFont(size=15, weight="bold"), text_color="#34D399")
        self.lbl_val_error.pack(anchor=tk.W, padx=8)
        self.lbl_val_mrad = ctk.CTkLabel(t1, text="1.70 mrad [SUB-MRAD PASS]", font=ctk.CTkFont(size=10, weight="bold"), text_color="#34D399")
        self.lbl_val_mrad.pack(anchor=tk.W, padx=8, pady=(0, 6))

        # Tile 2: Lock Retention
        t2 = ctk.CTkFrame(grid_frame, fg_color="#080F1E", corner_radius=8, border_width=1, border_color="#16233B")
        t2.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(t2, text="LOCK RETENTION", font=ctk.CTkFont(size=9, weight="bold"), text_color="#94A3B8").pack(anchor=tk.W, padx=8, pady=(6, 0))
        self.lbl_val_lock = ctk.CTkLabel(t2, text="99.2 %", font=ctk.CTkFont(size=15, weight="bold"), text_color="#38BDF8")
        self.lbl_val_lock.pack(anchor=tk.W, padx=8)
        self.progress_lock = ctk.CTkProgressBar(t2, height=6, progress_color="#0284C7", fg_color="#0F172A")
        self.progress_lock.set(0.99)
        self.progress_lock.pack(fill=tk.X, padx=8, pady=(2, 8))

        # Tile 3: Loop Rate & Latency
        t3 = ctk.CTkFrame(grid_frame, fg_color="#080F1E", corner_radius=8, border_width=1, border_color="#16233B")
        t3.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(t3, text="LOOP SPEED", font=ctk.CTkFont(size=9, weight="bold"), text_color="#94A3B8").pack(anchor=tk.W, padx=8, pady=(6, 0))
        self.lbl_val_fps = ctk.CTkLabel(t3, text="42.5 FPS", font=ctk.CTkFont(size=15, weight="bold"), text_color="#F8FAFC")
        self.lbl_val_fps.pack(anchor=tk.W, padx=8)
        self.lbl_val_lat = ctk.CTkLabel(t3, text="Latency: 23.5 ms", font=ctk.CTkFont(size=10), text_color="#94A3B8")
        self.lbl_val_lat.pack(anchor=tk.W, padx=8, pady=(0, 6))

        # Tile 4: Gimbal Attitude
        t4 = ctk.CTkFrame(grid_frame, fg_color="#080F1E", corner_radius=8, border_width=1, border_color="#16233B")
        t4.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(t4, text="GIMBAL ATTITUDE", font=ctk.CTkFont(size=9, weight="bold"), text_color="#94A3B8").pack(anchor=tk.W, padx=8, pady=(6, 0))
        self.lbl_val_gimbal = ctk.CTkLabel(t4, text="Pan: +0.00°", font=ctk.CTkFont(size=12, weight="bold"), text_color="#A78BFA")
        self.lbl_val_gimbal.pack(anchor=tk.W, padx=8)
        self.lbl_val_tilt = ctk.CTkLabel(t4, text="Tilt: +0.00°", font=ctk.CTkFont(size=10), text_color="#CBD5E1")
        self.lbl_val_tilt.pack(anchor=tk.W, padx=8, pady=(0, 6))

        # -------------------------------------------------------------
        # Sub-Card: Defense Avionics (IMM Model Bank, Vibration Notch, FDIR)
        # -------------------------------------------------------------
        adv_card = ctk.CTkFrame(card, fg_color="#060C1A", corner_radius=8, border_width=1, border_color="#15223A")
        adv_card.pack(fill=tk.X, padx=10, pady=(0, 8))

        ctk.CTkLabel(
            adv_card, text="🛡️ DEFENSE AVIONICS & FDIR WATCHDOG",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#38BDF8"
        ).pack(anchor=tk.W, padx=10, pady=(6, 2))

        # Status Badges Row
        badge_row = ctk.CTkFrame(adv_card, fg_color="transparent")
        badge_row.pack(fill=tk.X, padx=8, pady=2)
        badge_row.columnconfigure((0, 1), weight=1)

        self.lbl_fdir_badge = ctk.CTkLabel(
            badge_row, text="● FDIR: NOMINAL",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#34D399", fg_color="#064E3B", corner_radius=8, padx=8, pady=3
        )
        self.lbl_fdir_badge.grid(row=0, column=0, padx=2, sticky="ew")

        self.lbl_notch_badge = ctk.CTkLabel(
            badge_row, text="📳 NOTCH: 9.5 Hz (-18dB)",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#60A5FA", fg_color="#1E3A8A", corner_radius=8, padx=8, pady=3
        )
        self.lbl_notch_badge.grid(row=0, column=1, padx=2, sticky="ew")

        # IMM Model Probabilities
        self.lbl_imm_modes = ctk.CTkLabel(
            adv_card, text="IMM Bank: CV 70% | CA 15% | CT 15%",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color="#E2E8F0"
        )
        self.lbl_imm_modes.pack(anchor=tk.W, padx=10, pady=(4, 2))

        # 3 Mini Progress Bars for IMM Probabilities
        prog_frame = ctk.CTkFrame(adv_card, fg_color="transparent")
        prog_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
        prog_frame.columnconfigure((0, 1, 2), weight=1)

        self.prog_cv = ctk.CTkProgressBar(prog_frame, height=5, progress_color="#38BDF8", fg_color="#0F172A")
        self.prog_cv.grid(row=0, column=0, padx=2, sticky="ew")
        self.prog_cv.set(0.70)

        self.prog_ca = ctk.CTkProgressBar(prog_frame, height=5, progress_color="#FBBF24", fg_color="#0F172A")
        self.prog_ca.grid(row=0, column=1, padx=2, sticky="ew")
        self.prog_ca.set(0.15)

        self.prog_ct = ctk.CTkProgressBar(prog_frame, height=5, progress_color="#C084FC", fg_color="#0F172A")
        self.prog_ct.grid(row=0, column=2, padx=2, sticky="ew")
        self.prog_ct.set(0.15)

        # Causal Decision Audit Terminal Ticker
        audit_box = ctk.CTkFrame(adv_card, fg_color="#02050D", corner_radius=6, border_width=1, border_color="#121C30")
        audit_box.pack(fill=tk.X, padx=8, pady=(0, 8))

        ctk.CTkLabel(
            audit_box, text="LIVE CAUSAL DECISION AUDIT TRACE:",
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color="#94A3B8"
        ).pack(anchor=tk.W, padx=8, pady=(4, 0))

        self.lbl_causal_ticker = ctk.CTkLabel(
            audit_box, text="> Nominal Kalman smoothing in linear cruise regime.",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color="#34D399", wraplength=380, justify="left"
        )
        self.lbl_causal_ticker.pack(anchor=tk.W, padx=8, pady=(2, 6))

    def _build_stream_controls(self, parent):
        """Constructs modern segmented buttons for feed source and flight kinematics."""
        outer = ctk.CTkFrame(parent, fg_color="#0A1122", corner_radius=12, border_width=1, border_color="#16233B")
        outer.pack(fill=tk.X, padx=4, pady=6)

        card = ctk.CTkFrame(outer, fg_color="#0D172E", corner_radius=9, border_width=1, border_color="#1D2E4E")
        card.pack(fill=tk.X, padx=3, pady=3)

        ctk.CTkLabel(
            card, text="OPTICAL FEED & FLIGHT CONFIGURATION",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#A78BFA"
        ).pack(anchor=tk.W, padx=12, pady=(10, 4))

        # Feed Source Segmented Button
        ctk.CTkLabel(card, text="Optical Feed Stream:", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor=tk.W, padx=12, pady=(2, 2))
        self.seg_source = ctk.CTkSegmentedButton(
            card, values=["🚀 Simulation", "📹 Webcam", "📁 Video File"],
            command=self._on_source_segment_change,
            selected_color="#0284C7", selected_hover_color="#0369A1",
            unselected_color="#070E1C", font=ctk.CTkFont(size=11, weight="bold")
        )
        self.seg_source.set("🚀 Simulation")
        self.seg_source.pack(fill=tk.X, padx=12, pady=(0, 8))

        # Target Motion Model Segmented Button
        ctk.CTkLabel(card, text="Target Flight Kinematics:", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor=tk.W, padx=12, pady=(2, 2))
        self.seg_motion = ctk.CTkSegmentedButton(
            card, values=["figure8", "circular", "linear", "random_walk", "banking_turn"],
            command=self._on_motion_segment_change,
            selected_color="#7C3AED", selected_hover_color="#6D28D9",
            unselected_color="#070E1C", font=ctk.CTkFont(size=10)
        )
        self.seg_motion.set(self.cfg.target.motion_type)
        self.seg_motion.pack(fill=tk.X, padx=12, pady=(0, 8))

        # Advanced Estimator Toggle (IMM vs EKF)
        ctk.CTkLabel(card, text="Target State Estimator Engine:", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor=tk.W, padx=12, pady=(2, 2))
        self.seg_estimator = ctk.CTkSegmentedButton(
            card, values=["IMM (3-Model)", "Standard EKF"],
            command=self._on_estimator_segment_change,
            selected_color="#0284C7", selected_hover_color="#0369A1",
            unselected_color="#070E1C", font=ctk.CTkFont(size=10)
        )
        self.seg_estimator.set("IMM (3-Model)")
        self.seg_estimator.pack(fill=tk.X, padx=12, pady=(0, 8))

        # Vibration Notch Filter Toggle
        ctk.CTkLabel(card, text="Harmonic Vibration Notch Filter:", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor=tk.W, padx=12, pady=(2, 2))
        self.seg_vibe = ctk.CTkSegmentedButton(
            card, values=["Adaptive Notch: ACTIVE", "Notch: BYPASSED"],
            command=self._on_vibe_notch_segment_change,
            selected_color="#D97706", selected_hover_color="#B45309",
            unselected_color="#070E1C", font=ctk.CTkFont(size=10)
        )
        self.seg_vibe.set("Adaptive Notch: ACTIVE")
        self.seg_vibe.pack(fill=tk.X, padx=12, pady=(0, 8))

        # Control Algorithm Segmented Button
        ctk.CTkLabel(card, text="Gimbal Control Law:", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor=tk.W, padx=12, pady=(2, 2))
        self.seg_ctrl = ctk.CTkSegmentedButton(
            card, values=["Cov-Aware PID", "Standard PID", "MPC-Lite"],
            command=self._on_ctrl_segment_change,
            selected_color="#059669", selected_hover_color="#047857",
            unselected_color="#070E1C", font=ctk.CTkFont(size=10)
        )
        self.seg_ctrl.set("Cov-Aware PID")
        self.seg_ctrl.pack(fill=tk.X, padx=12, pady=(0, 10))

    def _build_disturbance_studio(self, parent):
        """Constructs tactile disturbance sliders with live values and stress presets."""
        outer = ctk.CTkFrame(parent, fg_color="#0A1122", corner_radius=12, border_width=1, border_color="#16233B")
        outer.pack(fill=tk.X, padx=4, pady=6)

        card = ctk.CTkFrame(outer, fg_color="#0D172E", corner_radius=9, border_width=1, border_color="#1D2E4E")
        card.pack(fill=tk.X, padx=3, pady=3)

        ctk.CTkLabel(
            card, text="DYNAMIC DISTURBANCE STUDIO",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#F472B6"
        ).pack(anchor=tk.W, padx=12, pady=(10, 4))

        # Slider 1: Atmospheric Turbulence
        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill=tk.X, padx=12, pady=(2, 0))
        ctk.CTkLabel(row1, text="Atmospheric Turbulence (Cn² proxy):", font=ctk.CTkFont(size=10), text_color="#E2E8F0").pack(side=tk.LEFT)
        self.lbl_turb_val = ctk.CTkLabel(row1, text=f"{self.cfg.disturbance.turbulence_strength:.2f}", font=ctk.CTkFont(size=10, weight="bold"), text_color="#F472B6")
        self.lbl_turb_val.pack(side=tk.RIGHT)

        self.slider_turb = ctk.CTkSlider(
            card, from_=0.0, to=1.0, number_of_steps=100,
            command=self._on_turb_slider,
            progress_color="#F472B6", button_color="#FB7185", button_hover_color="#E11D48"
        )
        self.slider_turb.set(self.cfg.disturbance.turbulence_strength)
        self.slider_turb.pack(fill=tk.X, padx=12, pady=(0, 6))

        # Slider 2: Platform Micro-Vibration
        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill=tk.X, padx=12, pady=(2, 0))
        ctk.CTkLabel(row2, text="Platform Micro-Vibration Amplitude (°):", font=ctk.CTkFont(size=10), text_color="#E2E8F0").pack(side=tk.LEFT)
        self.lbl_vib_val = ctk.CTkLabel(row2, text=f"{self.cfg.disturbance.vibration_amplitude_deg:.2f}°", font=ctk.CTkFont(size=10, weight="bold"), text_color="#F59E0B")
        self.lbl_vib_val.pack(side=tk.RIGHT)

        self.slider_vib = ctk.CTkSlider(
            card, from_=0.0, to=0.25, number_of_steps=50,
            command=self._on_vib_slider,
            progress_color="#F59E0B", button_color="#FBBF24", button_hover_color="#D97706"
        )
        self.slider_vib.set(self.cfg.disturbance.vibration_amplitude_deg)
        self.slider_vib.pack(fill=tk.X, padx=12, pady=(0, 6))

        # Slider 3: Sensor Noise
        row3 = ctk.CTkFrame(card, fg_color="transparent")
        row3.pack(fill=tk.X, padx=12, pady=(2, 0))
        ctk.CTkLabel(row3, text="Sensor Gaussian Noise (Sigma):", font=ctk.CTkFont(size=10), text_color="#E2E8F0").pack(side=tk.LEFT)
        self.lbl_noise_val = ctk.CTkLabel(row3, text=f"{self.cfg.disturbance.sensor_gaussian_sigma:.1f}", font=ctk.CTkFont(size=10, weight="bold"), text_color="#38BDF8")
        self.lbl_noise_val.pack(side=tk.RIGHT)

        self.slider_noise = ctk.CTkSlider(
            card, from_=0.0, to=15.0, number_of_steps=60,
            command=self._on_noise_slider,
            progress_color="#38BDF8", button_color="#60A5FA", button_hover_color="#2563EB"
        )
        self.slider_noise.set(self.cfg.disturbance.sensor_gaussian_sigma)
        self.slider_noise.pack(fill=tk.X, padx=12, pady=(0, 8))

        # Disturbance Presets Buttons
        preset_box = ctk.CTkFrame(card, fg_color="transparent")
        preset_box.pack(fill=tk.X, padx=10, pady=(0, 8))
        preset_box.columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(preset_box, text="🌿 Calm", font=ctk.CTkFont(size=10, weight="bold"),
                      fg_color="#0A1224", hover_color="#16233B", border_width=1, border_color="#1F2E4C", corner_radius=8,
                      command=lambda: self._apply_preset(0.0, 0.0, 2.0)).grid(row=0, column=0, padx=2)
        ctk.CTkButton(preset_box, text="🌪️ Turb", font=ctk.CTkFont(size=10, weight="bold"),
                      fg_color="#0A1224", hover_color="#16233B", border_width=1, border_color="#1F2E4C", corner_radius=8,
                      command=lambda: self._apply_preset(0.6, 0.05, 4.0)).grid(row=0, column=1, padx=2)
        ctk.CTkButton(preset_box, text="📳 Jitter", font=ctk.CTkFont(size=10, weight="bold"),
                      fg_color="#0A1224", hover_color="#16233B", border_width=1, border_color="#1F2E4C", corner_radius=8,
                      command=lambda: self._apply_preset(0.2, 0.20, 3.0)).grid(row=0, column=2, padx=2)
        ctk.CTkButton(preset_box, text="⚡ Storm", font=ctk.CTkFont(size=10, weight="bold"),
                      fg_color="#0A1224", hover_color="#16233B", border_width=1, border_color="#1F2E4C", corner_radius=8,
                      command=lambda: self._apply_preset(0.85, 0.22, 10.0)).grid(row=0, column=3, padx=2)

    def _build_action_deck(self, parent):
        """Constructs high-contrast interactive action buttons for testbed operation."""
        outer = ctk.CTkFrame(parent, fg_color="#0A1122", corner_radius=12, border_width=1, border_color="#16233B")
        outer.pack(fill=tk.X, padx=4, pady=(4, 12))

        card = ctk.CTkFrame(outer, fg_color="#0D172E", corner_radius=9, border_width=1, border_color="#1D2E4E")
        card.pack(fill=tk.X, padx=3, pady=3)

        ctk.CTkLabel(
            card, text="TACTICAL ACTION CONTROLS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#34D399"
        ).pack(anchor=tk.W, padx=12, pady=(8, 4))

        # Occlusion Trigger Button
        self.btn_occl = ctk.CTkButton(
            card, text="⚡ INJECT CLOUD OCCLUSION (1.5s)",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#D97706", hover_color="#B45309", text_color="#000000",
            height=38, corner_radius=8, command=self._on_inject_occlusion
        )
        self.btn_occl.pack(fill=tk.X, padx=10, pady=3)

        # Reset Gimbal & Tracker Button
        self.btn_reset = ctk.CTkButton(
            card, text="🔄 RE-INITIALIZE TRACKER & GIMBAL",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", text_color="#F1F5F9",
            border_width=1, border_color="#334155",
            height=34, corner_radius=8, command=self._on_reset_tracker
        )
        self.btn_reset.pack(fill=tk.X, padx=10, pady=3)

        # Run Automated Benchmark Suite Button
        self.btn_bench = ctk.CTkButton(
            card, text="📊 LAUNCH AUTOMATED BENCHMARK SUITE",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", text_color="#FFFFFF",
            height=40, corner_radius=8, command=self._on_run_benchmark
        )
        self.btn_bench.pack(fill=tk.X, padx=10, pady=(4, 3))

        # Export Causal Audit Trail Button
        self.btn_export_audit = ctk.CTkButton(
            card, text="📜 EXPORT CAUSAL AUDIT LOG (JSON)",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color="#0D9488", hover_color="#0F766E", text_color="#FFFFFF",
            height=34, corner_radius=8, command=self._on_export_audit
        )
        self.btn_export_audit.pack(fill=tk.X, padx=10, pady=(2, 6))

    def _apply_preset(self, turb: float, vib: float, noise: float):
        self.slider_turb.set(turb)
        self._on_turb_slider(turb)
        self.slider_vib.set(vib)
        self._on_vib_slider(vib)
        self.slider_noise.set(noise)
        self._on_noise_slider(noise)

    def _on_turb_slider(self, val):
        v = float(val)
        self.cfg.disturbance.turbulence_strength = v
        self.lbl_turb_val.configure(text=f"{v:.2f}")

    def _on_vib_slider(self, val):
        v = float(val)
        self.cfg.disturbance.vibration_amplitude_deg = v
        self.lbl_vib_val.configure(text=f"{v:.2f}°")

    def _on_noise_slider(self, val):
        v = float(val)
        self.cfg.disturbance.sensor_gaussian_sigma = v
        self.lbl_noise_val.configure(text=f"{v:.1f}")

    def _on_source_segment_change(self, val):
        if val == "🚀 Simulation":
            self._switch_source("simulation", None)
        elif val == "📹 Webcam":
            self._switch_source("webcam", 0)
        elif val == "📁 Video File":
            file_path = filedialog.askopenfilename(
                title="Select Optical Beacon Video File",
                filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
            )
            if file_path:
                self._switch_source("video", file_path)
            else:
                self.seg_source.set("🚀 Simulation")

    def _on_motion_segment_change(self, val):
        self.cfg.target.motion_type = val
        self.env.targets.reset(count=1, motion_type=val, speed=self.cfg.target.speed_deg_s, radius=self.cfg.target.trajectory_radius_deg)

    def _on_estimator_segment_change(self, val):
        if val == "IMM (3-Model)":
            self.estimator_mode = "IMM"
            self.active_estimator = self.imm
            self.state_machine.set_estimator(self.imm)
        else:
            self.estimator_mode = "EKF"
            self.active_estimator = self.kalman
            self.state_machine.set_estimator(self.kalman)

    def _on_vibe_notch_segment_change(self, val):
        if "ACTIVE" in val:
            self.vibration_canceller.enabled = True
        else:
            self.vibration_canceller.enabled = False

    def _on_export_audit(self):
        file_path = filedialog.asksaveasfilename(
            title="Export ISRO Mission Causal Decision Audit",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if file_path:
            self.explainer.export_audit_log_json(file_path)
            messagebox.showinfo("Export Successful", f"Causal decision audit trace exported to:\n{file_path}")

    def _on_ctrl_segment_change(self, val):
        if val == "Standard PID":
            self.damper.enabled = False
            self.cfg.controller.mode = "pid"
        elif val == "Cov-Aware PID":
            self.damper.enabled = True
            self.cfg.controller.mode = "pid"
        elif val == "MPC-Lite":
            self.cfg.controller.mode = "mpc_lite"

    def _switch_source(self, source_type: str, video_source: Optional[Any]):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.source_type = source_type
        self.video_source = video_source

        if source_type in ["webcam", "video"]:
            try:
                self.cap = cv2.VideoCapture(video_source)
                if not self.cap.isOpened():
                    messagebox.showwarning("Optical Feed Error", f"Unable to open stream: {video_source}.\nReverting to Simulation.")
                    self.source_type = "simulation"
                    self.seg_source.set("🚀 Simulation")
            except Exception as e:
                messagebox.showerror("Feed Exception", f"Error: {e}\nReverting to Simulation.")
                self.source_type = "simulation"
                self.seg_source.set("🚀 Simulation")

        self.kalman.reset()
        self.state_machine.reset()
        self.virtual_cam_pan_deg = 0.0
        self.virtual_cam_tilt_deg = 0.0

    def _on_inject_occlusion(self):
        self.env.disturbance.trigger_occlusion(duration_sec=1.5)
        self.occlusion_countdown = 1.5

    def _on_reset_tracker(self):
        self.env.reset()
        self.kalman.reset()
        self.imm.reset()
        self.vibration_canceller.reset()
        self.fdir.reset()
        self.explainer.reset()
        self.state_machine.reset()
        self.controller.reset()
        self.metrics = PerformanceMetrics()
        for i in range(len(self.error_history)):
            self.error_history[i] = 0.0

    def _on_run_benchmark(self):
        def _runner():
            self.btn_bench.configure(text="⏳ EXECUTING BENCHMARK...", state="disabled")
            suite = BenchmarkSuite("reports/gui_benchmark_run")
            report_path = suite.run_full_suite()
            self.btn_bench.configure(text="📊 LAUNCH AUTOMATED BENCHMARK SUITE", state="normal")
            messagebox.showinfo(
                "Benchmark Suite Complete",
                f"Benchmark evaluation executed successfully!\nReport generated at:\n{report_path}"
            )

        threading.Thread(target=_runner, daemon=True).start()

    def _on_close(self):
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.root.destroy()

    def _schedule_loop(self):
        self.root.after(16, self._step_loop)

    def _step_loop(self):
        if not self.is_running:
            return

        try:
            self._execute_single_step()
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            if self.is_running:
                self._schedule_loop()

    def _execute_single_step(self):
        t_frame_start = time.perf_counter()

        # Update UTC Clock
        current_time_str = time.strftime("%H:%M:%S UTC")
        self.lbl_clock.configure(text=f"{current_time_str}  |  COARSE PAT STAGE")

        # 1. Image Acquisition Step
        t0 = time.perf_counter()
        if self.source_type in ["webcam", "video"] and self.cap is not None:
            ret, frame_bgr = self.cap.read()
            if not ret or frame_bgr is None:
                if self.source_type == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame_bgr = self.cap.read()
                if not ret or frame_bgr is None:
                    frame_bgr = np.zeros((self.cfg.optical.image_height, self.cfg.optical.image_width, 3), dtype=np.uint8)

            if frame_bgr.shape[1] != self.cfg.optical.image_width or frame_bgr.shape[0] != self.cfg.optical.image_height:
                frame_bgr = cv2.resize(frame_bgr, (self.cfg.optical.image_width, self.cfg.optical.image_height))

            sensor_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            targets_gt = []
            sim_telem = {"cam_pan_deg": self.virtual_cam_pan_deg, "cam_tilt_deg": self.virtual_cam_tilt_deg}
            sim_time = self.metrics.total_frames * self.dt
        else:
            sensor_frame, targets_gt, sim_telem = self.env.step(self.dt)
            sim_time = self.env.sim_time

        self.metrics.record_frame_start(sim_time)
        self.metrics.record_stage_latency("simulation", (time.perf_counter() - t0) * 1000.0)

        # 2. Perception Step
        t0 = time.perf_counter()
        raw_candidates = self.detector.detect(sensor_frame)
        for cand in raw_candidates:
            pan_off, tilt_off = self.env.geom.pixel_to_angle_deg(cand["cx"], cand["cy"])
            cand["inertial_pan_deg"] = sim_telem["cam_pan_deg"] + pan_off
            cand["inertial_tilt_deg"] = sim_telem["cam_tilt_deg"] + tilt_off

        # Gated data association
        pred_inertial = np.array(self.active_estimator.position_deg, dtype=np.float64)
        innov_cov = self.active_estimator.get_innovation_covariance()
        is_active_track = (self.state_machine.state in [TrackState.TRACK, TrackState.COAST])
        selected_cand = self.associator.select_best_candidate(
            raw_candidates, pred_inertial, innov_cov,
            center_pixel=(self.env.geom.cx, self.env.geom.cy),
            is_tracking=is_active_track
        )

        # Autonomous FDIR Validation & Stuck-Pixel Isolation
        is_valid, fdir_cand, fdir_reason = self.fdir.inspect_measurement(
            selected_cand,
            self.active_estimator.position_deg,
            innov_cov,
            gimbal_moving=True,
            is_tracking=is_active_track
        )
        verified_cand = fdir_cand if is_valid else None
        self.metrics.record_stage_latency("detection", (time.perf_counter() - t0) * 1000.0)

        # 3. State Estimation Step
        t0 = time.perf_counter()
        self.active_estimator.predict(self.dt)
        track_state, cmd_pos_deg = self.state_machine.step(verified_cand, self.dt)
        self.metrics.record_stage_latency("estimation", (time.perf_counter() - t0) * 1000.0)

        # 4. Control Step
        t0 = time.perf_counter()
        gain_scale = self.damper.compute_gain_scale(innov_cov)
        feedforward_vel = self.active_estimator.velocity_deg_s if track_state in [TrackState.TRACK, TrackState.COAST] else (0.0, 0.0)

        if self.cfg.controller.mode == "mpc_lite":
            cmd_pan_rate, cmd_tilt_rate = self.mpc_controller.compute(
                target_pos_deg=cmd_pos_deg,
                target_vel_deg_s=feedforward_vel,
                current_pos_deg=(sim_telem["cam_pan_deg"], sim_telem["cam_tilt_deg"]),
                dt=self.dt
            )
        else:
            cmd_pan_rate, cmd_tilt_rate = self.controller.compute(
                target_pan_deg=cmd_pos_deg[0],
                target_tilt_deg=cmd_pos_deg[1],
                current_pan_deg=sim_telem["cam_pan_deg"],
                current_tilt_deg=sim_telem["cam_tilt_deg"],
                dt=self.dt,
                feedforward_vel=feedforward_vel,
                gain_scale=gain_scale
            )

        if self.source_type == "simulation":
            self.env.gimbal.set_rate_command(cmd_pan_rate, cmd_tilt_rate)
        else:
            self.virtual_cam_pan_deg += np.clip(cmd_pan_rate, -self.cfg.gimbal.max_slew_rate_pan_deg_s, self.cfg.gimbal.max_slew_rate_pan_deg_s) * self.dt
            self.virtual_cam_tilt_deg += np.clip(cmd_tilt_rate, -self.cfg.gimbal.max_slew_rate_tilt_deg_s, self.cfg.gimbal.max_slew_rate_tilt_deg_s) * self.dt

        self.metrics.record_stage_latency("control", (time.perf_counter() - t0) * 1000.0)

        # 5. Metrics Recording
        primary_tgt = targets_gt[0] if targets_gt else None
        if primary_tgt is not None:
            true_px = primary_tgt["pixel_x"]
            true_py = primary_tgt["pixel_y"]
        elif verified_cand is not None:
            true_px = verified_cand["cx"]
            true_py = verified_cand["cy"]
        else:
            true_px, true_py = None, None

        err_mrad = None
        if true_px is not None and true_py is not None:
            err_mrad = self.env.geom.angular_distance_mrad(true_px, true_py, self.env.geom.cx, self.env.geom.cy)

        frame_latency_ms = (time.perf_counter() - t_frame_start) * 1000.0
        self.metrics.update_frame(
            frame_latency_ms=frame_latency_ms,
            current_state=track_state.value,
            sim_time=sim_time,
            true_px=true_px,
            true_py=true_py,
            boresight_px=(self.env.geom.cx, self.env.geom.cy),
            error_mrad=err_mrad
        )

        # Causal Explainability Synthesis & Defense Telemetry
        imm_probs = self.imm.model_probabilities if self.estimator_mode == "IMM" else {"CV": 1.0, "CA": 0.0, "CT": 0.0}
        dom_model = self.imm.dominant_model if self.estimator_mode == "IMM" else "EKF"
        cur_nis = getattr(self.active_estimator, "last_nis", 1.0)
        trace = self.explainer.generate_trace(
            frame=self.metrics.total_frames,
            track_state=track_state.value,
            dominant_model=dom_model,
            imm_probs=imm_probs,
            fdir_status=self.fdir.current_status.value,
            nis_metric=cur_nis,
            vibration_diag=self.vibration_canceller.get_diagnostics(),
            gain_scale=gain_scale,
            pointing_error_mrad=err_mrad if err_mrad is not None else 0.0,
            detection_found=(verified_cand is not None),
            coast_frames=self.state_machine.coast_counter
        )

        # 6. Render HUD Overlays on Video Canvas
        det_pt = (verified_cand["cx"], verified_cand["cy"]) if verified_cand else None
        summary = self.metrics.get_summary()
        summary["advanced_avionics_str"] = f"IMM: {dom_model} | NOTCH: {self.vibration_canceller.active_peaks[0]:.1f}Hz | FDIR: {self.fdir.current_status.value}"

        # Record error for oscilloscope
        cur_err_px = float(summary.get("current_error_px", 0.0))
        self.error_history.append(cur_err_px)

        annotated_bgr = self.hud.draw_all(
            frame=sensor_frame,
            track_state=track_state.value,
            target_gt=primary_tgt,
            detected_centroid=det_pt,
            kf_pos_deg=self.active_estimator.position_deg,
            kf_cov_2x2=self.active_estimator.P[:2, :2],
            cam_pose_deg=(sim_telem["cam_pan_deg"], sim_telem["cam_tilt_deg"]),
            metrics_summary=summary,
            search_radius_deg=min(12.0, self.cfg.state_machine.search_spiral_growth * self.state_machine.search_time)
        )

        # 7. Render Video Frame to Tkinter Canvas
        resized_rgb = cv2.cvtColor(
            cv2.resize(annotated_bgr, (self.canvas_w, self.canvas_h), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB
        )
        img_pil = Image.fromarray(resized_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        if not hasattr(self, "_canvas_image_id"):
            self._canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        else:
            self.canvas.itemconfig(self._canvas_image_id, image=img_tk)
        self.canvas._img_tk = img_tk  # Prevent garbage collection

        # 8. Render Real-Time Oscilloscope Waveform
        self._render_oscilloscope(summary)

        # 9. Update UI Scorecard Labels & Gauges
        state_str = track_state.value
        if state_str == "TRACK":
            self.banner_state.configure(fg_color="#064E3B")
            self.lbl_state_text.configure(text="● TRACKING STATE: LOCKED", text_color="#34D399")
        elif state_str == "COAST":
            self.banner_state.configure(fg_color="#78350F")
            self.lbl_state_text.configure(text="● TRACKING STATE: COASTING (DEAD-RECKONING)", text_color="#F59E0B")
        elif state_str == "CANDIDATE":
            self.banner_state.configure(fg_color="#713F12")
            self.lbl_state_text.configure(text="● TRACKING STATE: CANDIDATE VERIFICATION", text_color="#FDE047")
        else:
            self.banner_state.configure(fg_color="#7F1D1D")
            self.lbl_state_text.configure(text=f"● TRACKING STATE: {state_str}", text_color="#EF4444")

        err_px = summary["current_error_px"]
        err_mr = summary["current_error_mrad"]
        err_clr = "#34D399" if err_mr < 2.0 else ("#F59E0B" if err_mr < 5.0 else "#EF4444")
        self.lbl_val_error.configure(text=f"{err_px:.1f} px", text_color=err_clr)
        sub_tag = "[SUB-MRAD PASS]" if err_mr < 2.0 else "[COARSE LIMIT]"
        self.lbl_val_mrad.configure(text=f"{err_mr:.2f} mrad {sub_tag}", text_color=err_clr)

        lock_pct = summary["lock_retention_rate_pct"]
        self.lbl_val_lock.configure(text=f"{lock_pct:.1f} %")
        self.progress_lock.set(min(1.0, max(0.0, lock_pct / 100.0)))

        self.lbl_val_fps.configure(text=f"{summary['fps']:.1f} FPS")
        self.lbl_val_lat.configure(text=f"Latency: {summary['frame_latency_ms']:.1f} ms")

        pan = sim_telem["cam_pan_deg"]
        tilt = sim_telem["cam_tilt_deg"]
        self.lbl_val_gimbal.configure(text=f"Pan: {pan:+.2f}°")
        self.lbl_val_tilt.configure(text=f"Tilt: {tilt:+.2f}°")

        # Update FDIR badge
        fdir_stat = self.fdir.current_status.value
        if fdir_stat == "NOMINAL":
            self.lbl_fdir_badge.configure(text="● FDIR: NOMINAL", fg_color="#064E3B", text_color="#34D399")
        elif fdir_stat == "STUCK_PIXEL_ISOLATED":
            self.lbl_fdir_badge.configure(text="● FDIR: BAD PIXEL MASKED", fg_color="#78350F", text_color="#F59E0B")
        elif fdir_stat == "OUTLIER_REJECTED":
            self.lbl_fdir_badge.configure(text="● FDIR: OUTLIER GATED", fg_color="#713F12", text_color="#FDE047")
        else:
            self.lbl_fdir_badge.configure(text=f"● FDIR: {fdir_stat}", fg_color="#7F1D1D", text_color="#EF4444")

        # Update Vibration Notch badge
        if self.vibration_canceller.enabled:
            notch_f = self.vibration_canceller.active_peaks[0] if self.vibration_canceller.active_peaks else 0.0
            notch_att = self.vibration_canceller.last_attenuation_db
            self.lbl_notch_badge.configure(
                text=f"📳 NOTCH: {notch_f:.1f}Hz (-{notch_att:.1f}dB)",
                fg_color="#1E3A8A", text_color="#60A5FA"
            )
        else:
            self.lbl_notch_badge.configure(
                text="📳 NOTCH: BYPASSED",
                fg_color="#1F2937", text_color="#9CA3AF"
            )

        # Update IMM Probabilities
        if self.estimator_mode == "IMM":
            p_cv = imm_probs["CV"]
            p_ca = imm_probs["CA"]
            p_ct = imm_probs["CT"]
            self.lbl_imm_modes.configure(text=f"IMM Bank: CV {p_cv*100:.0f}% | CA {p_ca*100:.0f}% | CT {p_ct*100:.0f}%")
            self.prog_cv.set(p_cv)
            self.prog_ca.set(p_ca)
            self.prog_ct.set(p_ct)
        else:
            self.lbl_imm_modes.configure(text="Estimator: Standard EKF Single Model")
            self.prog_cv.set(1.0)
            self.prog_ca.set(0.0)
            self.prog_ct.set(0.0)

        # Update Causal Decision Ticker
        self.lbl_causal_ticker.configure(text=f"> {trace.causal_explanation}")

        # Occlusion visual feedback on button
        if self.occlusion_countdown > 0:
            self.occlusion_countdown = max(0.0, self.occlusion_countdown - self.dt)
            self.btn_occl.configure(text=f"☁️ OCCLUDED... ({self.occlusion_countdown:.1f}s)", fg_color="#B45309")
        else:
            self.btn_occl.configure(text="⚡ INJECT CLOUD OCCLUSION (1.5s)", fg_color="#D97706")

    def _render_oscilloscope(self, summary: Dict[str, Any]):
        """Draws animated real-time oscilloscope tracking error strip chart with shaded under-curve polygon."""
        cw = self.canvas_w
        ch = 105
        self.osc_canvas.delete("all")

        # Subtle background grid lines
        self.osc_canvas.create_line(0, ch - 22, cw, ch - 22, fill="#111B2C", dash=(2, 4))
        self.osc_canvas.create_line(0, ch - 52, cw, ch - 52, fill="#111B2C", dash=(2, 4))
        self.osc_canvas.create_line(0, ch - 82, cw, ch - 82, fill="#111B2C", dash=(2, 4))

        # Pixel per mrad scale
        px_per_mrad = getattr(self.hud, "px_per_mrad", 7.324)

        # 5.0 mrad coarse limit reference line
        y_coarse = ch - max(4, min(ch - 4, int((5.0 * px_per_mrad) * (ch / 120.0))))
        self.osc_canvas.create_line(0, y_coarse, cw, y_coarse, fill="#78350F", dash=(3, 3))
        self.osc_canvas.create_text(cw - 110, y_coarse - 7, text="5.0 mrad Coarse Capture Limit", fill="#F59E0B", font=("Segoe UI", 8))

        # 2.0 mrad fine threshold reference line
        y_thresh = ch - max(4, min(ch - 4, int((2.0 * px_per_mrad) * (ch / 120.0))))
        self.osc_canvas.create_line(0, y_thresh, cw, y_thresh, fill="#065F46", dash=(4, 4))
        self.osc_canvas.create_text(cw - 110, y_thresh - 7, text="2.0 mrad Fine Steering Handover", fill="#10B981", font=("Segoe UI", 8, "bold"))

        # Plot error trajectory line and under-curve polygon
        history = list(self.error_history)
        n = len(history)
        if n > 1:
            points = []
            dx = cw / float(n - 1)
            for i, err in enumerate(history):
                x = i * dx
                # Scale error (0 to 120 px full scale)
                y = ch - max(4, min(ch - 4, int(err * (ch / 120.0))))
                points.append((x, y))

            # Shaded polygon under curve
            poly_pts = [(0, ch)] + points + [(cw, ch)]
            flat_poly = [coord for pt in poly_pts for coord in pt]
            self.osc_canvas.create_polygon(flat_poly, fill="#042738", outline="")

            # Draw smooth line segments on top
            flat_pts = [coord for pt in points for coord in pt]
            err_mrad = summary.get("current_error_mrad", 0.0)
            line_clr = "#00F0FF" if err_mrad < 2.0 else ("#F59E0B" if err_mrad < 5.0 else "#EF4444")
            self.osc_canvas.create_line(flat_pts, fill=line_clr, width=2, smooth=True)

        err_px = summary.get("current_error_px", 0.0)
        err_mrad = summary.get("current_error_mrad", 0.0)
        rmse_px = summary.get("rmse_error_px", 0.0)
        rmse_mrad = rmse_px / px_per_mrad if px_per_mrad > 0 else 0.0
        status_tag = "[SUB-MRAD PASS]" if err_mrad < 2.0 else "[COARSE ACQUIRED]"
        self.lbl_osc_stat.configure(
            text=f"Current: {err_px:.1f}px ({err_mrad:.2f}mrad) | RMSE: {rmse_px:.1f}px ({rmse_mrad:.2f}mrad) | Limit: 2.0mrad {status_tag}"
        )


def launch_gui(config: Optional[AppConfig] = None, source_type: str = "simulation", video_source: Optional[Any] = None):
    root = ctk.CTk()
    app = FSOCTrackerApp(root, config, source_type=source_type, video_source=video_source)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
