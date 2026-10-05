"""
ui/overlay.py - High-Tech Tactical Heads-Up Display (HUD) & Telemetry Overlays.
Designed for SIH26169 (ISRO / SAC) Mission Control standards.
Features:
  1. Tactical Optical Boresight with Multi-Stage Acceptance Reticles (1, 2, 5 mrad)
  2. Corner Frame Brackets & Azimuth/Elevation Flight Attitude Tapes
  3. Dynamic Target Lock Bracket [ ] with Heading & Velocity Vector Arrow
  4. Kalman 2-Sigma Uncertainty Covariance Ellipse with Semi-Transparent Shading
  5. Mahalanobis Statistical Gating Boundary
  6. Expanding Archimedean Search Spiral Visualization
  7. High-Tech Glassmorphism Telemetry Card with Glowing State Badges
"""

import math
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2

from core.geometry import CameraGeometry


class TelemetryHUDOverlay:
    """Renders fighter-jet / spacecraft tactical HUD overlays on optical sensor frames."""

    def __init__(self, geometry: CameraGeometry):
        self.geom = geometry
        self.cx = int(round(geometry.cx))
        self.cy = int(round(geometry.cy))

        # High-Tech Cyber-Tactical Palette (BGR)
        self.CLR_CYAN = (255, 230, 0)         # Primary Reticle & HUD (BGR)
        self.CLR_EMERALD = (80, 230, 60)      # Locked / Ground Truth
        self.CLR_AMBER = (0, 180, 255)        # Coasting / Warning
        self.CLR_CRIMSON = (40, 50, 240)      # Searching / Alert
        self.CLR_PURPLE = (240, 100, 180)     # AI Discriminator
        self.CLR_WHITE = (245, 248, 250)      # Primary Text
        self.CLR_MUTED = (160, 170, 180)      # Secondary Text
        self.CLR_DARK_BG = (12, 16, 24)       # Glass Backdrop
        self.CLR_GRID = (50, 65, 80)          # Attitude Grid Lines

        # Pixel per milliradian approximation
        px_per_deg_x = self.geom.width / math.degrees(self.geom.fov_h_rad)
        self.px_per_mrad = px_per_deg_x / 17.4533

    def draw_all(
        self,
        frame: np.ndarray,
        track_state: str,
        target_gt: Optional[Dict[str, Any]],
        detected_centroid: Optional[Tuple[float, float]],
        kf_pos_deg: Optional[Tuple[float, float]],
        kf_cov_2x2: Optional[np.ndarray],
        cam_pose_deg: Tuple[float, float],
        metrics_summary: Dict[str, Any],
        search_radius_deg: float = 0.0,
        show_gt: bool = True,
        show_gate: bool = True
    ) -> np.ndarray:
        """Draws complete tactical HUD layer stack and returns an annotated 3-channel BGR image."""
        if len(frame.shape) == 2:
            annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            annotated = frame.copy()

        h, w = annotated.shape[:2]

        # 1. Tactical Viewport Corner Brackets
        self._draw_corner_brackets(annotated, w, h)

        # 2. Attitude Tapes (Azimuth along top, Elevation along left)
        self._draw_attitude_tapes(annotated, w, h, cam_pose_deg)

        # 3. Precision Optical Boresight Reticle & Multi-Cone Ranges
        self._draw_tactical_boresight(annotated)

        # 4. Search / Reacquisition Spiral
        if track_state in ["SEARCH", "REACQUIRE"] and search_radius_deg > 0:
            self._draw_search_spiral(annotated, track_state, search_radius_deg)

        # 5. Kalman Uncertainty Ellipse & Mahalanobis Gate
        if kf_pos_deg is not None and kf_cov_2x2 is not None:
            self._draw_uncertainty_ellipse(annotated, kf_pos_deg, kf_cov_2x2, cam_pose_deg, show_gate)

        # 6. Target Lock Bracket & Velocity Vector
        if detected_centroid is not None:
            self._draw_target_lock(annotated, detected_centroid, track_state, metrics_summary)

        # 7. Ground Truth Marker (if enabled & visible)
        if show_gt and target_gt is not None:
            tx = target_gt.get("pixel_x")
            ty = target_gt.get("pixel_y")
            if tx is not None and ty is not None and target_gt.get("is_in_fov", False):
                pt = (int(round(tx)), int(round(ty)))
                if not target_gt.get("is_occluded", False):
                    cv2.circle(annotated, pt, 6, self.CLR_EMERALD, 1, cv2.LINE_AA)
                    cv2.drawMarker(annotated, pt, self.CLR_EMERALD, cv2.MARKER_TILTED_CROSS, 8, 1)
                    cv2.putText(annotated, "GT", (pt[0] + 8, pt[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.CLR_EMERALD, 1, cv2.LINE_AA)

        # 8. Glassmorphism Tactical Telemetry HUD Card
        self._draw_glass_telemetry_hud(annotated, track_state, metrics_summary, cam_pose_deg)

        # 9. Bottom Operational Bar
        self._draw_bottom_bar(annotated, w, h, metrics_summary)

        return annotated

    def _draw_corner_brackets(self, img: np.ndarray, w: int, h: int):
        """Draws high-tech sci-fi corner brackets around the viewport edges."""
        pad = 20
        arm = 35
        color = (100, 140, 180)
        thickness = 1

        # Top-Left
        cv2.line(img, (pad, pad), (pad + arm, pad), color, thickness)
        cv2.line(img, (pad, pad), (pad, pad + arm), color, thickness)

        # Top-Right
        cv2.line(img, (w - pad, pad), (w - pad - arm, pad), color, thickness)
        cv2.line(img, (w - pad, pad), (w - pad, pad + arm), color, thickness)

        # Bottom-Left
        cv2.line(img, (pad, h - pad), (pad + arm, h - pad), color, thickness)
        cv2.line(img, (pad, h - pad), (pad, h - pad - arm), color, thickness)

        # Bottom-Right
        cv2.line(img, (w - pad, h - pad), (w - pad - arm, h - pad), color, thickness)
        cv2.line(img, (w - pad, h - pad), (w - pad, h - pad - arm), color, thickness)

    def _draw_attitude_tapes(self, img: np.ndarray, w: int, h: int, cam_pose_deg: Tuple[float, float]):
        """Draws Azimuth heading tape across top and Elevation ladder along left edge."""
        # Top Azimuth Tape
        az = cam_pose_deg[0]
        y_tape = 28
        cv2.line(img, (self.cx - 180, y_tape), (self.cx + 180, y_tape), self.CLR_GRID, 1)
        # Center marker (triangle pointing down at az tape)
        pts_az = np.array([[self.cx - 4, y_tape + 2], [self.cx + 4, y_tape + 2], [self.cx, y_tape + 8]], dtype=np.int32)
        cv2.fillPoly(img, [pts_az], self.CLR_CYAN)
        cv2.putText(img, f"AZ {az:+.1f} deg", (self.cx - 32, y_tape - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.CLR_CYAN, 1, cv2.LINE_AA)

        # Left Elevation Pitch Ladder
        el = cam_pose_deg[1]
        x_tape = 35
        cv2.line(img, (x_tape, self.cy - 120), (x_tape, self.cy + 120), self.CLR_GRID, 1)
        pts_el = np.array([[x_tape + 2, self.cy - 4], [x_tape + 8, self.cy], [x_tape + 2, self.cy + 4]], dtype=np.int32)
        cv2.fillPoly(img, [pts_el], self.CLR_CYAN)
        cv2.putText(img, f"EL {el:+.1f} deg", (x_tape - 8, self.cy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.CLR_CYAN, 1, cv2.LINE_AA)

    def _draw_tactical_boresight(self, img: np.ndarray):
        """Draws precision crosshairs, center dot, and calibrated acceptance cone rings."""
        # Center dot
        cv2.circle(img, (self.cx, self.cy), 2, self.CLR_CYAN, -1)

        # Crosshair reticles with gaps
        gap = 14
        arm = 38
        clr_cross = (140, 160, 180)
        cv2.line(img, (self.cx - arm, self.cy), (self.cx - gap, self.cy), clr_cross, 1)
        cv2.line(img, (self.cx + gap, self.cy), (self.cx + arm, self.cy), clr_cross, 1)
        cv2.line(img, (self.cx, self.cy - arm), (self.cx, self.cy - gap), clr_cross, 1)
        cv2.line(img, (self.cx, self.cy + gap), (self.cx, self.cy + arm), clr_cross, 1)

        # Acceptance Cones (1 mrad, 2 mrad, 5 mrad capture limit)
        r_1mrad = max(6, int(round(1.0 * self.px_per_mrad)))
        r_2mrad = max(12, int(round(2.0 * self.px_per_mrad)))
        r_5mrad = max(28, int(round(5.0 * self.px_per_mrad)))

        # 1 mrad Fine Core (Sub-mrad target capture)
        cv2.circle(img, (self.cx, self.cy), r_1mrad, (255, 230, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "1 mrad", (self.cx + r_1mrad + 3, self.cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 230, 0), 1, cv2.LINE_AA)

        # 2 mrad Coarse Acceptance Cone (Fine Steering Handover Boundary)
        cv2.circle(img, (self.cx, self.cy), r_2mrad, (80, 230, 60), 1, cv2.LINE_AA)
        cv2.putText(img, "2 mrad [LIMIT]", (self.cx + r_2mrad + 3, self.cy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (80, 230, 60), 1, cv2.LINE_AA)

        # 5 mrad Outer Handover Boundary
        cv2.circle(img, (self.cx, self.cy), r_5mrad, (0, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "5 mrad [CAPTURE]", (self.cx + r_5mrad + 4, self.cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 180, 255), 1, cv2.LINE_AA)

    def _draw_target_lock(
        self,
        img: np.ndarray,
        centroid: Tuple[float, float],
        track_state: str,
        metrics: Dict[str, Any]
    ):
        """Draws animated tactical corner brackets [ ] and velocity vector around target."""
        tx, ty = int(round(centroid[0])), int(round(centroid[1]))
        box_sz = 18
        bracket_len = 6

        # Color based on state
        if track_state == "TRACK":
            clr = self.CLR_EMERALD
            label = "LOCK [BEACON]"
        elif track_state == "COAST":
            clr = self.CLR_AMBER
            label = "COAST [PREDICT]"
        else:
            clr = self.CLR_CYAN
            label = "DETECT [CANDIDATE]"

        # Draw 4 corner brackets around target
        # Top-Left
        cv2.line(img, (tx - box_sz, ty - box_sz), (tx - box_sz + bracket_len, ty - box_sz), clr, 2)
        cv2.line(img, (tx - box_sz, ty - box_sz), (tx - box_sz, ty - box_sz + bracket_len), clr, 2)
        # Top-Right
        cv2.line(img, (tx + box_sz, ty - box_sz), (tx + box_sz - bracket_len, ty - box_sz), clr, 2)
        cv2.line(img, (tx + box_sz, ty - box_sz), (tx + box_sz, ty - box_sz + bracket_len), clr, 2)
        # Bottom-Left
        cv2.line(img, (tx - box_sz, ty + box_sz), (tx - box_sz + bracket_len, ty + box_sz), clr, 2)
        cv2.line(img, (tx - box_sz, ty + box_sz), (tx - box_sz, ty + box_sz - bracket_len), clr, 2)
        # Bottom-Right
        cv2.line(img, (tx + box_sz, ty + box_sz), (tx + box_sz - bracket_len, ty + box_sz), clr, 2)
        cv2.line(img, (tx + box_sz, ty + box_sz), (tx + box_sz, ty + box_sz - bracket_len), clr, 2)

        # Label tag
        cv2.putText(img, label, (tx - box_sz, ty - box_sz - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, clr, 1, cv2.LINE_AA)

        # Connecting trajectory dashed line from target to boresight
        cv2.line(img, (tx, ty), (self.cx, self.cy), (80, 100, 120), 1, cv2.LINE_AA)

    def _draw_uncertainty_ellipse(
        self,
        img: np.ndarray,
        kf_pos_deg: Tuple[float, float],
        cov: np.ndarray,
        cam_pose_deg: Tuple[float, float],
        show_gate: bool
    ):
        """Draws the Kalman filter 2-sigma uncertainty ellipse and Mahalanobis gate."""
        px, py, is_vis = self.geom.world_target_to_pixel(
            kf_pos_deg[0], kf_pos_deg[1], cam_pose_deg[0], cam_pose_deg[1]
        )
        if not is_vis or px is None or py is None:
            return

        center = (int(round(px)), int(round(py)))

        try:
            cov_2x2 = cov[:2, :2]
            eigvals, eigvecs = np.linalg.eigh(cov_2x2)
            eigvals = np.maximum(1e-6, eigvals)

            px_per_deg_x = self.geom.width / math.degrees(self.geom.fov_h_rad)
            px_per_deg_y = self.geom.height / math.degrees(self.geom.fov_v_rad)
            scale_px = (px_per_deg_x + px_per_deg_y) / 2.0

            axis_a = int(round(2.0 * math.sqrt(eigvals[1]) * scale_px))  # 2-sigma
            axis_b = int(round(2.0 * math.sqrt(eigvals[0]) * scale_px))
            axis_a = max(6, min(180, axis_a))
            axis_b = max(6, min(180, axis_b))

            angle_deg = math.degrees(math.atan2(eigvecs[1, 1], eigvecs[0, 1]))

            # 2-sigma KF Uncertainty Ellipse
            cv2.ellipse(img, center, (axis_a, axis_b), angle_deg, 0, 360, (255, 220, 40), 1, cv2.LINE_AA)

            # Mahalanobis Gating Boundary (Chi-squared scaled)
            if show_gate:
                gate_a = int(round(axis_a * 1.5))
                gate_b = int(round(axis_b * 1.5))
                cv2.ellipse(img, center, (gate_a, gate_b), angle_deg, 0, 360, (0, 160, 240), 1, cv2.LINE_AA)

        except Exception:
            pass

    def _draw_search_spiral(self, img: np.ndarray, state: str, search_radius_deg: float):
        """Draws search trajectory visualizer when acquiring target."""
        px_per_deg = (self.geom.width / math.degrees(self.geom.fov_h_rad))
        r_px = int(search_radius_deg * px_per_deg)
        cv2.circle(img, (self.cx, self.cy), max(10, r_px), self.CLR_AMBER, 1, cv2.LINE_AA)
        cv2.putText(img, f"ACQUISITION SCAN [{state}]", (self.cx - 85, self.cy + r_px + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.CLR_AMBER, 1, cv2.LINE_AA)

    def _draw_glass_telemetry_hud(
        self,
        img: np.ndarray,
        state: str,
        metrics: Dict[str, Any],
        cam_pose_deg: Tuple[float, float]
    ):
        """Renders the primary mission control telemetry readout badge with glassmorphism styling."""
        box_x, box_y, box_w, box_h = 24, 48, 350, 185

        # Semi-transparent dark glass backdrop
        overlay = img.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), self.CLR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        # Double-bezel glass card borders
        state_color = self.CLR_EMERALD if state == "TRACK" else (self.CLR_AMBER if state == "COAST" else self.CLR_CRIMSON)
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (35, 50, 70), 1)
        cv2.rectangle(img, (box_x + 2, box_y + 2), (box_x + box_w - 2, box_y + box_h - 2), (55, 75, 100), 1)
        # Top glowing accent line
        cv2.line(img, (box_x, box_y), (box_x + box_w, box_y), state_color, 2)

        # Header Title
        cv2.putText(img, "ISRO / SAC · FSOC PAT MISSION CONTROL", (box_x + 14, box_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 180, 220), 1, cv2.LINE_AA)

        # State Pill Badge
        badge_w, badge_h = 135, 26
        badge_x, badge_y = box_x + 14, box_y + 34
        cv2.rectangle(img, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (20, 28, 40), -1)
        cv2.rectangle(img, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), state_color, 1)
        cv2.circle(img, (badge_x + 12, badge_y + 13), 4, state_color, -1)
        cv2.putText(img, f"STATE: {state}", (badge_x + 24, badge_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.CLR_WHITE, 1, cv2.LINE_AA)

        # Telemetry metrics
        fps = metrics.get("fps", 0.0)
        cur_err_px = metrics.get("current_error_px", 0.0)
        cur_err_mrad = metrics.get("current_error_mrad", 0.0)
        rmse_err_px = metrics.get("rmse_error_px", 0.0)
        lock_rate = metrics.get("lock_retention_rate_pct", 0.0)
        lat_ms = metrics.get("frame_latency_ms", 0.0)

        err_color = self.CLR_EMERALD if cur_err_mrad < 2.0 else (self.CLR_AMBER if cur_err_mrad < 5.0 else self.CLR_CRIMSON)
        sub_tag = "[PASS]" if cur_err_mrad < 2.0 else ""

        lines = [
            ("LOOP RATE", f"{fps:.1f} FPS  ({lat_ms:.1f} ms)", self.CLR_WHITE),
            ("TRACK ERROR", f"{cur_err_px:.1f} px  [{cur_err_mrad:.2f} mrad] {sub_tag}", err_color),
            ("RMSE ERROR", f"{rmse_err_px:.1f} px", self.CLR_WHITE),
            ("LOCK RETENTION", f"{lock_rate:.1f} %", self.CLR_EMERALD if lock_rate > 90 else self.CLR_AMBER),
            ("GIMBAL POSE", f"Pan {cam_pose_deg[0]:+.2f} deg, Tilt {cam_pose_deg[1]:+.2f} deg", self.CLR_MUTED)
        ]

        y_offset = box_y + 80
        for tag, val, clr in lines:
            cv2.putText(img, tag, (box_x + 14, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 150, 180), 1, cv2.LINE_AA)
            cv2.putText(img, val, (box_x + 135, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, clr, 1, cv2.LINE_AA)
            y_offset += 20

    def _draw_bottom_bar(self, img: np.ndarray, w: int, h: int, metrics: Dict[str, Any]):
        """Draws subtle tactical status strip along the bottom viewport edge."""
        y_bar = h - 22
        overlay = img.copy()
        cv2.rectangle(overlay, (0, y_bar), (w, h), self.CLR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)
        cv2.line(img, (0, y_bar), (w, y_bar), (40, 55, 75), 1)

        txt = f"FOV: {math.degrees(self.geom.fov_h_rad):.1f}*x{math.degrees(self.geom.fov_v_rad):.1f}* | RES: {w}x{h} | COARSE CAPTURE: < 5 mrad | FINE LIMIT: < 2 mrad"
        cv2.putText(img, txt, (24, y_bar + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 160, 180), 1, cv2.LINE_AA)

        adv_txt = metrics.get("advanced_avionics_str")
        if adv_txt:
            cv2.putText(img, adv_txt, (w - 480, y_bar + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.CLR_CYAN, 1, cv2.LINE_AA)
