"""
simulation/environment.py - Virtual Sensor Frame Generator & Optical Digital Twin.
Combines camera projection, beacon Gaussian PSF rendering, starfield sky,
atmospheric turbulence, platform vibration, and sensor noise into realistic frames.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from core.config import AppConfig
from core.geometry import CameraGeometry
from simulation.camera import VirtualCameraGimbal
from simulation.targets import TargetManager
from simulation.disturbances import DisturbanceEngine


class VirtualEnvironment:
    """End-to-end simulation of the optical acquisition channel."""

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.geom = CameraGeometry(
            config.optical.image_width,
            config.optical.image_height,
            config.optical.fov_horizontal_deg,
            config.optical.fov_vertical_deg
        )

        self.gimbal = VirtualCameraGimbal(config.gimbal)
        self.targets = TargetManager(
            count=config.target.target_count,
            motion_type=config.target.motion_type,
            speed=config.target.speed_deg_s,
            radius=config.target.trajectory_radius_deg
        )
        self.disturbance = DisturbanceEngine(config.disturbance)

        # Starfield background (fixed catalog of stars in deep space world coordinates)
        self._stars: List[Tuple[float, float, int]] = []  # (world_pan_deg, world_tilt_deg, brightness)
        self._init_starfield()

        # Canvas buffer
        self._canvas = np.zeros((config.optical.image_height, config.optical.image_width), dtype=np.uint8)
        self.sim_time = 0.0

    def _init_starfield(self):
        rng = np.random.RandomState(1337)
        self._stars.clear()
        # Cover a +/- 30 deg world area around boresight
        for _ in range(self.cfg.optical.starfield_density):
            pan = rng.uniform(-25.0, 25.0)
            tilt = rng.uniform(-15.0, 15.0)
            brightness = int(rng.randint(40, 115))
            self._stars.append((pan, tilt, brightness))

    def step(self, dt: float) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Advances the simulation by dt seconds.
        Returns:
            (sensor_frame, target_ground_truth_list, sim_telemetry)
        """
        self.sim_time += dt

        # 1. Update platform micro-vibration & gimbal pose
        vib_pan, vib_tilt = self.disturbance.compute_platform_vibration(self.sim_time)
        self.gimbal.set_vibration_jitter(vib_pan, vib_tilt)
        cam_pan_deg, cam_tilt_deg = self.gimbal.step(dt)

        # 2. Update kinematic targets
        target_states = self.targets.update(dt)

        # 3. Check occlusion status
        is_occluded = self.disturbance.update_occlusion(dt)

        # 4. Compute atmospheric turbulence effects
        aoa_x, aoa_y, scint_factor, blur_sigma = self.disturbance.compute_atmospheric_disturbances(dt)

        # 5. Clear frame canvas to dark sky baseline
        self._canvas.fill(int(self.cfg.optical.ambient_background_level))

        # 6. Render background stars (projected through current camera attitude)
        for star_pan, star_tilt, brightness in self._stars:
            spx, spy, s_vis = self.geom.world_target_to_pixel(star_pan, star_tilt, cam_pan_deg, cam_tilt_deg)
            if s_vis and 0 <= int(spy) < self.cfg.optical.image_height and 0 <= int(spx) < self.cfg.optical.image_width:
                self._canvas[int(spy), int(spx)] = brightness

        # 7. Render decoy bright spots (if any)
        for d_pan, d_tilt, d_int in self.disturbance.decoys:
            dpx, dpy, d_vis = self.geom.world_target_to_pixel(d_pan, d_tilt, cam_pan_deg, cam_tilt_deg)
            if d_vis:
                self._draw_gaussian_psf(dpx, dpy, sigma=self.cfg.optical.beacon_sigma_px * 0.8,
                                        peak_intensity=d_int * 0.8)

        # 8. Render optical beacon targets
        target_ground_truth = []
        for tgt in target_states:
            tgt_pan = tgt["pan_deg"]
            tgt_tilt = tgt["tilt_deg"]
            true_px, true_py, is_in_fov = self.geom.world_target_to_pixel(
                tgt_pan, tgt_tilt, cam_pan_deg, cam_tilt_deg
            )

            # Ground truth record
            tgt_record = {
                "id": tgt["id"],
                "world_pan_deg": tgt_pan,
                "world_tilt_deg": tgt_tilt,
                "world_pan_vel_deg_s": tgt["pan_vel_deg_s"],
                "world_tilt_vel_deg_s": tgt["tilt_vel_deg_s"],
                "pixel_x": true_px,
                "pixel_y": true_py,
                "is_in_fov": is_in_fov,
                "is_occluded": is_occluded,
                "priority": tgt["priority"]
            }
            target_ground_truth.append(tgt_record)

            # Render beacon PSF spot if in FOV and not occluded
            if is_in_fov and not is_occluded and true_px is not None and true_py is not None:
                # Apply AoA jitter and scintillation
                spot_px = true_px + aoa_x
                spot_py = true_py + aoa_y
                spot_intensity = min(255.0, tgt["intensity"] * scint_factor)
                spot_sigma = self.cfg.optical.beacon_sigma_px + blur_sigma

                self._draw_gaussian_psf(spot_px, spot_py, sigma=spot_sigma, peak_intensity=spot_intensity)

        # 9. Apply sensor readout noise (Gaussian + salt/pepper)
        sensor_frame = self.disturbance.apply_sensor_noise(self._canvas)

        sim_telemetry = {
            "time_sec": self.sim_time,
            "cam_pan_deg": cam_pan_deg,
            "cam_tilt_deg": cam_tilt_deg,
            "cam_pan_rate_deg_s": self.gimbal.pan_rate_deg_s,
            "cam_tilt_rate_deg_s": self.gimbal.tilt_rate_deg_s,
            "is_occluded": is_occluded,
            "scintillation_factor": scint_factor,
            "aoa_jitter_px": (aoa_x, aoa_y),
            "vibration_jitter_deg": (vib_pan, vib_tilt)
        }

        return sensor_frame, target_ground_truth, sim_telemetry

    def _draw_gaussian_psf(self, center_x: float, center_y: float, sigma: float, peak_intensity: float):
        """Draws a sub-pixel accurate Gaussian PSF blob on the canvas buffer."""
        cx = int(round(center_x))
        cy = int(round(center_y))
        
        radius = int(math.ceil(3.5 * sigma))
        if radius < 3:
            radius = 3

        x_min = max(0, cx - radius)
        x_max = min(self.cfg.optical.image_width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(self.cfg.optical.image_height, cy + radius + 1)

        if x_max <= x_min or y_max <= y_min:
            return

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        psf = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
        patch = (psf * peak_intensity).astype(np.uint8)

        roi = self._canvas[y_min:y_max, x_min:x_max]
        self._canvas[y_min:y_max, x_min:x_max] = np.maximum(roi, patch)

    def reset(self):
        self.sim_time = 0.0
        self.gimbal.reset()
        self.targets.reset(
            count=self.cfg.target.target_count,
            motion_type=self.cfg.target.motion_type,
            speed=self.cfg.target.speed_deg_s,
            radius=self.cfg.target.trajectory_radius_deg
        )
