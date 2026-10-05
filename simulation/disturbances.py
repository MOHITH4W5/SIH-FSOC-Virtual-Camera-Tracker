"""
simulation/disturbances.py - Disturbance & Environmental Degradation Engine.
Implements physically grounded proxies for:
  1. Atmospheric Turbulence:
     - Angle of Arrival (AoA) jitter via an Ornstein-Uhlenbeck stochastic process
     - Scintillation (intensity flicker) via log-normal distribution
     - Variable Gaussian beam blur tied to a Cn^2 / Fried parameter r0 proxy
  2. Platform Micro-Vibration:
     - Multi-harmonic sinusoidal jitter on gimbal pan/tilt pose
  3. Sensor Degradation:
     - Additive Gaussian readout noise + salt-and-pepper impulses
  4. Dynamic Occlusions:
     - Scripted or interactive passing blockers (clouds / structural obstructions)
  5. Decoy Bright Artifacts:
     - Stray reflections, stars, and lens flares to test false-positive rejection
"""

import math
from typing import Tuple, List, Optional
import numpy as np
import cv2

from core.config import DisturbanceConfig


class OrnsteinUhlenbeckProcess:
    """Mean-reverting stochastic process for realistic physical Angle-of-Arrival wander."""
    def __init__(self, theta: float = 0.3, sigma: float = 2.0):
        self.theta = theta
        self.sigma = sigma
        self.state = 0.0

    def step(self, dt: float, scale: float = 1.0) -> float:
        # dx = -theta * x * dt + sigma * sqrt(dt) * dW
        dw = np.random.normal(0, 1.0)
        eff_sigma = self.sigma * scale
        dx = -self.theta * self.state * dt + eff_sigma * math.sqrt(max(dt, 1e-4)) * dw
        self.state += dx
        return self.state

    def reset(self):
        self.state = 0.0


class DisturbanceEngine:
    """Orchestrates all physical disturbances affecting the optical channel and sensor."""

    def __init__(self, config: DisturbanceConfig):
        self.cfg = config
        
        # Ornstein-Uhlenbeck generators for horizontal and vertical AoA jitter
        self.ou_aoa_x = OrnsteinUhlenbeckProcess(theta=config.ou_theta, sigma=config.ou_sigma_px)
        self.ou_aoa_y = OrnsteinUhlenbeckProcess(theta=config.ou_theta, sigma=config.ou_sigma_px)

        # Vibration phases (random initial phase per frequency)
        self._vib_phases_pan = [np.random.uniform(0, 2*math.pi) for _ in config.vibration_frequencies_hz]
        self._vib_phases_tilt = [np.random.uniform(0, 2*math.pi) for _ in config.vibration_frequencies_hz]

        # Occlusion management
        self.occlusion_active = config.occlusion_active
        self._occlusion_timer = 0.0

        # Decoy positions (world degrees)
        self._decoys: List[Tuple[float, float, float]] = []  # (pan_deg, tilt_deg, intensity)
        self._init_decoys()

    def _init_decoys(self):
        self._decoys.clear()
        for i in range(self.cfg.decoy_count):
            self._decoys.append((
                np.random.uniform(-4.0, 4.0),
                np.random.uniform(-3.0, 3.0),
                255.0 * self.cfg.decoy_intensity_ratio
            ))

    def trigger_occlusion(self, duration_sec: float = 1.5):
        """Manually triggers a temporary target blackout/occlusion."""
        self.occlusion_active = True
        self._occlusion_timer = duration_sec

    def update_occlusion(self, dt: float) -> bool:
        """Updates occlusion timer and returns True if target is currently occluded."""
        if self._occlusion_timer > 0.0:
            self._occlusion_timer -= dt
            if self._occlusion_timer <= 0.0:
                self.occlusion_active = False
                self._occlusion_timer = 0.0
        return self.occlusion_active

    def compute_platform_vibration(self, time_sec: float) -> Tuple[float, float]:
        """
        Computes the platform micro-vibration perturbation (degrees) to be added to camera mount.
        Sum of sinusoids at characteristic structural frequencies.
        """
        if not self.cfg.enabled or self.cfg.vibration_amplitude_deg <= 0:
            return 0.0, 0.0

        vib_pan = 0.0
        vib_tilt = 0.0
        n_harmonics = len(self.cfg.vibration_frequencies_hz)
        amp_per_harmonic = self.cfg.vibration_amplitude_deg / max(1, math.sqrt(n_harmonics))

        for idx, freq in enumerate(self.cfg.vibration_frequencies_hz):
            vib_pan += amp_per_harmonic * math.sin(2.0 * math.pi * freq * time_sec + self._vib_phases_pan[idx])
            vib_tilt += amp_per_harmonic * math.cos(2.0 * math.pi * freq * time_sec + self._vib_phases_tilt[idx])

        # Add small high-frequency jitter
        vib_pan += np.random.normal(0, 0.05 * amp_per_harmonic)
        vib_tilt += np.random.normal(0, 0.05 * amp_per_harmonic)

        return vib_pan, vib_tilt

    def compute_atmospheric_disturbances(self, dt: float) -> Tuple[float, float, float, float]:
        """
        Computes atmospheric turbulence effects:
        Returns:
            (aoa_jitter_x_px, aoa_jitter_y_px, scintillation_factor, blur_sigma_px)
        """
        if not self.cfg.enabled or self.cfg.turbulence_strength <= 0:
            return 0.0, 0.0, 1.0, 0.0

        turb = self.cfg.turbulence_strength  # Scale [0.0, 1.0]

        # 1. Angle of Arrival jitter (stochastic spot displacement)
        jitter_x = self.ou_aoa_x.step(dt, scale=turb)
        jitter_y = self.ou_aoa_y.step(dt, scale=turb)

        # 2. Scintillation: Log-normal intensity modulation
        # Var(ln(I)) proportional to turbulence strength
        sigma_chi = math.sqrt(self.cfg.scintillation_variance * turb)
        scintillation_factor = math.exp(np.random.normal(-0.5 * sigma_chi**2, sigma_chi))
        # Clamp to realistic bounds [0.15, 2.0]
        scintillation_factor = float(np.clip(scintillation_factor, 0.15, 2.2))

        # 3. Dynamic Gaussian beam blur expansion
        blur_sigma = turb * (self.cfg.blur_kernel_max / 2.0)

        return jitter_x, jitter_y, scintillation_factor, blur_sigma

    def apply_sensor_noise(self, frame: np.ndarray) -> np.ndarray:
        """Applies readout Gaussian noise and impulse noise to the rendered sensor frame."""
        if not self.cfg.enabled:
            return frame

        # Gaussian readout noise
        if self.cfg.sensor_gaussian_sigma > 0:
            noise = np.random.normal(0, self.cfg.sensor_gaussian_sigma, frame.shape)
            noisy = frame.astype(np.float32) + noise
        else:
            noisy = frame.astype(np.float32)

        # Salt and pepper impulse noise (hot pixels / cosmic ray strikes)
        if self.cfg.salt_pepper_prob > 0:
            num_sp = int(self.cfg.salt_pepper_prob * frame.size)
            if num_sp > 0:
                y_idx = np.random.randint(0, frame.shape[0], num_sp)
                x_idx = np.random.randint(0, frame.shape[1], num_sp)
                noisy[y_idx, x_idx] = 255.0

        return np.clip(noisy, 0, 255).astype(np.uint8)

    @property
    def decoys(self) -> List[Tuple[float, float, float]]:
        return self._decoys
