"""
core/config.py - Central Configuration Management for SIH26169 FSOC Virtual Camera Tracker.
Every physical, optical, algorithmic, and benchmark parameter is configurable at runtime.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional
import json
import math


@dataclass
class OpticalConfig:
    """Camera and optical sensor specifications."""
    image_width: int = 1280
    image_height: int = 720
    # Field of view in degrees (typical wide-FOV acquisition camera: ~10° x 5.6°)
    fov_horizontal_deg: float = 10.0
    fov_vertical_deg: float = 5.625
    # Optical beacon nominal properties
    beacon_sigma_px: float = 3.5  # Gaussian PSF standard deviation in pixels
    beacon_peak_intensity: float = 245.0  # 8-bit saturation scale (0-255)
    ambient_background_level: float = 12.0  # Baseline dark sky background
    starfield_density: int = 150  # Number of background star points

    @property
    def fov_horizontal_rad(self) -> float:
        return math.radians(self.fov_horizontal_deg)

    @property
    def fov_vertical_rad(self) -> float:
        return math.radians(self.fov_vertical_deg)

    @property
    def focal_length_x(self) -> float:
        return (self.image_width / 2.0) / math.tan(self.fov_horizontal_rad / 2.0)

    @property
    def focal_length_y(self) -> float:
        return (self.image_height / 2.0) / math.tan(self.fov_vertical_rad / 2.0)


@dataclass
class GimbalConfig:
    """Virtual 2-axis Pan/Tilt gimbal mechanical specifications."""
    pan_min_deg: float = -90.0
    pan_max_deg: float = 90.0
    tilt_min_deg: float = -45.0
    tilt_max_deg: float = 45.0
    # Maximum slew rate (deg/s)
    max_slew_rate_pan_deg_s: float = 25.0
    max_slew_rate_tilt_deg_s: float = 20.0
    # Maximum angular acceleration (deg/s^2)
    max_accel_deg_s2: float = 50.0


@dataclass
class DisturbanceConfig:
    """Atmospheric turbulence proxy, platform vibration, and sensor noise parameters."""
    enabled: bool = True
    # Atmospheric Turbulence Proxy:
    # Approximate index representing Cn^2 / Fried parameter r0 scaling [0.0 (calm) to 1.0 (severe)]
    turbulence_strength: float = 0.35
    # Angle of Arrival (AoA) jitter: Ornstein-Uhlenbeck mean-reverting process
    ou_theta: float = 0.25      # Reversion rate
    ou_sigma_px: float = 4.0    # Jitter standard deviation (scaled by turbulence_strength)
    # Scintillation: Log-normal intensity modulation
    scintillation_variance: float = 0.2
    # Thermal beam blur variation
    blur_kernel_max: int = 7

    # Platform micro-vibration (camera pose jitter, sum of sinusoids)
    vibration_amplitude_deg: float = 0.08
    vibration_frequencies_hz: List[float] = field(default_factory=lambda: [1.2, 3.7, 9.5])

    # Sensor noise
    sensor_gaussian_sigma: float = 4.0
    salt_pepper_prob: float = 0.0005

    # Occlusion injection
    occlusion_active: bool = False
    occlusion_duration_sec: float = 1.5

    # Decoy bright artifacts (lens flare / stray light)
    decoy_count: int = 1
    decoy_intensity_ratio: float = 0.50


@dataclass
class DetectorConfig:
    """Detection algorithms (classical, discriminator, gating)."""
    # Mode: 'classical', 'hybrid', 'discriminator'
    mode: str = "hybrid"
    threshold_adaptive: bool = True
    threshold_min_val: int = 180
    min_blob_area_px: int = 4
    max_blob_area_px: int = 1200
    min_circularity: float = 0.4
    min_beacon_flux: float = 3000.0
    discriminator_confidence_threshold: float = 0.45
    mahalanobis_gate_threshold: float = 9.21  # Chi-squared 2 DOF (99% confidence level)


@dataclass
class KalmanConfig:
    """Kalman filter parameters."""
    dt: float = 1.0 / 60.0
    process_noise_accel_std: float = 8.0   # std dev of acceleration noise in degrees/s^2
    measurement_noise_std: float = 0.08    # measurement noise std dev in degrees
    initial_covariance_pos: float = 10.0
    initial_covariance_vel: float = 50.0


@dataclass
class StateMachineConfig:
    """Track state manager parameters."""
    candidate_confirm_frames: int = 3   # M of N frames to lock
    candidate_window_frames: int = 5
    coast_max_frames: int = 45          # ~0.75s at 60 FPS
    search_spiral_omega: float = 3.5    # Angular velocity of search spiral
    search_spiral_growth: float = 0.45  # Radial expansion rate (deg/s)
    velocity_bias_weight: float = 0.75  # Weight given to KF last known velocity in reacquisition


@dataclass
class ControllerConfig:
    """Dual-axis PID and adaptive control parameters."""
    kp_pan: float = 15.0
    ki_pan: float = 0.5
    kd_pan: float = 0.75
    kp_tilt: float = 15.0
    ki_tilt: float = 0.5
    kd_tilt: float = 0.75
    # Integrator anti-windup clamping (deg)
    integral_limit_deg: float = 1.0
    # Derivative filter coefficient (alpha in low-pass filter on d_error)
    derivative_filter_alpha: float = 0.60
    # Covariance-aware gain damping factor (alpha_cov)
    covariance_damping_enabled: bool = True
    covariance_damping_alpha: float = 0.002
    # Alternative controller: 'pid' or 'mpc_lite'
    mode: str = "pid"


@dataclass
class TargetConfig:
    """Kinematic target trajectory configuration."""
    # Motion type: 'linear', 'circular', 'figure8', 'random_walk', 'multi_target'
    motion_type: str = "figure8"
    speed_deg_s: float = 2.5
    trajectory_radius_deg: float = 2.0
    initial_pan_deg: float = 0.0
    initial_tilt_deg: float = 0.0
    target_count: int = 1


@dataclass
class AppConfig:
    """Global application configuration combining all sub-systems."""
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    target_fps: float = 60.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "AppConfig":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = cls()
        if "optical" in data:
            cfg.optical = OpticalConfig(**data["optical"])
        if "gimbal" in data:
            cfg.gimbal = GimbalConfig(**data["gimbal"])
        if "disturbance" in data:
            cfg.disturbance = DisturbanceConfig(**data["disturbance"])
        if "detector" in data:
            cfg.detector = DetectorConfig(**data["detector"])
        if "kalman" in data:
            cfg.kalman = KalmanConfig(**data["kalman"])
        if "state_machine" in data:
            cfg.state_machine = StateMachineConfig(**data["state_machine"])
        if "controller" in data:
            cfg.controller = ControllerConfig(**data["controller"])
        if "target" in data:
            cfg.target = TargetConfig(**data["target"])
        if "target_fps" in data:
            cfg.target_fps = data["target_fps"]
        return cfg
