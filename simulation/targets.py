"""
simulation/targets.py - Kinematic Target Motion Models for FSOC Mobile Terminals.
Supports multiple trajectory types:
  1. Linear / Constant Velocity (with boundary reflections)
  2. Circular / Orbital
  3. Lemniscate of Bernoulli (Figure-8)
  4. Random Walk / Evasive Maneuver
  5. Multi-Target Crossing Scenarios
"""

import math
from typing import List, Tuple, Dict, Any
import numpy as np


class TargetKinematics:
    """Represents a single optical beacon target in world space (pan, tilt in degrees)."""

    def __init__(
        self,
        target_id: int,
        motion_type: str = "figure8",
        speed_deg_s: float = 3.0,
        trajectory_radius_deg: float = 5.0,
        initial_pan_deg: float = 0.0,
        initial_tilt_deg: float = 0.0,
        intensity: float = 245.0,
        priority: int = 1
    ):
        self.target_id = target_id
        self.motion_type = motion_type
        self.speed = speed_deg_s
        self.radius = trajectory_radius_deg
        self.base_pan = initial_pan_deg
        self.base_tilt = initial_tilt_deg
        self.intensity = intensity
        self.priority = priority

        # Current state
        self.pan_deg = initial_pan_deg
        self.tilt_deg = initial_tilt_deg
        self.pan_vel_deg_s = 0.0
        self.tilt_vel_deg_s = 0.0

        # Internal state for specific motion types
        self._linear_heading_rad = np.random.uniform(0, 2.0 * math.pi)
        self._rw_vel_pan = 0.0
        self._rw_vel_tilt = 0.0
        self._time = 0.0

    def update(self, dt: float) -> Tuple[float, float, float, float]:
        """
        Advances the target state by dt seconds.
        Returns:
            (pan_deg, tilt_deg, pan_vel_deg_s, tilt_vel_deg_s)
        """
        self._time += dt

        if self.motion_type == "figure8":
            # Lemniscate of Bernoulli:
            # x(t) = R * cos(omega * t)
            # y(t) = R * sin(2 * omega * t) / 2
            omega = (self.speed / self.radius) if self.radius > 0 else 0.5
            t = self._time * omega
            self.pan_deg = self.base_pan + self.radius * math.cos(t)
            self.tilt_deg = self.base_tilt + (self.radius * 0.55) * math.sin(2.0 * t) / 2.0

            # Analytical velocities
            self.pan_vel_deg_s = -self.radius * omega * math.sin(t)
            self.tilt_vel_deg_s = (self.radius * 0.55) * omega * math.cos(2.0 * t)

        elif self.motion_type == "circular":
            # Circular orbit
            omega = (self.speed / self.radius) if self.radius > 0 else 0.5
            t = self._time * omega
            self.pan_deg = self.base_pan + self.radius * math.cos(t)
            self.tilt_deg = self.base_tilt + self.radius * math.sin(t)

            self.pan_vel_deg_s = -self.radius * omega * math.sin(t)
            self.tilt_vel_deg_s = self.radius * omega * math.cos(t)

        elif self.motion_type == "linear":
            # Constant velocity with soft bouncing within a bounding box
            self.pan_deg += self.speed * math.cos(self._linear_heading_rad) * dt
            self.tilt_deg += self.speed * math.sin(self._linear_heading_rad) * dt

            # Bound checks (rebound within +/- 15 deg from base)
            if abs(self.pan_deg - self.base_pan) > 12.0:
                self._linear_heading_rad = math.pi - self._linear_heading_rad
            if abs(self.tilt_deg - self.base_tilt) > 8.0:
                self._linear_heading_rad = -self._linear_heading_rad

            self.pan_vel_deg_s = self.speed * math.cos(self._linear_heading_rad)
            self.tilt_vel_deg_s = self.speed * math.sin(self._linear_heading_rad)

        elif self.motion_type == "random_walk":
            # Gauss-Markov acceleration random walk
            accel_pan = np.random.normal(0, 3.5)
            accel_tilt = np.random.normal(0, 3.5)
            
            # Spring restore to base to prevent drifting infinitely
            restore_pan = -0.3 * (self.pan_deg - self.base_pan)
            restore_tilt = -0.3 * (self.tilt_deg - self.base_tilt)

            self._rw_vel_pan += (accel_pan + restore_pan) * dt
            self._rw_vel_tilt += (accel_tilt + restore_tilt) * dt

            # Damp velocities
            self._rw_vel_pan *= 0.96
            self._rw_vel_tilt *= 0.96

            self.pan_deg += self._rw_vel_pan * dt
            self.tilt_deg += self._rw_vel_tilt * dt

            self.pan_vel_deg_s = self._rw_vel_pan
            self.tilt_vel_deg_s = self._rw_vel_tilt

        elif self.motion_type == "banking_turn":
            # Tactical high-g banking maneuver: linear ingress -> sharp coordinated turn -> linear egress
            cycle_t = self._time % 12.0
            if cycle_t < 3.5:
                # Linear cruise phase
                self.pan_vel_deg_s = self.speed * 1.0
                self.tilt_vel_deg_s = 0.0
                self.pan_deg = self.base_pan - 3.0 + self.pan_vel_deg_s * cycle_t
                self.tilt_deg = self.base_tilt
            elif cycle_t < 8.5:
                # 180° Coordinated banking turn with turn rate omega ~ 0.63 rad/s
                turn_t = cycle_t - 3.5
                omega = math.pi / 5.0
                turn_r = (self.speed * 1.0) / omega
                self.pan_deg = (self.base_pan - 3.0 + self.speed * 3.5) + turn_r * math.sin(omega * turn_t)
                self.tilt_deg = self.base_tilt + turn_r * (1.0 - math.cos(omega * turn_t))
                self.pan_vel_deg_s = self.speed * 1.0 * math.cos(omega * turn_t)
                self.tilt_vel_deg_s = self.speed * 1.0 * math.sin(omega * turn_t)
            else:
                # Linear egress phase
                egress_t = cycle_t - 8.5
                self.pan_vel_deg_s = -self.speed * 1.0
                self.tilt_vel_deg_s = 0.0
                turn_r = (self.speed * 1.0) / (math.pi / 5.0)
                self.pan_deg = (self.base_pan - 3.0 + self.speed * 3.5) + self.pan_vel_deg_s * egress_t
                self.tilt_deg = self.base_tilt + 2.0 * turn_r

        else:
            # Stationary beacon
            self.pan_vel_deg_s = 0.0
            self.tilt_vel_deg_s = 0.0

        return self.pan_deg, self.tilt_deg, self.pan_vel_deg_s, self.tilt_vel_deg_s


class TargetManager:
    """Manages one or more targets in the virtual environment."""

    def __init__(self, count: int = 1, motion_type: str = "figure8", speed: float = 3.0, radius: float = 5.0):
        self.targets: List[TargetKinematics] = []
        self._init_targets(count, motion_type, speed, radius)

    def _init_targets(self, count: int, motion_type: str, speed: float, radius: float):
        self.targets.clear()
        # Primary target (ID: 1)
        self.targets.append(
            TargetKinematics(
                target_id=1,
                motion_type=motion_type,
                speed_deg_s=speed,
                trajectory_radius_deg=radius,
                initial_pan_deg=2.0,
                initial_tilt_deg=-1.0,
                intensity=245.0,
                priority=1
            )
        )

        # Optional secondary targets (crossing or decoy targets)
        if count > 1:
            for i in range(2, count + 1):
                offset_pan = (i - 1) * 3.5 * (-1)**i
                offset_tilt = (i - 1) * 2.0 * (-1)**(i+1)
                self.targets.append(
                    TargetKinematics(
                        target_id=i,
                        motion_type="circular" if motion_type != "circular" else "linear",
                        speed_deg_s=speed * 0.8,
                        trajectory_radius_deg=radius * 0.7,
                        initial_pan_deg=offset_pan,
                        initial_tilt_deg=offset_tilt,
                        intensity=190.0,
                        priority=i
                    )
                )

    def reset(self, count: int = 1, motion_type: str = "figure8", speed: float = 3.0, radius: float = 5.0):
        self._init_targets(count, motion_type, speed, radius)

    def update(self, dt: float) -> List[Dict[str, Any]]:
        """Updates all targets and returns their current state list."""
        states = []
        for tgt in self.targets:
            pan, tilt, v_pan, v_tilt = tgt.update(dt)
            states.append({
                "id": tgt.target_id,
                "pan_deg": pan,
                "tilt_deg": tilt,
                "pan_vel_deg_s": v_pan,
                "tilt_vel_deg_s": v_tilt,
                "intensity": tgt.intensity,
                "priority": tgt.priority
            })
        return states

    @property
    def primary_target(self) -> TargetKinematics:
        return self.targets[0]
