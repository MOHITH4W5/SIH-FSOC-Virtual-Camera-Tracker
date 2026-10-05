"""
core/geometry.py - Nonlinear Optical Projection & Coordinate Transformations.
Implements the exact arctangent pixel-to-angle and tangent angle-to-pixel equations
to prevent the spherical distortion errors of naive linear FOV models.
"""

import math
from typing import Tuple, Optional
import numpy as np


class CameraGeometry:
    """Handles projection between 2D sensor pixel coordinates and angular space."""

    def __init__(self, image_width: int, image_height: int, fov_h_deg: float, fov_v_deg: float):
        self.width = image_width
        self.height = image_height
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

        self.fov_h_rad = math.radians(fov_h_deg)
        self.fov_v_rad = math.radians(fov_v_deg)

        self.tan_half_fov_h = math.tan(self.fov_h_rad / 2.0)
        self.tan_half_fov_v = math.tan(self.fov_v_rad / 2.0)

        # Equivalent focal lengths in pixels (for standard pinhole matrix)
        self.fx = self.cx / self.tan_half_fov_h
        self.fy = self.cy / self.tan_half_fov_v

    def pixel_to_angle(self, px: float, py: float) -> Tuple[float, float]:
        """
        Converts sensor pixel coordinate (px, py) to angular offset (theta_pan, theta_tilt)
        from optical boresight using the exact arctangent mapping.
        
        Returns:
            (theta_pan_rad, theta_tilt_rad) in radians.
            Positive pan is to the right (+x), positive tilt is downward (+y in image, or upward depending on convention).
            Standard convention: pan = azimuth (+x), tilt = elevation (conventionally up = negative py from center).
        """
        dx = px - self.cx
        dy = py - self.cy

        # Normalized coordinates [-1, 1]
        norm_x = dx / self.cx
        norm_y = dy / self.cy

        theta_pan = math.atan(norm_x * self.tan_half_fov_h)
        # Note: image y grows downwards; here tilt_err > 0 means target is below boresight (need tilt down)
        theta_tilt = math.atan(norm_y * self.tan_half_fov_v)

        return theta_pan, theta_tilt

    def pixel_to_angle_deg(self, px: float, py: float) -> Tuple[float, float]:
        """Converts sensor pixel coordinate to angular offset in degrees."""
        pan_rad, tilt_rad = self.pixel_to_angle(px, py)
        return math.degrees(pan_rad), math.degrees(tilt_rad)

    def angle_to_pixel(self, theta_pan_rad: float, theta_tilt_rad: float) -> Tuple[float, float]:
        """
        Projects an angular offset from boresight back onto the sensor pixel coordinate plane.
        Inverse of pixel_to_angle.
        """
        norm_x = math.tan(theta_pan_rad) / self.tan_half_fov_h
        norm_y = math.tan(theta_tilt_rad) / self.tan_half_fov_v

        px = self.cx + norm_x * self.cx
        py = self.cy + norm_y * self.cy

        return px, py

    def world_target_to_pixel(
        self,
        target_pan_deg: float,
        target_tilt_deg: float,
        cam_pan_deg: float,
        cam_tilt_deg: float
    ) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Projects a world target position (pan, tilt in degrees) into the virtual camera sensor plane,
        given the camera's current pan and tilt attitude.
        
        Returns:
            (px, py, is_visible)
            is_visible is True if the projected centroid lies strictly inside the sensor bounds.
        """
        delta_pan_rad = math.radians(target_pan_deg - cam_pan_deg)
        delta_tilt_rad = math.radians(target_tilt_deg - cam_tilt_deg)

        # Check if target is behind camera or beyond FOV limits
        if abs(delta_pan_rad) >= math.pi / 2.0 or abs(delta_tilt_rad) >= math.pi / 2.0:
            return None, None, False

        px, py = self.angle_to_pixel(delta_pan_rad, delta_tilt_rad)

        is_visible = (0.0 <= px < self.width) and (0.0 <= py < self.height)
        return px, py, is_visible

    def angular_distance_mrad(self, px1: float, py1: float, px2: float, py2: float) -> float:
        """Computes true angular separation in milliradians between two pixel locations."""
        pan1, tilt1 = self.pixel_to_angle(px1, py1)
        pan2, tilt2 = self.pixel_to_angle(px2, py2)
        d_pan = pan1 - pan2
        d_tilt = tilt1 - tilt2
        angle_rad = math.hypot(d_pan, d_tilt)
        return angle_rad * 1000.0  # mrad
