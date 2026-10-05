"""
tests/test_geometry.py - Unit tests for nonlinear optical geometry transformations.
"""

import math
import pytest
from core.geometry import CameraGeometry


def test_geometry_projection_inversion():
    geom = CameraGeometry(image_width=1280, image_height=720, fov_h_deg=10.0, fov_v_deg=5.625)
    
    # Test boresight center
    pan_rad, tilt_rad = geom.pixel_to_angle(640.0, 360.0)
    assert abs(pan_rad) < 1e-7
    assert abs(tilt_rad) < 1e-7

    px, py = geom.angle_to_pixel(0.0, 0.0)
    assert abs(px - 640.0) < 1e-5
    assert abs(py - 360.0) < 1e-5

    # Test arbitrary off-center points and invertibility
    test_points = [(100.0, 150.0), (1200.0, 600.0), (640.0, 100.0), (300.0, 360.0)]
    for tx, ty in test_points:
        p_rad, t_rad = geom.pixel_to_angle(tx, ty)
        rx, ry = geom.angle_to_pixel(p_rad, t_rad)
        assert abs(rx - tx) < 1e-4
        assert abs(ry - ty) < 1e-4


def test_nonlinear_vs_linear_edge_effect():
    """Verifies that nonlinear atan mapping produces true angular angles rather than linear distortion."""
    geom = CameraGeometry(image_width=1280, image_height=720, fov_h_deg=10.0, fov_v_deg=5.625)
    
    # At exact right edge of sensor (px = 1280), angle should be exactly half FOV (+5.0 degrees)
    pan_deg, _ = geom.pixel_to_angle_deg(1280.0, 360.0)
    assert abs(pan_deg - 5.0) < 1e-5

    # At exact left edge (px = 0), angle should be exactly -5.0 degrees
    pan_deg_left, _ = geom.pixel_to_angle_deg(0.0, 360.0)
    assert abs(pan_deg_left - (-5.0)) < 1e-5
