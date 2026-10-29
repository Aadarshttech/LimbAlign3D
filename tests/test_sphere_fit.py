"""
Unit tests for 3D sphere fitting algorithms.
"""

import numpy as np
import pytest
from limbalign.geometry.sphere_fit import fit_sphere_least_squares, fit_sphere_ransac


def test_fit_sphere_least_squares_exact():
    """Tests exact recovery of known center and radius from clean points."""
    true_center = np.array([25.0, -10.0, 150.0])
    true_radius = 24.0

    # Generate points uniformly distributed on sphere surface
    rng = np.random.RandomState(42)
    u = rng.uniform(0, 1, 500)
    v = rng.uniform(0, 1, 500)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    x = true_center[0] + true_radius * np.sin(phi) * np.cos(theta)
    y = true_center[1] + true_radius * np.sin(phi) * np.sin(theta)
    z = true_center[2] + true_radius * np.cos(phi)
    points = np.column_stack([x, y, z])

    est_center, est_radius, rmse = fit_sphere_least_squares(points)

    assert np.allclose(est_center, true_center, atol=1e-3)
    assert np.isclose(est_radius, true_radius, atol=1e-3)
    assert rmse < 1e-3


def test_fit_sphere_with_noise():
    """Tests recovery under mild Gaussian surface noise."""
    true_center = np.array([0.0, 0.0, 0.0])
    true_radius = 22.5

    rng = np.random.RandomState(42)
    u = rng.uniform(0, 1, 300)
    v = rng.uniform(0, 1, 300)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    r = true_radius + rng.normal(0, 0.2, 300)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.column_stack([x, y, z])

    est_center, est_radius, rmse = fit_sphere_ransac(points, seed=42)

    assert np.allclose(est_center, true_center, atol=0.25)
    assert np.isclose(est_radius, true_radius, atol=0.25)
    assert rmse < 0.50

