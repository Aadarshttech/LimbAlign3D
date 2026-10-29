"""
3D sphere fitting algorithms for femoral head center localization.
Provides algebraic least-squares and RANSAC robust fitting.
"""

from typing import Tuple, Optional
import numpy as np


def fit_sphere_least_squares(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Fits a 3D sphere to a set of Cartesian surface points using linear least-squares.

    Args:
        points: (N, 3) array of Cartesian coordinates in millimeters [X, Y, Z].

    Returns:
        center: (3,) array [x_c, y_c, z_c] in millimeters.
        radius: float radius R in millimeters.
        rmse: float root-mean-square radial error in millimeters.
    """
    if len(points) < 4:
        raise ValueError(f"At least 4 non-coplanar points required for 3D sphere fit, got {len(points)}.")

    # Linear system formulation:
    # 2*x*x_c + 2*y*y_c + 2*z*z_c + (R^2 - x_c^2 - y_c^2 - z_c^2) = x^2 + y^2 + z^2
    A = np.column_stack([
        2.0 * points[:, 0],
        2.0 * points[:, 1],
        2.0 * points[:, 2],
        np.ones(len(points), dtype=np.float64),
    ])
    b = np.sum(points**2, axis=1)

    # Solve least-squares
    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    x_c, y_c, z_c, w = sol

    center = np.array([x_c, y_c, z_c], dtype=np.float64)
    radius_sq = w + np.sum(center**2)

    if radius_sq <= 0:
        # Fallback to centroid and mean distance if matrix is degenerate
        center = np.mean(points, axis=0)
        radius = float(np.mean(np.linalg.norm(points - center, axis=1)))
    else:
        radius = float(np.sqrt(radius_sq))

    distances = np.linalg.norm(points - center, axis=1)
    rmse = float(np.sqrt(np.mean((distances - radius) ** 2)))

    return center, radius, rmse


def fit_sphere_ransac(
    points: np.ndarray,
    max_iterations: int = 100,
    inlier_threshold_mm: float = 1.5,
    min_inlier_ratio: float = 0.6,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, float, float]:
    """
    Robust sphere fitting using RANSAC to eliminate femoral neck and trochanteric outliers.

    Args:
        points: (N, 3) point cloud in millimeters.
        max_iterations: Maximum RANSAC trials.
        inlier_threshold_mm: Distance tolerance to sphere boundary for inlier selection.
        min_inlier_ratio: Minimum ratio of inliers to accept fit.
        seed: Random seed for reproducibility.

    Returns:
        center: (3,) array [x_c, y_c, z_c].
        radius: float radius R.
        rmse: float root-mean-square error of inliers.
    """
    if len(points) < 10:
        return fit_sphere_least_squares(points)

    rng = np.random.RandomState(seed)
    n_points = len(points)
    best_inliers = None
    best_center = None
    best_radius = None
    best_rmse = float("inf")

    for _ in range(max_iterations):
        sample_idx = rng.choice(n_points, size=4, replace=False)
        sample = points[sample_idx]

        try:
            center, radius, _ = fit_sphere_least_squares(sample)
        except Exception:
            continue

        if not (10.0 <= radius <= 40.0):  # Plausible human femoral head bounds (mm)
            continue

        distances = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inliers = distances < inlier_threshold_mm
        inlier_count = np.sum(inliers)

        if inlier_count > (best_inliers.sum() if best_inliers is not None else 0):
            best_inliers = inliers
            best_center = center
            best_radius = radius

    # Re-fit using all inliers
    if best_inliers is not None and best_inliers.sum() >= int(min_inlier_ratio * n_points):
        inlier_pts = points[best_inliers]
        return fit_sphere_least_squares(inlier_pts)

    # Fallback to standard least-squares if RANSAC does not find adequate consensus
    return fit_sphere_least_squares(points)

