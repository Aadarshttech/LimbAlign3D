"""
Automated anatomical landmark extraction from segmented lower-limb bones.
Computes femoral head center, knee centers, condyle points, and ankle center.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_erosion
from .sphere_fit import fit_sphere_ransac, fit_sphere_least_squares


@dataclass
class LimbLandmarks:
    """Anatomical landmarks for a single lower limb (Left or Right)."""
    side: str  # "Left" or "Right"
    femoral_head_center: np.ndarray        # [x, y, z] in mm
    femoral_head_radius_mm: float
    femoral_knee_center: np.ndarray        # [x, y, z] in mm
    medial_femoral_condyle: np.ndarray     # [x, y, z] in mm
    lateral_femoral_condyle: np.ndarray    # [x, y, z] in mm
    tibial_knee_center: np.ndarray         # [x, y, z] in mm
    medial_tibial_plateau: np.ndarray      # [x, y, z] in mm
    lateral_tibial_plateau: np.ndarray     # [x, y, z] in mm
    ankle_center: np.ndarray               # [x, y, z] in mm
    medial_malleolus: np.ndarray           # [x, y, z] in mm


def voxel_to_world(voxel_coords: np.ndarray, spacing: Tuple[float, float, float], origin: Optional[np.ndarray] = None) -> np.ndarray:
    """Converts (N, 3) voxel indices [z, y, x] or [x, y, z] to physical millimeters."""
    pts = voxel_coords * np.asarray(spacing)
    if origin is not None:
        pts += np.asarray(origin)
    return pts


def extract_femoral_head(femur_mask: np.ndarray, spacing: Tuple[float, float, float], side: str = "Right") -> Tuple[np.ndarray, float]:
    """
    Isolates the proximal femoral head and fits a 3D sphere.

    Args:
        femur_mask: 3D binary array where 1 is the target femur.
        spacing: Voxel zooms (dx, dy, dz) in mm.
        side: "Left" or "Right".

    Returns:
        head_center: [x, y, z] in physical mm.
        head_radius: radius in mm.
    """
    coords = np.argwhere(femur_mask)  # shape (N, 3) [z, y, x] or [x, y, z] depending on indexing
    if len(coords) < 100:
        raise ValueError("Insufficient femur voxels for landmark detection.")

    # Convert to physical mm (assuming coords are [x, y, z])
    pts_mm = coords.astype(np.float64) * np.asarray(spacing)

    # In standard orientation, Z is superior/inferior. Superior is proximal.
    z_min, z_max = np.min(pts_mm[:, 2]), np.max(pts_mm[:, 2])
    z_height = z_max - z_min

    # Proximal 25% of femur contains the head, neck, and greater trochanter
    proximal_cutoff = z_max - 0.25 * z_height
    proximal_mask = pts_mm[:, 2] >= proximal_cutoff
    prox_pts = pts_mm[proximal_mask]

    # Medial/Lateral differentiation:
    # Femoral head projects medially compared to greater trochanter.
    # In canonical RAS, Right medial is -X (towards body midline), Left medial is +X (towards midline)
    x_mean = np.mean(prox_pts[:, 0])
    if side.lower() == "right":
        # For Right femur, medial head is towards +X or -X depending on coordinate convention
        # Here we isolate the medial extrema cluster
        x_cutoff = np.percentile(prox_pts[:, 0], 35)
        head_pts = prox_pts[prox_pts[:, 0] <= x_cutoff]
    else:
        x_cutoff = np.percentile(prox_pts[:, 0], 65)
        head_pts = prox_pts[prox_pts[:, 0] >= x_cutoff]

    if len(head_pts) < 20:
        head_pts = prox_pts

    center, radius, _ = fit_sphere_ransac(head_pts)
    return center, radius


def extract_femoral_condyles(femur_mask: np.ndarray, spacing: Tuple[float, float, float], side: str = "Right") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifies medial and lateral distal condyles and computes the femoral knee center.

    Returns:
        knee_center: [x, y, z] in mm.
        medial_condyle: [x, y, z] in mm.
        lateral_condyle: [x, y, z] in mm.
    """
    coords = np.argwhere(femur_mask)
    pts_mm = coords.astype(np.float64) * np.asarray(spacing)

    z_min = np.min(pts_mm[:, 2])
    # Distal 12% of femur contains the condyles
    distal_cutoff = z_min + 0.12 * (np.max(pts_mm[:, 2]) - z_min)
    dist_pts = pts_mm[pts_mm[:, 2] <= distal_cutoff]

    # Split into medial and lateral halves along X axis
    x_mid = np.median(dist_pts[:, 0])
    pts_part1 = dist_pts[dist_pts[:, 0] < x_mid]
    pts_part2 = dist_pts[dist_pts[:, 0] >= x_mid]

    # Identify most distal point (lowest Z) in each condyle cluster
    condyle1 = pts_part1[np.argmin(pts_part1[:, 2])]
    condyle2 = pts_part2[np.argmin(pts_part2[:, 2])]

    if side.lower() == "right":
        medial_condyle = condyle1 if condyle1[0] < condyle2[0] else condyle2
        lateral_condyle = condyle2 if condyle1[0] < condyle2[0] else condyle1
    else:
        medial_condyle = condyle2 if condyle2[0] > condyle1[0] else condyle1
        lateral_condyle = condyle1 if condyle2[0] > condyle1[0] else condyle2

    knee_center = 0.5 * (medial_condyle + lateral_condyle)
    return knee_center, medial_condyle, lateral_condyle


def extract_tibial_landmarks(tibia_mask: np.ndarray, spacing: Tuple[float, float, float], side: str = "Right") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts proximal tibial knee center, plateau points, distal ankle center, and medial malleolus.

    Returns:
        tibial_knee_center: [x, y, z]
        medial_plateau: [x, y, z]
        lateral_plateau: [x, y, z]
        ankle_center: [x, y, z]
        medial_malleolus: [x, y, z]
    """
    coords = np.argwhere(tibia_mask)
    pts_mm = coords.astype(np.float64) * np.asarray(spacing)

    z_min, z_max = np.min(pts_mm[:, 2]), np.max(pts_mm[:, 2])
    total_height = z_max - z_min

    # Proximal 8% is the tibial plateau
    prox_pts = pts_mm[pts_mm[:, 2] >= z_max - 0.08 * total_height]
    x_mid = np.median(prox_pts[:, 0])
    plat1 = prox_pts[prox_pts[:, 0] < x_mid]
    plat2 = prox_pts[prox_pts[:, 0] >= x_mid]

    top1 = plat1[np.argmax(plat1[:, 2])]
    top2 = plat2[np.argmax(plat2[:, 2])]

    if side.lower() == "right":
        medial_plateau = top1 if top1[0] < top2[0] else top2
        lateral_plateau = top2 if top1[0] < top2[0] else top1
    else:
        medial_plateau = top2 if top2[0] > top1[0] else top1
        lateral_plateau = top1 if top2[0] > top1[0] else top2

    tibial_knee_center = 0.5 * (medial_plateau + lateral_plateau)

    # Distal 8% contains ankle plafond and medial malleolus
    dist_pts = pts_mm[pts_mm[:, 2] <= z_min + 0.08 * total_height]
    # Medial malleolus is the most inferior/distal extension
    medial_malleolus = dist_pts[np.argmin(dist_pts[:, 2])]

    # Ankle center: centroid of articular plafond (slightly proximal to malleolus tip)
    plafond_pts = dist_pts[dist_pts[:, 2] > (z_min + 0.02 * total_height)]
    if len(plafond_pts) > 0:
        ankle_center = np.mean(plafond_pts, axis=0)
    else:
        ankle_center = np.mean(dist_pts, axis=0)

    return tibial_knee_center, medial_plateau, lateral_plateau, ankle_center, medial_malleolus


def extract_limb_landmarks(
    femur_mask: np.ndarray,
    tibia_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    side: str = "Right",
) -> LimbLandmarks:
    """
    Extracts all clinical landmarks for a single limb.
    """
    head_center, head_radius = extract_femoral_head(femur_mask, spacing, side=side)
    fem_knee, med_cond, lat_cond = extract_femoral_condyles(femur_mask, spacing, side=side)
    tib_knee, med_plat, lat_plat, ankle, med_mall = extract_tibial_landmarks(tibia_mask, spacing, side=side)

    return LimbLandmarks(
        side=side,
        femoral_head_center=head_center,
        femoral_head_radius_mm=head_radius,
        femoral_knee_center=fem_knee,
        medial_femoral_condyle=med_cond,
        lateral_femoral_condyle=lat_cond,
        tibial_knee_center=tib_knee,
        medial_tibial_plateau=med_plat,
        lateral_tibial_plateau=lat_plat,
        ankle_center=ankle,
        medial_malleolus=med_mall,
    )

