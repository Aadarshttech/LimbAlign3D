"""
Geometric processing, 3D sphere fitting, landmark extraction, and alignment angles.
"""

from .sphere_fit import fit_sphere_least_squares, fit_sphere_ransac
from .landmarks import (
    LimbLandmarks,
    extract_limb_landmarks,
    extract_femoral_head,
    extract_femoral_condyles,
    extract_tibial_landmarks,
)
from .angles import (
    AlignmentReport,
    compute_limb_alignment,
    angle_between_vectors_deg,
)

__all__ = [
    "fit_sphere_least_squares",
    "fit_sphere_ransac",
    "LimbLandmarks",
    "extract_limb_landmarks",
    "extract_femoral_head",
    "extract_femoral_condyles",
    "extract_tibial_landmarks",
    "AlignmentReport",
    "compute_limb_alignment",
    "angle_between_vectors_deg",
]

