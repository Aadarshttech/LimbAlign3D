"""
Validation, perturbation stress-testing, and quality control gates.
"""

from .perturbations import (
    apply_gaussian_noise,
    apply_voxel_downsampling,
    apply_fov_truncation,
    apply_rotation_perturbation,
)
from .qc import QCReport, run_quality_control_checks

__all__ = [
    "apply_gaussian_noise",
    "apply_voxel_downsampling",
    "apply_fov_truncation",
    "apply_rotation_perturbation",
    "QCReport",
    "run_quality_control_checks",
]

