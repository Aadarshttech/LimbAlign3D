"""
LimbAlign3D: Automated 3D Lower-Limb Alignment, Landmark Extraction, and Deformity Analysis from CT.
==================================================================================================
"""

__version__ = "0.1.0"
__author__ = "Aadarsh Pandita"

from .geometry import (
    fit_sphere_least_squares,
    fit_sphere_ransac,
    LimbLandmarks,
    extract_limb_landmarks,
    AlignmentReport,
    compute_limb_alignment,
)
from .seg import (
    LABELS,
    LABEL_NAMES,
    split_bilateral_bones,
    compute_dice,
    compute_hd95,
    evaluate_segmentation,
    build_limbalign_unet,
)
from .validation import (
    apply_gaussian_noise,
    apply_voxel_downsampling,
    apply_fov_truncation,
    apply_rotation_perturbation,
    QCReport,
    run_quality_control_checks,
)
from .viz import (
    extract_surface_mesh,
    export_binary_stl,
    render_mechanical_axes_3d,
    plot_coronal_alignment_projection,
    plot_robustness_curves,
)

__all__ = [
    "fit_sphere_least_squares",
    "fit_sphere_ransac",
    "LimbLandmarks",
    "extract_limb_landmarks",
    "AlignmentReport",
    "compute_limb_alignment",
    "LABELS",
    "LABEL_NAMES",
    "split_bilateral_bones",
    "compute_dice",
    "compute_hd95",
    "evaluate_segmentation",
    "build_limbalign_unet",
    "apply_gaussian_noise",
    "apply_voxel_downsampling",
    "apply_fov_truncation",
    "apply_rotation_perturbation",
    "QCReport",
    "run_quality_control_checks",
    "extract_surface_mesh",
    "export_binary_stl",
    "render_mechanical_axes_3d",
    "plot_coronal_alignment_projection",
    "plot_robustness_curves",
]

