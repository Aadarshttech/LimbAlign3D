"""
Segmentation models, labels, and physical metrics for lower-limb bones.
"""

from .labels import LABELS, LABEL_NAMES, split_bilateral_bones
from .metrics import compute_dice, compute_hd95, evaluate_segmentation
from .model import build_limbalign_unet, infer_sliding_window

__all__ = [
    "LABELS",
    "LABEL_NAMES",
    "split_bilateral_bones",
    "compute_dice",
    "compute_hd95",
    "evaluate_segmentation",
    "build_limbalign_unet",
    "infer_sliding_window",
]

