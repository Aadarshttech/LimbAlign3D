"""
Dataset conversion, loading, and synthetic generation for lower-limb CT scans.
"""

from .convert import merge_segments, crop_to_labels
from .synthetic import generate_synthetic_lower_limb_case
from .loader import load_splits, find_lower_limb_cases

__all__ = [
    "merge_segments",
    "crop_to_labels",
    "generate_synthetic_lower_limb_case",
    "load_splits",
    "find_lower_limb_cases",
]

