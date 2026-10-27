"""
Segmentation evaluation metrics: Dice Similarity Coefficient and 95% Hausdorff Distance (HD95 in mm).
Implements the exact formulations specified in the build guide.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from .labels import LABELS


def compute_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Computes binary Dice coefficient between two boolean or binary arrays."""
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    s = a_bool.sum() + b_bool.sum()
    if s == 0:
        return 1.0
    intersection = np.logical_and(a_bool, b_bool).sum()
    return float(2.0 * intersection / s)


def compute_hd95(a: np.ndarray, b: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    """
    Symmetric 95th percentile Hausdorff Distance (HD95) in physical millimeters.
    Computes maximum of the two directed 95th percentiles.

    Args:
        a: boolean 3D array (z, y, x) or (x, y, z).
        b: boolean 3D array (z, y, x) or (x, y, z).
        spacing_zyx: Voxel spacing along the array axes in millimeters.

    Returns:
        HD95 value in millimeters. Returns NaN if either mask is empty.
    """
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)

    if not a_bool.any() or not b_bool.any():
        return float("nan")

    # Surface voxel extraction via binary erosion
    sa = a_bool & ~ndi.binary_erosion(a_bool)
    sb = b_bool & ~ndi.binary_erosion(b_bool)

    # Euclidean distance transforms in physical space
    d_to_a = ndi.distance_transform_edt(~sa, sampling=spacing_zyx)
    d_to_b = ndi.distance_transform_edt(~sb, sampling=spacing_zyx)

    dist_b_to_a = d_to_a[sb]
    dist_a_to_b = d_to_b[sa]

    if len(dist_b_to_a) == 0 or len(dist_a_to_b) == 0:
        return float("nan")

    p95_b_to_a = np.percentile(dist_b_to_a, 95)
    p95_a_to_b = np.percentile(dist_a_to_b, 95)

    return float(max(p95_b_to_a, p95_a_to_b))


def evaluate_segmentation(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    classes: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Evaluates multi-class bone segmentation across all labels.

    Returns:
        pd.DataFrame with columns: ['Structure', 'Label_ID', 'Dice', 'HD95_mm']
    """
    target_classes = classes or {k: v for k, v in LABELS.items() if v > 0}
    rows = []

    for name, val in target_classes.items():
        p_bin = (pred_mask == val)
        g_bin = (gt_mask == val)

        dice_score = compute_dice(p_bin, g_bin)
        hd95_score = compute_hd95(p_bin, g_bin, spacing_zyx=spacing)

        rows.append({
            "Structure": name.capitalize(),
            "Label_ID": val,
            "Dice": round(dice_score, 4),
            "HD95_mm": round(hd95_score, 2) if not np.isnan(hd95_score) else None,
        })

    df = pd.DataFrame(rows)
    return df

