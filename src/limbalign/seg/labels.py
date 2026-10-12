"""
Anatomical label definitions and bilateral (Left / Right) separation.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.ndimage import label as nd_label

# 4-class merged label scheme prescribed by the build guide
LABELS: Dict[str, int] = {
    "background": 0,
    "pelvis": 1,
    "femur": 2,
    "tibia": 3,
    "fibula": 4,
}

LABEL_NAMES = {v: k for k, v in LABELS.items()}


def split_bilateral_bones(
    multilabel_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, np.ndarray]:
    """
    Separates merged left/right bones into discrete anatomical instances
    using 3D connected components and physical midline coordinates.

    Returns:
        dict with keys:
            'pelvis', 'femur_left', 'femur_right', 'tibia_left', 'tibia_right',
            'fibula_left', 'fibula_right'
    """
    out = {
        "pelvis": (multilabel_mask == LABELS["pelvis"]).astype(np.uint8),
    }

    # Find patient midline along X axis based on pelvis or overall mask centroid
    body_voxels = np.where(multilabel_mask > 0)
    if len(body_voxels[0]) > 0:
        midline_x = np.median(body_voxels[0])
    else:
        midline_x = multilabel_mask.shape[0] // 2

    for bone_name in ["femur", "tibia", "fibula"]:
        val = LABELS[bone_name]
        bone_bin = (multilabel_mask == val)

        if not np.any(bone_bin):
            out[f"{bone_name}_left"] = np.zeros_like(multilabel_mask, dtype=np.uint8)
            out[f"{bone_name}_right"] = np.zeros_like(multilabel_mask, dtype=np.uint8)
            continue

        # Connected component analysis to find two dominant clusters
        labeled_cc, num_features = nd_label(bone_bin)
        if num_features >= 2:
            # Sort components by volume
            sizes = [(labeled_cc == i).sum() for i in range(1, num_features + 1)]
            top_two = np.argsort(sizes)[-2:] + 1

            c1_mask = (labeled_cc == top_two[0])
            c2_mask = (labeled_cc == top_two[1])

            c1_x = np.mean(np.where(c1_mask)[0])
            c2_x = np.mean(np.where(c2_mask)[0])

            # In canonical RAS: Right is -X (lower index if 0 is patient right), Left is +X
            # Here lower X is patient Right, higher X is patient Left
            if c1_x < c2_x:
                out[f"{bone_name}_right"] = c1_mask.astype(np.uint8)
                out[f"{bone_name}_left"] = c2_mask.astype(np.uint8)
            else:
                out[f"{bone_name}_right"] = c2_mask.astype(np.uint8)
                out[f"{bone_name}_left"] = c1_mask.astype(np.uint8)
        else:
            # Spatial half split fallback
            x_indices = np.where(bone_bin)[0]
            right_mask = bone_bin.copy()
            left_mask = bone_bin.copy()

            right_mask[midline_x:, :, :] = 0
            left_mask[:midline_x, :, :] = 0

            out[f"{bone_name}_right"] = right_mask.astype(np.uint8)
            out[f"{bone_name}_left"] = left_mask.astype(np.uint8)

    return out

