"""
Dataset conversion, multi-label merging, and anatomical cropping utilities.
Implements the exact logic specified in Section 3.4 of the build guide.
"""

from typing import Dict, List, Tuple
import numpy as np
import SimpleITK as sitk
from ..seg.labels import LABELS


def merge_segments(ref_ct: sitk.Image, segment_files: Dict[str, List[str]]) -> sitk.Image:
    """
    Merges individual bone segmentations (e.g. left and right STL/NRRD masks)
    onto the reference CT grid into a single multi-label image.

    Args:
        ref_ct: SimpleITK reference CT volume.
        segment_files: Dictionary mapping structure name to list of file paths.
                       Example: {"femur": [left_path, right_path], "pelvis": [pelvis_path]}

    Returns:
        sitk.Image: Single multi-label segmentation mask matching reference CT geometry.
    """
    ct_arr = sitk.GetArrayFromImage(ref_ct)
    out_arr = np.zeros(ct_arr.shape, dtype=np.uint8)

    for name, paths in segment_files.items():
        if name not in LABELS:
            continue
        label_val = LABELS[name]
        for p in paths:
            seg_img = sitk.ReadImage(str(p))
            # Resample mask to reference CT grid
            resampled = sitk.Resample(
                seg_img,
                ref_ct,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
            mask_arr = sitk.GetArrayFromImage(resampled)
            out_arr[mask_arr > 0] = label_val

    res_img = sitk.GetImageFromArray(out_arr)
    res_img.CopyInformation(ref_ct)
    return res_img


def crop_to_labels(ct: sitk.Image, lab: sitk.Image, pad_mm: float = 100.0) -> Tuple[sitk.Image, sitk.Image]:
    """
    Crops full-body scan to the lower-limb region of interest plus a margin.
    A 100 mm margin ensures the network observes surrounding anatomical context
    (e.g., lower spine, paraspinal and gluteal soft tissue).

    Args:
        ct: SimpleITK CT image.
        lab: SimpleITK multi-label segmentation image.
        pad_mm: Margin padding in millimeters.

    Returns:
        (cropped_ct, cropped_lab)
    """
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(sitk.Cast(lab > 0, sitk.sitkUInt8))

    if not f.HasLabel(1):
        # If no labels found, return original images
        return ct, lab

    x, y, z, sx, sy, sz = f.GetBoundingBox(1)
    spacing = ct.GetSpacing()

    pad = [int(round(pad_mm / s)) for s in spacing]
    start = [max(0, b - p) for b, p in zip((x, y, z), pad)]
    end = [
        min(n, b + e + p)
        for n, b, e, p in zip(ct.GetSize(), (x, y, z), (sx, sy, sz), pad)
    ]
    size = [e - s for s, e in zip(start, end)]

    cropped_ct = sitk.RegionOfInterest(ct, size, start)
    cropped_lab = sitk.RegionOfInterest(lab, size, start)

    return cropped_ct, cropped_lab

