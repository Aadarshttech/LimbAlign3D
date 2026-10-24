"""
Dataset loading, train/val/test splitting, and NIfTI I/O.
"""

import os
import json
import glob
from typing import Dict, List, Tuple, Optional
import nibabel as nib
import numpy as np


def load_splits(config_path: str = "configs/splits.json") -> Dict[str, List[str]]:
    """Loads fixed stratified train/val/test split configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_lower_limb_cases(data_dir: str) -> List[Dict[str, str]]:
    """
    Finds paired CT scans and multi-label segmentation masks.
    Supports both nnU-Net convention (case_001_0000.nii.gz and case_001.nii.gz)
    and standard clinical directory trees.
    """
    all_nii = glob.glob(os.path.join(data_dir, "**", "*.nii*"), recursive=True)
    ct_candidates = [
        f for f in all_nii
        if ("_0000.nii" in f or "_ct." in f.lower() or "image" in f.lower())
        and "seg" not in f.lower() and "mask" not in f.lower()
    ]

    paired = []
    for ct in sorted(ct_candidates):
        base_name = os.path.basename(ct)
        case_id = base_name.replace("_0000.nii.gz", "").replace("_0000.nii", "").replace("_ct.nii.gz", "")

        ct_dir = os.path.dirname(ct)
        # Search parallel or same folder for matching mask
        possible_masks = [
            os.path.join(ct_dir, f"{case_id}.nii.gz"),
            os.path.join(ct_dir, f"{case_id}.nii"),
            os.path.join(ct_dir, f"{case_id}_seg.nii.gz"),
            os.path.join(ct_dir, "..", "labelsTr", f"{case_id}.nii.gz"),
            os.path.join(ct_dir, "..", "labelsTs", f"{case_id}.nii.gz"),
        ]

        found_mask = None
        for pm in possible_masks:
            if os.path.exists(pm) and pm != ct:
                found_mask = pm
                break

        if found_mask:
            paired.append({
                "case_id": case_id,
                "ct": os.path.abspath(ct),
                "seg": os.path.abspath(found_mask),
            })

    return paired

