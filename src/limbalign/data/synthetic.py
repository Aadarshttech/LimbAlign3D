"""
Procedural synthetic lower-limb CT generator for testing and offline execution.
Produces anatomically plausible pelvis, bilateral femurs, tibias, and fibulas.
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np
import nibabel as nib
from ..seg.labels import LABELS


def generate_synthetic_lower_limb_case(
    output_dir: str,
    case_id: str = "case_syn001",
    shape: Tuple[int, int, int] = (128, 128, 256),
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0),
    seed: int = 42,
) -> Dict[str, str]:
    """
    Generates a full bilateral lower-limb CT volume and multi-label segmentation mask.

    Anatomical layout along Z-axis (superior to inferior, 0 to Z_max):
    - Z = 200..250: Pelvis and hip joints
    - Z = 120..220: Femurs (femoral heads at ~220, shafts down to condyles at ~120)
    - Z = 20..115:  Tibias & Fibulas (knee plateau at ~115, ankle plafond at ~20)
    - Z = 0..20:    Feet & calcaneus (background/tarsals)
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    W, H, D = shape
    dx, dy, dz = spacing
    affine = np.diag([dx, dy, dz, 1.0])

    # Base volume: Soft tissue background (+35 HU)
    ct_arr = rng.normal(35.0, 15.0, size=(W, H, D)).astype(np.float32)
    seg_arr = np.zeros((W, H, D), dtype=np.uint8)

    # Ambient air outside bilateral thigh/leg soft tissue cylinders
    cx_right, cx_left = int(0.35 * W), int(0.65 * W)
    cy = int(0.50 * H)

    # Soft tissue envelope
    y_grid, x_grid = np.ogrid[:W, :H]
    body_mask = np.zeros((W, H, D), dtype=bool)

    # Scale factors relative to canonical 256 depth
    scale_z = D / 256.0

    # Upper pelvis soft tissue ellipse
    pelvis_ellipse = ((x_grid - cy)**2 / (0.35 * H)**2 + (y_grid - W//2)**2 / (0.42 * W)**2) <= 1.0
    z_pelvis_start = int(190 * scale_z)
    for z in range(z_pelvis_start, D):
        body_mask[:, :, z] = pelvis_ellipse

    # Bilateral leg cylinders
    r_leg = int(0.14 * H)
    leg_right = ((x_grid - cy)**2 + (y_grid - cx_right)**2) <= r_leg**2
    leg_left = ((x_grid - cy)**2 + (y_grid - cx_left)**2) <= r_leg**2
    for z in range(0, z_pelvis_start):
        body_mask[:, :, z] = (leg_right | leg_left)

    # Set air (-950 HU) outside body
    for z in range(D):
        non_body = ~body_mask[:, :, z]
        ct_arr[non_body, z] = rng.normal(-950.0, 20.0, size=non_body.sum())

    # -------------------------------------------------------------
    # 1. Pelvis (Label 1): Iliac wings, acetabular cups, sacrum
    # -------------------------------------------------------------
    z_p_min, z_p_max = int(200 * scale_z), min(D, int(245 * scale_z))
    for z in range(z_p_min, z_p_max):
        z_norm = (z - z_p_min) / max(1, z_p_max - z_p_min)
        # Iliac crest curve
        pelvis_bone = (((x_grid - cy)**2 / (0.12 * H)**2 + (y_grid - W//2)**2 / (0.35 * W * (1.1 - 0.4*z_norm))**2) <= 1.0) & \
                      (((x_grid - cy)**2 / (0.06 * H)**2 + (y_grid - W//2)**2 / (0.18 * W)**2) >= 0.7)
        pelvis_bone = pelvis_bone & body_mask[:, :, z]
        seg_arr[pelvis_bone, z] = LABELS["pelvis"]
        ct_arr[pelvis_bone, z] = rng.normal(900.0, 120.0, size=pelvis_bone.sum())

    # -------------------------------------------------------------
    # 2. Femurs (Label 2): Bilateral
    # -------------------------------------------------------------
    for side_sign, cx_fem in [(-1, cx_right), (1, cx_left)]:
        # Femoral head: sphere at Z=215, offset medially
        head_cz = int(215 * scale_z)
        head_cx = cx_fem + side_sign * int(0.04 * W)
        head_cy = cy + int(0.02 * H)
        head_r = int(14.0 / dx)  # ~21 mm radius

        # Femoral head sphere
        z_grid, y_g, x_g = np.ogrid[:W, :H, :D]
        dist_head = ((z_grid - head_cx)**2 + (y_g - head_cy)**2 + (x_g - head_cz)**2)
        head_mask = (dist_head <= head_r**2) & body_mask
        seg_arr[head_mask] = LABELS["femur"]

        cortical_head = (dist_head <= head_r**2) & (dist_head >= (0.75 * head_r)**2)
        cancellous_head = dist_head < (0.75 * head_r)**2
        ct_arr[cancellous_head & body_mask] = rng.normal(480.0, 45.0, size=(cancellous_head & body_mask).sum())
        ct_arr[cortical_head & body_mask] = rng.normal(1350.0, 90.0, size=(cortical_head & body_mask).sum())

        # Femoral shaft: from Z=130 to Z=210
        shaft_r = int(8.0 / dx)  # ~12 mm radius
        z_s_min, z_s_max = int(130 * scale_z), int(212 * scale_z)
        for z in range(z_s_min, z_s_max):
            z_ratio = (z - z_s_min) / max(1, z_s_max - z_s_min)
            curr_cx = cx_fem + int(side_sign * 0.04 * W * z_ratio)
            dist_shaft = ((x_grid - cy)**2 + (y_grid - curr_cx)**2)
            shaft_mask = (dist_shaft <= shaft_r**2) & body_mask[:, :, z]
            cortex_shaft = (dist_shaft <= shaft_r**2) & (dist_shaft >= (0.65 * shaft_r)**2) & body_mask[:, :, z]
            core_shaft = (dist_shaft < (0.65 * shaft_r)**2) & body_mask[:, :, z]

            seg_arr[shaft_mask, z] = LABELS["femur"]
            ct_arr[core_shaft, z] = rng.normal(380.0, 35.0, size=core_shaft.sum())
            ct_arr[cortex_shaft, z] = rng.normal(1500.0, 80.0, size=cortex_shaft.sum())

        # Distal condyles: Z=120 to Z=130
        z_c_min, z_c_max = int(120 * scale_z), int(131 * scale_z)
        for z in range(z_c_min, z_c_max):
            med_cx = cx_fem + side_sign * int(0.03 * W)
            lat_cx = cx_fem - side_sign * int(0.03 * W)
            cond_r = int(7.0 / dx)
            med_cond = (((x_grid - cy)**2 + (y_grid - med_cx)**2) <= cond_r**2) & body_mask[:, :, z]
            lat_cond = (((x_grid - cy)**2 + (y_grid - lat_cx)**2) <= cond_r**2) & body_mask[:, :, z]
            cond_mask = med_cond | lat_cond

            seg_arr[cond_mask, z] = LABELS["femur"]
            ct_arr[cond_mask, z] = rng.normal(1100.0, 95.0, size=cond_mask.sum())

    # -------------------------------------------------------------
    # 3. Tibias (Label 3) & Fibulas (Label 4): Bilateral
    # -------------------------------------------------------------
    for side_sign, cx_tib in [(-1, cx_right), (1, cx_left)]:
        # Tibial plateau
        z_tp_min, z_tp_max = int(108 * scale_z), int(116 * scale_z)
        for z in range(z_tp_min, z_tp_max):
            plat_r = int(12.0 / dx)
            plat_mask = (((x_grid - cy)**2 + (y_grid - cx_tib)**2) <= plat_r**2) & body_mask[:, :, z]
            seg_arr[plat_mask, z] = LABELS["tibia"]
            ct_arr[plat_mask, z] = rng.normal(950.0, 80.0, size=plat_mask.sum())

        # Tibial shaft
        z_ts_min, z_ts_max = int(30 * scale_z), int(108 * scale_z)
        tib_r = int(7.5 / dx)
        for z in range(z_ts_min, z_ts_max):
            shaft_mask = (((x_grid - cy)**2 + (y_grid - cx_tib)**2) <= tib_r**2) & body_mask[:, :, z]
            cortex = (((x_grid - cy)**2 + (y_grid - cx_tib)**2) >= (0.65 * tib_r)**2) & shaft_mask
            core = (((x_grid - cy)**2 + (y_grid - cx_tib)**2) < (0.65 * tib_r)**2) & shaft_mask

            seg_arr[shaft_mask, z] = LABELS["tibia"]
            ct_arr[core, z] = rng.normal(400.0, 40.0, size=core.sum())
            ct_arr[cortex, z] = rng.normal(1450.0, 75.0, size=cortex.sum())

        # Distal tibial plafond & medial malleolus
        z_ta_min, z_ta_max = int(20 * scale_z), int(30 * scale_z)
        for z in range(z_ta_min, z_ta_max):
            plaf_mask = (((x_grid - cy)**2 + (y_grid - (cx_tib + side_sign * 2))**2) <= (tib_r * 1.2)**2) & body_mask[:, :, z]
            seg_arr[plaf_mask, z] = LABELS["tibia"]
            ct_arr[plaf_mask, z] = rng.normal(1150.0, 85.0, size=plaf_mask.sum())

        # Fibula: thin lateral bone parallel to tibia
        cx_fib = cx_tib - side_sign * int(0.07 * W)
        fib_r = int(3.5 / dx)
        z_f_min, z_f_max = int(22 * scale_z), int(106 * scale_z)
        for z in range(z_f_min, z_f_max):
            fib_mask = (((x_grid - cy)**2 + (y_grid - cx_fib)**2) <= fib_r**2) & body_mask[:, :, z]
            seg_arr[fib_mask, z] = LABELS["fibula"]
            ct_arr[fib_mask, z] = rng.normal(1250.0, 90.0, size=fib_mask.sum())

    # Save NIfTI volumes
    ct_path = os.path.join(output_dir, f"{case_id}_0000.nii.gz")
    seg_path = os.path.join(output_dir, f"{case_id}.nii.gz")

    ct_nii = nib.Nifti1Image(ct_arr, affine)
    ct_nii.header.set_zooms(spacing)
    nib.save(ct_nii, ct_path)

    seg_nii = nib.Nifti1Image(seg_arr, affine)
    seg_nii.header.set_zooms(spacing)
    nib.save(seg_nii, seg_path)

    return {
        "case_id": case_id,
        "ct": os.path.abspath(ct_path),
        "seg": os.path.abspath(seg_path),
    }
