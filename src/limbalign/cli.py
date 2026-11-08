"""
Command-line interface for LimbAlign3D lower-limb alignment analysis.
"""

import os
import argparse
import json
import pandas as pd
import nibabel as nib
import numpy as np

from .seg.labels import split_bilateral_bones
from .seg.metrics import evaluate_segmentation
from .geometry.landmarks import extract_limb_landmarks
from .geometry.angles import compute_limb_alignment
from .validation.qc import run_quality_control_checks
from .validation.perturbations import apply_gaussian_noise, apply_voxel_downsampling
from .viz.mesh import extract_surface_mesh, export_binary_stl
from .viz.axis_overlay import render_mechanical_axes_3d
from .viz.plots import plot_coronal_alignment_projection, plot_robustness_curves
from .data.synthetic import generate_synthetic_lower_limb_case


def process_case(ct_path: str, mask_path: str, output_dir: str) -> None:
    """Executes full end-to-end alignment analysis on a single patient scan."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading CT:   {ct_path}")
    print(f"Loading Mask: {mask_path}")

    ct_nii = nib.load(ct_path)
    mask_nii = nib.load(mask_path)

    ct_arr = np.asanyarray(ct_nii.dataobj).astype(np.float32)
    mask_arr = np.asanyarray(mask_nii.dataobj).astype(np.uint8)
    spacing = tuple(float(s) for s in ct_nii.header.get_zooms())

    print(f"Volume shape: {ct_arr.shape} | Voxel spacing: {spacing}")

    # 1. Split bilateral bones
    print("Separating bilateral anatomical instances (Left vs Right)...")
    bones = split_bilateral_bones(mask_arr, spacing=spacing)

    reports = []
    qc_reports = []

    for side in ["Right", "Left"]:
        fem_key = f"femur_{side.lower()}"
        tib_key = f"tibia_{side.lower()}"

        if not np.any(bones[fem_key]) or not np.any(bones[tib_key]):
            print(f"Skipping {side} limb: incomplete segmentation mask.")
            continue

        print(f"Extracting 3D landmarks for {side} limb...")
        landmarks = extract_limb_landmarks(
            femur_mask=bones[fem_key],
            tibia_mask=bones[tib_key],
            spacing=spacing,
            side=side,
        )

        print(f"Computing mechanical alignment angles ({side})...")
        alignment = compute_limb_alignment(landmarks)
        reports.append(alignment.to_dict())

        # Quality Control audit
        qc = run_quality_control_checks(alignment)
        qc_reports.append({"side": side, **qc.to_dict()})

        print(f"  [{side} Limb]")
        print(f"    HKA (Coronal): {alignment.hka_coronal_deg:.2f}° ({alignment.alignment_type})")
        print(f"    mLDFA:         {alignment.mldfa_deg:.2f}° (Normal ~87°)")
        print(f"    mMPTA:         {alignment.mmpta_deg:.2f}° (Normal ~87°)")
        print(f"    JLCA:          {alignment.jlca_deg:.2f}°")
        print(f"    QC Status:     {qc.status}")

        # Render visualizations
        print(f"Rendering visualizations for {side} limb...")
        fem_v, fem_f, _ = extract_surface_mesh(bones[fem_key], spacing=spacing)
        tib_v, tib_f, _ = extract_surface_mesh(bones[tib_key], spacing=spacing)

        # 3D Mechanical Axis Plot
        render_mechanical_axes_3d(
            femur_verts=fem_v,
            tibia_verts=tib_v,
            landmarks=landmarks,
            alignment=alignment,
            save_path=os.path.join(output_dir, f"mechanical_axis_3d_{side.lower()}.png"),
        )

        # Coronal MIP Projection Plot
        plot_coronal_alignment_projection(
            ct_volume=ct_arr,
            landmarks=landmarks,
            alignment=alignment,
            spacing=spacing,
            save_path=os.path.join(output_dir, f"coronal_projection_{side.lower()}.png"),
        )

        # Export STL Meshes
        stl_fem = os.path.join(output_dir, f"{side.lower()}_femur.stl")
        stl_tib = os.path.join(output_dir, f"{side.lower()}_tibia.stl")
        export_binary_stl(fem_v, fem_f, stl_fem, header_title=f"{side} Femur Mesh")
        export_binary_stl(tib_v, tib_f, stl_tib, header_title=f"{side} Tibia Mesh")

    # Export CSV summary reports
    df_rep = pd.DataFrame(reports)
    csv_path = os.path.join(output_dir, "alignment_report.csv")
    df_rep.to_csv(csv_path, index=False)

    df_qc = pd.DataFrame(qc_reports)
    qc_csv_path = os.path.join(output_dir, "quality_control_report.csv")
    df_qc.to_csv(qc_csv_path, index=False)

    print(f"Alignment report saved to: {csv_path}")
    print(f"QC audit saved to:         {qc_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="LimbAlign3D: Lower-Limb Alignment and Deformity Analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: process
    proc_p = subparsers.add_parser("process", help="Analyze lower-limb CT alignment")
    proc_p.add_argument("--ct", required=True, help="Path to CT NIfTI volume")
    proc_p.add_argument("--mask", required=True, help="Path to multi-label bone mask")
    proc_p.add_argument("--output", default="reports", help="Output directory for reports and STLs")

    # Command: evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluate segmentation metrics (Dice, HD95)")
    eval_p.add_argument("--pred", required=True, help="Path to predicted mask")
    eval_p.add_argument("--gt", required=True, help="Path to ground truth mask")
    eval_p.add_argument("--output", default="reports/metrics.csv", help="Output metrics CSV path")

    # Command: synthetic
    syn_p = subparsers.add_parser("synthetic", help="Generate synthetic test case")
    syn_p.add_argument("--output", default="data/synthetic", help="Output directory")
    syn_p.add_argument("--case_id", default="case_syn001", help="Subject identifier")

    args = parser.parse_args()

    if args.command == "process":
        process_case(args.ct, args.mask, args.output)
    elif args.command == "evaluate":
        p_nii = nib.load(args.pred)
        g_nii = nib.load(args.gt)
        p_data = np.asanyarray(p_nii.dataobj).astype(np.uint8)
        g_data = np.asanyarray(g_nii.dataobj).astype(np.uint8)
        sp = tuple(float(s) for s in g_nii.header.get_zooms())
        df = evaluate_segmentation(p_data, g_data, spacing=sp)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(df.to_string(index=False))
    elif args.command == "synthetic":
        case = generate_synthetic_lower_limb_case(args.output, case_id=args.case_id)
        print(f"Generated synthetic case: {case}")


if __name__ == "__main__":
    main()

