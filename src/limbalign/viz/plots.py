"""
Publication plotting utilities for coronal radiograph projection and perturbation stability.
"""

from typing import Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from ..geometry.landmarks import LimbLandmarks
from ..geometry.angles import AlignmentReport


def plot_coronal_alignment_projection(
    ct_volume: np.ndarray,
    landmarks: LimbLandmarks,
    alignment: AlignmentReport,
    spacing: tuple = (1.5, 1.5, 2.0),
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Renders coronal Maximum Intensity Projection (MIP) simulating a full-leg radiograph,
    with overlaid clinical axes and angles.
    """
    # Coronal projection: average or max across Y axis
    # Shape of ct_volume: (W, H, D) [X, Y, Z]
    mip_coronal = np.max(ct_volume, axis=1)  # shape (X, Z)

    fig, ax = plt.subplots(figsize=(8, 14))
    # Window to bone [-200, 1500] HU
    mip_windowed = np.clip(mip_coronal, -200, 1500)
    mip_norm = (mip_windowed - (-200)) / (1700)

    dx, dy, dz = spacing
    extent = [0, ct_volume.shape[0] * dx, 0, ct_volume.shape[2] * dz]

    ax.imshow(mip_norm.T, cmap="bone", origin="lower", extent=extent)

    # Coordinates in physical mm
    hip = landmarks.femoral_head_center
    knee = landmarks.femoral_knee_center
    ankle = landmarks.ankle_center
    med_cond = landmarks.medial_femoral_condyle
    lat_cond = landmarks.lateral_femoral_condyle
    med_plat = landmarks.medial_tibial_plateau
    lat_plat = landmarks.lateral_tibial_plateau

    # 1. Femoral Mechanical Axis
    ax.plot([hip[0], knee[0]], [hip[2], knee[2]], color="#e74c3c", linewidth=2.8, label="Femoral Axis")
    # 2. Tibial Mechanical Axis
    ax.plot([knee[0], ankle[0]], [knee[2], ankle[2]], color="#2ecc71", linewidth=2.8, label="Tibial Axis")
    # 3. Distal femoral joint line
    ax.plot([med_cond[0], lat_cond[0]], [med_cond[2], lat_cond[2]], color="#f39c12", linewidth=2.0, linestyle="--", label="Distal Femur Line")
    # 4. Proximal tibial joint line
    ax.plot([med_plat[0], lat_plat[0]], [med_plat[2], lat_plat[2]], color="#1abc9c", linewidth=2.0, linestyle="--", label="Tibial Plateau Line")

    # Landmark points
    ax.scatter([hip[0]], [hip[2]], color="#e74c3c", s=120, edgecolors="white", zorder=5)
    ax.scatter([knee[0]], [knee[2]], color="#f39c12", s=100, edgecolors="white", zorder=5)
    ax.scatter([ankle[0]], [ankle[2]], color="#2ecc71", s=100, edgecolors="white", zorder=5)

    # Annotated clinical text box
    info_str = (
        f"Limb: {alignment.side}\n"
        f"HKA (Coronal): {alignment.hka_coronal_deg:.1f}°\n"
        f"Deformity: {alignment.alignment_type} ({alignment.mechanical_axis_deviation_deg:+.1f}°)\n"
        f"mLDFA: {alignment.mldfa_deg:.1f}°\n"
        f"mMPTA: {alignment.mmpta_deg:.1f}°\n"
        f"JLCA: {alignment.jlca_deg:.1f}°\n"
        f"Femur Length: {alignment.femoral_mechanical_length_mm:.0f} mm\n"
        f"Tibia Length: {alignment.tibial_mechanical_length_mm:.0f} mm"
    )
    ax.text(
        0.05, 0.95, info_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="black", alpha=0.75, edgecolor="white"),
        color="white",
        fontfamily="monospace",
    )

    ax.set_title(f"LimbAlign3D Coronal Mechanical Alignment ({alignment.side})", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (mm) [Medial / Lateral]")
    ax.set_ylabel("Z (mm) [Inferior / Superior]")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_robustness_curves(
    noise_levels: List[float],
    hka_noise: List[float],
    downsample_scales: List[float],
    hka_downsample: List[float],
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Renders stability curves showing resilience of computed HKA angles
    under increasing sensor noise and voxel downsampling.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Noise stability
    ax1.plot(noise_levels, hka_noise, marker="o", color="#e74c3c", linewidth=2.2)
    ax1.axhline(hka_noise[0], color="gray", linestyle="--", alpha=0.7, label=f"Baseline HKA ({hka_noise[0]:.2f}°)")
    ax1.set_xlabel(r"Gaussian CT Noise $\sigma$ (Hounsfield Units)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Measured HKA Angle (deg)", fontsize=11, fontweight="bold")
    ax1.set_title("Angle Stability vs CT Sensor Noise", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Downsampling stability
    ax2.plot(downsample_scales, hka_downsample, marker="s", color="#3498db", linewidth=2.2)
    ax2.axhline(hka_downsample[0], color="gray", linestyle="--", alpha=0.7, label=f"Baseline HKA ({hka_downsample[0]:.2f}°)")
    ax2.set_xlabel("Voxel Resolution Scaling Factor", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Measured HKA Angle (deg)", fontsize=11, fontweight="bold")
    ax2.set_title("Angle Stability vs Spatial Downsampling", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
