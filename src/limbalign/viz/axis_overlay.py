"""
3D rendering of lower-limb mechanical axes, joint lines, and anatomical landmark spheres.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..geometry.landmarks import LimbLandmarks
from ..geometry.angles import AlignmentReport


def render_mechanical_axes_3d(
    femur_verts: Optional[np.ndarray] = None,
    tibia_verts: Optional[np.ndarray] = None,
    landmarks: Optional[LimbLandmarks] = None,
    alignment: Optional[AlignmentReport] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Renders 3D anatomical bone surface point cloud with superimposed mechanical axis lines
    and clinical landmark spheres.
    """
    fig = plt.figure(figsize=(10, 14))
    ax = fig.add_subplot(111, projection="3d")

    # Downsample surface vertices for fast, crisp 3D visualization
    if femur_verts is not None and len(femur_verts) > 0:
        step_f = max(1, len(femur_verts) // 3000)
        ax.scatter(
            femur_verts[::step_f, 0],
            femur_verts[::step_f, 1],
            femur_verts[::step_f, 2],
            c="#d4ac0d",
            s=1.2,
            alpha=0.35,
            label="Femur Mesh",
        )

    if tibia_verts is not None and len(tibia_verts) > 0:
        step_t = max(1, len(tibia_verts) // 3000)
        ax.scatter(
            tibia_verts[::step_t, 0],
            tibia_verts[::step_t, 1],
            tibia_verts[::step_t, 2],
            c="#2980b9",
            s=1.2,
            alpha=0.35,
            label="Tibia Mesh",
        )

    if landmarks is not None:
        hip = landmarks.femoral_head_center
        knee = landmarks.femoral_knee_center
        ankle = landmarks.ankle_center
        med_cond = landmarks.medial_femoral_condyle
        lat_cond = landmarks.lateral_femoral_condyle
        med_plat = landmarks.medial_tibial_plateau
        lat_plat = landmarks.lateral_tibial_plateau

        # 1. Landmark spheres
        ax.scatter([hip[0]], [hip[1]], [hip[2]], c="#e74c3c", s=180, edgecolors="black", label=f"Hip Center (R={landmarks.femoral_head_radius_mm:.1f}mm)")
        ax.scatter([knee[0]], [knee[1]], [knee[2]], c="#f39c12", s=140, edgecolors="black", label="Knee Center")
        ax.scatter([ankle[0]], [ankle[1]], [ankle[2]], c="#2ecc71", s=140, edgecolors="black", label="Ankle Center")

        # 2. Femoral mechanical axis line (Hip -> Knee)
        ax.plot(
            [hip[0], knee[0]],
            [hip[1], knee[1]],
            [hip[2], knee[2]],
            color="#e74c3c",
            linewidth=3.2,
            label="Femoral Mechanical Axis",
        )

        # 3. Tibial mechanical axis line (Knee -> Ankle)
        ax.plot(
            [knee[0], ankle[0]],
            [knee[1], ankle[1]],
            [knee[2], ankle[2]],
            color="#27ae60",
            linewidth=3.2,
            label="Tibial Mechanical Axis",
        )

        # 4. Distal femoral joint line (Medial Condyle -> Lateral Condyle)
        ax.plot(
            [med_cond[0], lat_cond[0]],
            [med_cond[1], lat_cond[1]],
            [med_cond[2], lat_cond[2]],
            color="#8e44ad",
            linewidth=2.5,
            linestyle="--",
            label="Distal Femoral Joint Line",
        )

        # 5. Proximal tibial joint line (Medial Plateau -> Lateral Plateau)
        ax.plot(
            [med_plat[0], lat_plat[0]],
            [med_plat[1], lat_plat[1]],
            [med_plat[2], lat_plat[2]],
            color="#16a085",
            linewidth=2.5,
            linestyle="--",
            label="Proximal Tibial Joint Line",
        )

    title_text = "3D Lower-Limb Mechanical Alignment"
    if alignment is not None:
        title_text += f"\nHKA: {alignment.hka_coronal_deg:.1f}° ({alignment.alignment_type}) | mLDFA: {alignment.mldfa_deg:.1f}° | mMPTA: {alignment.mmpta_deg:.1f}°"

    ax.set_title(title_text, fontsize=12, fontweight="bold")
    ax.set_xlabel("X (mm) [Medial / Lateral]")
    ax.set_ylabel("Y (mm) [Anterior / Posterior]")
    ax.set_zlabel("Z (mm) [Inferior / Superior]")
    ax.view_init(elev=12, azim=-60)
    ax.legend(loc="lower left", fontsize=9, frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig

