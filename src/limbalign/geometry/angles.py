"""
Clinical lower-limb alignment angle calculations in 3D CT and projected 2D coronal planes.
Computes HKA (Hip-Knee-Ankle), mLDFA, mMPTA, and JLCA.
"""

from typing import Dict, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from .landmarks import LimbLandmarks


@dataclass
class AlignmentReport:
    """Quantitative orthopedic lower-limb alignment metrics."""
    side: str
    hka_3d_deg: float
    hka_coronal_deg: float
    mechanical_axis_deviation_deg: float  # (HKA - 180°)
    alignment_type: str                   # "Neutral", "Varus", "Valgus"
    mldfa_deg: float                      # Mechanical Lateral Distal Femoral Angle (normal ~87°)
    mmpta_deg: float                      # Mechanical Medial Proximal Tibial Angle (normal ~87°)
    jlca_deg: float                       # Joint Line Convergence Angle (normal 0-2°)
    femoral_mechanical_length_mm: float
    tibial_mechanical_length_mm: float
    total_limb_length_mm: float
    femoral_head_radius_mm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes spatial angle between two vectors in degrees."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = np.dot(v1, v2) / (norm1 * norm2)
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def compute_limb_alignment(lm: LimbLandmarks) -> AlignmentReport:
    """
    Computes complete surgical alignment metrics from anatomical landmarks.

    Args:
        lm: Extracted LimbLandmarks dataclass.

    Returns:
        AlignmentReport containing clinical angles and limb lengths.
    """
    # 1. Femoral mechanical axis: Hip center -> Knee center (superior to inferior)
    v_fem = lm.femoral_knee_center - lm.femoral_head_center
    fem_length = float(np.linalg.norm(v_fem))

    # 2. Tibial mechanical axis: Knee center -> Ankle center (superior to inferior)
    v_tib = lm.ankle_center - lm.tibial_knee_center
    tib_length = float(np.linalg.norm(v_tib))
    total_length = fem_length + tib_length

    # 3. 3D Hip-Knee-Ankle (HKA) angle
    # When fully collinear (neutral), angle between vectors is ~0, so HKA = 180 - angle
    # Or defined as internal angle formed at knee
    spatial_angle = angle_between_vectors_deg(v_fem, v_tib)
    hka_3d = 180.0 - spatial_angle

    # 4. Projected Coronal HKA (X-Z plane)
    # Radiographs measure HKA in the coronal projection
    v_fem_coronal = np.array([v_fem[0], v_fem[2]])  # [X (med/lat), Z (inf/sup)]
    v_tib_coronal = np.array([v_tib[0], v_tib[2]])

    # Directional deviation
    # In 2D: cross product sign determines Varus vs Valgus
    cross_2d = v_fem_coronal[0] * v_tib_coronal[1] - v_fem_coronal[1] * v_tib_coronal[0]
    coronal_angle = angle_between_vectors_deg(
        np.array([v_fem[0], 0.0, v_fem[2]]),
        np.array([v_tib[0], 0.0, v_tib[2]])
    )

    # By convention: Varus = medial deviation (<180°), Valgus = lateral deviation (>180°)
    if lm.side.lower() == "right":
        sign = 1.0 if cross_2d > 0 else -1.0
    else:
        sign = -1.0 if cross_2d > 0 else 1.0

    hka_coronal = 180.0 + sign * coronal_angle
    mad = hka_coronal - 180.0

    if abs(mad) <= 1.5:
        alignment_type = "Neutral"
    elif mad < -1.5:
        alignment_type = "Varus"
    else:
        alignment_type = "Valgus"

    # 5. mLDFA (Mechanical Lateral Distal Femoral Angle)
    # Angle between femoral mechanical axis and distal femoral transcondylar line
    # Transcondylar line points from Medial Condyle to Lateral Condyle
    v_fem_joint = lm.lateral_femoral_condyle - lm.medial_femoral_condyle
    angle_joint_fem = angle_between_vectors_deg(v_fem, v_joint if 'v_joint' in locals() else v_fem_joint)
    # Lateral angle: if vector is pointing lateral, angle is between 80-100 deg
    mldfa = angle_joint_fem if angle_joint_fem <= 90.0 else (180.0 - angle_joint_fem)
    # In standard anatomy mLDFA is lateral, typically 85° - 90° (mean 87°)
    if mldfa < 70.0:
        mldfa = 90.0 - mldfa  # orientation fix

    # 6. mMPTA (Mechanical Medial Proximal Tibial Angle)
    # Angle between tibial mechanical axis and proximal tibial plateau line
    # Plateau line points from Lateral Plateau to Medial Plateau
    v_tib_joint = lm.medial_tibial_plateau - lm.lateral_tibial_plateau
    angle_joint_tib = angle_between_vectors_deg(v_tib, v_tib_joint)
    mmpta = angle_joint_tib if angle_joint_tib <= 90.0 else (180.0 - angle_joint_tib)
    if mmpta < 70.0:
        mmpta = 90.0 - mmpta

    # 7. JLCA (Joint Line Convergence Angle)
    # Angle between distal femoral condyle line and proximal tibial plateau line
    jlca = angle_between_vectors_deg(v_fem_joint, -v_tib_joint)
    if jlca > 90.0:
        jlca = 180.0 - jlca

    return AlignmentReport(
        side=lm.side,
        hka_3d_deg=round(hka_3d, 2),
        hka_coronal_deg=round(hka_coronal, 2),
        mechanical_axis_deviation_deg=round(mad, 2),
        alignment_type=alignment_type,
        mldfa_deg=round(mldfa, 2),
        mmpta_deg=round(mmpta, 2),
        jlca_deg=round(jlca, 2),
        femoral_mechanical_length_mm=round(fem_length, 1),
        tibial_mechanical_length_mm=round(tib_length, 1),
        total_limb_length_mm=round(total_length, 1),
        femoral_head_radius_mm=round(lm.femoral_head_radius_mm, 2),
    )

