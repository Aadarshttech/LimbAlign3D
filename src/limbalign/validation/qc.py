"""
Clinical quality control (QC) gates for automated surgical alignment planning.
Flags anatomical anomalies, segmentation dropouts, and fitting failures.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from ..geometry.angles import AlignmentReport


@dataclass
class QCReport:
    """Clinical Quality Control audit result."""
    passed: bool
    status: str  # "ACCEPTED", "FLAGGED_FOR_REVIEW", "REJECTED"
    flags: List[str]
    sphere_radius_valid: bool
    hka_valid: bool
    length_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_quality_control_checks(
    alignment: AlignmentReport,
    min_head_radius_mm: float = 18.0,
    max_head_radius_mm: float = 30.0,
    min_hka_deg: float = 160.0,
    max_hka_deg: float = 200.0,
    min_limb_length_mm: float = 600.0,
    max_limb_length_mm: float = 1100.0,
) -> QCReport:
    """
    Evaluates clinical plausibility gates on computed lower-limb metrics.

    Threshold rationales:
    - Adult human femoral head diameter ranges 38–58 mm (radius 19–29 mm).
    - Clinical severe varus/valgus deformities rarely exceed 20° MAD (HKA 160–200°).
    - Adult human lower limb length ranges 650–1050 mm.
    """
    flags = []

    # 1. Sphere radius check
    r = alignment.femoral_head_radius_mm
    radius_valid = (min_head_radius_mm <= r <= max_head_radius_mm)
    if not radius_valid:
        flags.append(f"Femoral head radius ({r:.1f} mm) outside physiological range [{min_head_radius_mm}, {max_head_radius_mm}] mm.")

    # 2. HKA plausibility check
    hka = alignment.hka_coronal_deg
    hka_valid = (min_hka_deg <= hka <= max_hka_deg)
    if not hka_valid:
        flags.append(f"Coronal HKA ({hka:.1f}°) indicates extreme non-physiological angle or landmark dislocation.")

    # 3. Limb length check
    length = alignment.total_limb_length_mm
    length_valid = (min_limb_length_mm <= length <= max_limb_length_mm)
    if not length_valid:
        flags.append(f"Limb length ({length:.1f} mm) outside expected adult range [{min_limb_length_mm}, {max_limb_length_mm}] mm.")

    passed = (radius_valid and hka_valid and length_valid)
    if passed:
        status = "ACCEPTED"
    elif len(flags) == 1 and radius_valid:
        status = "FLAGGED_FOR_REVIEW"
    else:
        status = "REJECTED"

    return QCReport(
        passed=passed,
        status=status,
        flags=flags,
        sphere_radius_valid=radius_valid,
        hka_valid=hka_valid,
        length_valid=length_valid,
    )

