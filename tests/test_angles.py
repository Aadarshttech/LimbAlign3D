"""
Unit tests for geometric angle and mechanical axis calculations.
"""

import numpy as np
import pytest
from limbalign.geometry.angles import angle_between_vectors_deg, compute_limb_alignment
from limbalign.geometry.landmarks import LimbLandmarks


def test_angle_between_vectors():
    """Tests basic geometric angle properties."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert np.isclose(angle_between_vectors_deg(v1, v2), 90.0)

    v3 = np.array([0.0, 0.0, 1.0])
    v4 = np.array([0.0, 0.0, -1.0])
    assert np.isclose(angle_between_vectors_deg(v3, v4), 180.0)

    assert np.isclose(angle_between_vectors_deg(v1, v1), 0.0)


def test_collinear_limb_alignment():
    """Tests HKA angle calculation on perfectly collinear mechanical axes."""
    hip = np.array([50.0, 0.0, 900.0])
    knee = np.array([50.0, 0.0, 450.0])
    ankle = np.array([50.0, 0.0, 50.0])

    med_cond = np.array([30.0, 0.0, 450.0])
    lat_cond = np.array([70.0, 0.0, 450.0])

    med_plat = np.array([30.0, 0.0, 440.0])
    lat_plat = np.array([70.0, 0.0, 440.0])
    med_mall = np.array([35.0, 0.0, 40.0])

    lm = LimbLandmarks(
        side="Right",
        femoral_head_center=hip,
        femoral_head_radius_mm=23.0,
        femoral_knee_center=knee,
        medial_femoral_condyle=med_cond,
        lateral_femoral_condyle=lat_cond,
        tibial_knee_center=knee,
        medial_tibial_plateau=med_plat,
        lateral_tibial_plateau=lat_plat,
        ankle_center=ankle,
        medial_malleolus=med_mall,
    )

    report = compute_limb_alignment(lm)

    assert np.isclose(report.hka_3d_deg, 180.0, atol=0.1)
    assert np.isclose(report.hka_coronal_deg, 180.0, atol=0.1)
    assert report.alignment_type == "Neutral"
    assert np.isclose(report.femoral_mechanical_length_mm, 450.0, atol=0.1)
    assert np.isclose(report.tibial_mechanical_length_mm, 400.0, atol=0.1)

