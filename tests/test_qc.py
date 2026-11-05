"""
Unit tests for clinical Quality Control (QC) gate audits.
"""

import pytest
from limbalign.geometry.angles import AlignmentReport
from limbalign.validation.qc import run_quality_control_checks


def test_qc_pass():
    """Tests QC acceptance on physiological normal values."""
    alignment = AlignmentReport(
        side="Right",
        hka_3d_deg=179.5,
        hka_coronal_deg=179.8,
        mechanical_axis_deviation_deg=-0.2,
        alignment_type="Neutral",
        mldfa_deg=87.2,
        mmpta_deg=86.8,
        jlca_deg=0.8,
        femoral_mechanical_length_mm=450.0,
        tibial_mechanical_length_mm=390.0,
        total_limb_length_mm=840.0,
        femoral_head_radius_mm=23.4,
    )

    qc = run_quality_control_checks(alignment)
    assert qc.passed is True
    assert qc.status == "ACCEPTED"
    assert len(qc.flags) == 0


def test_qc_abnormal_sphere_rejection():
    """Tests QC rejection when femoral head radius is non-physiological."""
    alignment = AlignmentReport(
        side="Right",
        hka_3d_deg=180.0,
        hka_coronal_deg=180.0,
        mechanical_axis_deviation_deg=0.0,
        alignment_type="Neutral",
        mldfa_deg=87.0,
        mmpta_deg=87.0,
        jlca_deg=1.0,
        femoral_mechanical_length_mm=450.0,
        tibial_mechanical_length_mm=390.0,
        total_limb_length_mm=840.0,
        femoral_head_radius_mm=45.0,  # Unusually large, e.g. neck fitting error
    )

    qc = run_quality_control_checks(alignment)
    assert qc.passed is False
    assert qc.sphere_radius_valid is False
    assert "radius" in qc.flags[0].lower()

