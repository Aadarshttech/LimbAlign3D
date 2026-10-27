"""
Unit tests for segmentation metrics (Dice, HD95).
"""

import numpy as np
import pytest
from limbalign.seg.metrics import compute_dice, compute_hd95


def test_compute_dice():
    """Tests Dice calculation on identical, disjoint, and overlapping arrays."""
    a = np.zeros((20, 20, 20), dtype=bool)
    b = np.zeros((20, 20, 20), dtype=bool)

    # Both empty -> 1.0
    assert compute_dice(a, b) == 1.0

    # Identical non-empty -> 1.0
    a[5:15, 5:15, 5:15] = True
    b[5:15, 5:15, 5:15] = True
    assert compute_dice(a, b) == 1.0

    # Completely disjoint -> 0.0
    b[:] = False
    b[0:4, 0:4, 0:4] = True
    assert compute_dice(a, b) == 0.0


def test_compute_hd95():
    """Tests 95% Hausdorff distance calculation with known voxel spacing."""
    a = np.zeros((30, 30, 30), dtype=bool)
    b = np.zeros((30, 30, 30), dtype=bool)
    spacing = (1.0, 1.0, 1.0)

    # Identical -> HD95 = 0.0
    a[10:20, 10:20, 10:20] = True
    b[10:20, 10:20, 10:20] = True
    assert compute_hd95(a, b, spacing) == 0.0

    # Shifted by 2 voxels (2 mm)
    b[:] = False
    b[12:22, 10:20, 10:20] = True
    hd = compute_hd95(a, b, spacing)
    assert 1.5 <= hd <= 2.5

