"""
Robustness stress testing and perturbation analysis for lower-limb alignment.
Tests resilience to sensor noise, spatial downsampling, rotation, and FOV truncation.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import zoom, rotate


def apply_gaussian_noise(ct_volume: np.ndarray, std_hu: float = 25.0, seed: int = 42) -> np.ndarray:
    """Injects zero-mean Gaussian CT sensor noise."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, std_hu, size=ct_volume.shape).astype(np.float32)
    return ct_volume + noise


def apply_voxel_downsampling(
    volume: np.ndarray,
    scale_factor: float = 0.5,
    order: int = 1,
) -> np.ndarray:
    """
    Simulates low-dose or thick-slice CT acquisitions by downsampling
    and upsampling back to original shape.
    """
    down = zoom(volume, scale_factor, order=order)
    reconstructed = zoom(down, [volume.shape[i] / down.shape[i] for i in range(3)], order=order)
    return reconstructed


def apply_fov_truncation(
    volume: np.ndarray,
    crop_fraction_inferior: float = 0.08,
) -> np.ndarray:
    """
    Simulates clinical scan field-of-view truncation where the ankle
    or distal tibia is partially truncated.
    """
    out = volume.copy()
    z_cutoff = int(crop_fraction_inferior * volume.shape[2])
    out[:, :, :z_cutoff] = -1000.0 if volume.min() < -500 else 0
    return out


def apply_rotation_perturbation(
    volume: np.ndarray,
    angle_deg: float = 5.0,
    axes: Tuple[int, int] = (0, 1),
    order: int = 1,
) -> np.ndarray:
    """Simulates patient leg malpositioning / rotation in the scanner gantry."""
    return rotate(volume, angle=angle_deg, axes=axes, reshape=False, order=order)

