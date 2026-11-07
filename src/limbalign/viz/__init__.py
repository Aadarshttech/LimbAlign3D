"""
Visualization, surface mesh extraction, and rendering utilities.
"""

from .mesh import extract_surface_mesh, export_binary_stl
from .axis_overlay import render_mechanical_axes_3d
from .plots import plot_coronal_alignment_projection, plot_robustness_curves

__all__ = [
    "extract_surface_mesh",
    "export_binary_stl",
    "render_mechanical_axes_3d",
    "plot_coronal_alignment_projection",
    "plot_robustness_curves",
]

