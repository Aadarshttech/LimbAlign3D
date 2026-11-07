"""
3D triangle surface mesh extraction via Marching Cubes and STL export.
Produces watertight boundary models directly compatible with 3D Slicer and CAD software.
"""

import os
from typing import Tuple, Dict, Optional
import struct
import numpy as np
from skimage.measure import marching_cubes


def extract_surface_mesh(
    binary_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    level: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts high-resolution triangular isosurface from a 3D binary bone mask.

    Returns:
        verts: (N, 3) vertex coordinates scaled to physical millimeters [X, Y, Z].
        faces: (M, 3) triangular face vertex indices.
        normals: (N, 3) vertex normal vectors.
    """
    if not np.any(binary_mask):
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))

    verts, faces, normals, _ = marching_cubes(
        volume=binary_mask,
        level=level,
        spacing=spacing,
    )
    return verts, faces, normals


def export_binary_stl(
    verts: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    header_title: str = "LimbAlign3D Mesh",
) -> str:
    """
    Exports a 3D triangle mesh to standard binary STL format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    header = header_title.encode("ascii")[:80].ljust(80, b"\0")
    n_triangles = len(faces)

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n_triangles))

        for face in faces:
            v0 = verts[face[0]]
            v1 = verts[face[1]]
            v2 = verts[face[2]]

            # Normal vector calculation
            norm = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(norm)
            if norm_len > 0:
                norm = norm / norm_len
            else:
                norm = np.array([0.0, 0.0, 0.0])

            # Write normal (12 bytes) and 3 vertices (36 bytes) + attribute byte count (2 bytes)
            f.write(struct.pack("<3f", *norm))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))

    return output_path

