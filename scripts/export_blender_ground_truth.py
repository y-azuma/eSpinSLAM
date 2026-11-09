#!/usr/bin/env python3
"""Export a deterministic target-surface point cloud from an opened .blend file.

Usage:
  blender --background sample/scene.blend \
    --python scripts/export_blender_ground_truth.py -- output.npz 30000
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


def _arguments() -> tuple[Path, int]:
    tail = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not tail:
        raise ValueError("output path is required after --")
    return Path(tail[0]).expanduser().resolve(), int(tail[1]) if len(tail) > 1 else 30000


def _triangles_in_rig_frame() -> np.ndarray:
    rig = bpy.data.objects.get("TargetRig_COM")
    if rig is None:
        raise RuntimeError("TargetRig_COM was not found")
    inverse_rig = rig.matrix_world.inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    triangles: list[np.ndarray] = []
    for source in bpy.data.objects:
        if source.type != "MESH" or not bool(source.get("target_component", False)):
            continue
        evaluated = source.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            transform = inverse_rig @ evaluated.matrix_world
            vertices = [transform @ vertex.co for vertex in mesh.vertices]
            for face in mesh.loop_triangles:
                triangles.append(np.asarray([
                    tuple(vertices[index]) for index in face.vertices
                ], dtype=np.float64))
        finally:
            evaluated.to_mesh_clear()
    if not triangles:
        raise RuntimeError("no target triangles were found")
    return np.asarray(triangles, dtype=np.float64)


def _sample_surface(triangles: np.ndarray, count: int, seed: int = 2701) -> np.ndarray:
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(first, second), axis=1)
    valid = areas > 1e-12
    triangles, areas = triangles[valid], areas[valid]
    probabilities = areas / areas.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(triangles.shape[0], size=count, p=probabilities)
    selected = triangles[indices]
    u, v = rng.random(count), rng.random(count)
    reflected = u + v > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    return (
        selected[:, 0]
        + u[:, None] * (selected[:, 1] - selected[:, 0])
        + v[:, None] * (selected[:, 2] - selected[:, 0])
    ).astype(np.float32)


def main() -> None:
    output, count = _arguments()
    if count < 100:
        raise ValueError("point count must be at least 100")
    triangles = _triangles_in_rig_frame()
    points = _sample_surface(triangles, count)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points=points,
        coordinate_frame=np.asarray("TargetRig_COM body frame"),
        unit=np.asarray("meter"),
        source_blend=np.asarray(bpy.data.filepath),
    )
    print(f"Saved {points.shape[0]} reference points to {output}")


if __name__ == "__main__":
    main()

