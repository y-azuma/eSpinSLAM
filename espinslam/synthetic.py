"""Synthetic event streams that satisfy the paper's assumptions exactly.

A static pinhole event camera observes a rigid object that spins at constant
frequency ``f`` about a fixed axis. By the duality of Sec. 3 this is the
orbit model of :mod:`espinslam.geometry` with a constant ``R0``; the camera is
elevated and aimed at the spin axis, so ``R0`` is a genuine tilt (not the
identity). The object is a set of textured square patches whose corners are
the trackable features. Patches are planar squares in 3D, so their corners
(:attr:`SyntheticScene.feature_points`) are rigid world points that serve as
structure ground truth; a patch is visible only while facing the camera.
Events follow the standard contrast-threshold model with per-pixel log
intensity references and linear timestamp interpolation between frames.

The generator is used by the unit tests and by
``scripts/synthetic_benchmark.py``, which checks the whole pipeline (frequency,
R0, structure, reprojection error) against ground truth without TOPSPIN.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import orbit_camera_frames


@dataclass(frozen=True)
class SyntheticScene:
    points: np.ndarray
    normals: np.ndarray
    corners: np.ndarray
    intensities: np.ndarray
    rotation: np.ndarray
    frequency_hz: float
    orbit_radius: float
    intrinsic: np.ndarray
    width: int
    height: int

    @property
    def feature_points(self) -> np.ndarray:
        """Rigid 3D patch corners (structure ground truth)."""

        return self.corners.reshape(-1, 3)


@dataclass(frozen=True)
class SyntheticStream:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    scene: SyntheticScene


def make_scene(*, frequency_hz: float = 0.5, elevation_deg: float = 20.0,
               pan_deg: float = 2.0, roll_deg: float = 3.0, point_count: int = 60,
               object_radius: float = 1.5, orbit_radius: float = 10.0,
               width: int = 346, height: int = 260, focal: float = 528.6,
               patch_half_size_px: tuple[float, float] = (3.0, 6.0),
               seed: int = 0) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    axis_offset = orbit_radius * math.tan(math.radians(elevation_deg))
    centre = np.asarray((0.0, 0.0, -axis_offset))
    points = centre + rng.uniform(-1.0, 1.0, (point_count, 3)) * np.asarray(
        (object_radius, object_radius, 0.6 * object_radius),
    )
    normals = points - centre
    normals[:, 2] *= 0.3
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    # Square patches in the plane orthogonal to each normal (random in-plane
    # orientation; half-size in pixels at the orbit-centre depth). eFAST needs
    # corners at least ~4 px apart, so very small patches fire as one blob.
    helper = np.where(np.abs(normals[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    first = np.cross(normals, helper)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(normals, first)
    angle = rng.uniform(0.0, np.pi / 2.0, point_count)[:, None]
    u = np.cos(angle) * first + np.sin(angle) * second
    v = np.cross(normals, u)
    half = rng.uniform(*patch_half_size_px, point_count)[:, None] * orbit_radius / focal
    signs = np.asarray(((1, 1), (1, -1), (-1, -1), (-1, 1)), dtype=np.float64)
    corners = points[:, None, :] + half[:, None, :] * (
        signs[None, :, 0:1] * u[:, None, :] + signs[None, :, 1:2] * v[:, None, :])
    aim, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [[0.0, axis_offset, orbit_radius]])
    rotation = (Rotation.from_euler("yz", [pan_deg, roll_deg], degrees=True) * aim).as_matrix()
    intrinsic = np.asarray(((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0),
                            (0.0, 0.0, 1.0)))
    return SyntheticScene(
        points=points, normals=normals, corners=corners,
        intensities=rng.uniform(0.6, 1.0, point_count),
        rotation=rotation, frequency_hz=float(frequency_hz),
        orbit_radius=float(orbit_radius), intrinsic=intrinsic,
        width=int(width), height=int(height),
    )


_SUBPIXEL_BITS = 4


def render(scene: SyntheticScene, time_seconds: float, background: float = 0.15) -> np.ndarray:
    """Anti-aliased intensity image of the visible patches at one instant."""

    import cv2
    image = np.full((scene.height, scene.width), background, dtype=np.float64)
    count = scene.points.shape[0]
    rotations, centres = orbit_camera_frames(
        np.full(count, time_seconds), scene.frequency_hz, scene.orbit_radius, scene.rotation,
    )
    rotation, centre = rotations[0], centres[0]
    camera = (scene.corners.reshape(-1, 3) - centre) @ rotation.T
    depth = camera[:, 2].reshape(count, 4)
    pixels = (camera @ scene.intrinsic.T)
    pixels = (pixels[:, :2] / np.maximum(pixels[:, 2:3], 1e-9)).reshape(count, 4, 2)
    facing = np.einsum("ij,ij->i", scene.normals, centre - scene.points) > 0
    scale = 1 << _SUBPIXEL_BITS
    for index in np.argsort(-depth.mean(axis=1)):
        if not facing[index] or np.any(depth[index] <= 0):
            continue
        quad = pixels[index]
        x0 = max(0, int(np.floor(quad[:, 0].min())) - 1)
        y0 = max(0, int(np.floor(quad[:, 1].min())) - 1)
        x1 = min(scene.width, int(np.ceil(quad[:, 0].max())) + 2)
        y1 = min(scene.height, int(np.ceil(quad[:, 1].max())) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        local = np.rint((quad - (x0, y0)) * scale).astype(np.int32)
        cv2.fillConvexPoly(mask, local, 255, lineType=cv2.LINE_AA, shift=_SUBPIXEL_BITS)
        alpha = mask.astype(np.float64) / 255.0
        window = image[y0:y1, x0:x1]
        window[:] = window * (1.0 - alpha) + scene.intensities[index] * alpha
    return image


def simulate_events(scene: SyntheticScene, *, duration_seconds: float = 5.0,
                    frame_rate: float = 1000.0, contrast_threshold: float = 0.2,
                    contrast_threshold_off: float | None = None,
                    noise_rate_hz_per_pixel: float = 0.0, seed: int = 0) -> SyntheticStream:
    """Generate an event stream with the contrast-threshold sensor model.

    ``contrast_threshold`` applies to ON events and ``contrast_threshold_off``
    (default: the same value) to OFF events; real sensors differ (Sec. 4.1).
    """

    rng = np.random.default_rng(seed)
    frame_times = np.arange(0.0, duration_seconds, 1.0 / frame_rate)
    epsilon = 0.05
    reference = np.log(render(scene, 0.0) + epsilon)
    previous = reference.copy()
    chunks_t, chunks_x, chunks_y, chunks_p = [], [], [], []
    for index in range(1, frame_times.size):
        current = np.log(render(scene, frame_times[index]) + epsilon)
        delta = current - reference
        threshold = np.where(delta > 0, contrast_threshold,
                             contrast_threshold if contrast_threshold_off is None
                             else contrast_threshold_off)
        counts = np.floor(np.abs(delta) / threshold).astype(np.int64)
        ys, xs = np.nonzero(counts)
        if ys.size:
            n = counts[ys, xs]
            owner = np.repeat(np.arange(ys.size), n)
            crossing = np.arange(owner.size) - np.repeat(np.cumsum(n) - n, n) + 1
            sign = np.sign(delta[ys, xs])
            step = threshold[ys, xs]
            target = reference[ys, xs][owner] + sign[owner] * crossing * step[owner]
            start = previous[ys, xs][owner]
            change = (current[ys, xs] - previous[ys, xs])[owner]
            with np.errstate(divide="ignore", invalid="ignore"):
                fraction = np.where(np.abs(change) > 1e-12, (target - start) / change, 1.0)
            times = frame_times[index - 1] + np.clip(fraction, 0.0, 1.0) / frame_rate
            reference[ys, xs] += sign * n * step
            chunks_t.append(times)
            chunks_x.append(xs[owner])
            chunks_y.append(ys[owner])
            chunks_p.append((sign[owner] > 0).astype(np.uint8))
        if noise_rate_hz_per_pixel > 0:
            expected = noise_rate_hz_per_pixel * scene.width * scene.height / frame_rate
            noise = rng.poisson(expected)
            chunks_t.append(frame_times[index - 1] + rng.random(noise) / frame_rate)
            chunks_x.append(rng.integers(0, scene.width, noise))
            chunks_y.append(rng.integers(0, scene.height, noise))
            chunks_p.append(rng.integers(0, 2, noise).astype(np.uint8))
        previous = current
    t = np.concatenate(chunks_t) if chunks_t else np.empty(0)
    order = np.argsort(t, kind="stable")
    return SyntheticStream(
        t=t[order],
        x=np.concatenate(chunks_x)[order].astype(np.int32),
        y=np.concatenate(chunks_y)[order].astype(np.int32),
        p=np.concatenate(chunks_p)[order].astype(np.uint8),
        scene=scene,
    )
