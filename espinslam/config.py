"""Validated JSONC configuration for the eSpinSLAM reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


def _strip_jsonc_comments(source: str) -> str:
    """Replace `//` comments with spaces while preserving quoted strings."""

    result = list(source)
    in_string = escaped = in_comment = False
    for index, character in enumerate(source):
        if in_comment:
            if character in "\r\n":
                in_comment = False
            else:
                result[index] = " "
        elif in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "/" and source[index:index + 2] == "//":
            result[index] = " "
            in_comment = True
    return "".join(result)


def _finite(name: str, value: object, *, positive: bool = False,
            non_negative: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if non_negative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass
class DataConfig:
    """Input geometry and leakage-safe evaluation split."""

    sensor_size: list[int] = field(default_factory=lambda: [346, 260])
    max_duration_seconds: float = 5.0
    evaluation_split: str = "test"

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_size, list) or len(self.sensor_size) != 2:
            raise ValueError("data.sensor_size must be [width, height]")
        for value in self.sensor_size:
            _positive_integer("data.sensor_size", value)
        _finite("data.max_duration_seconds", self.max_duration_seconds, positive=True)
        if self.evaluation_split != "test":
            raise ValueError("data.evaluation_split must remain 'test'")


@dataclass
class CameraConfig:
    """Pinhole intrinsics used by the continuous-time backend."""

    fx: float = 528.6111111111
    fy: float = 706.1728395062
    cx: float = 173.0
    cy: float = 130.0

    def __post_init__(self) -> None:
        _finite("camera.fx", self.fx, positive=True)
        _finite("camera.fy", self.fy, positive=True)
        _finite("camera.cx", self.cx, non_negative=True)
        _finite("camera.cy", self.cy, non_negative=True)

    def matrix(self):
        import numpy as np
        return np.asarray(
            ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )


@dataclass
class FrontendConfig:
    """Event chunk, corner, density, clustering, and track settings."""

    chunk_events: int = 1000
    surface_decay_seconds: float = 0.030
    fast_threshold: int = 18
    max_corners_per_chunk: int = 100
    density_radius_px: float = 6.0
    cluster_min_samples: int = 3
    track_merge_radius_px: float = 9.0
    track_max_gap_seconds: float = 0.080
    max_tracks: int = 500
    retain_observations_per_track: int = 1000
    fallback_frame_seconds: float = 0.020
    fallback_max_features: int = 300

    def __post_init__(self) -> None:
        for name in (
            "chunk_events", "fast_threshold", "max_corners_per_chunk",
            "cluster_min_samples", "max_tracks", "retain_observations_per_track",
            "fallback_max_features",
        ):
            _positive_integer(f"frontend.{name}", getattr(self, name))
        for name in (
            "surface_decay_seconds", "density_radius_px",
            "track_merge_radius_px", "track_max_gap_seconds",
            "fallback_frame_seconds",
        ):
            _finite(f"frontend.{name}", getattr(self, name), positive=True)


@dataclass
class FrequencyConfig:
    """Loop-closure frequency search without using rotation labels."""

    min_hz: float = 0.10
    max_hz: float = 1.00
    candidate_count: int = 91
    max_alignment_samples: int = 4000
    spatial_tolerance_px: float = 6.0
    temporal_tolerance_seconds: float = 0.030
    equivalent_score_fraction: float = 0.0
    loop_std_threshold_hz: float = 0.001
    loop_history: int = 20

    def __post_init__(self) -> None:
        _finite("frequency.min_hz", self.min_hz, positive=True)
        _finite("frequency.max_hz", self.max_hz, positive=True)
        if self.max_hz <= self.min_hz:
            raise ValueError("frequency.max_hz must exceed frequency.min_hz")
        for name in ("candidate_count", "max_alignment_samples", "loop_history"):
            _positive_integer(f"frequency.{name}", getattr(self, name))
        if self.candidate_count < 3:
            raise ValueError("frequency.candidate_count must be at least three")
        _finite(
            "frequency.spatial_tolerance_px",
            self.spatial_tolerance_px,
            positive=True,
        )
        _finite(
            "frequency.temporal_tolerance_seconds",
            self.temporal_tolerance_seconds,
            positive=True,
        )
        _finite(
            "frequency.equivalent_score_fraction",
            self.equivalent_score_fraction,
            non_negative=True,
        )
        _finite(
            "frequency.loop_std_threshold_hz",
            self.loop_std_threshold_hz,
            positive=True,
        )


@dataclass
class BackendConfig:
    """Continuous-time orbit and robust reconstruction settings."""

    orbit_radius: float = 10.0
    min_track_observations: int = 3
    max_landmarks: int = 250
    huber_scale_px: float = 2.0
    max_mean_reprojection_error_px: float = 10.0
    landmark_merge_radius: float = 0.01
    optimization_cadence_revolutions: float = 0.1
    optimization_iterations: int = 60

    def __post_init__(self) -> None:
        for name in ("min_track_observations", "max_landmarks", "optimization_iterations"):
            _positive_integer(f"backend.{name}", getattr(self, name))
        for name in (
            "orbit_radius", "huber_scale_px", "max_mean_reprojection_error_px",
            "landmark_merge_radius", "optimization_cadence_revolutions",
        ):
            _finite(f"backend.{name}", getattr(self, name), positive=True)


@dataclass
class EvaluationConfig:
    """Held-out rotation and point-cloud report settings."""

    observation_times_seconds: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    )
    pointcloud_max_reference_points: int = 30000
    failure_policy: str = "report_and_mark_invalid"

    def __post_init__(self) -> None:
        if not isinstance(self.observation_times_seconds, list) or not self.observation_times_seconds:
            raise ValueError("evaluation.observation_times_seconds must be non-empty")
        previous = 0.0
        for value in self.observation_times_seconds:
            current = _finite("evaluation.observation_times_seconds", value, positive=True)
            if current <= previous:
                raise ValueError("evaluation observation times must be strictly increasing")
            previous = current
        _positive_integer(
            "evaluation.pointcloud_max_reference_points",
            self.pointcloud_max_reference_points,
        )
        if self.failure_policy != "report_and_mark_invalid":
            raise ValueError("evaluation.failure_policy must be 'report_and_mark_invalid'")


@dataclass
class LoggingConfig:
    """W&B options matching event_snn_torch's experiment interface."""

    wandb: bool = False
    wandb_mode: str = "online"
    wandb_project: str = "snn-research"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.wandb, bool):
            raise TypeError("logging.wandb must be a boolean")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("logging.wandb_mode is invalid")
        if not isinstance(self.wandb_project, str) or not self.wandb_project.strip():
            raise ValueError("logging.wandb_project must be non-empty")
        if not isinstance(self.wandb_tags, list) or not all(
                isinstance(value, str) and value.strip() for value in self.wandb_tags):
            raise ValueError("logging.wandb_tags must contain non-empty strings")


@dataclass
class ESpinSLAMConfig:
    """Complete independent eSpinSLAM reproduction configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "data": DataConfig,
    "camera": CameraConfig,
    "frontend": FrontendConfig,
    "frequency": FrequencyConfig,
    "backend": BackendConfig,
    "evaluation": EvaluationConfig,
    "logging": LoggingConfig,
}


def load_config(path: str | Path) -> ESpinSLAMConfig:
    """Load JSON/JSONC and reject unknown configuration keys."""

    path = Path(path)
    try:
        value = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid eSpinSLAM config: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = set(value) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    sections: dict[str, Any] = {}
    for name, factory in _SECTIONS.items():
        supplied = value.get(name, {})
        if not isinstance(supplied, dict):
            raise ValueError(f"config section {name!r} must be an object")
        known = set(factory.__dataclass_fields__)
        extra = set(supplied) - known
        if extra:
            raise ValueError(f"unknown {name} keys: {sorted(extra)}")
        sections[name] = factory(**supplied)
    return ESpinSLAMConfig(**sections)
