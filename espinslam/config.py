"""Validated JSONC configuration for the eSpinSLAM reproduction.

Every hyperparameter listed in Table 1 of the paper has a dedicated key. The
mapping is documented next to each field so that the configuration can be
audited against the paper:

========================================  ==========================================
Paper (Table 1)                           Configuration key
========================================  ==========================================
Standard deviation f estimation sigma     frequency.loop_std_threshold_hz
Frequency estimation samples aleph        frequency.loop_history
Inlier threshold for rho                  backend.huber_threshold_px
Forgetting threshold sigma_forget         backend.forget_revolutions
Circular buffer length sigma_retain       frontend.retain_events_per_track
Track length N_match                      backend.min_track_events
Orbit radius r                            backend.orbit_radius
Maximum landmark mean reproj. error gamma backend.max_mean_reprojection_error_px
Landmark merge radius delta_fuse          backend.landmark_merge_radius
Cadence of optimization epsilon           backend.optimization_cadence_revolutions
========================================  ==========================================
"""

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


def _non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


_TIME_UNITS = {"auto", "s", "ms", "us", "ns"}


@dataclass
class DataConfig:
    """Input geometry, timestamp units, and the debris evaluation split."""

    sensor_size: list[int] = field(default_factory=lambda: [346, 260])
    # None processes the full stream (TOPSPIN); a value truncates causally.
    max_duration_seconds: float | None = None
    # Unit of raw timestamps for generic readers; "auto" infers s/us from range.
    time_unit: str = "auto"
    # Column order of plain-text event files.
    text_columns: str = "txyp"
    evaluation_split: str = "test"

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_size, list) or len(self.sensor_size) != 2:
            raise ValueError("data.sensor_size must be [width, height]")
        for value in self.sensor_size:
            _positive_integer("data.sensor_size", value)
        if self.max_duration_seconds is not None:
            _finite("data.max_duration_seconds", self.max_duration_seconds, positive=True)
        if self.time_unit not in _TIME_UNITS:
            raise ValueError(f"data.time_unit must be one of {sorted(_TIME_UNITS)}")
        if sorted(self.text_columns) != sorted("txyp"):
            raise ValueError("data.text_columns must be a permutation of 'txyp'")
        if self.evaluation_split != "test":
            raise ValueError("data.evaluation_split must remain 'test'")


@dataclass
class CameraConfig:
    """Pinhole intrinsics (and optional OpenCV distortion) of the event camera.

    ``None`` intrinsics mean "must be supplied per sequence" (e.g. by the
    TOPSPIN manifest); :meth:`matrix` refuses to run without them.
    """

    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    # OpenCV order [k1, k2, p1, p2, k3]; empty means an ideal pinhole camera.
    distortion: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("fx", "fy"):
            value = getattr(self, name)
            if value is not None:
                _finite(f"camera.{name}", value, positive=True)
        for name in ("cx", "cy"):
            value = getattr(self, name)
            if value is not None:
                _finite(f"camera.{name}", value, non_negative=True)
        if not isinstance(self.distortion, list) or len(self.distortion) not in (0, 4, 5, 8):
            raise ValueError("camera.distortion must have 0, 4, 5, or 8 coefficients")
        for value in self.distortion:
            _finite("camera.distortion", value)

    @property
    def complete(self) -> bool:
        return None not in (self.fx, self.fy, self.cx, self.cy)

    def matrix(self):
        import numpy as np
        if not self.complete:
            raise ValueError(
                "camera intrinsics are missing; supply fx, fy, cx, cy in the config "
                "or in the manifest",
            )
        return np.asarray(
            ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )


@dataclass
class FrontendConfig:
    """Section 4.1: chunking, eFAST, density filtering, HDBSCAN, track pool."""

    # |C|: events per chunk pushed by the camera driver (~1000 for Metavision).
    chunk_events: int = 1000
    # lambda in Eq. (6): radius of the same-polarity density neighbourhood.
    density_radius_px: float = 3.0
    # HDBSCAN parameters for spatio-temporal tracklet clustering.
    cluster_min_size: int = 5
    cluster_min_samples: int = 3
    cluster_time_scale_px_per_second: float = 1000.0
    # Sec. 4.1 only says G is clustered "spatio-temporally". Clustering each
    # polarity separately (consistent with the per-polarity density of Eq. 6)
    # keeps the ON/OFF corners of small features apart; mixing them doubled the
    # structure error on the low-resolution synthetic scene (docs/).
    cluster_by_polarity: bool = True
    # Tracklet-to-track association (merging into the global pool H).
    association_radius_px: float = 5.0
    association_max_gap_seconds: float = 0.05
    association_head_events: int = 5
    # sigma_retain: circular buffer length per track.
    retain_events_per_track: int = 1000

    def __post_init__(self) -> None:
        for name in (
            "chunk_events", "cluster_min_size", "cluster_min_samples",
            "association_head_events", "retain_events_per_track",
        ):
            _positive_integer(f"frontend.{name}", getattr(self, name))
        if self.cluster_min_size < 2:
            raise ValueError("frontend.cluster_min_size must be at least two")
        for name in (
            "density_radius_px", "cluster_time_scale_px_per_second",
            "association_radius_px", "association_max_gap_seconds",
        ):
            _finite(f"frontend.{name}", getattr(self, name), positive=True)
        _boolean("frontend.cluster_by_polarity", self.cluster_by_polarity)


_FREQUENCY_METHODS = {"loop_closure", "fft"}


@dataclass
class FrequencyConfig:
    """Section 4.2.1: loop-closure frequency estimation and loop detection."""

    # "loop_closure" is the paper's method; "fft" is the Table 3 ablation.
    method: str = "loop_closure"
    # The estimator re-runs whenever H changed and at least this much event
    # time elapsed since the previous estimate (0 means after every chunk).
    update_interval_seconds: float = 0.1
    min_period_seconds: float = 1.0
    max_period_seconds: float = 60.0
    # Pixels that one second of temporal offset is worth in Eq. (9).
    time_scale_px_per_second: float = 100.0
    max_alignment_samples: int = 10000
    coarse_alignment_samples: int = 2000
    coarse_relative_step: float = 0.01
    icp_iterations: int = 30
    icp_tolerance_seconds: float = 1e-5
    # Local minima within this relative cost of the best are treated as equal
    # and the shortest period (the fundamental) is selected.
    harmonic_tolerance: float = 0.15
    min_samples_per_half: int = 50
    # sigma and aleph of Table 1.
    loop_std_threshold_hz: float = 1e-3
    loop_history: int = 20
    fft_bin_seconds: float = 0.01

    def __post_init__(self) -> None:
        if self.method not in _FREQUENCY_METHODS:
            raise ValueError(f"frequency.method must be one of {sorted(_FREQUENCY_METHODS)}")
        _finite("frequency.update_interval_seconds", self.update_interval_seconds,
                non_negative=True)
        for name in (
            "min_period_seconds", "max_period_seconds", "time_scale_px_per_second",
            "coarse_relative_step", "icp_tolerance_seconds", "loop_std_threshold_hz",
            "fft_bin_seconds",
        ):
            _finite(f"frequency.{name}", getattr(self, name), positive=True)
        _finite("frequency.harmonic_tolerance", self.harmonic_tolerance, non_negative=True)
        if self.max_period_seconds <= self.min_period_seconds:
            raise ValueError("frequency.max_period_seconds must exceed min_period_seconds")
        for name in (
            "max_alignment_samples", "coarse_alignment_samples", "icp_iterations",
            "min_samples_per_half", "loop_history",
        ):
            _positive_integer(f"frequency.{name}", getattr(self, name))
        if self.loop_history < 2:
            raise ValueError("frequency.loop_history must be at least two")


@dataclass
class BackendConfig:
    """Sections 4.2.2-4.2.7: continuous-time orbit optimisation and map upkeep."""

    orbit_radius: float = 10.0
    min_track_events: int = 3
    huber_threshold_px: float = 2.0
    max_mean_reprojection_error_px: float = 10.0
    landmark_merge_radius: float = 0.01
    optimization_cadence_revolutions: float = 0.1
    forget_revolutions: float = 3.0
    max_iterations: int = 100
    # Multi-view triangulation (Sec. 4.2.4) is undefined without parallax.
    # Tracks whose events span less orbit arc than this are under-constrained
    # and wait (in H) until they grow. 1.5 deg is COLMAP's default minimum
    # triangulation angle; the paper does not state its triangulator's check.
    min_triangulation_angle_deg: float = 1.5
    # 0 uses every retained event of every track, as in Eq. (11).
    max_observations_per_track: int = 0
    # Ablations of Table 3.
    optimize_frequency: bool = False
    landmark_association: bool = True

    def __post_init__(self) -> None:
        for name in ("min_track_events", "max_iterations"):
            _positive_integer(f"backend.{name}", getattr(self, name))
        _non_negative_integer("backend.max_observations_per_track",
                              self.max_observations_per_track)
        _finite("backend.min_triangulation_angle_deg", self.min_triangulation_angle_deg,
                non_negative=True)
        for name in (
            "orbit_radius", "huber_threshold_px", "max_mean_reprojection_error_px",
            "landmark_merge_radius", "optimization_cadence_revolutions",
            "forget_revolutions",
        ):
            _finite(f"backend.{name}", getattr(self, name), positive=True)
        _boolean("backend.optimize_frequency", self.optimize_frequency)
        _boolean("backend.landmark_association", self.landmark_association)


@dataclass
class EvaluationConfig:
    """Report settings shared by the debris and TOPSPIN evaluations."""

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
        _boolean("logging.wandb", self.wandb)
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("logging.wandb_mode is invalid")
        if not isinstance(self.wandb_project, str) or not self.wandb_project.strip():
            raise ValueError("logging.wandb_project must be non-empty")
        if not isinstance(self.wandb_tags, list) or not all(
                isinstance(value, str) and value.strip() for value in self.wandb_tags):
            raise ValueError("logging.wandb_tags must contain non-empty strings")


@dataclass
class ESpinSLAMConfig:
    """Complete eSpinSLAM reproduction configuration."""

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


def config_from_dict(value: dict[str, Any]) -> ESpinSLAMConfig:
    """Build a configuration from a plain mapping, rejecting unknown keys."""

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


def load_config(path: str | Path) -> ESpinSLAMConfig:
    """Load JSON/JSONC and reject unknown configuration keys."""

    path = Path(path)
    try:
        value = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid eSpinSLAM config: {path}") from error
    return config_from_dict(value)
