"""Ground-truth-free D-003 loop-artefact diagnostics."""

from dataclasses import replace
import math

import numpy as np

from espinslam.backend import OptimizationReport, empty_reconstruction
from espinslam.config import FrequencyConfig
from espinslam.frequency import FrequencyEstimate
from espinslam.pipeline import (
    ESpinSLAMResult,
    FrequencyRecord,
    _frequency_record_diagnostics,
    _loop_artefact_diagnostic,
)
from espinslam.topspin import (
    TOPSPIN_DEV_SCENES,
    _d003_viability_summary,
    _sequence_row,
)


def _record(*, boundary: bool = False, half_span: bool = False) -> FrequencyRecord:
    return FrequencyRecord(
        event_time=1.0,
        wall_seconds=0.1,
        frequency_hz=0.5,
        loop_closed=False,
        period_seconds=2.0,
        at_search_boundary=boundary,
        near_half_span=half_span,
    )


def test_frequency_record_reports_span_ratio_and_frozen_flags():
    config = FrequencyConfig(
        min_period_seconds=0.1,
        coarse_relative_step=0.01,
        icp_tolerance_seconds=1e-5,
    )
    estimate = FrequencyEstimate(0.5, 2.0, 1.0, True)
    ratio, boundary, half_span = _frequency_record_diagnostics(
        estimate, 4.0, config,
    )
    assert ratio == 0.5
    assert boundary is False
    assert half_span is True

    lower = FrequencyEstimate(10.0, 0.1, 1.0, True)
    _, boundary, half_span = _frequency_record_diagnostics(lower, 4.0, config)
    assert boundary is True and half_span is False

    outside_one_grid_step = FrequencyEstimate(1.0 / 2.03, 2.03, 1.0, True)
    _, boundary, half_span = _frequency_record_diagnostics(
        outside_one_grid_step, 4.0, config,
    )
    assert boundary is False and half_span is False


def test_confirming_window_marks_only_uniform_frozen_artefact_signatures():
    suspected, reason = _loop_artefact_diagnostic(
        [_record(half_span=True) for _ in range(3)], 3,
    )
    assert suspected is True
    assert reason == "all_confirming_periods_within_one_grid_step_of_half_span"

    suspected, reason = _loop_artefact_diagnostic(
        [_record(boundary=True) for _ in range(3)], 3,
    )
    assert suspected is True
    assert reason == "all_confirming_periods_at_search_boundary"

    # A mixed window is retained as a raw loop and is not classified by either
    # of the two preregistered signatures.
    suspected, reason = _loop_artefact_diagnostic(
        [_record(boundary=True), _record(half_span=True), _record()], 3,
    )
    assert suspected is False and reason == ""

    # Invalid estimator updates are recorded for audit, but the paper path
    # neither adds them to nor clears the aleph-sized stability deque.
    invalid = FrequencyRecord(
        event_time=1.5,
        wall_seconds=0.15,
        frequency_hz=math.nan,
        loop_closed=False,
        period_seconds=math.nan,
    )
    suspected, reason = _loop_artefact_diagnostic(
        [_record(half_span=True), invalid, _record(half_span=True),
         invalid, _record(half_span=True)],
        3,
    )
    assert suspected is True
    assert reason == "all_confirming_periods_within_one_grid_step_of_half_span"


def test_suspected_loop_is_reported_but_excluded_from_d003_viability():
    reconstruction = replace(
        empty_reconstruction("fixture"),
        points=np.zeros((4, 3)),
        track_ids=np.arange(4),
        observation_count=12,
        valid=True,
        reason="",
        mean_observation_error_px=1.0,
        mean_reprojection_error_px=1.0,
    )
    optimization = OptimizationReport(
        event_time=2.0,
        frequency_hz=0.5,
        landmarks=4,
        observations=12,
        mean_reprojection_error_px=1.0,
        removed_short=0,
        removed_reprojection=0,
        removed_radius=0,
        fused=0,
        iterations=3,
        termination_status="converged",
    )
    result = ESpinSLAMResult(
        sequence_id="dev-scene",
        observation_seconds=4.0,
        event_count=100,
        frequency=FrequencyEstimate(0.5, 2.0, 1.0, True),
        loop_closed=True,
        loop_closure_time=2.0,
        reconstruction=reconstruction,
        optimizations=[optimization],
        loop_artefact_suspected=True,
        loop_artefact_reason="all_confirming_periods_within_one_grid_step_of_half_span",
    )
    row = _sequence_row(
        {"sequence_id": "dev-scene", "object": "Hubble", "split": "dev"},
        result,
        icp_rmse_mm=None,
        chamfer_mm=None,
        icp_reason="unavailable",
        scale_source="",
    )
    assert row["loop_closed"] is True
    assert row["loop_artefact_suspected"] is True
    assert row["d003_viability_eligible"] is False

    clean_row = _sequence_row(
        {"sequence_id": "dev-scene", "object": "Hubble", "split": "dev"},
        replace(result, loop_artefact_suspected=False, loop_artefact_reason=""),
        icp_rmse_mm=None,
        chamfer_mm=None,
        icp_reason="unavailable",
        scale_source="",
    )
    assert clean_row["d003_viability_eligible"] is True
    assert math.isfinite(clean_row["frequency_est_hz"])

    not_run_row = _sequence_row(
        {"sequence_id": "dev-scene", "object": "Hubble", "split": "dev"},
        replace(
            result,
            loop_artefact_suspected=False,
            loop_artefact_reason="",
            optimizations=[replace(optimization, termination_status="not_run")],
        ),
        icp_rmse_mm=None,
        chamfer_mm=None,
        icp_reason="unavailable",
        scale_source="",
    )
    assert not_run_row["last_optimization_status"] == "not_run"
    assert not_run_row["d003_viability_eligible"] is False


def test_d003_summary_does_not_count_suspected_raw_loop():
    names = sorted(TOPSPIN_DEV_SCENES)
    rows = [
        {"sequence_id": names[0], "split": "dev", "loop_artefact_suspected": True,
         "d003_viability_eligible": False},
        {"sequence_id": names[1], "split": "dev", "loop_artefact_suspected": False,
         "d003_viability_eligible": True},
        {"sequence_id": names[2], "split": "dev", "loop_artefact_suspected": False,
         "d003_viability_eligible": True},
        {"sequence_id": "test", "split": "test", "loop_artefact_suspected": False,
         "d003_viability_eligible": True},
    ]
    summary = _d003_viability_summary(
        rows, reference_mode=True, verified_acquisition=True,
    )
    assert summary["complete"] is True
    assert summary["reference_profile"] is True
    assert summary["verified_acquisition"] is True
    assert summary["canonical_selection_complete"] is True
    assert summary["qualification_reasons"] == []
    assert summary["eligible_scene_count"] == 2
    assert summary["passed"] is True
    assert summary["artefact_suspected_sequences"] == [names[0]]
    assert summary["canonical_scene_ids"] == names
    assert summary["missing_canonical_scenes"] == []
    assert summary["unexpected_dev_scenes"] == []


def test_d003_summary_rejects_three_noncanonical_or_duplicate_dev_rows():
    common = {
        "split": "dev",
        "loop_artefact_suspected": False,
        "d003_viability_eligible": True,
    }
    noncanonical = _d003_viability_summary(
        [{"sequence_id": name, **common} for name in ("a", "b", "c")],
        reference_mode=True,
        verified_acquisition=True,
    )
    assert noncanonical["complete"] is False
    assert noncanonical["passed"] is None
    assert set(noncanonical["missing_canonical_scenes"]) == TOPSPIN_DEV_SCENES
    assert noncanonical["unexpected_dev_scenes"] == ["a", "b", "c"]

    canonical = sorted(TOPSPIN_DEV_SCENES)
    duplicate = _d003_viability_summary(
        [
            {"sequence_id": canonical[0], **common},
            {"sequence_id": canonical[0], **common},
            {"sequence_id": canonical[1], **common},
        ],
        reference_mode=True,
        verified_acquisition=True,
    )
    assert duplicate["scene_count"] == 3
    assert duplicate["complete"] is False
    assert duplicate["passed"] is None
    assert duplicate["missing_canonical_scenes"] == [canonical[2]]


def test_d003_summary_rejects_experimental_or_unverified_runs():
    common = {
        "split": "dev",
        "loop_artefact_suspected": False,
        "d003_viability_eligible": True,
    }
    rows = [
        {"sequence_id": name, **common} for name in sorted(TOPSPIN_DEV_SCENES)
    ]

    experimental = _d003_viability_summary(
        rows, reference_mode=False, verified_acquisition=True,
    )
    assert experimental["canonical_selection_complete"] is True
    assert experimental["reference_profile"] is False
    assert experimental["verified_acquisition"] is True
    assert experimental["complete"] is False
    assert experimental["passed"] is None
    assert experimental["qualification_reasons"] == [
        "method is not the exact full eSpinSLAM reference profile",
    ]

    unverified = _d003_viability_summary(
        rows, reference_mode=True, verified_acquisition=False,
    )
    assert unverified["canonical_selection_complete"] is True
    assert unverified["reference_profile"] is True
    assert unverified["verified_acquisition"] is False
    assert unverified["complete"] is False
    assert unverified["passed"] is None
    assert unverified["qualification_reasons"] == [
        "verified TOPSPIN acquisition audit is unavailable",
    ]
