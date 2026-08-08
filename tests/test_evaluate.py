"""Tests for the forecast machinery.

The leakage test is the important one. Every claim the package makes rests on
forecasts being made from the past only, and that property is easy to break by
accident and invisible when it is broken -- the numbers just quietly get better.
"""

from datetime import timedelta, timezone

import numpy as np
import pytest

from adherence import EventLog, RoutineModel
from adherence.baselines import (
    HomogeneousModel,
    LastTimeModel,
    interdaily_stability,
    intradaily_variability,
    sleep_regularity_index,
    social_rhythm_hit_rate,
    timing_entropy_bits,
)
from adherence.evaluate import auc, calibration_table, compare, metrics, rolling_forecast
from conftest import START, make


def test_no_future_leakage():
    """Forecasts must be identical when only the future differs.

    Two logs share a history up to day 60 and diverge wildly after it. Every
    forecast issued at an origin at or before day 60 must be bit-identical; if
    any of them differ, information has flowed backwards through time.
    """
    shared = [START.replace(tzinfo=timezone.utc) + timedelta(days=d, hours=8) for d in range(60)]
    future_a = [START.replace(tzinfo=timezone.utc) + timedelta(days=d, hours=8)
                for d in range(60, 90)]
    future_b = [START.replace(tzinfo=timezone.utc) + timedelta(days=d, hours=3, minutes=17)
                for d in range(60, 90)]
    t0 = shared[0].timestamp()
    t_end = (START.replace(tzinfo=timezone.utc) + timedelta(days=90)).timestamp()

    log_a = EventLog.from_records(shared + future_a, t_start=t0, t_end=t_end)
    log_b = EventLog.from_records(shared + future_b, t_start=t0, t_end=t_end)

    kw = dict(bin_min=30, warmup_days=14, horizon_hours=24)
    fa = rolling_forecast(log_a, RoutineModel(), **kw)
    fb = rolling_forecast(log_b, RoutineModel(), **kw)

    cutoff = START.replace(tzinfo=timezone.utc).timestamp() + 59 * 86400
    ma, mb = fa.origin <= cutoff, fb.origin <= cutoff
    assert ma.sum() > 40
    np.testing.assert_array_equal(fa.origin[ma], fb.origin[mb])
    np.testing.assert_allclose(fa.p[ma], fb.p[mb], rtol=0, atol=0)


def test_forecasts_are_calibrated(clockwork):
    """When the model says 0.3, it should happen about three times in ten."""
    f = rolling_forecast(clockwork, RoutineModel(bandwidth_min=12), bin_min=30, warmup_days=21)
    m = metrics(f)
    assert m.ece < 0.02
    for row in m.calibration:
        if row["n"] >= 50:
            assert abs(row["predicted"] - row["observed"]) < 0.12


def test_predicted_event_count_matches_observed(clockwork):
    f = rolling_forecast(clockwork, RoutineModel(bandwidth_min=12), bin_min=30, warmup_days=21)
    assert f.p.sum() == pytest.approx(f.y.sum(), rel=0.2)


def test_routine_model_beats_the_null_on_a_real_routine(clockwork):
    res = compare(
        clockwork,
        {"routine": RoutineModel(bandwidth_min=12), "homogeneous": HomogeneousModel()},
        bin_min=30, warmup_days=21,
    )
    assert res["routine"][1]["log_loss_bits_skill"] > 0.4
    assert res["routine"][0].auc > 0.9


def test_no_model_beats_the_null_on_noise(chaotic):
    """Guards against a metric that flatters the model when there is nothing there."""
    res = compare(
        chaotic,
        {"routine": RoutineModel(), "homogeneous": HomogeneousModel()},
        bin_min=30, warmup_days=21,
    )
    assert abs(res["routine"][1]["log_loss_bits_skill"]) < 0.15
    assert res["routine"][0].auc < 0.75


def test_last_time_baseline_is_beaten_on_a_two_slot_routine(bimodal):
    """A memory-of-one forecaster should lose to a model that holds both slots."""
    res = compare(
        bimodal,
        {
            "routine": RoutineModel(bandwidth_min=20),
            "last_time": LastTimeModel(),
            "homogeneous": HomogeneousModel(),
        },
        bin_min=30, warmup_days=21,
    )
    assert res["routine"][1]["log_loss_bits_skill"] > res["last_time"][1]["log_loss_bits_skill"]


def test_auc_matches_a_known_case():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(1.0)
    assert auc(y, np.array([0.4, 0.3, 0.2, 0.1])) == pytest.approx(0.0)
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_calibration_table_is_exact_on_constructed_data():
    p = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    y = np.concatenate([np.zeros(80), np.ones(20), np.zeros(20), np.ones(80)])
    rows, ece = calibration_table(y, p, n_bins=2)
    assert len(rows) == 2
    assert ece < 1e-9


def test_empty_forecast_is_handled():
    log, _ = make("clockwork", days=5)
    f = rolling_forecast(log, RoutineModel(), warmup_days=30)
    assert len(f) == 0
    assert np.isnan(metrics(f).log_loss_bits)


# ------------------------------------------------------- classic index comparators
def test_classic_indices_agree_on_the_extremes(clockwork, chaotic):
    assert interdaily_stability(clockwork) > interdaily_stability(chaotic)
    assert sleep_regularity_index(clockwork) > sleep_regularity_index(chaotic)
    assert social_rhythm_hit_rate(clockwork) > social_rhythm_hit_rate(chaotic)
    assert timing_entropy_bits(clockwork) < timing_entropy_bits(chaotic)
    assert intradaily_variability(clockwork) > 0


def _jitter_log(jitter, days=150, seed=3):
    from adherence.simulate import PersonProfile, Slot, simulate_person

    return simulate_person(
        PersonProfile(slots=[Slot(hour=8.0, jitter_min=jitter, p=1.0)]),
        START, days=days, rng=seed, dropout=False,
    )[0]


def test_hard_window_saturates_where_the_kernel_score_still_discriminates():
    """The +/-45 minute box cannot tell 5 minutes of jitter from 20.

    Nearly every session lands inside the window either way, so the hit rate is
    ~1.0 for both. The kernel score keeps separating them because a session 20
    minutes late is worth less than one 5 minutes late instead of being worth
    exactly as much. This is the concrete gain from grading the window.
    """
    from adherence.scores import timing_consistency

    tight, looser = _jitter_log(5.0), _jitter_log(20.0)
    srm_gap = social_rhythm_hit_rate(tight) - social_rhythm_hit_rate(looser)
    kernel_gap = timing_consistency(tight) - timing_consistency(looser)
    assert srm_gap < 0.06
    assert kernel_gap > 2 * srm_gap


def test_binned_indices_saturate_at_fine_jitter():
    """IS and SRI are blind below their bin width; the kernel score is not."""
    from adherence.scores import timing_consistency

    a, b = _jitter_log(3.0), _jitter_log(10.0)
    assert interdaily_stability(a) == pytest.approx(interdaily_stability(b), abs=1e-9)
    assert sleep_regularity_index(a) == pytest.approx(sleep_regularity_index(b), abs=1e-9)
    assert timing_consistency(a) > timing_consistency(b) + 0.02


def test_sri_dynamic_range_is_narrow():
    """SRI moves ~3 points over a 30x change in jitter; the kernel score moves 0.7.

    Not a flaw in SRI -- it was built for binary sleep/wake state, where most
    epochs agree trivially -- but it is why it makes a poor engagement-timing
    score.
    """
    from adherence.scores import timing_consistency

    a, b = _jitter_log(3.0), _jitter_log(90.0)
    sri_range = abs(sleep_regularity_index(a) - sleep_regularity_index(b))
    kernel_range = abs(timing_consistency(a) - timing_consistency(b))
    assert sri_range < 5.0  # on a -100..100 scale
    assert kernel_range > 0.5  # on a 0..1 scale
