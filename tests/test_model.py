from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from adherence import EventLog, RoutineModel
from adherence.events import SECONDS_PER_DAY, clock_to_phase, phase_to_clock
from adherence.simulate import PersonProfile, Slot, simulate_person
from conftest import START


def test_empty_log_is_uniform_and_silent():
    t0 = datetime(2025, 1, 6, tzinfo=timezone.utc).timestamp()
    log = EventLog.from_records([], t_start=t0, t_end=t0 + 30 * SECONDS_PER_DAY)
    m = RoutineModel().fit(log)
    assert m.rate == 0.0
    assert m.p_engage_next(24.0) == 0.0
    np.testing.assert_allclose(m.timing_density(np.linspace(0, 6.28, 20)), 1 / (2 * np.pi))


def test_fit_excludes_events_at_or_after_now(clockwork):
    """The model must not see the event it is about to be asked to predict."""
    t = clockwork.t[50]
    m = RoutineModel().fit(clockwork, now=t)
    assert m.n_events == 50
    assert m._t.max() < t


def test_expected_events_recovers_the_rate(clockwork):
    """Integrating the intensity over a window must reproduce the observed rate."""
    m = RoutineModel(half_life_days=10_000).fit(clockwork)
    window = 28 * SECONDS_PER_DAY
    expected = m.expected_events(m.now, m.now + window)
    observed_rate = len(clockwork) / clockwork.span_days
    assert expected / 28.0 == pytest.approx(observed_rate, rel=0.15)


def test_p_engage_is_a_probability_and_increases_with_horizon(clockwork):
    m = RoutineModel().fit(clockwork)
    ps = [m.p_engage_next(h) for h in (1, 6, 24, 72, 168)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert all(a <= b + 1e-12 for a, b in zip(ps, ps[1:]))


def test_best_window_lands_on_the_true_slot():
    """The 07:30 person should get a reminder window containing 07:30."""
    log, _ = simulate_person(
        PersonProfile(slots=[Slot(hour=7.5, jitter_min=10.0, p=0.95)]),
        START, days=90, rng=5, dropout=False,
    )
    m = RoutineModel(bandwidth_min=20).fit(log)
    start, end, p = m.best_window(horizon_hours=24, window_min=60)
    from adherence.events import daily_phase
    from zoneinfo import ZoneInfo

    lo = daily_phase(np.array([start]), ZoneInfo("UTC"))[0]
    hi = daily_phase(np.array([end]), ZoneInfo("UTC"))[0]
    assert lo <= clock_to_phase(7, 30) <= hi
    assert p > 0.5


def test_weekday_structure_is_learned(mwf):
    """Mon/Wed/Fri rates must exceed Tue/Thu/weekend rates."""
    m = RoutineModel().fit(mwf)
    on = m.rate_by_dow[[0, 2, 4]].mean()
    off = m.rate_by_dow[[1, 3, 5, 6]].mean()
    assert on > 5 * off


def test_weekday_shrinkage_prevents_zero_rates(mwf):
    """A never-observed weekday gets a small rate, not an impossible one."""
    m = RoutineModel().fit(mwf)
    assert (m.rate_by_dow > 0).all()
    assert m.rate_by_dow[[1, 3]].max() < m.rate


def test_recency_weighting_tracks_a_moved_routine():
    """After a schedule change, a short memory should follow; a long one should lag."""
    early = [START.replace(tzinfo=timezone.utc) + timedelta(days=d, hours=7) for d in range(60)]
    late = [START.replace(tzinfo=timezone.utc) + timedelta(days=d, hours=20) for d in range(60, 90)]
    log = EventLog.from_records(early + late)

    short = RoutineModel(half_life_days=7, bandwidth_min=30).fit(log)
    long = RoutineModel(half_life_days=365, bandwidth_min=30).fit(log)
    p20, p7 = clock_to_phase(20), clock_to_phase(7)

    assert short.timing_density(np.array([p20]))[0] > short.timing_density(np.array([p7]))[0]
    assert long.timing_density(np.array([p7]))[0] > short.timing_density(np.array([p7]))[0]


def test_uniform_floor_bounds_the_surprise(clockwork):
    """An event at an unprecedented hour must stay finitely surprising."""
    m = RoutineModel(uniform_floor=0.02).fit(clockwork)
    worst = float(m.log_timing_density(np.linspace(0, 2 * np.pi, 500)).min())
    assert np.isfinite(worst)
    assert worst > np.log(0.02 / (2 * np.pi)) - 1e-9


def test_exposure_matches_span_for_long_half_life(clockwork):
    m = RoutineModel(half_life_days=100_000).fit(clockwork)
    assert m._exposure_days == pytest.approx(clockwork.span_days, rel=1e-3)
    assert m._exposure_by_dow.sum() == pytest.approx(m._exposure_days, rel=1e-6)


def test_rate_accounts_for_exposure_not_just_count():
    """Two events in a week is a different rate from two events in a year."""
    base = datetime(2025, 1, 6, 8, tzinfo=timezone.utc)
    ts = [base, base + timedelta(days=3)]
    dense = EventLog.from_records(ts, t_start=base.timestamp(),
                                  t_end=(base + timedelta(days=7)).timestamp())
    sparse = EventLog.from_records(ts, t_start=base.timestamp(),
                                   t_end=(base + timedelta(days=365)).timestamp())
    r_dense = RoutineModel(half_life_days=100_000).fit(dense).rate
    r_sparse = RoutineModel(half_life_days=100_000).fit(sparse).rate
    assert r_dense > 10 * r_sparse


def test_intensity_is_wall_clock_stable_across_dst():
    """A fixed 08:00 local routine keeps its peak at 08:00 after the clocks change."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Europe/London")
    ts = [datetime(2025, 2, 1, tzinfo=tz) + timedelta(days=d, hours=8) for d in range(80)]
    log = EventLog.from_records(ts, tz=tz)
    m = RoutineModel(bandwidth_min=20).fit(log)
    hours, dens = m.daily_profile(n=1440)
    assert phase_to_clock(2 * np.pi * hours[int(np.argmax(dens))] / 24.0).startswith("08:")
