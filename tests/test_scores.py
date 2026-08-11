"""Does the score measure what it claims to measure?

These tests are the argument for the metric. Each one pins a property the
concept requires: ordering by regularity, recovery of a known jitter,
invariance to things that should not matter (which hour, how many sessions,
which timezone), and sensitivity to things that should (drift, multiple slots).
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from adherence import EventLog, RoutineModel
from adherence.scores import (
    anchor_precision,
    consistency_report,
    find_anchors,
    prequential_timing_bits,
    resultant_length,
    timing_consistency,
    timing_consistency_track,
    timing_consistency_track_permuted,
    weekday_regularity,
)
from adherence.simulate import PersonProfile, Slot, simulate_person
from conftest import START, make


def test_ordering_matches_true_regularity(clockwork, loose, chaotic):
    a = timing_consistency(clockwork)
    b = timing_consistency(loose)
    c = timing_consistency(chaotic)
    assert a > b > c
    assert a > 0.85
    assert c < 0.15


def test_chaotic_scores_near_zero(chaotic):
    """Uniform engagement times must sit at the bottom of the scale, not mid-range."""
    assert timing_consistency(chaotic) < 0.1
    assert consistency_report(chaotic).timing_bits < 0.6


def test_score_is_monotone_in_jitter():
    scores = []
    for jitter in (5.0, 15.0, 40.0, 90.0):
        log, _ = simulate_person(
            PersonProfile(slots=[Slot(hour=9.0, jitter_min=jitter, p=0.95)]),
            START, days=120, rng=4, dropout=False,
        )
        scores.append(timing_consistency(log))
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("jitter", [10.0, 25.0, 60.0])
def test_jitter_is_recovered(jitter):
    """The reported jitter should be the jitter that was simulated."""
    log, _ = simulate_person(
        PersonProfile(slots=[Slot(hour=9.0, jitter_min=jitter, p=0.95)]),
        START, days=150, rng=9, dropout=False,
    )
    anchors = find_anchors(log, RoutineModel(bandwidth_min=20, half_life_days=10_000))
    assert len(anchors) == 1
    assert anchors[0].jitter_min == pytest.approx(jitter, rel=0.3)


def test_anchor_time_is_recovered():
    log, _ = simulate_person(
        PersonProfile(slots=[Slot(hour=6.25, jitter_min=12.0, p=0.95)]),
        START, days=120, rng=2, dropout=False,
    )
    anchors = find_anchors(log, RoutineModel(bandwidth_min=20))
    assert anchors[0].clock.startswith("06:1") or anchors[0].clock.startswith("06:2")


def test_drift_is_recovered_and_separated_from_jitter(drifter):
    """A sliding routine reports its slide, and is not charged for it as noise."""
    anchors = find_anchors(drifter, RoutineModel(bandwidth_min=30, half_life_days=10_000))
    a = anchors[0]
    assert a.drift_min_per_week == pytest.approx(25.0, rel=0.25)
    assert a.jitter_min < 30.0  # true jitter is 15 min
    assert a.spread_min > a.jitter_min  # undetrended spread is inflated by the drift


def test_two_tight_slots_are_not_one_sloppy_habit(bimodal, loose):
    """The failure a single number cannot avoid, and why two are reported."""
    r_bi = consistency_report(bimodal)
    r_lo = consistency_report(loose)
    # Comparable on raw predictability of the next session...
    assert abs(r_bi.timing_consistency - r_lo.timing_consistency) < 0.25
    # ...but the bimodal person's slots are far tighter.
    assert r_bi.anchor_precision > 0.75
    assert r_bi.anchor_precision > r_lo.anchor_precision + 0.3
    assert len(r_bi.anchors) == 2


def test_anchor_precision_collapses_on_uniform_times():
    """The permutation null must remove the concentration that basins invent.

    Cutting the circle into basins around density peaks makes any set of times
    look tight *within its own basin*. Uncorrected, uniform random engagement
    scored ~0.48 here -- above a real but sloppy routine.
    """
    scores = []
    for seed in range(5):
        log, _ = simulate_person(
            PersonProfile(slots=[], off_schedule_per_day=1.0, off_schedule_hours=(0.0, 24.0)),
            START, days=150, rng=seed, dropout=False,
        )
        scores.append(anchor_precision(log))
    assert np.mean(scores) < 0.15


def test_anchor_precision_is_deterministic(bimodal):
    """A score that moved between calls on identical data would be unusable."""
    assert len({round(anchor_precision(bimodal), 12) for _ in range(3)}) == 1


def test_anchor_precision_orders_slots_by_tightness(bimodal, loose, clockwork):
    assert anchor_precision(clockwork) > anchor_precision(bimodal) - 0.15
    assert anchor_precision(bimodal) > anchor_precision(loose) + 0.3


def test_bimodal_anchors_are_at_the_right_hours(bimodal):
    clocks = sorted(a.clock for a in find_anchors(bimodal, RoutineModel(bandwidth_min=20)))
    assert clocks[0].startswith("07:")
    assert clocks[1].startswith("2")  # 21:00-ish


def test_resultant_length_fails_where_the_kernel_score_succeeds(bimodal):
    """Motivates the kernel: classic circular concentration collapses on two slots."""
    assert resultant_length(bimodal.daily_phase) < 0.3
    assert anchor_precision(bimodal) > 0.75


def test_weekday_regularity_detects_mwf(mwf, clockwork):
    """Mon/Wed/Fri carries weekday information; every-day-at-07:30 does not."""
    assert weekday_regularity(mwf) > 0.3
    assert weekday_regularity(clockwork) < 0.1


def test_weekday_conditioning_helps_the_weekend_shifter():
    log, _ = make("weekend_shifter", days=150, seed=6)
    plain = timing_consistency(log)
    by_dow = timing_consistency(log, by_weekday=True)
    assert by_dow > plain + 0.1


def test_score_is_invariant_to_which_hour():
    """07:00 and 22:00 routines of equal tightness must score equally."""
    scores = []
    for hour in (7.0, 13.0, 22.0, 0.25):  # includes one that straddles midnight
        log, _ = simulate_person(
            PersonProfile(slots=[Slot(hour=hour, jitter_min=20.0, p=0.95)]),
            START, days=120, rng=13, dropout=False,
        )
        scores.append(timing_consistency(log))
    assert max(scores) - min(scores) < 0.1


def test_score_is_invariant_to_timezone():
    """The same wall-clock routine scores the same wherever it is lived."""
    utc, _ = simulate_person(
        PersonProfile(slots=[Slot(hour=8.0, jitter_min=15.0, p=0.9)]),
        START, days=120, tz="UTC", rng=21, dropout=False,
    )
    tokyo, _ = simulate_person(
        PersonProfile(slots=[Slot(hour=8.0, jitter_min=15.0, p=0.9)]),
        START, days=120, tz="Asia/Tokyo", rng=21, dropout=False,
    )
    assert timing_consistency(utc) == pytest.approx(timing_consistency(tokyo), abs=0.02)


def test_dst_does_not_manufacture_irregularity():
    """A perfectly rigid local routine must not lose points for a clock change."""
    tz = ZoneInfo("Europe/London")
    ts = [datetime(2025, 2, 1, tzinfo=tz) + timedelta(days=d, hours=8) for d in range(120)]
    local = EventLog.from_records(ts, tz=tz)
    naive = EventLog.from_records(ts, tz="UTC")  # same instants, read in UTC
    assert timing_consistency(local) > 0.99
    assert timing_consistency(local) > timing_consistency(naive)


def test_score_does_not_reward_sheer_volume():
    """Frequency and regularity are different axes; the score must not conflate them."""
    once = PersonProfile(slots=[Slot(hour=8.0, jitter_min=20.0, p=0.5)])
    thrice = PersonProfile(slots=[Slot(hour=8.0, jitter_min=20.0, p=1.0)])
    a, _ = simulate_person(once, START, days=150, rng=8, dropout=False)
    b, _ = simulate_person(thrice, START, days=150, rng=8, dropout=False)
    assert len(b) > 1.5 * len(a)
    assert timing_consistency(a) == pytest.approx(timing_consistency(b), abs=0.12)


def test_prequential_bits_are_positive_for_a_real_routine(clockwork):
    bits, n = prequential_timing_bits(clockwork)
    assert n > 50
    assert bits > 2.0


def test_prequential_bits_collapse_for_noise(chaotic):
    bits, n = prequential_timing_bits(chaotic)
    assert n > 20
    assert bits < 0.5


def test_prequential_bits_can_go_negative_when_the_routine_moves():
    """Honest scoring must be able to say 'your history is now misleading'."""
    early = [START.replace(tzinfo=timezone.utc) + timedelta(days=d, hours=7) for d in range(40)]
    late = [START.replace(tzinfo=timezone.utc) + timedelta(days=d, hours=19) for d in range(40, 45)]
    log = EventLog.from_records(early + late)
    model = RoutineModel(bandwidth_min=15, half_life_days=10_000, uniform_floor=0.01)

    # Score only the post-move sessions, but let each forecast use the full past.
    bits_after, n = prequential_timing_bits(log, model, score_from=log.t[40])
    bits_before, _ = prequential_timing_bits(log, model, score_to=log.t[40])
    assert n == 5
    assert bits_before > 3.0  # the old routine was highly predictable
    assert bits_after < 0.0  # and is now actively misleading


def test_report_is_serialisable_and_printable(clockwork):
    r = consistency_report(clockwork)
    d = r.to_dict()
    assert set(["timing_consistency", "anchors", "weekday_rates"]).issubset(d)
    assert isinstance(str(r), str) and "timing consistency" in str(r)


def test_short_history_is_flagged_not_guessed():
    ts = [START.replace(tzinfo=timezone.utc) + timedelta(days=d, hours=8) for d in range(3)]
    log = EventLog.from_records(ts)
    r = consistency_report(log)
    assert r.warmup
    assert np.isnan(r.timing_bits)


def test_score_is_invariant_to_the_scale_of_recency_weights():
    """Only weight ratios may matter, never their absolute size.

    Recency weights are 2^(-age/half-life), so an observation window ending
    years after someone's last event drives every weight to ~1e-16. An absolute
    floor in the leave-one-out denominator then clamped, and a genuinely tight
    routine scored exactly 0.0000 instead of 0.7425. Real data hit this the
    moment a dataset spanned years rather than weeks.
    """
    base = datetime(2012, 1, 5, tzinfo=timezone.utc).timestamp()
    rng = np.random.default_rng(0)
    t = np.array([base + d * 86400 + 8 * 3600 + rng.normal(0, 1800) for d in range(200)])
    model = RoutineModel(bandwidth_min=45.0, half_life_days=28.0)

    own = EventLog.from_records(t, t_start=t[0], t_end=t[-1])
    stretched = EventLog.from_records(t, t_start=t[0], t_end=t[-1] + 4 * 365 * 86400)

    score = timing_consistency(own, model)
    assert score > 0.5, "a tight routine must score well in the first place"
    assert timing_consistency(stretched, model) == pytest.approx(score, abs=1e-9)


def test_score_is_invariant_to_a_constant_weight_multiplier():
    """The same property stated directly on per-event weights."""
    base = datetime(2020, 1, 6, tzinfo=timezone.utc).timestamp()
    t = np.array([base + d * 86400 + 9 * 3600 for d in range(60)])
    model = RoutineModel(bandwidth_min=30.0, half_life_days=10_000)
    plain = EventLog.from_records(t, t_start=t[0], t_end=t[-1])
    tiny = EventLog.from_records(t, weights=np.full(t.size, 1e-18),
                                 t_start=t[0], t_end=t[-1])
    assert timing_consistency(tiny, model) == pytest.approx(
        timing_consistency(plain, model), abs=1e-9)


# ------------------------------------------------ scoring the same person repeatedly
def _decaying_person(seed=0, tight_days=200, loose_days=160):
    """Tight at 07:00, then coming apart. The trajectory is the point."""
    base = datetime(2014, 1, 6, tzinfo=timezone.utc).timestamp()
    rng = np.random.default_rng(seed)
    t = [base + d * 86400 + 7 * 3600 + rng.normal(0, 10 * 60)
         for d in range(tight_days)]
    t += [base + (tight_days + d) * 86400 + rng.uniform(5, 22) * 3600
          for d in range(loose_days)]
    t = np.array(sorted(t))
    return EventLog.from_records(t, tz="Europe/London", t_start=t[0], t_end=t[-1])


def test_track_matches_scoring_one_moment_at_a_time():
    """The fast path exists for speed only; it must change no number.

    It shares the pairwise kernel weights across moments instead of rebuilding
    them, which is worth an order of magnitude and would be worth nothing if it
    quietly returned something else.
    """
    log = _decaying_person()
    model = RoutineModel(bandwidth_min=30.0, half_life_days=28.0)
    at = np.linspace(log.t[0] + 90 * 86400, log.t[-1], 20)

    one_at_a_time = np.array([
        timing_consistency(log.slice(t_to=a), model, now=a) for a in at])
    np.testing.assert_allclose(
        timing_consistency_track(log, at, model), one_at_a_time, atol=1e-12)


def test_track_falls_when_the_routine_comes_apart():
    log = _decaying_person()
    model = RoutineModel(bandwidth_min=30.0, half_life_days=28.0)
    at = np.array([log.t[0] + d * 86400 for d in (150.0, 200.0, 260.0, 340.0)])
    track = timing_consistency_track(log, at, model, min_events=10)
    assert min(track[0], track[1]) > 0.6, "should score well while the routine holds"
    assert track[3] < 0.15, "and near zero once the times are scattered"
    assert track[2] < track[1], "with the fall beginning as soon as it does"


def test_track_needs_history_before_it_reports_anything():
    log = _decaying_person()
    at = np.array([log.t[0] - 86400, log.t[0] + 3 * 86400, log.t[0] + 200 * 86400])
    track = timing_consistency_track(log, at, min_events=10)
    assert np.isnan(track[0]) and np.isnan(track[1])
    assert np.isfinite(track[2])


def test_permuted_track_equals_scoring_the_reordered_phases():
    """The batched permutation must agree with the obvious implementation."""
    log = _decaying_person(seed=3)
    model = RoutineModel(bandwidth_min=30.0, half_life_days=28.0)
    at = np.linspace(log.t[0] + 90 * 86400, log.t[-1], 12)

    perms = timing_consistency_track_permuted(log, at, 4, model, min_events=10, seed=5)
    rng = np.random.default_rng(5)
    phase = log.daily_phase
    for r in range(4):
        order = rng.permutation(phase.size)
        direct = timing_consistency_track(log, at, model, min_events=10,
                                          phase=phase[order])
        np.testing.assert_allclose(perms[r], direct, atol=1e-12)


def test_permutation_holds_the_level_but_destroys_the_trend():
    """What the null is for: it must keep how regular someone is overall while
    removing *when* they were regular.

    Reordering a person's own times of day cannot change their marginal
    distribution of times, so a permuted person is about as regular on average.
    What it does remove is the decay -- so the real track ends far below where
    it started and the permuted tracks do not.
    """
    log = _decaying_person(seed=1)
    model = RoutineModel(bandwidth_min=30.0, half_life_days=28.0)
    at = np.array([log.t[0] + d * 86400 for d in (150.0, 340.0)])

    real = timing_consistency_track(log, at, model, min_events=10)
    null = timing_consistency_track_permuted(log, at, 30, model, min_events=10, seed=0)

    real_drop = real[0] - real[1]
    null_drop = null[:, 0] - null[:, 1]
    assert real_drop > 0.5, "the real routine must visibly come apart"
    assert abs(np.median(null_drop)) < 0.1, "a shuffled history must not trend"
    assert real_drop > np.quantile(null_drop, 0.99)


def test_permutation_leaves_a_stable_routine_alone():
    """A person whose timing never changes has nothing for the shuffle to break.

    This is the property that makes the null specific: it targets the *trend* in
    someone's timing, not their regularity, so it cannot be accused of simply
    destroying the signal it is testing.
    """
    base = datetime(2014, 1, 6, tzinfo=timezone.utc).timestamp()
    rng = np.random.default_rng(2)
    t = np.array([base + d * 86400 + 7 * 3600 + rng.normal(0, 12 * 60)
                  for d in range(300)])
    log = EventLog.from_records(t, t_start=t[0], t_end=t[-1])
    model = RoutineModel(bandwidth_min=30.0, half_life_days=28.0)
    at = np.array([t[0] + 250 * 86400])

    real = timing_consistency_track(log, at, model, min_events=10)[0]
    null = timing_consistency_track_permuted(log, at, 20, model, min_events=10)
    assert real == pytest.approx(float(np.mean(null)), abs=0.05)


def test_permuted_track_falls_back_correctly_for_a_large_history():
    """Above the memory guard a different code path runs; it must agree."""
    from adherence import scores as _scores

    log = _decaying_person(seed=4, tight_days=60, loose_days=40)
    model = RoutineModel(bandwidth_min=30.0, half_life_days=28.0)
    at = np.linspace(log.t[0] + 30 * 86400, log.t[-1], 6)

    batched = timing_consistency_track_permuted(log, at, 3, model, min_events=10, seed=9)
    original = _scores._MAX_GRAM
    try:
        _scores._MAX_GRAM = 1  # force the low-memory path
        looped = timing_consistency_track_permuted(log, at, 3, model,
                                                   min_events=10, seed=9)
    finally:
        _scores._MAX_GRAM = original
    np.testing.assert_allclose(batched, looped, atol=1e-12)
