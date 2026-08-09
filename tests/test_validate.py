"""Tests for the reliability analysis.

The negative control is the one that matters. An analysis that reports "people
differ" whenever scores differ would pass every other test here and be worthless
on real data, because sampling noise alone spreads scores across a cohort of
identical people. So the suite checks both directions: signal is found when it
exists, and *not* found when it does not.
"""

import numpy as np

from adherence import RoutineModel
from adherence.datasets import load_duolingo, write_synthetic_duolingo
from adherence.validate import (
    SCORERS,
    reliability_report,
    score_cohort,
    split_alternate,
)

MODEL = RoutineModel(bandwidth_min=45.0)


def _cohort(tmp_path, identical, n_users=250, seed=5, days=14):
    p = write_synthetic_duolingo(str(tmp_path / f"c{identical}{seed}.csv.gz"),
                                 n_users=n_users, days=days,
                                 identical_people=identical, seed=seed)
    return load_duolingo(p, sample_pct=100.0, min_events=8, min_days=5,
                         verbose=False).logs


# ------------------------------------------------------------------- mechanics
def test_split_alternate_partitions_events(clockwork):
    a, b = split_alternate(clockwork)
    assert len(a) + len(b) == len(clockwork)
    assert abs(len(a) - len(b)) <= 1
    np.testing.assert_allclose(np.sort(np.concatenate([a.t, b.t])), clockwork.t)


def test_split_alternate_preserves_the_window(clockwork):
    """Both halves must span the same calendar window as the whole."""
    a, b = split_alternate(clockwork)
    assert a.t_start == b.t_start == clockwork.t_start
    assert a.t_end == b.t_end == clockwork.t_end


def test_score_cohort_returns_one_value_per_person(clockwork, loose, chaotic):
    logs = {"a": clockwork, "b": loose, "c": chaotic}
    out = score_cohort(logs, MODEL)
    assert set(out) == set(SCORERS)
    assert all(v.shape == (3,) for v in out.values())
    assert out["timing_consistency"][0] > out["timing_consistency"][2]


def test_scorer_failure_yields_nan_not_a_crash(clockwork):
    bad = {"explodes": lambda log, m: 1 / 0, "fine": lambda log, m: 1.0}
    out = score_cohort({"a": clockwork}, MODEL, bad)
    assert np.isnan(out["explodes"][0])
    assert out["fine"][0] == 1.0


# ---------------------------------------------------------- the two controls
def test_signal_is_found_when_people_genuinely_differ(tmp_path):
    logs = _cohort(tmp_path, identical=False)
    rep = reliability_report(logs, MODEL, verbose=False)
    i = rep.get("timing_consistency")
    assert i.reliability > 0.3
    assert i.reliable_sd > 0.05
    assert rep.verdict().startswith("PROCEED")


def test_no_signal_is_claimed_when_everyone_is_identical(tmp_path):
    """The control that stops the analysis fooling itself.

    Every simulated person here has the same true consistency, so any spread in
    their scores is sampling noise. The observed SD is nonetheless clearly
    nonzero -- which is exactly why "do the scores differ?" is the wrong
    question -- and reliability must collapse anyway.
    """
    logs = _cohort(tmp_path, identical=True)
    rep = reliability_report(logs, MODEL, verbose=False)
    i = rep.get("timing_consistency")
    assert i.sd > 0.05, "the noise-only cohort should still show apparent spread"
    assert i.reliability < 0.25
    assert i.reliable_sd < 0.08
    assert rep.verdict().startswith("NOT ESTABLISHED")


def test_reliability_separates_the_two_worlds(tmp_path):
    real = reliability_report(_cohort(tmp_path, False), MODEL, verbose=False)
    noise = reliability_report(_cohort(tmp_path, True), MODEL, verbose=False)
    assert (real.get("timing_consistency").reliability
            > noise.get("timing_consistency").reliability + 0.25)


# ------------------------------------------------------------------- reporting
def test_report_flags_a_frequency_confound(tmp_path):
    """An index that tracks event count must be visibly marked as doing so."""
    logs = _cohort(tmp_path, identical=False)
    scorers = dict(SCORERS)
    scorers["pure_frequency"] = lambda log, m: float(len(log))
    rep = reliability_report(logs, MODEL, scorers, verbose=False)
    assert abs(rep.frequency_confound["pure_frequency"]) > 0.95
    assert abs(rep.frequency_confound["timing_consistency"]) < 0.95


def test_report_renders_and_lists_every_index(tmp_path):
    rep = reliability_report(_cohort(tmp_path, False, n_users=80), MODEL, verbose=False)
    text = str(rep)
    for name in SCORERS:
        assert name in text
    assert "reliab." in text and "vs freq" in text
    assert rep.n_people > 0


def test_verdict_is_inconclusive_without_usable_people(clockwork):
    rep = reliability_report({"a": clockwork}, MODEL, verbose=False)
    assert "INCONCLUSIVE" in rep.verdict() or "NOT ESTABLISHED" in rep.verdict()


def test_reliability_is_bounded(tmp_path):
    rep = reliability_report(_cohort(tmp_path, False, n_users=80), MODEL, verbose=False)
    for i in rep.indices:
        assert np.isnan(i.reliability) or 0.0 <= i.reliability <= 1.0


def test_bandwidth_scan_prefers_a_width_matched_to_the_population(tmp_path):
    """A loose-routine population must not be scored best by a narrow kernel."""
    from adherence.validate import bandwidth_scan, format_bandwidth_scan

    logs = _cohort(tmp_path, identical=False, n_users=120, seed=7)
    rows = bandwidth_scan(logs, bandwidths=(15.0, 45.0, 120.0))
    assert len(rows) == 3
    assert all(np.isfinite(r["reliability"]) for r in rows)
    # A wider kernel always raises the mean; that is why reliability, not the
    # mean, is what the scan is read on.
    means = [r["mean"] for r in rows]
    assert means == sorted(means)
    assert "most reliable" in format_bandwidth_scan(rows)


def test_anchor_scale_diagnosis_reads_the_curve_shape():
    """The shape, not the peak, separates a real routine from a time-of-day habit."""
    from adherence.validate import diagnose_anchor_scale

    # Peaks narrow then declines: habit-scale anchors present.
    anchored = [{"bandwidth_min": b, "reliability": r, "mean": 0.0, "sd": 0.0,
                 "half_correlation": 0.0, "reliable_sd": 0.0}
                for b, r in [(15, 0.874), (30, 0.887), (60, 0.893), (120, 0.878),
                             (240, 0.857), (360, 0.848)]]
    assert diagnose_anchor_scale(anchored)[0] == "anchored"

    # Rises monotonically and plateaus: only a broad part-of-day preference.
    diffuse = [{"bandwidth_min": b, "reliability": r, "mean": 0.0, "sd": 0.0,
                "half_correlation": 0.0, "reliable_sd": 0.0}
               for b, r in [(15, 0.630), (30, 0.748), (60, 0.830), (120, 0.881),
                            (240, 0.898), (360, 0.899)]]
    assert diagnose_anchor_scale(diffuse)[0] == "diffuse"

    # The real Duolingo curve.
    real = [{"bandwidth_min": b, "reliability": r, "mean": 0.0, "sd": 0.0,
             "half_correlation": 0.0, "reliable_sd": 0.0}
            for b, r in [(15, 0.474), (30, 0.558), (45, 0.575), (60, 0.582),
                         (90, 0.603), (120, 0.621), (180, 0.637), (240, 0.639),
                         (360, 0.638)]]
    key, text = diagnose_anchor_scale(real)
    assert key == "diffuse"
    assert "does not have the phenomenon" in text

    assert diagnose_anchor_scale(real[:2])[0] == "unknown"


def test_anchored_verdict_survives_saturated_reliability():
    """Long histories push reliability near 1 everywhere, shrinking the decline.

    The peak stays narrow, so the population is still anchored; a threshold on
    the size of the fall-off would misfile precisely the best-measured cohorts.
    """
    from adherence.validate import diagnose_anchor_scale

    saturated = [{"bandwidth_min": b, "reliability": r, "mean": 0.0, "sd": 0.0,
                  "half_correlation": 0.0, "reliable_sd": 0.0}
                 for b, r in [(15, 0.970), (30, 0.971), (45, 0.972), (60, 0.971),
                              (120, 0.963), (240, 0.953), (360, 0.949)]]
    key, text = diagnose_anchor_scale(saturated)
    assert key == "anchored"
    assert "45 min" in text
