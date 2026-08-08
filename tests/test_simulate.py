import numpy as np
import pytest

from adherence.events import local_dow
from adherence.simulate import (
    PRESETS,
    PersonProfile,
    Slot,
    simulate_cohort,
    simulate_linked_cohort,
    simulate_person,
)
from conftest import START


def test_seed_is_reproducible():
    a, _ = simulate_person(PRESETS["clockwork"], START, days=60, rng=42)
    b, _ = simulate_person(PRESETS["clockwork"], START, days=60, rng=42)
    np.testing.assert_array_equal(a.t, b.t)


def test_different_seeds_differ():
    a, _ = simulate_person(PRESETS["clockwork"], START, days=60, rng=1)
    b, _ = simulate_person(PRESETS["clockwork"], START, days=60, rng=2)
    assert len(a) != len(b) or not np.array_equal(a.t, b.t)


def test_dropout_truncates_and_is_recorded():
    prof = PersonProfile(slots=[Slot(hour=8.0, p=1.0)], dropout_hazard=0.02)
    log, truth = simulate_person(prof, START, days=400, rng=3)
    assert truth["dropout_day"] is not None
    assert truth["survived_days"] < 400
    assert len(log) > 0
    assert (log.t.max() - log.t_start) / 86400 <= truth["dropout_day"] + 1


def test_immediate_dropout_yields_an_empty_but_valid_log():
    prof = PersonProfile(slots=[Slot(hour=8.0)], dropout_hazard=1.0)
    log, truth = simulate_person(prof, START, days=90, rng=3)
    assert len(log) == 0
    assert truth["dropout_day"] == 0
    assert log.span_days == pytest.approx(90.0, abs=1.0)


def test_dropout_can_be_disabled():
    prof = PersonProfile(slots=[Slot(hour=8.0, p=1.0)], dropout_hazard=0.2)
    log, truth = simulate_person(prof, START, days=100, rng=3, dropout=False)
    assert truth["dropout_day"] is None
    assert len(log) > 90


def test_weekday_restriction_is_respected():
    prof = PersonProfile(slots=[Slot(hour=8.0, jitter_min=5.0, days=(0, 2, 4), p=1.0)])
    log, _ = simulate_person(prof, START, days=90, rng=5, dropout=False)
    assert set(np.unique(local_dow(log.t, log.tz))) <= {0, 2, 4}


def test_observation_window_covers_the_requested_span():
    log, _ = simulate_person(PRESETS["clockwork"], START, days=45, rng=1, dropout=False)
    assert log.span_days == pytest.approx(45.0, abs=1.0)


def test_cohort_shapes():
    cohort = simulate_cohort(n_per_profile=2, days=40, rng=0)
    assert len(cohort) == 2 * len(PRESETS)
    assert all(isinstance(truth, dict) for _, truth in cohort)


def test_linked_cohort_builds_in_the_assumed_dependence():
    """Sanity check on the power-analysis generator: regular people last longer.

    This is true by construction -- it is what the generator was told to do.
    The test confirms the wiring, not the hypothesis.
    """
    cohort = simulate_linked_cohort(n=120, days=180, log_hazard_ratio=1.0, rng=0)
    jit = np.array([t["true_jitter_min"] for _, t in cohort])
    surv = np.array([t["survived_days"] for _, t in cohort])
    r = np.corrcoef(np.log(jit), surv)[0, 1]
    assert r < -0.15
