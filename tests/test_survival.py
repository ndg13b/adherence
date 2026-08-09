"""The Cox fit is checked against data with a known hazard ratio.

A survival routine that is subtly wrong would silently corrupt every power
calculation built on it, so it is validated by simulation rather than trusted.
"""

import math

import numpy as np
import pytest

from adherence.survival import cox_ph


def _exponential_survival(n, beta, rng, censor_at=None):
    """Exponential times with hazard exp(x*beta); optional administrative censoring."""
    x = rng.normal(size=n)
    hazard = np.exp(x * beta)
    t = rng.exponential(1.0 / hazard)
    if censor_at is None:
        return x, t, np.ones(n)
    event = (t <= censor_at).astype(float)
    return x, np.minimum(t, censor_at), event


@pytest.mark.parametrize("beta", [-1.0, -0.4, 0.0, 0.7, 1.5])
def test_recovers_a_known_log_hazard_ratio(beta):
    rng = np.random.default_rng(0)
    x, t, e = _exponential_survival(4000, beta, rng)
    fit = cox_ph(x[:, None], t, e)
    assert fit.converged
    assert fit.coef[0] == pytest.approx(beta, abs=0.12)


def test_confidence_interval_covers_the_truth():
    rng = np.random.default_rng(1)
    covered = 0
    for s in range(40):
        rng = np.random.default_rng(s)
        x, t, e = _exponential_survival(800, 0.6, rng)
        lo, hi = cox_ph(x[:, None], t, e).ci()[0]
        covered += lo <= 0.6 <= hi
    assert covered >= 33  # nominal 95%, allow sampling slack


def test_handles_censoring():
    rng = np.random.default_rng(2)
    x, t, e = _exponential_survival(3000, 0.8, rng, censor_at=0.5)
    assert 0 < e.mean() < 1  # some censored, some not
    fit = cox_ph(x[:, None], t, e)
    assert fit.coef[0] == pytest.approx(0.8, abs=0.15)
    assert fit.n_events == int(e.sum())


def test_handles_heavy_ties():
    """Event times in whole days produce ties everywhere; Efron must cope."""
    rng = np.random.default_rng(3)
    x, t, e = _exponential_survival(2000, 0.9, rng)
    t_days = np.ceil(t * 20)  # coarse grid -> many ties
    fit = cox_ph(x[:, None], t_days, e)
    assert fit.coef[0] == pytest.approx(0.9, abs=0.2)


def test_null_effect_is_not_significant():
    rng = np.random.default_rng(4)
    x, t, e = _exponential_survival(1500, 0.0, rng)
    fit = cox_ph(x[:, None], t, e)
    assert abs(fit.z[0]) < 2.5


def test_multiple_covariates():
    rng = np.random.default_rng(5)
    n = 4000
    X = rng.normal(size=(n, 2))
    beta = np.array([0.5, -0.8])
    t = rng.exponential(1.0 / np.exp(X @ beta))
    fit = cox_ph(X, t, np.ones(n), names=["a", "b"])
    np.testing.assert_allclose(fit.coef, beta, atol=0.15)
    assert "a" in str(fit) and "HR" in str(fit)


# ------------------------------------------------- time-varying (Andersen-Gill)
def _time_varying_data(n, beta, rng, n_intervals=20, base_hazard=0.05):
    """Discrete-time survival where the covariate is redrawn each interval.

    The covariate genuinely moves within a person, so a model that only used
    each person's first value would be measuring something else.
    """
    starts, stops, events, xs = [], [], [], []
    for _ in range(n):
        for k in range(n_intervals):
            x = rng.normal()
            h = base_hazard * math.exp(beta * x)
            failed = rng.random() < h
            starts.append(k)
            stops.append(k + 1)
            events.append(float(failed))
            xs.append(x)
            if failed:
                break
    return (np.array(xs)[:, None], np.array(starts, dtype=float),
            np.array(stops, dtype=float), np.array(events))


@pytest.mark.parametrize("beta", [-0.7, 0.0, 0.8])
def test_time_varying_recovers_a_known_coefficient(beta):
    from adherence.survival import cox_ph_time_varying

    rng = np.random.default_rng(7)
    X, start, stop, event = _time_varying_data(1500, beta, rng)
    fit = cox_ph_time_varying(X, start, stop, event)
    assert fit.converged
    assert fit.coef[0] == pytest.approx(beta, abs=0.15)


def test_time_varying_uses_the_current_value_not_the_first():
    """The whole point: risk must track the covariate as it moves.

    Each person's covariate is a constant plus a large within-person swing that
    carries the real effect. A model keyed to baseline sees only the constant
    and finds nothing; the time-varying model must find the swing.
    """
    from adherence.survival import cox_ph, cox_ph_time_varying

    rng = np.random.default_rng(11)
    starts, stops, events, xs, first_x, tot, ev = [], [], [], [], [], [], []
    for _ in range(1200):
        baseline = rng.normal()  # pure noise, unrelated to hazard
        seen_first = None
        for k in range(20):
            swing = rng.normal()  # this is what actually drives the hazard
            x = baseline + swing
            if seen_first is None:
                seen_first = x
            failed = rng.random() < 0.05 * math.exp(1.0 * swing)
            starts.append(k); stops.append(k + 1)
            events.append(float(failed)); xs.append(swing)
            if failed:
                break
        first_x.append(seen_first)
        tot.append(stops[-1])
        ev.append(events[-1])

    tv = cox_ph_time_varying(np.array(xs)[:, None], np.array(starts, dtype=float),
                             np.array(stops, dtype=float), np.array(events))
    baseline_only = cox_ph(np.array(first_x)[:, None], np.array(tot, dtype=float),
                           np.array(ev))
    assert tv.coef[0] == pytest.approx(1.0, abs=0.2)
    assert abs(baseline_only.coef[0]) < 0.4
    assert tv.coef[0] > baseline_only.coef[0] + 0.4


def test_time_varying_matches_fixed_cox_when_nothing_varies():
    """With a constant covariate the two models must agree."""
    from adherence.survival import cox_ph, cox_ph_time_varying

    rng = np.random.default_rng(3)
    n = 800
    x = rng.normal(size=n)
    t = rng.exponential(1.0 / np.exp(x * 0.6))
    e = np.ones(n)
    fixed = cox_ph(x[:, None], t, e)
    tv = cox_ph_time_varying(x[:, None], np.zeros(n), t, e)
    assert tv.coef[0] == pytest.approx(fixed.coef[0], abs=0.02)


def test_decaying_routine_is_seen_only_by_the_time_varying_model():
    """The case the frozen-baseline design cannot see, by construction.

    Everyone starts equally regular. Half of them loosen over their final months
    and then stop. A score frozen at month three is identical across the two
    groups and must find nothing; a score that moves with the person must find
    the decay.
    """
    import importlib.util
    import pathlib
    from datetime import datetime, timezone

    from adherence import EventLog, RoutineModel, timing_consistency
    from adherence.survival import cox_ph, cox_ph_time_varying

    spec = importlib.util.spec_from_file_location(
        "tv", str(pathlib.Path(__file__).parent.parent / "examples"
                  / "fitrec_timevarying.py"))
    tv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tv)

    base = datetime(2014, 1, 6, tzinfo=timezone.utc).timestamp()
    rng = np.random.default_rng(0)
    logs, baseline_score, dur, ev = {}, [], [], []
    model = RoutineModel(bandwidth_min=30.0, half_life_days=365.0)

    for u in range(240):
        decays = u % 2 == 0
        n_days = int(rng.integers(420, 640))
        hour = rng.uniform(6, 20)
        ts = []
        for d in range(n_days):
            if rng.random() > 0.55:
                continue
            frac = max(0.0, (d - (n_days - 120)) / 120) if decays else 0.0
            jitter = 20.0 + 220.0 * frac
            ts.append(base + d * 86400 + hour * 3600 + rng.normal(0, jitter * 60))
        t = np.array(sorted(ts))
        log = EventLog.from_records(t, t_start=t[0], t_end=t[-1])
        logs[u] = log

        cut = t[0] + 90 * 86400
        baseline_score.append(timing_consistency(log.slice(t_to=cut), model, now=cut))
        dur.append((t[-1] - cut) / 86400.0)
        ev.append(1.0)

    end = max(v.t[-1] for v in logs.values()) + 200 * 86400
    rows = tv.build_intervals(logs, end, 90.0, 30.0, 30.0, 180.0, 10)
    assert sum(r["event"] for r in rows) > 50

    s = np.array([r["score"] for r in rows])
    x = -(s - s.mean()) / s.std()
    tv_fit = cox_ph_time_varying(
        x[:, None], np.array([r["start"] for r in rows]),
        np.array([r["stop"] for r in rows]), np.array([r["event"] for r in rows]))

    b = np.array(baseline_score)
    frozen = cox_ph((-(b - b.mean()) / b.std())[:, None],
                    np.array(dur), np.array(ev))

    assert tv_fit.coef[0] > 0.3, "moving score should detect the decay"
    assert tv_fit.p[0] < 0.01
    assert abs(frozen.coef[0]) < tv_fit.coef[0] / 2, "frozen score should see far less"
