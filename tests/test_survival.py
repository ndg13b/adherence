"""The Cox fit is checked against data with a known hazard ratio.

A survival routine that is subtly wrong would silently corrupt every power
calculation built on it, so it is validated by simulation rather than trusted.
"""

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
