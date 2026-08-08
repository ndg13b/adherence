"""A minimal Cox proportional-hazards fit, for study planning.

The hypothesis this package is built around -- that a more consistent routine
survives longer -- is a survival claim, and testing it means relating a
consistency score measured during a run-in window to time until disengagement,
with the still-active participants right-censored rather than dropped.

This is deliberately small: Newton-Raphson on the Efron partial likelihood, no
diagnostics, no time-varying covariates, no penalisation. It exists so that
:mod:`adherence.simulate` can answer "how many participants would I need", and
so the worked example runs with no dependencies beyond numpy. For an analysis
you intend to publish, refit in ``lifelines`` or R ``survival`` -- and check
the proportional-hazards assumption, which nothing here does.

Efron's handling of ties is used rather than Breslow's because event times here
are typically recorded in whole days, so ties are the rule, not the exception.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoxResult:
    coef: np.ndarray
    se: np.ndarray
    z: np.ndarray
    p: np.ndarray
    loglik: float
    n: int
    n_events: int
    converged: bool
    names: list[str] | None = None

    @property
    def hazard_ratio(self) -> np.ndarray:
        return np.exp(self.coef)

    def ci(self, level: float = 0.95) -> np.ndarray:
        from scipy.stats import norm

        z = norm.ppf(0.5 + level / 2)
        return np.stack([self.coef - z * self.se, self.coef + z * self.se], axis=1)

    def __str__(self) -> str:
        names = self.names or [f"x{i}" for i in range(len(self.coef))]
        lines = [f"Cox PH: n={self.n}, events={self.n_events}, loglik={self.loglik:.2f}"]
        lines.append(f"{'term':>16s} {'coef':>8s} {'HR':>7s} {'se':>7s} {'z':>7s} {'p':>9s}")
        for i, nm in enumerate(names):
            lines.append(
                f"{nm:>16s} {self.coef[i]:8.4f} {self.hazard_ratio[i]:7.3f} "
                f"{self.se[i]:7.4f} {self.z[i]:7.2f} {self.p[i]:9.2g}"
            )
        if not self.converged:
            lines.append("  [did not converge]")
        return "\n".join(lines)


def cox_ph(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    names: list[str] | None = None,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> CoxResult:
    """Fit ``h(t) = h0(t) exp(X beta)`` by Newton-Raphson on the Efron likelihood.

    Parameters
    ----------
    X:
        ``(n, p)`` covariates. Standardise them if you want the coefficient read
        as "per SD".
    time:
        Follow-up duration per participant.
    event:
        1 if the participant disengaged, 0 if still active at the end (censored).
    """
    from scipy.stats import norm

    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(time):
        X = X.T
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n, p = X.shape

    order = np.argsort(-time, kind="stable")  # descending: risk sets accumulate
    Xs, ts, es = X[order], time[order], event[order]

    beta = np.zeros(p)
    loglik = -np.inf
    converged = False
    info = np.eye(p)

    for _ in range(max_iter):
        eta = Xs @ beta
        eta -= eta.max()  # stabilise the exponential
        w = np.exp(eta)

        ll = 0.0
        grad = np.zeros(p)
        info = np.zeros((p, p))
        s0 = 0.0
        s1 = np.zeros(p)
        s2 = np.zeros((p, p))

        i = 0
        while i < n:
            j = i
            while j + 1 < n and ts[j + 1] == ts[i]:
                j += 1
            # everyone in [i, j] shares this time; add them all to the risk set
            for k in range(i, j + 1):
                s0 += w[k]
                s1 += w[k] * Xs[k]
                s2 += w[k] * np.outer(Xs[k], Xs[k])

            d = int(es[i : j + 1].sum())
            if d > 0:
                dead = np.arange(i, j + 1)[es[i : j + 1] > 0]
                d0 = float(w[dead].sum())
                d1 = (w[dead, None] * Xs[dead]).sum(axis=0)
                d2 = np.einsum("i,ij,ik->jk", w[dead], Xs[dead], Xs[dead])
                ll += float((Xs[dead] @ beta).sum())
                grad += Xs[dead].sum(axis=0)
                for l in range(d):
                    c = l / d
                    a0 = s0 - c * d0
                    a1 = s1 - c * d1
                    a2 = s2 - c * d2
                    ll -= np.log(a0)
                    m = a1 / a0
                    grad -= m
                    info += a2 / a0 - np.outer(m, m)
            i = j + 1

        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if abs(ll - loglik) < tol:
            loglik = ll
            converged = True
            break
        loglik = ll

    try:
        cov = np.linalg.inv(info)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    pval = 2 * norm.sf(np.abs(z))

    return CoxResult(
        coef=beta, se=se, z=z, p=pval, loglik=float(loglik),
        n=n, n_events=int(event.sum()), converged=converged, names=names,
    )
