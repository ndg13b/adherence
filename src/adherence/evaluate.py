"""Rolling-origin forecast evaluation.

The claim "this person is regular" is only worth as much as the forecast it
supports, so the model is evaluated the way a forecast is: stand at a point in
time, use only what was known then, predict the next 24 hours bin by bin, and
score against what happened. The origin then advances a day and the model is
refit. Nothing here ever sees its own future.

Proper scoring rules only (log loss, Brier) plus calibration, because the
practical question is not "did it rank the right hour first" but "when it says
0.3, does it happen three times in ten" -- a reminder system built on
overconfident probabilities fires at the wrong moments and trains people to
ignore it.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .events import SECONDS_PER_DAY, EventLog, local_day_index, local_midnight_epoch
from .model import RoutineModel

LOG2 = math.log(2.0)
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class Forecast:
    """Predicted probabilities and outcomes for a set of time bins."""

    bin_start: np.ndarray
    bin_min: float
    p: np.ndarray
    y: np.ndarray
    origin: np.ndarray
    n_origins: int = 0

    def __len__(self) -> int:
        return int(self.p.size)


def rolling_forecast(
    log: EventLog,
    model=None,
    bin_min: float = 30.0,
    warmup_days: float = 14.0,
    horizon_hours: float = 24.0,
    step_days: float = 1.0,
    max_origins: int = 1000,
) -> Forecast:
    """Refit daily, forecast the next ``horizon_hours`` in ``bin_min`` bins."""
    model = model if model is not None else RoutineModel()
    bin_s = bin_min * 60.0
    horizon_s = horizon_hours * 3600.0

    d0 = int(local_day_index(np.array([log.t_start + warmup_days * SECONDS_PER_DAY]), log.tz)[0])
    d1 = int(local_day_index(np.array([log.t_end - horizon_s]), log.tz)[0])
    if d1 < d0:
        return Forecast(np.zeros(0), bin_min, np.zeros(0), np.zeros(0), np.zeros(0), 0)
    days = np.arange(d0, d1 + 1, max(int(step_days), 1))[:max_origins]
    origins = local_midnight_epoch(days, log.tz)

    starts, ps, ys, ogs = [], [], [], []
    n_bins = int(round(horizon_s / bin_s))
    for o in origins:
        m = _fit(model, log, o)
        edges = o + np.arange(n_bins + 1) * bin_s
        expected = _bin_integrals(m, edges)
        p = -np.expm1(-expected)
        idx = np.searchsorted(edges, log.t, side="right") - 1
        hit = (log.t >= edges[0]) & (log.t < edges[-1])
        y = np.zeros(n_bins)
        y[np.unique(idx[hit])] = 1.0
        starts.append(edges[:-1])
        ps.append(p)
        ys.append(y)
        ogs.append(np.full(n_bins, o))

    return Forecast(
        bin_start=np.concatenate(starts),
        bin_min=bin_min,
        p=np.clip(np.concatenate(ps), 1e-12, 1 - 1e-12),
        y=np.concatenate(ys),
        origin=np.concatenate(ogs),
        n_origins=len(origins),
    )


def _fit(model, log: EventLog, now: float):
    """Refit a fresh copy of ``model`` on history strictly before ``now``.

    The copy matters: reusing one instance would let state from a later origin
    leak into an earlier forecast, which is the exact leakage this module exists
    to prevent.
    """
    return copy.deepcopy(model).fit(log, now=now)


def _bin_integrals(m, edges: np.ndarray) -> np.ndarray:
    """Integrate the intensity over consecutive bins on a shared fine grid."""
    sub = 6  # sub-samples per bin; the intensity is smooth on this scale
    fine = np.linspace(edges[0], edges[-1], (len(edges) - 1) * sub + 1)
    lam = m.intensity(fine)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(fine))])
    return np.diff(np.interp(edges, fine, cum))


# ------------------------------------------------------------------- metrics
@dataclass
class Metrics:
    n: int
    base_rate: float
    log_loss_bits: float
    brier: float
    auc: float
    ece: float
    calibration: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def metrics(f: Forecast, n_calibration_bins: int = 10) -> Metrics:
    y, p = f.y, f.p
    if y.size == 0:
        return Metrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    ll = -(y * np.log2(p) + (1 - y) * np.log2(1 - p)).mean()
    brier = float(((p - y) ** 2).mean())
    table, ece = calibration_table(y, p, n_calibration_bins)
    return Metrics(
        n=int(y.size),
        base_rate=float(y.mean()),
        log_loss_bits=float(ll),
        brier=brier,
        auc=auc(y, p),
        ece=ece,
        calibration=table,
    )


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), tie-aware."""
    pos, neg = y > 0.5, y <= 0.5
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, p.size + 1)
    # average ranks within ties
    sp = p[order]
    i = 0
    while i < sp.size:
        j = i
        while j + 1 < sp.size and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Reliability table plus expected calibration error."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return [], float("nan")
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
    rows, ece = [], 0.0
    for b in range(edges.size - 1):
        m = idx == b
        if not m.any():
            continue
        pm, ym, n = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        rows.append({"predicted": pm, "observed": ym, "n": n})
        ece += n * abs(pm - ym)
    return rows, float(ece / y.size)


def skill(model_metrics: Metrics, baseline_metrics: Metrics) -> dict:
    """Fractional reduction in loss versus a baseline. 0 = no better, 1 = perfect."""
    out = {}
    for key in ("log_loss_bits", "brier"):
        b = getattr(baseline_metrics, key)
        m = getattr(model_metrics, key)
        out[key + "_skill"] = float(1.0 - m / b) if b and np.isfinite(b) and b > 0 else float("nan")
    return out


def compare(
    log: EventLog,
    models: dict,
    bin_min: float = 30.0,
    warmup_days: float = 14.0,
    horizon_hours: float = 24.0,
    baseline: str = "homogeneous",
    **kw,
) -> dict:
    """Score several forecasters on identical bins and report skill vs a baseline."""
    results = {}
    for name, mdl in models.items():
        f = rolling_forecast(
            log, mdl, bin_min=bin_min, warmup_days=warmup_days,
            horizon_hours=horizon_hours, **kw
        )
        results[name] = metrics(f)
    if baseline in results:
        base = results[baseline]
        bins_per_day = 1440.0 / bin_min
        for name, mt in results.items():
            sk = skill(mt, base)
            sk["bits_saved_per_day"] = (
                (base.log_loss_bits - mt.log_loss_bits) * bins_per_day
                if np.isfinite(base.log_loss_bits) and np.isfinite(mt.log_loss_bits)
                else float("nan")
            )
            results[name] = (mt, sk)
    return results
