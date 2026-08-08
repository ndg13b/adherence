"""Established comparators, and simpler forecasters to beat.

Any new regularity index has to earn its place against the ones already in use,
so they are implemented here on the same data structure. Two groups:

*Descriptive indices* from the chronobiology and social-rhythm literature --
Interdaily Stability, the Sleep Regularity Index, the Social Rhythm Metric hit
rate, and hour-of-day entropy. These are the standard of comparison; if the
kernel score merely reproduces them, it is not worth the extra machinery.

*Forecasters* with the same interface as :class:`~adherence.model.RoutineModel`,
so :mod:`adherence.evaluate` can score them head to head. A model that cannot
beat "this person's average rate, spread evenly over the day" has demonstrated
nothing.
"""

from __future__ import annotations

import math

import numpy as np

from .events import SECONDS_PER_DAY, TWO_PI, EventLog, daily_phase, local_day_index
from .kernels import wrap_to_pi
from .model import RoutineModel

# ---------------------------------------------------------------- binned views
def binned_counts(log: EventLog, bin_min: float = 60.0) -> tuple[np.ndarray, int]:
    """Counts per clock bin, shaped ``(n_days, bins_per_day)``.

    Bins are anchored to local midnight, so the grid follows wall-clock time
    through DST changes the way a routine does.
    """
    per_day = int(round(1440.0 / bin_min))
    if abs(per_day * bin_min - 1440.0) > 1e-6:
        raise ValueError("bin_min must divide 1440")
    day0 = int(local_day_index(np.array([log.t_start]), log.tz)[0])
    day1 = int(local_day_index(np.array([log.t_end]), log.tz)[0])
    n_days = max(day1 - day0 + 1, 1)
    grid = np.zeros((n_days, per_day))
    if len(log) == 0:
        return grid, per_day
    d = local_day_index(log.t, log.tz) - day0
    b = (daily_phase(log.t, log.tz) / TWO_PI * per_day).astype(int) % per_day
    ok = (d >= 0) & (d < n_days)
    np.add.at(grid, (d[ok], b[ok]), log.weight[ok])
    return grid, per_day


def interdaily_stability(log: EventLog, bin_min: float = 60.0) -> float:
    """Interdaily Stability (Van Someren et al.), on engagement counts.

    Ratio of the variance of the average 24-hour profile to the overall
    variance: 1 means every day is an exact copy, 0 means no day-to-day
    structure. Standard in actigraphy; applied here to event counts.
    """
    grid, p = binned_counts(log, bin_min)
    x = grid.ravel()
    n = x.size
    if n == 0 or x.var() == 0:
        return float("nan")
    profile = grid.mean(axis=0)
    num = n * ((profile - x.mean()) ** 2).sum()
    den = p * ((x - x.mean()) ** 2).sum()
    return float(num / den) if den > 0 else float("nan")


def intradaily_variability(log: EventLog, bin_min: float = 60.0) -> float:
    """Intradaily Variability: fragmentation of activity within days."""
    grid, _ = binned_counts(log, bin_min)
    x = grid.ravel()
    if x.size < 2 or x.var() == 0:
        return float("nan")
    return float((np.diff(x) ** 2).sum() / (x.size - 1) / x.var())


def sleep_regularity_index(log: EventLog, bin_min: float = 30.0) -> float:
    """SRI-style concordance (Phillips et al. 2017), on engaged/not-engaged state.

    Percentage agreement between the state now and the state exactly 24 hours
    later, rescaled to ``[-100, 100]``. Binary and epoch-based, so it is blind
    to *how far* off a late session was -- the limitation the kernel score is
    meant to remove.
    """
    grid, _ = binned_counts(log, bin_min)
    s = (grid > 0).ravel().astype(int)
    per_day = grid.shape[1]
    if s.size <= per_day:
        return float("nan")
    agree = (s[:-per_day] == s[per_day:]).mean()
    return float(-100.0 + 200.0 * agree)


def social_rhythm_hit_rate(
    log: EventLog, window_min: float = 45.0, habitual_phase: float | None = None
) -> float:
    """Social Rhythm Metric-style hit rate (Monk et al. 1990).

    Fraction of active days on which an event fell within ``+/-window_min`` of the
    habitual time. The habitual time defaults to the circular mean. This is the
    direct ancestor of the kernel score: same idea, hard edges.
    """
    if len(log) == 0:
        return float("nan")
    phase = log.daily_phase
    if habitual_phase is None:
        c, s = np.cos(phase).sum(), np.sin(phase).sum()
        habitual_phase = math.atan2(s, c) % TWO_PI
    tol = TWO_PI * window_min / 1440.0
    hit = np.abs(wrap_to_pi(phase - habitual_phase)) <= tol
    days = local_day_index(log.t, log.tz)
    n_days = len(np.unique(days))
    return float(len(np.unique(days[hit])) / n_days) if n_days else float("nan")


def timing_entropy_bits(log: EventLog, bin_min: float = 60.0) -> float:
    """Shannon entropy of the hour-of-day distribution, in bits.

    Low entropy = concentrated routine. Comparable to the kernel score in
    spirit but discretisation-dependent, and it ignores recency entirely.
    """
    grid, per_day = binned_counts(log, bin_min)
    p = grid.sum(axis=0)
    total = p.sum()
    if total <= 0:
        return float("nan")
    p = p / total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


# ------------------------------------------------------------------ forecasters
class HomogeneousModel:
    """The null forecaster: right rate, no idea when.

    Any claim that timing structure is predictive has to be a claim relative to
    this.
    """

    def __init__(self, half_life_days: float = 28.0):
        self.half_life_days = half_life_days

    def fit(self, log: EventLog, now: float | None = None) -> "HomogeneousModel":
        self.tz = log.tz
        self.now = float(now) if now is not None else log.t_end
        past = log.t < self.now
        w = log.weight[past] * np.exp2(
            -(self.now - log.t[past]) / SECONDS_PER_DAY / self.half_life_days
        )
        span = max((self.now - log.t_start) / SECONDS_PER_DAY, 1e-9)
        eff = min(span, self.half_life_days / math.log(2.0) * (1 - 2 ** (-span / self.half_life_days)))
        self.rate = float(w.sum() / max(eff, 1e-9))
        return self

    def intensity(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.full(t.shape, self.rate / SECONDS_PER_DAY)


class HourHistogramModel:
    """Hour-of-day histogram with Laplace smoothing: no kernel, no recency.

    The point of comparison that isolates what the kernel and the recency
    weighting actually buy, since everything else about the pipeline is shared.
    """

    def __init__(self, bin_min: float = 60.0, alpha: float = 1.0):
        self.bin_min = bin_min
        self.alpha = alpha

    def fit(self, log: EventLog, now: float | None = None) -> "HourHistogramModel":
        self.tz = log.tz
        self.now = float(now) if now is not None else log.t_end
        self.per_day = int(round(1440.0 / self.bin_min))
        past = log.slice(t_to=self.now)
        counts = np.zeros(self.per_day)
        if len(past):
            b = (past.daily_phase / TWO_PI * self.per_day).astype(int) % self.per_day
            np.add.at(counts, b, past.weight)
        span = max((self.now - log.t_start) / SECONDS_PER_DAY, 1e-9)
        self.rate = float(counts.sum() / span)
        self.p_bin = (counts + self.alpha) / (counts.sum() + self.alpha * self.per_day)
        return self

    def intensity(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        b = (daily_phase(t, self.tz) / TWO_PI * self.per_day).astype(int) % self.per_day
        # density per second: rate/day * P(bin) / bin length
        return self.rate * self.p_bin[b] / (self.bin_min * 60.0)


class LastTimeModel:
    """"Same time as last session, give or take." A memory-of-one forecaster.

    Surprisingly hard to beat over short horizons, and a useful check that the
    full model is doing more than tracking the most recent event.
    """

    def __init__(self, spread_min: float = 60.0, half_life_days: float = 28.0):
        self.spread_min = spread_min
        self.half_life_days = half_life_days

    def fit(self, log: EventLog, now: float | None = None) -> "LastTimeModel":
        self.tz = log.tz
        self.now = float(now) if now is not None else log.t_end
        past = log.slice(t_to=self.now)
        self._inner = RoutineModel(
            bandwidth_min=self.spread_min,
            half_life_days=self.half_life_days,
            weekday=False,
        )
        if len(past) == 0:
            self._inner.fit(log, now=self.now)
            self._only_last = False
        else:
            last = past.slice(t_from=past.t[-1])
            self._inner.fit(
                EventLog(
                    t=last.t, tz=log.tz, weight=last.weight,
                    t_start=log.t_start, t_end=self.now,
                ),
                now=self.now,
            )
            # keep the person's overall rate rather than the one-event rate
            span = max((self.now - log.t_start) / SECONDS_PER_DAY, 1e-9)
            self._inner.rate_by_dow[:] = len(past) / span
        return self

    def intensity(self, t) -> np.ndarray:
        return self._inner.intensity(t)
