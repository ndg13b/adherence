"""The routine model: a recency-weighted, periodic point process.

The object of study is a person's engagement history, and the model treats it as
an inhomogeneous Poisson process whose intensity is periodic in wall-clock time:

    lambda(t) = mu[dow(t)] * f_dow(t)(phase(t)) * (2*pi / 86400)

with two estimated pieces:

``mu[d]``
    How many sessions this person does on a day of week ``d`` (events/day),
    shrunk toward their overall rate so that a single missed Tuesday does not
    convince the model that Tuesdays are dead.

``f_d(phase)``
    *When* within the day those sessions land -- a circular kernel density over
    past events, so each event contributes a bump with tails on either side.
    Also shrunk toward the person's pooled time-of-day density.

Everything is weighted by recency (exponential half-life), so the model tracks a
routine that moves, and the estimate at time ``now`` uses only events before
``now`` -- which is what makes honest out-of-sample scoring possible.

The factorisation matters. Rate and timing answer different questions ("do they
show up on Tuesdays at all?" vs. "when on a Tuesday?"), they degrade
differently, and a single blurred intensity would confuse a person who trains
three times a week like clockwork with one who trains daily at random hours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .events import (
    SECONDS_PER_DAY,
    TWO_PI,
    EventLog,
    daily_phase,
    local_day_index,
    local_dow,
    local_midnight_epoch,
)
from .kernels import LOG_UNIFORM, CircularKernel, make_kernel

LOG2 = math.log(2.0)

# numpy renamed trapz -> trapezoid in 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class RoutineModel:
    """Fit a periodic intensity to an :class:`~adherence.events.EventLog`.

    Parameters
    ----------
    kernel:
        Kernel name (see :func:`~adherence.kernels.make_kernel`).
    bandwidth_min:
        Timing tolerance in minutes -- how far off their usual time a session can
        land and still count as "the same slot". 45 minutes is a reasonable
        default and matches the window used by the Social Rhythm Metric; pass a
        ``(before, after)`` tuple for asymmetric tails.
    half_life_days:
        Recency half-life. An event 28 days old counts half as much as today's
        by default. Shorter means the score reacts faster to a schedule change
        and is noisier; longer means it is stable but stale.
    weekday:
        Whether to model day-of-week structure at all. ``False`` pools every day
        together, which is right for a daily-prescription intervention.
    rate_shrinkage_days, timing_shrinkage_events:
        Strength of the pull toward the pooled estimate, in units of prior
        observations. Larger = more conservative about weekday-specific claims.
    uniform_floor:
        Mixture weight on a uniform "they could show up any time" component.
        Without it a single surprising session has unbounded log-loss and one
        3 a.m. outlier dominates the score.
    """

    kernel: str = "vonmises"
    bandwidth_min: float | tuple[float, float] = 45.0
    half_life_days: float = 28.0
    weekday: bool = True
    rate_shrinkage_days: float = 3.0
    timing_shrinkage_events: float = 4.0
    uniform_floor: float = 0.02

    # ----------------------------------------------------------- fitted state
    fitted: bool = field(default=False, init=False)
    now: float = field(default=0.0, init=False)
    _k: CircularKernel = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.uniform_floor < 1.0:
            raise ValueError("uniform_floor must be in [0, 1)")
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self._k = make_kernel(self.kernel, self.bandwidth_min)

    # -------------------------------------------------------------------- fit
    def fit(self, log: EventLog, now: float | None = None) -> "RoutineModel":
        """Estimate the intensity from events strictly before ``now``."""
        self.tz = log.tz
        self.now = float(now) if now is not None else float(max(log.t_end, log.t[-1] if len(log) else log.t_end))
        self.t_start = float(log.t_start)

        past = log.t < self.now
        self._t = log.t[past]
        self._phase = daily_phase(self._t, log.tz) if self._t.size else np.zeros(0)
        self._dow = local_dow(self._t, log.tz) if self._t.size else np.zeros(0, dtype=np.int64)

        age_days = (self.now - self._t) / SECONDS_PER_DAY
        self._w = log.weight[past] * np.exp2(-age_days / self.half_life_days)
        self.w_total = float(self._w.sum())
        self.n_events = int(self._t.size)
        self.n_effective = float(self.w_total**2 / np.sum(self._w**2)) if self.w_total > 0 else 0.0

        self._exposure_days, self._exposure_by_dow = _weighted_exposure(
            self.t_start, self.now, log.tz, self.half_life_days
        )
        self._w_by_dow = np.zeros(7)
        for d in range(7):
            self._w_by_dow[d] = float(self._w[self._dow == d].sum()) if self.n_events else 0.0

        # Rates (events per day), with an empirical-Bayes pull toward the pooled rate.
        self.rate = self.w_total / self._exposure_days if self._exposure_days > 0 else 0.0
        s = self.rate_shrinkage_days
        if self.weekday:
            self.rate_by_dow = (self._w_by_dow + s * self.rate) / (self._exposure_by_dow + s)
        else:
            self.rate_by_dow = np.full(7, self.rate)

        self.fitted = True
        return self

    # ------------------------------------------------------------- components
    def log_timing_density(self, phase, dow=None) -> np.ndarray:
        """Log density over time-of-day, in radians^-1, conditional on weekday."""
        self._check()
        phase = np.atleast_1d(np.asarray(phase, dtype=float))
        if self.n_events == 0 or self.w_total <= 0:
            return np.full(phase.shape, LOG_UNIFORM)

        pooled = _weighted_kde(phase, self._phase, self._w, self._k)
        if not self.weekday or dow is None:
            dens = pooled
        else:
            dow = np.broadcast_to(np.asarray(dow, dtype=np.int64), phase.shape)
            dens = np.array(pooled, copy=True)
            for d in np.unique(dow):
                m = dow == d
                sel = self._dow == d
                n_d = float(self._w[sel].sum())
                if n_d <= 0:
                    continue
                own = _weighted_kde(phase[m], self._phase[sel], self._w[sel], self._k)
                a = n_d / (n_d + self.timing_shrinkage_events)
                dens[m] = a * own + (1.0 - a) * pooled[m]

        eps = self.uniform_floor
        return np.log((1.0 - eps) * dens + eps / TWO_PI)

    def timing_density(self, phase, dow=None) -> np.ndarray:
        return np.exp(self.log_timing_density(phase, dow))

    def intensity(self, t) -> np.ndarray:
        """Expected events per second at absolute times ``t``."""
        self._check()
        t = np.atleast_1d(np.asarray(t, dtype=float))
        ph = daily_phase(t, self.tz)
        dw = local_dow(t, self.tz)
        dens = self.timing_density(ph, dw)
        return self.rate_by_dow[dw] * dens * TWO_PI / SECONDS_PER_DAY

    # ------------------------------------------------------------ predictions
    def expected_events(self, t_from: float, t_to: float, max_steps: int = 200_000) -> float:
        """Integral of the intensity over ``[t_from, t_to)``."""
        if t_to <= t_from:
            return 0.0
        step = _integration_step(self._k, self.bandwidth_min)
        n = int(min(max(math.ceil((t_to - t_from) / step), 2), max_steps)) + 1
        grid = np.linspace(t_from, t_to, n)
        return float(_trapz(self.intensity(grid), grid))

    def p_engage(self, t_from: float, t_to: float) -> float:
        """Probability of at least one engagement in a window.

        This is the number the model exists to produce: it turns "how regular is
        this person" into "will they show up between 7 and 9 tomorrow", which is
        both testable and directly actionable for reminder timing.
        """
        return float(-np.expm1(-self.expected_events(t_from, t_to)))

    def p_engage_next(self, hours: float = 24.0, t_from: float | None = None) -> float:
        t0 = self.now if t_from is None else t_from
        return self.p_engage(t0, t0 + hours * 3600.0)

    def best_window(
        self, horizon_hours: float = 24.0, window_min: float = 60.0, t_from: float | None = None
    ) -> tuple[float, float, float]:
        """Highest-probability window of a given width within the horizon.

        Returns ``(start, end, probability)``. Use it to place a nudge where the
        person is already most likely to act, rather than where it is convenient
        to send.
        """
        t0 = self.now if t_from is None else t_from
        t1 = t0 + horizon_hours * 3600.0
        step = _integration_step(self._k, self.bandwidth_min)
        grid = np.arange(t0, t1 + step, step)
        lam = self.intensity(grid)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(grid))])
        width = window_min * 60.0
        starts = grid[grid <= t1 - width]
        if starts.size == 0:
            return t0, t0 + width, self.p_engage(t0, t0 + width)
        ends = starts + width
        integral = np.interp(ends, grid, cum) - np.interp(starts, grid, cum)
        i = int(np.argmax(integral))
        return float(starts[i]), float(ends[i]), float(-np.expm1(-integral[i]))

    def daily_profile(self, dow: int | None = None, n: int = 288) -> tuple[np.ndarray, np.ndarray]:
        """Time-of-day density on a grid, for plotting. Returns (hours, density)."""
        ph = np.linspace(0.0, TWO_PI, n, endpoint=False)
        return ph / TWO_PI * 24.0, self.timing_density(ph, dow)

    # ------------------------------------------------------------------ utils
    def _check(self) -> None:
        if not self.fitted:
            raise RuntimeError("call fit() before using the model")


def _weighted_kde(
    at: np.ndarray, points: np.ndarray, weights: np.ndarray, kernel: CircularKernel
) -> np.ndarray:
    """Circular KDE evaluated at ``at``, chunked to bound memory."""
    at = np.atleast_1d(at)
    out = np.zeros(at.shape, dtype=float)
    total = weights.sum()
    if total <= 0 or points.size == 0 or at.size == 0:
        return np.full(at.shape, 1.0 / TWO_PI)
    chunk = max(1, int(4_000_000 // max(points.size, 1)))
    for i in range(0, at.size, chunk):
        d = at[i : i + chunk, None] - points[None, :]
        out[i : i + chunk] = np.exp(kernel.log_pdf(d)) @ weights
    return out / total


def _integration_step(kernel: CircularKernel, bandwidth_min) -> float:
    """Grid step (seconds) fine enough to resolve the kernel's narrowest feature."""
    bw = min(bandwidth_min) if isinstance(bandwidth_min, (tuple, list)) else bandwidth_min
    return float(np.clip(bw * 60.0 / 5.0, 60.0, 900.0))


def _weighted_exposure(
    t_start: float, now: float, tz, half_life_days: float
) -> tuple[float, np.ndarray]:
    """Recency-weighted observation time, in days, overall and per weekday.

    Without this the model cannot tell "no Tuesday sessions" from "never
    observed on a Tuesday", and every rate would be biased by how the
    observation window happens to align with the week.
    """
    by_dow = np.zeros(7)
    if now <= t_start:
        return 0.0, by_dow

    h = half_life_days * SECONDS_PER_DAY
    scale = h / LOG2

    def cum(s):
        # integral of 2^-((now - s)/h) ds from -inf up to s, anchored at `now`
        return scale * np.exp2(-(now - np.asarray(s, dtype=float)) / h)

    # Walk local calendar days so that DST-shortened days get the right exposure.
    day0 = int(local_day_index(np.array([t_start]), tz)[0])
    day1 = int(local_day_index(np.array([now]), tz)[0])
    n_days = day1 - day0 + 1
    if n_days > 200_000:  # pathological window; fall back to a pooled estimate
        total = (cum(now) - cum(t_start)) / SECONDS_PER_DAY
        return total, np.full(7, total / 7.0)

    edges = local_midnight_epoch(np.arange(day0, day1 + 2), tz)
    dows = ((np.arange(day0, day1 + 1) + 3) % 7).astype(np.int64)

    lo = np.clip(edges[:-1], t_start, now)
    hi = np.clip(edges[1:], t_start, now)
    contrib = np.where(hi > lo, (cum(hi) - cum(lo)) / SECONDS_PER_DAY, 0.0)
    for d in range(7):
        by_dow[d] = float(contrib[dows == d].sum())
    return float(contrib.sum()), by_dow
