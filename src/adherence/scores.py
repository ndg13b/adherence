"""Consistency scores.

Three questions get separate numbers here, because they behave differently and
collapsing them hides the thing you care about:

``timing_consistency``
    Do sessions land at the same time of day? Scaled so that 0 is "times drawn
    uniformly across the 24 hours" and 1 is "identical to the minute". This is
    the direct generalisation of the classic Social Rhythm Metric hit-count:
    instead of a hard +/-45 minute box, every event gets a kernel with tails, so
    a session 50 minutes late is worth slightly less than one 40 minutes late
    rather than nothing at all.

``weekday_regularity``
    Are the *days* predictable -- Mon/Wed/Fri rather than three days scattered
    through the week? Measured in bits per day against the person's own overall
    rate.

``timing_bits``
    The honest version of the first score: at each event, refit the model on
    strictly earlier events and ask how much better than chance it predicted
    this one. Reported in bits per event, prequentially, so it cannot be inflated
    by fitting and evaluating on the same data.

The descriptive score is the readable one; the prequential one is the
defensible one. They usually agree, and when they diverge it is informative --
a person whose routine is drifting scores well descriptively (their sessions do
cluster) and poorly prequentially (yesterday's cluster did not locate today's).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .events import SECONDS_PER_DAY, TWO_PI, EventLog, local_dow, phase_to_clock
from .kernels import LOG_UNIFORM, wrap_to_pi
from .model import RoutineModel, _weighted_kde

LOG2 = math.log(2.0)
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------- circular stats
def circular_mean(phase: np.ndarray, weights: np.ndarray | None = None) -> float:
    w = np.ones_like(phase) if weights is None else weights
    if w.sum() <= 0:
        return float("nan")
    return float(np.arctan2((w * np.sin(phase)).sum(), (w * np.cos(phase)).sum()) % TWO_PI)


def resultant_length(phase: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mean resultant length R in [0, 1] -- the textbook circular concentration.

    Reported for comparison, not used as the headline score: R collapses for
    genuinely bimodal routines. Someone who reliably trains at 07:00 and 19:00
    has R near 0 despite being perfectly regular, which is exactly the failure
    the kernel score is designed to avoid.
    """
    w = np.ones_like(phase) if weights is None else weights
    if w.sum() <= 0:
        return float("nan")
    c, s = (w * np.cos(phase)).sum(), (w * np.sin(phase)).sum()
    return float(math.hypot(c, s) / w.sum())


def circular_sd_minutes(deviations_rad: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Circular standard deviation of signed deviations, expressed in minutes."""
    w = np.ones_like(deviations_rad) if weights is None else weights
    if w.sum() <= 0 or deviations_rad.size == 0:
        return float("nan")
    r = resultant_length(deviations_rad, w)
    r = min(max(r, 1e-12), 1.0 - 1e-12)
    sd_rad = math.sqrt(-2.0 * math.log(r))
    return float(sd_rad / TWO_PI * 1440.0)


# ------------------------------------------------------------------- containers
@dataclass
class Anchor:
    """A recurring slot in the day."""

    clock: str
    phase: float
    share: float  #: fraction of (recency-weighted) events belonging to this slot
    jitter_min: float  #: SD of arrival times about the trend -- day-to-day noise
    drift_min_per_week: float  #: signed trend in arrival time
    n_events: float
    spread_min: float = float("nan")  #: SD about the mean -- jitter and drift together

    def __str__(self) -> str:
        return (
            f"{self.clock} (+/-{self.jitter_min:.0f} min, {100 * self.share:.0f}% of sessions, "
            f"drift {self.drift_min_per_week:+.1f} min/week)"
        )


@dataclass
class ConsistencyReport:
    n_events: int
    span_days: float
    n_effective: float

    timing_consistency: float
    timing_consistency_by_weekday: float
    anchor_precision: float
    weekday_regularity: float
    timing_bits: float
    timing_bits_normalised: float

    rate_per_day: float
    active_day_fraction: float
    jitter_min: float
    drift_min_per_week: float
    anchors: list[Anchor] = field(default_factory=list)
    weekday_rates: dict = field(default_factory=dict)

    resultant_length: float = float("nan")
    warmup: bool = False
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"{self.n_events} events over {self.span_days:.0f} days "
            f"({self.rate_per_day:.2f}/day, active on {100 * self.active_day_fraction:.0f}% of days)",
            f"  timing consistency   {self.timing_consistency:.3f}   "
            f"(0 = times spread uniformly, 1 = same minute every time)",
            f"  anchor precision     {self.anchor_precision:.3f}   "
            f"(tightness around each slot, ignoring how many slots)",
            f"  weekday regularity   {self.weekday_regularity:.3f}",
            f"  out-of-sample timing {self.timing_bits:+.2f} bits/event "
            f"(normalised {self.timing_bits_normalised:.3f})",
            f"  typical jitter       +/-{self.jitter_min:.0f} min, "
            f"drift {self.drift_min_per_week:+.1f} min/week",
        ]
        if self.anchors:
            lines.append("  anchors:")
            lines += [f"    - {a}" for a in self.anchors]
        if self.warmup:
            lines.append("  [warm-up: too little history for the out-of-sample score]")
        return "\n".join(lines)


# ------------------------------------------------------------- descriptive score
def _loo_kernel_score(phase, w, kern, groups) -> float:
    """Weighted leave-one-out kernel self-similarity, rescaled to ``[0, 1]``.

    Each event is scored by the density the *other* events place at its time, so
    a lone event cannot vouch for itself. The raw density is then rescaled
    against two fixed reference points -- the uniform density (no routine) and
    the kernel peak (perfect coincidence) -- which is what makes the number
    comparable between people who engage at very different frequencies. An
    unnormalised KDE height is not: it grows with sample size and shrinks with
    bandwidth, and would rank a busy irregular user above a sparse punctual one.
    """
    peak = math.exp(kern.peak_log_pdf)
    uniform = 1.0 / TWO_PI
    num = den = 0.0
    for g in groups:
        if g.sum() < 2:
            continue
        p, ww = phase[g], w[g]
        total = ww.sum()
        if total <= 0:
            continue
        # Full KDE at each event (self included), then subtract the self term.
        dens_all = _weighted_kde(p, p, ww, kern) * total
        loo = (dens_all - ww * peak) / np.maximum(total - ww, 1e-12)
        num += float((ww * loo).sum())
        den += float(total)
    if den <= 0:
        return float("nan")
    return float(np.clip((num / den - uniform) / (peak - uniform), 0.0, 1.0))


def timing_consistency(
    log: EventLog,
    model: RoutineModel | None = None,
    now: float | None = None,
    by_weekday: bool = False,
) -> float:
    """How predictable the time of day is, on ``[0, 1]``.

    Note what this deliberately does *not* reward: a person with two tight daily
    slots scores around 0.5, not 1.0, because knowing their history still leaves
    you a coin flip about which slot comes next. That is the correct answer to
    "how predictable is their next session", and the wrong answer to "how tight
    are their rituals" -- for which see :func:`anchor_precision`. Reporting both
    is what separates two crisp routines from one sloppy one; either number
    alone confuses them.
    """
    model = model or RoutineModel()
    t_ref = now if now is not None else max(log.t_end, log.t[-1] if len(log) else log.t_end)
    if len(log) < 2:
        return float("nan")
    phase = log.daily_phase
    w = log.weight * np.exp2(-(t_ref - log.t) / SECONDS_PER_DAY / model.half_life_days)
    groups = (
        [log.dow == d for d in range(7)] if by_weekday else [np.ones(len(log), dtype=bool)]
    )
    return _loo_kernel_score(phase, w, model._k, groups)


def _within_anchor_score(phase, w, kern, bw, min_share, grid, n_max=None) -> tuple[float, int]:
    """Find slots in these phases, then score tightness within each slot."""
    centres = _anchor_centres(phase, w, kern, bw, min_share, grid, n_max)
    if centres.size == 0:
        return float("nan"), 0
    owner = np.argmin(np.abs(wrap_to_pi(phase[:, None] - centres[None, :])), axis=1)
    score = _loo_kernel_score(phase, w, kern, [owner == j for j in range(centres.size)])
    return score, int(centres.size)


def anchor_precision(
    log: EventLog,
    model: RoutineModel | None = None,
    now: float | None = None,
    null_reps: int = 24,
    seed: int = 0,
) -> float:
    """Tightness of sessions around their *own* slot, ignoring how many slots.

    The complement to :func:`timing_consistency`: a two-slot routine with 12
    minutes of jitter scores near 1 here and near 0.5 there, while a one-slot
    routine with 75 minutes of jitter scores low on both. Splitting the two
    stops a multi-session-per-day user from being mislabelled as irregular
    merely for engaging more than once.

    Corrected against a permutation null, which is not optional. Cutting the
    circle into basins around density peaks makes *any* set of times look
    concentrated within its own basin -- including times drawn uniformly at
    random, which scored ~0.48 raw, above a genuine but sloppy single-slot
    routine. The null is the same pipeline run on uniform phases with the same
    count and weights, so the reported value is the excess over what the
    partitioning alone would manufacture. ``null_reps=0`` returns the raw score.
    """
    model = model or RoutineModel()
    if len(log) < 2:
        return float("nan")
    t_ref = now if now is not None else max(log.t_end, log.t[-1])
    phase = log.daily_phase
    w = log.weight * np.exp2(-(t_ref - log.t) / SECONDS_PER_DAY / model.half_life_days)
    args = (model._k, model._k.sigma, 0.08, 1440)

    observed, k = _within_anchor_score(phase, w, *args)
    if not null_reps or not np.isfinite(observed):
        return observed

    rng = np.random.default_rng(seed)  # seeded: the same log must give the same score
    nulls = [
        _within_anchor_score(rng.uniform(0.0, TWO_PI, phase.size), w, *args, n_max=k)[0]
        for _ in range(null_reps)
    ]
    e0 = float(np.nanmean(nulls))
    if not np.isfinite(e0) or e0 >= 1.0:
        return observed
    return float(np.clip((observed - e0) / (1.0 - e0), 0.0, 1.0))


# ------------------------------------------------------------ prequential score
def prequential_timing_bits(
    log: EventLog,
    model: RoutineModel | None = None,
    min_history: int = 5,
    min_history_days: float = 7.0,
    stride: int = 1,
    score_from: float | None = None,
    score_to: float | None = None,
) -> tuple[float, int]:
    """Mean out-of-sample bits per event, and the number of events scored.

    ``score_from`` / ``score_to`` restrict which events are *scored* without
    restricting the history each forecast may use -- the way to ask "how well
    does this person's past predict their present" over a recent window.

    For each event in turn the model is refit on strictly earlier events and
    asked for the log density at the event's actual time. The score is the
    improvement over a uniform "any time of day" forecast:

        bits_i = log2 f_{<i}(phase_i) - log2 (1 / 2*pi)

    Positive means the past genuinely located the present. Zero means their
    history told you nothing. Negative -- which is possible and worth
    surfacing -- means the routine has moved and yesterday's pattern is now
    actively misleading.
    """
    model = model or RoutineModel()
    if len(log) <= min_history:
        return float("nan"), 0

    phase = log.daily_phase
    dow = log.dow
    bits = []
    for i in range(min_history, len(log), stride):
        t_i = log.t[i]
        if (t_i - log.t_start) / SECONDS_PER_DAY < min_history_days:
            continue
        if score_from is not None and t_i < score_from:
            continue
        if score_to is not None and t_i >= score_to:
            continue
        m = RoutineModel(**_model_params(model)).fit(log, now=t_i)
        if m.n_events < min_history:
            continue
        lp = float(m.log_timing_density(np.array([phase[i]]), np.array([dow[i]]))[0])
        bits.append((lp - LOG_UNIFORM) / LOG2)
    if not bits:
        return float("nan"), 0
    return float(np.mean(bits)), len(bits)


def bits_ceiling(model: RoutineModel) -> float:
    """Best achievable bits/event given bandwidth and uniform floor."""
    peak = math.exp(model._k.peak_log_pdf)
    eps = model.uniform_floor
    best = (1.0 - eps) * peak + eps / TWO_PI
    return (math.log(best) - LOG_UNIFORM) / LOG2


# ------------------------------------------------------------ weekday structure
def weekday_regularity(log: EventLog, model: RoutineModel | None = None,
                       now: float | None = None) -> float:
    """How much knowing the day of week improves the forecast, as a skill score.

    Compares a weekday-specific Bernoulli forecast of "any engagement today"
    against the person's own pooled rate, normalised by the pooled entropy.
    1.0 means the weekday alone tells you exactly which days are active; 0 means
    the days carry no information beyond the overall rate.
    """
    model = model or RoutineModel()
    if len(log) < 2:
        return float("nan")
    m = RoutineModel(**{**_model_params(model), "weekday": True}).fit(log, now=now)
    if m._exposure_days <= 0:
        return float("nan")

    p_dow = -np.expm1(-m.rate_by_dow)
    p_pool = -np.expm1(-m.rate)
    if not 0.0 < p_pool < 1.0:
        return 0.0

    # Observed activity per weekday, recency-weighted the same way as exposure.
    day_idx = log.local_day_index
    uniq_days, first = np.unique(day_idx, return_index=True)
    active_dow = local_dow(log.t[first], log.tz)
    weight_by_dow_active = np.zeros(7)
    ref = m.now
    for d, t0 in zip(active_dow, log.t[first]):
        weight_by_dow_active[d] += math.exp2(-(ref - t0) / SECONDS_PER_DAY / m.half_life_days)

    exp_dow = m._exposure_by_dow
    eps = 1e-9
    ll_model = 0.0
    ll_pool = 0.0
    for d in range(7):
        n_active = min(weight_by_dow_active[d], exp_dow[d])
        n_idle = max(exp_dow[d] - n_active, 0.0)
        pm = min(max(p_dow[d], eps), 1 - eps)
        pp = min(max(p_pool, eps), 1 - eps)
        ll_model += n_active * math.log2(pm) + n_idle * math.log2(1 - pm)
        ll_pool += n_active * math.log2(pp) + n_idle * math.log2(1 - pp)

    total = exp_dow.sum()
    if total <= 0:
        return float("nan")
    gain = (ll_model - ll_pool) / total
    entropy = -(p_pool * math.log2(p_pool) + (1 - p_pool) * math.log2(1 - p_pool))
    return float(np.clip(gain / entropy, 0.0, 1.0)) if entropy > 0 else 0.0


# -------------------------------------------------------------------- anchors
def find_anchors(
    log: EventLog,
    model: RoutineModel | None = None,
    now: float | None = None,
    min_share: float = 0.08,
    grid: int = 1440,
) -> list[Anchor]:
    """Locate the recurring slots in the day and describe each one.

    Handles the multi-session case the plain "average time" cannot: a person who
    trains at 07:00 and again at 21:00 is not a person who trains at 14:00.
    Peaks in the kernel density become anchors, each event is assigned to its
    nearest anchor, and jitter and drift are then computed *within* a slot.
    """
    model = model or RoutineModel()
    if len(log) < 3:
        return []
    t_ref = now if now is not None else max(log.t_end, log.t[-1])
    phase = log.daily_phase
    w = log.weight * np.exp2(-(t_ref - log.t) / SECONDS_PER_DAY / model.half_life_days)
    if w.sum() <= 0:
        return []

    centres = _anchor_centres(phase, w, model._k, model._k.sigma, min_share, grid)
    if centres.size == 0:
        return []

    # Assign every event to its nearest centre.
    d = np.abs(wrap_to_pi(phase[:, None] - centres[None, :]))
    owner = np.argmin(d, axis=1)

    anchors: list[Anchor] = []
    for j, c in enumerate(centres):
        m = owner == j
        if not m.any():
            continue
        ww, share = w[m], float(w[m].sum() / w.sum())
        if share < min_share and len(centres) > 1:
            continue
        mean = circular_mean(phase[m], ww)
        dev = wrap_to_pi(phase[m] - mean)
        # Jitter is measured around the *trend*, not around the overall mean. A
        # routine sliding 25 min/week is tight day to day; charging that slide to
        # jitter would report it as sloppy and hide the one thing worth acting on.
        drift, resid = _drift(log.t[m], dev, ww)
        anchors.append(
            Anchor(
                clock=phase_to_clock(mean),
                phase=float(mean),
                share=share,
                jitter_min=circular_sd_minutes(resid, ww),
                spread_min=circular_sd_minutes(dev, ww),
                drift_min_per_week=drift,
                n_events=float(m.sum()),
            )
        )
    return sorted(anchors, key=lambda a: -a.share)


def _anchor_centres(phase, w, kern, bw, min_share, grid, n_max: int | None = None) -> np.ndarray:
    """Peaks of the circular density, merged if closer than one bandwidth.

    ``n_max`` caps how many are kept, strongest first. The permutation null needs
    it: cutting the circle into *k* basins is what inflates within-basin
    concentration, so a null free to pick its own *k* would not be comparable.
    """
    if phase.size == 0 or w.sum() <= 0:
        return np.zeros(0)
    g = np.linspace(0.0, TWO_PI, grid, endpoint=False)
    dens = _weighted_kde(g, phase, w, kern)
    peaks = np.nonzero((dens >= np.roll(dens, 1)) & (dens > np.roll(dens, -1)))[0]
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(dens))])
    keep: list[int] = []
    for p in peaks[np.argsort(-dens[peaks])]:
        if n_max is not None and len(keep) >= n_max:
            break
        if all(abs(wrap_to_pi(g[p] - g[q])) > bw for q in keep):
            keep.append(int(p))
    return np.sort(g[np.array(keep)])


def _drift(t: np.ndarray, dev_rad: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted least-squares trend in arrival time. Returns (min/week, residuals).

    A drifting routine and a noisy routine look identical to a variance-based
    measure, but they mean opposite things: drift is a schedule migrating (often
    benign, and predictable if you model it), noise is a routine dissolving.
    """
    if t.size < 3 or w.sum() <= 0:
        return 0.0, dev_rad
    x = (t - t.mean()) / SECONDS_PER_DAY
    y = dev_rad / TWO_PI * 1440.0  # minutes
    sw = w.sum()
    xm = (w * x).sum() / sw
    ym = (w * y).sum() / sw
    var = (w * (x - xm) ** 2).sum()
    if var <= 1e-12:
        return 0.0, dev_rad
    slope = float((w * (x - xm) * (y - ym)).sum() / var)
    resid_min = y - (ym + slope * (x - xm))
    return slope * 7.0, resid_min / 1440.0 * TWO_PI


# --------------------------------------------------------------------- summary
def consistency_report(
    log: EventLog,
    model: RoutineModel | None = None,
    now: float | None = None,
    prequential: bool = True,
    stride: int = 1,
) -> ConsistencyReport:
    """Everything above, in one pass."""
    model = model or RoutineModel()
    t_ref = now if now is not None else max(log.t_end, log.t[-1] if len(log) else log.t_end)
    m = RoutineModel(**_model_params(model)).fit(log, now=t_ref)

    n_days = max(len(np.unique(log.local_day_index)), 0)
    span = max((t_ref - log.t_start) / SECONDS_PER_DAY, 0.0)

    if prequential and len(log) > 5:
        bits, n_scored = prequential_timing_bits(log, model, stride=stride)
    else:
        bits, n_scored = float("nan"), 0
    ceiling = bits_ceiling(m)
    bits_norm = float(np.clip(bits / ceiling, 0.0, 1.0)) if np.isfinite(bits) else float("nan")

    anchors = find_anchors(log, model, now=t_ref)
    if anchors:
        shares = np.array([a.share for a in anchors])
        jit = float(np.nansum(shares * np.array([a.jitter_min for a in anchors])) / shares.sum())
        dr = float(np.nansum(shares * np.array([a.drift_min_per_week for a in anchors])) / shares.sum())
    else:
        jit, dr = float("nan"), float("nan")

    return ConsistencyReport(
        n_events=len(log),
        span_days=span,
        n_effective=m.n_effective,
        timing_consistency=timing_consistency(log, model, now=t_ref),
        timing_consistency_by_weekday=timing_consistency(log, model, now=t_ref, by_weekday=True),
        anchor_precision=anchor_precision(log, model, now=t_ref),
        weekday_regularity=weekday_regularity(log, model, now=t_ref),
        timing_bits=bits,
        timing_bits_normalised=bits_norm,
        rate_per_day=m.rate,
        active_day_fraction=(n_days / span) if span > 0 else float("nan"),
        jitter_min=jit,
        drift_min_per_week=dr,
        anchors=anchors,
        weekday_rates={DAY_NAMES[d]: float(m.rate_by_dow[d]) for d in range(7)},
        resultant_length=resultant_length(log.daily_phase, log.weight) if len(log) else float("nan"),
        warmup=n_scored < 5,
        params={
            "kernel": model.kernel,
            "bandwidth_min": model.bandwidth_min,
            "half_life_days": model.half_life_days,
            "uniform_floor": model.uniform_floor,
            "bits_ceiling": ceiling,
            "n_scored": n_scored,
        },
    )


@dataclass
class Outlook:
    """Forward-looking summary for one person at one moment."""

    p_next_24h: float
    p_next_72h: float
    expected_sessions_7d: float
    best_window_start: float
    best_window_end: float
    best_window_p: float
    days_since_last: float
    typical_gap_days: float

    def to_dict(self) -> dict:
        return asdict(self)


def outlook(model: RoutineModel, log: EventLog, window_min: float = 60.0) -> Outlook:
    """Turn the fitted model into the decisions it is meant to support.

    ``best_window`` answers "when should the reminder fire"; ``p_next_24h``
    against ``typical_gap_days`` answers "is this person slipping".
    """
    now = model.now
    s, e, p = model.best_window(horizon_hours=24.0, window_min=window_min)
    gaps = np.diff(log.t) / SECONDS_PER_DAY if len(log) > 1 else np.array([np.nan])
    return Outlook(
        p_next_24h=model.p_engage_next(24.0),
        p_next_72h=model.p_engage_next(72.0),
        expected_sessions_7d=model.expected_events(now, now + 7 * SECONDS_PER_DAY),
        best_window_start=s,
        best_window_end=e,
        best_window_p=p,
        days_since_last=float((now - log.t[-1]) / SECONDS_PER_DAY) if len(log) else float("nan"),
        typical_gap_days=float(np.nanmedian(gaps)) if gaps.size else float("nan"),
    )


def _model_params(model: RoutineModel) -> dict:
    return {
        "kernel": model.kernel,
        "bandwidth_min": model.bandwidth_min,
        "half_life_days": model.half_life_days,
        "weekday": model.weekday,
        "rate_shrinkage_days": model.rate_shrinkage_days,
        "timing_shrinkage_events": model.timing_shrinkage_events,
        "uniform_floor": model.uniform_floor,
    }
