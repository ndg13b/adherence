"""Does a consistency score measure a stable property of a person?

The naive check on real data -- "do people's scores differ?" -- cannot answer
this, and will say yes even when the answer is no. With two weeks of history a
score is estimated from a dozen events, so it carries substantial sampling
noise, and noise produces spread between people all by itself. A cohort of
identical robots would show a spread of scores.

The question that can be answered is the psychometric one: **split-half
reliability**. Score each person twice from disjoint halves of their own
events, and correlate the two. If the halves agree, the score is tracking
something about the person; if they do not, the spread was noise and no amount
of downstream modelling will rescue it.

That correlation then does a second job. Observed variance between people is
inflated by measurement error, and the classical correction removes it:

    reliable SD = observed SD * sqrt(reliability)

which is the quantity that actually decides whether the concept is viable. A
large observed spread with reliability near zero means there is nothing there.

Every index is put through the same procedure, so the comparison against the
Social Rhythm Metric and friends is like for like -- and the correlation matrix
at the end asks the question a new measure should always have to answer: does
it differ from the ones that already exist?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .baselines import (
    interdaily_stability,
    sleep_regularity_index,
    social_rhythm_hit_rate,
    timing_entropy_bits,
)
from .events import EventLog
from .model import RoutineModel
from .scores import anchor_precision, resultant_length, timing_consistency

#: Index name -> callable(log, model) -> float. Higher or lower may be "better";
#: reliability and spread do not care about direction.
SCORERS = {
    "timing_consistency": lambda log, m: timing_consistency(log, m),
    "srm_hit_rate": lambda log, m: social_rhythm_hit_rate(log),
    "interdaily_stability": lambda log, m: interdaily_stability(log),
    "sleep_regularity_index": lambda log, m: sleep_regularity_index(log),
    "timing_entropy_bits": lambda log, m: timing_entropy_bits(log),
    "resultant_length": lambda log, m: resultant_length(log.daily_phase),
}

#: Off by default: the permutation null makes it far slower than the rest.
OPTIONAL_SCORERS = {
    "anchor_precision": lambda log, m: anchor_precision(log, m, null_reps=8),
}


def split_alternate(log: EventLog) -> tuple[EventLog, EventLog]:
    """Split a person's events into two interleaved halves.

    Alternating rather than first-half/second-half on purpose: both halves then
    cover the same calendar window, so the comparison is between two estimates
    of one routine rather than between an earlier routine and a later one. A
    chronological split would confound unreliability with genuine change, and
    those need to stay separable.
    """
    def take(idx):
        return EventLog(
            t=log.t[idx], tz=log.tz, weight=log.weight[idx],
            t_start=log.t_start, t_end=log.t_end, meta=dict(log.meta),
        )

    i = np.arange(len(log))
    return take(i % 2 == 0), take(i % 2 == 1)


def score_cohort(
    logs: dict[str, EventLog],
    model: RoutineModel | None = None,
    scorers: dict | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Apply every scorer to every person. Returns ``{index_name: array}``."""
    model = model or RoutineModel()
    scorers = scorers or SCORERS
    out = {k: np.full(len(logs), np.nan) for k in scorers}
    for i, log in enumerate(logs.values()):
        if verbose and i and i % 500 == 0:
            print(f"    ...scored {i:,}/{len(logs):,}", flush=True)
        for name, fn in scorers.items():
            try:
                out[name][i] = fn(log, model)
            except Exception:
                pass
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    x, y = a[ok], b[ok]
    if x.std() == 0 or y.std() == 0:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(ok.sum())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    r = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    return _pearson(r(a[ok]), r(b[ok]))[0]


@dataclass
class IndexReliability:
    name: str
    n: int
    mean: float
    sd: float
    iqr: tuple[float, float]
    half_correlation: float
    reliability: float  #: Spearman-Brown corrected to full history length
    rank_correlation: float
    reliable_sd: float  #: between-person SD with measurement error removed

    def row(self) -> str:
        return (
            f"{self.name:24s} {self.mean:8.3f} {self.sd:7.3f} "
            f"{self.half_correlation:8.3f} {self.reliability:9.3f} "
            f"{self.reliable_sd:9.3f}  {self.n:6d}"
        )


@dataclass
class CohortReport:
    indices: list[IndexReliability]
    correlations: dict = field(default_factory=dict)
    frequency_confound: dict = field(default_factory=dict)
    n_people: int = 0
    median_events: float = 0.0
    span_days: float = 0.0
    primary: str = "timing_consistency"

    def get(self, name: str) -> IndexReliability | None:
        return next((i for i in self.indices if i.name == name), None)

    def verdict(self, min_reliability: float = 0.3, min_reliable_sd: float = 0.05) -> str:
        """Plain-language read on whether the concept survives this dataset.

        The thresholds are conventions, not laws: 0.3 is a low bar for
        reliability (0.7+ is what you would want for an individual-level
        decision), and the SD floor asks whether the reliable spread is large
        enough to rank people at all on a 0-1 scale.
        """
        p = self.get(self.primary)
        if p is None or not np.isfinite(p.reliability):
            return "INCONCLUSIVE - the score could not be computed on enough people."
        if p.reliability < min_reliability:
            return (
                f"NOT ESTABLISHED - reliability {p.reliability:.2f} is too low to treat "
                f"the score as a person-level trait at this history length.\n"
                "  The spread between people here is mostly measurement noise. This is a\n"
                "  statement about the data, not the metric: with ~2 weeks of events there\n"
                "  may simply be too little per person. Retry on a longer dataset before\n"
                "  concluding anything about the concept."
            )
        if p.reliable_sd < min_reliable_sd:
            return (
                f"CONCEPT LOOKS WEAK - the score is reliable ({p.reliability:.2f}) but the "
                f"reliable spread is small (SD {p.reliable_sd:.3f}).\n"
                "  People genuinely differ, but not by enough to rank them usefully.\n"
                "  A predictor with this little variance cannot carry a survival model."
            )
        return (
            f"PROCEED - reliability {p.reliability:.2f}, reliable between-person SD "
            f"{p.reliable_sd:.3f}.\n"
            "  Timing consistency behaves like a stable individual difference here, with\n"
            "  enough spread to rank people. The retention question is next, and needs a\n"
            "  dataset long enough to observe dropout."
        )

    def __str__(self) -> str:
        lines = [
            f"{self.n_people:,} people, median {self.median_events:.0f} sessions each, "
            f"{self.span_days:.0f}-day window",
            "",
            f"{'index':24s} {'mean':>8s} {'SD':>7s} {'half r':>8s} {'reliab.':>9s} "
            f"{'true SD':>9s}  {'n':>6s} {'rank r':>8s} {'vs freq':>8s}",
        ]
        for i in self.indices:
            conf = self.frequency_confound.get(i.name, float("nan"))
            lines.append(f"  {i.row()} {i.rank_correlation:8.3f} {conf:+8.2f}")
        lines += [
            "",
            "  half r    = correlation between scores from two interleaved halves",
            "  reliab.   = Spearman-Brown corrected to full history length",
            "  true SD   = between-person SD with measurement error removed",
            "  rank r    = split-half agreement on ranks (robust to skew)",
            "  vs freq   = correlation with log(sessions). Near +/-1 means the index is",
            "              largely measuring how OFTEN someone engages, not how",
            "              REGULARLY -- a different construct wearing the same name.",
        ]
        if self.correlations:
            lines += ["", f"Correlation with {self.primary} (does it add anything?):"]
            for k, v in sorted(self.correlations.items(), key=lambda kv: -abs(kv[1])):
                lines.append(f"    {k:24s} {v:+.3f}")
        return "\n".join(lines)


def reliability_report(
    logs: dict[str, EventLog],
    model: RoutineModel | None = None,
    scorers: dict | None = None,
    primary: str = "timing_consistency",
    verbose: bool = True,
) -> CohortReport:
    """Score every person whole and in halves, then report reliability and spread."""
    model = model or RoutineModel()
    scorers = scorers or SCORERS

    if verbose:
        print("  scoring full histories...", flush=True)
    full = score_cohort(logs, model, scorers, verbose=verbose)

    halves = {k: (np.full(len(logs), np.nan), np.full(len(logs), np.nan)) for k in scorers}
    if verbose:
        print("  scoring split halves...", flush=True)
    for i, log in enumerate(logs.values()):
        if verbose and i and i % 500 == 0:
            print(f"    ...split {i:,}/{len(logs):,}", flush=True)
        a, b = split_alternate(log)
        for name, fn in scorers.items():
            for j, part in enumerate((a, b)):
                try:
                    halves[name][j][i] = fn(part, model)
                except Exception:
                    pass

    indices = []
    for name in scorers:
        v = full[name]
        ok = np.isfinite(v)
        r_half, n = _pearson(*halves[name])
        # Spearman-Brown: the halves are half-length, so correct up to full length.
        r_full = (2 * r_half / (1 + r_half)) if np.isfinite(r_half) and r_half > -1 else float("nan")
        r_full = float(np.clip(r_full, 0.0, 1.0)) if np.isfinite(r_full) else float("nan")
        sd = float(np.std(v[ok], ddof=1)) if ok.sum() > 1 else float("nan")
        indices.append(
            IndexReliability(
                name=name,
                n=int(ok.sum()),
                mean=float(np.mean(v[ok])) if ok.any() else float("nan"),
                sd=sd,
                iqr=(
                    float(np.percentile(v[ok], 25)) if ok.any() else float("nan"),
                    float(np.percentile(v[ok], 75)) if ok.any() else float("nan"),
                ),
                half_correlation=r_half,
                reliability=r_full,
                rank_correlation=_spearman(*halves[name]),
                reliable_sd=sd * math.sqrt(r_full) if np.isfinite(r_full) and np.isfinite(sd)
                else float("nan"),
            )
        )

    corr = {}
    if primary in full:
        for name in scorers:
            if name != primary:
                corr[name] = _pearson(full[primary], full[name])[0]

    # An index that tracks engagement count is a frequency measure, not a
    # regularity one, however it is labelled. Worth knowing before building on it.
    n_events = np.log(np.array([max(len(v), 1) for v in logs.values()], dtype=float))
    confound = {name: _spearman(full[name], n_events) for name in scorers}

    ev = np.array([len(v) for v in logs.values()]) if logs else np.zeros(1)
    span = max((max(v.t_end for v in logs.values()) - min(v.t_start for v in logs.values()))
               / 86400.0, 0.0) if logs else 0.0
    return CohortReport(
        indices=indices, correlations=corr, frequency_confound=confound,
        n_people=len(logs), median_events=float(np.median(ev)), span_days=span,
        primary=primary,
    )
