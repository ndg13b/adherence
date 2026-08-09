"""One screening pass over a loaded cohort, shared by every dataset.

The sequence is deliberate. Look at the timestamps before scoring them, score
against the established indices rather than alone, and only then ask the
bandwidth question -- because a headline score means nothing until you know the
data carries time-of-day information at all and the kernel is at the right
resolution to see it.
"""

from __future__ import annotations

import numpy as np

from .datasets import CohortLoadResult, hour_of_day_histogram
from .model import RoutineModel
from .validate import (
    OPTIONAL_SCORERS,
    SCORERS,
    bandwidth_scan,
    format_bandwidth_scan,
    format_half_life_scan,
    half_life_scan,
    reliability_report,
)


def print_hour_profile(logs) -> float:
    """Sanity check on the timestamps before trusting anything downstream."""
    h = hour_of_day_histogram(logs)
    peak = h.max()
    print("\nPooled engagement by hour (all people):")
    for i, v in enumerate(h):
        bar = "#" * int(round(40 * v / peak)) if peak > 0 else ""
        print(f"  {i:02d}:00 {v:6.3f} {bar}")
    flat = float(h.min() / h.max()) if h.max() > 0 else float("nan")
    print(f"  trough/peak ratio {flat:.2f}", end="  ")
    if flat > 0.6:
        print("-- flat: users span many timezones, or times were shifted per user.\n"
              "     Harmless for the scores (they are shift-invariant) but clock\n"
              "     labels on anchors are not interpretable.")
    else:
        print("-- a diurnal shape is present, so the timestamps carry real\n"
              "     time-of-day information.")
    return flat


def screen(
    res: CohortLoadResult,
    bandwidth_min: float = 45.0,
    half_life_days: float = 28.0,
    with_anchor_precision: bool = False,
    do_bandwidth_scan: bool = False,
    do_half_life_scan: bool = False,
    short_span_days: float = 60.0,
) -> dict:
    """Print the full screening report. Returns the pieces for further use."""
    print("\n" + res.summary())
    if res.span_days < short_span_days:
        print(f"\n  NOTE: the data spans {res.span_days:.0f} days. Long enough to ask "
              "whether\n        people differ; too short to observe dropout.")
    if not res.logs:
        print("\nNo person met the inclusion thresholds. Try lowering "
              "--min-events / --min-days.")
        return {}

    print_hour_profile(res.logs)

    scorers = dict(SCORERS)
    if with_anchor_precision:
        scorers.update(OPTIONAL_SCORERS)

    model = RoutineModel(bandwidth_min=bandwidth_min, half_life_days=half_life_days)
    print(f"\nScoring (fixed bandwidth {bandwidth_min:.0f} min, so people stay "
          "comparable)...")
    report = reliability_report(res.logs, model, scorers)
    print("\n" + str(report))

    print("\n" + "=" * 72)
    print(report.verdict())
    print("=" * 72)

    scan = None
    if do_bandwidth_scan:
        print("\n\nBandwidth scan -- is this resolution right for these people?")
        print("(reusing the loaded data, so this is fast)\n")
        scan = bandwidth_scan(res.logs)
        print(format_bandwidth_scan(scan))

    hl_scan = None
    if do_half_life_scan:
        print("\n\nHalf-life scan -- how much of each history is the score using?\n")
        hl_scan = half_life_scan(res.logs, bandwidth_min=bandwidth_min)
        print(format_half_life_scan(hl_scan))

    return {"report": report, "model": model, "scan": scan, "half_life_scan": hl_scan}


def write_scores(path: str, logs, model, scorers=None) -> None:
    """Per-person scores as CSV, for your own downstream analysis."""
    import csv

    from .validate import score_cohort

    scorers = scorers or SCORERS
    scores = score_cohort(logs, model, scorers)
    names = list(scorers)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "n_sessions", "span_days"] + names)
        for i, (uid, log) in enumerate(logs.items()):
            span = (log.t[-1] - log.t[0]) / 86400.0 if len(log) else 0.0
            w.writerow([uid, len(log), f"{span:.2f}"]
                       + [f"{scores[n][i]:.6f}" for n in names])


def survival_table(logs, censor_at: float | None = None) -> np.ndarray:
    """Time-to-last-event and censoring flag per person, for a retention analysis.

    Disengagement is not recorded in these datasets -- it has to be inferred from
    silence. Someone whose last workout falls well before the end of the
    observation window has stopped; someone still active at the end is censored,
    not a non-event. Getting that distinction wrong is the classic way to
    manufacture a survival result.
    """
    end = censor_at if censor_at is not None else max(v.t_end for v in logs.values())
    rows = []
    for log in logs.values():
        last = log.t[-1] if len(log) else log.t_start
        gaps = np.diff(log.t) if len(log) > 1 else np.array([np.nan])
        typical = float(np.nanmedian(gaps)) if gaps.size else np.nan
        # "Stopped" = silent for far longer than their own usual gap.
        quiet = end - last
        event = 1.0 if np.isfinite(typical) and quiet > max(4 * typical, 30 * 86400) else 0.0
        rows.append(((last - log.t_start) / 86400.0, event))
    return np.array(rows)
