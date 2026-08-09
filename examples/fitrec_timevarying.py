"""Is *slipping* consistency a warning sign, even when baseline consistency is not?

    python examples/fitrec_timevarying.py endomondoHR.json.gz

WHY THIS IS A DIFFERENT QUESTION

The earlier analysis froze one score per person from a 90-day run-in and asked
whether people who *were* regular lasted longer. It found nothing: HR 0.955 per
SD, with the interval excluding anything above 1.10.

But that design can only detect a stable trait. It cannot see the thing the
concept actually describes -- a routine coming apart. People rarely quit out of
a steady habit; the habit usually loosens first. So the natural question is not
"were they regular in month one" but "is their regularity falling right now",
and answering it needs a covariate that moves with the person.

THE DESIGN

Every ``--interval`` days, each person is re-scored using only their history up
to that moment. That score, and its change since the previous assessment,
become the covariates for the *following* interval. Anyone who disengages
contributes an event in the interval where their last run falls.

REVERSE CAUSATION, AND THE LAG

Someone about to quit slows down first. Measure their consistency in the days
immediately before they stop and "prediction" becomes near-tautological -- the
score is picking up the beginning of the very event it claims to forecast.

So the model is fitted twice: with no lag, and with the covariates lagged so
that a score computed at day T is used to predict disengagement in
``(T + lag, T + lag + interval]``. If an effect survives the lag it is a genuine
early warning. If it appears only without one, it was measuring the slowdown
itself.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Run straight from a clone, with or without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adherence import RoutineModel, timing_consistency  # noqa: E402
from adherence.datasets import load_fitrec  # noqa: E402
from adherence.events import SECONDS_PER_DAY  # noqa: E402
from adherence.survival import cox_ph_time_varying  # noqa: E402

DAY = SECONDS_PER_DAY


def cessation_time(log, dataset_end: float, quiet_days: float):
    """When (if ever) this person stopped. Returns ``(time, stopped)``.

    Silence far longer than their own usual gap counts as having stopped; anyone
    still going at the end of the data is censored, not a non-event.
    """
    gaps = np.diff(log.t)
    typical = float(np.median(gaps)) if gaps.size else np.nan
    threshold = max(4 * typical, quiet_days * DAY) if np.isfinite(typical) \
        else quiet_days * DAY
    if dataset_end - log.t[-1] > threshold:
        return float(log.t[-1]), True
    return float(dataset_end), False


def build_intervals(logs, dataset_end: float, warmup_days: float,
                    interval_days: float, lag_days: float, quiet_days: float,
                    min_events: int):
    """One row per person per interval, covariates knowable at the interval start."""
    model = RoutineModel(bandwidth_min=30.0, half_life_days=365.0)
    rows = []

    for log in logs.values():
        end_t, stopped = cessation_time(log, dataset_end, quiet_days)
        origin = log.t[0]
        assess = origin + warmup_days * DAY
        prev_score = None

        while True:
            risk_start = assess + lag_days * DAY
            risk_stop = risk_start + interval_days * DAY
            if risk_start >= end_t:
                break

            # Everything below uses only events strictly at or before `assess`.
            hist = log.slice(t_to=assess)
            if len(hist) < min_events:
                assess += interval_days * DAY
                continue
            score = timing_consistency(hist, model, now=assess)
            if not np.isfinite(score):
                assess += interval_days * DAY
                continue

            recent = hist.t[hist.t > assess - interval_days * DAY]
            rate = recent.size / interval_days * 7.0  # runs per week
            delta = 0.0 if prev_score is None else score - prev_score

            # The event lands in the interval containing their final run.
            event = 1.0 if (stopped and risk_start < end_t <= risk_stop) else 0.0
            stop = min(risk_stop, end_t)
            if stop > risk_start:
                rows.append({
                    "person": id(log), "start": (risk_start - origin) / DAY,
                    "stop": (stop - origin) / DAY, "event": event,
                    "score": score, "delta": delta, "rate": rate,
                })
            prev_score = score
            assess += interval_days * DAY

    return rows


def fit(rows, terms: list[str], label: str) -> None:
    """Fit and print one specification."""
    z = lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * 0.0  # noqa: E731
    score = np.array([r["score"] for r in rows])
    cols, names = [], []
    if "irregularity" in terms:
        cols.append(-z(score))            # flipped: higher = less regular
        names.append("irregularity")
    if "falling" in terms:
        # Negative delta = consistency dropped, so flip for "amount it fell".
        cols.append(-z(np.array([r["delta"] for r in rows])))
        names.append("consistency falling")
    if "rate" in terms:
        cols.append(z(np.log(np.array([r["rate"] for r in rows]) + 0.1)))
        names.append("log run rate")

    fitted = cox_ph_time_varying(
        np.column_stack(cols),
        np.array([r["start"] for r in rows]),
        np.array([r["stop"] for r in rows]),
        np.array([r["event"] for r in rows]),
        names=names,
    )
    print(f"\n{label}")
    print("  " + str(fitted).replace("\n", "\n  "))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="endomondoHR.json.gz")
    p.add_argument("--sport", default="run")
    p.add_argument("--warmup", type=float, default=90.0,
                   help="days of history before the first assessment (default 90)")
    p.add_argument("--interval", type=float, default=30.0,
                   help="days between re-scorings (default 30)")
    p.add_argument("--lag", type=float, default=30.0,
                   help="days between measuring the score and the interval it "
                        "predicts (default 30). The model is also fitted at lag 0")
    p.add_argument("--quiet-days", type=float, default=180.0)
    p.add_argument("--min-events", type=int, default=10)
    p.add_argument("--min-days", type=float, default=120.0)
    args = p.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"No such file: {args.file}")
        return 2

    t0 = time.time()
    print(f"Loading {args.file} (sport={args.sport})")
    res = load_fitrec(args.file, sport=args.sport, min_events=20,
                      min_days=args.min_days)
    print("\n" + res.summary())
    dataset_end = max(v.t[-1] for v in res.logs.values())

    for lag in (0.0, args.lag):
        rows = build_intervals(res.logs, dataset_end, args.warmup, args.interval,
                               lag, args.quiet_days, args.min_events)
        if not rows:
            print(f"\nNo usable intervals at lag {lag:.0f}d.")
            continue
        n_people = len({r["person"] for r in rows})
        n_ev = sum(r["event"] for r in rows)
        print(f"\n{'=' * 72}")
        print(f"LAG {lag:.0f} DAYS -- score measured at T, predicts "
              f"({lag:.0f}, {lag + args.interval:.0f}] days later")
        print(f"{'=' * 72}")
        print(f"  {len(rows):,} intervals from {n_people:,} people, "
              f"{n_ev:.0f} disengagements")
        if n_ev < 20:
            print("  Too few events to fit.")
            continue

        fit(rows, ["irregularity"], "Current irregularity alone:")
        fit(rows, ["irregularity", "rate"],
            "Current irregularity, holding current run rate constant:")
        fit(rows, ["irregularity", "falling", "rate"],
            "Adding whether consistency is FALLING (the early-warning test):")

    print(
        "\n\nHow to read this. The lag-0 fit can be inflated by reverse causation:\n"
        "someone already winding down looks both irregular and about to quit. The\n"
        "lagged fit is the honest one -- an effect that survives there is a genuine\n"
        "early warning, and one that vanishes was measuring the slowdown itself.\n"
        "\nNothing here is causal either way: people whose routines come apart differ\n"
        "from people whose routines hold in ways a workout log cannot record."
    )
    print(f"\nTotal time {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
