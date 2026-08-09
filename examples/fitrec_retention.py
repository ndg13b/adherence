"""Does timing consistency predict who keeps running?

    python examples/fitrec_retention.py endomondoHR.json.gz

This is the question the whole package was built for, and the FitRec running
subset is the first real data able to address it: 626 people, a median of 116
runs each, spanning years, and -- unlike Duolingo -- carrying genuine
habit-scale anchors (the bandwidth scan peaks at 30 minutes and declines).

THE DESIGN

1. **Run-in.** Score consistency from each person's first ``--run-in`` days
   only.
2. **Follow-up.** The clock starts where the run-in ends. Anyone who had already
   stopped during the run-in is excluded -- they cannot be scored by it, and
   including them would credit the score with an outcome it got to observe.
3. **Outcome.** Time until their last run, censored for anyone still active near
   the end of the dataset. Disengagement is never recorded here; it has to be
   inferred from silence, and treating a censored person as a non-event is the
   classic way to manufacture a result.

THE CONFOUND THAT MATTERS

People who run *often* keep running. If consistency is entered alone it will
absorb that, and the finding would be trivial. Baseline frequency during the
run-in is therefore included as a covariate, so the coefficient on consistency
is what remains after frequency is accounted for. The unadjusted model is
printed alongside precisely so the difference is visible.

WHAT THIS CANNOT SHOW

Nothing here is causal. Regular runners differ from irregular ones in ways no
covariate available in a workout log can capture -- injury, job, motivation,
whether they were training for a race. A positive result says the score carries
retention information beyond frequency, which is what a screening instrument
needs to do; it does not say that making someone regular would keep them
running.
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
from adherence.survival import cox_ph  # noqa: E402


def build_cohort(logs, run_in_days: float, dataset_end: float,
                 quiet_days: float, min_run_in_events: int):
    """Run-in scores and follow-up outcomes, with the two clocks kept separate."""
    model = RoutineModel(bandwidth_min=30.0, half_life_days=365.0)
    rows, skipped = [], {"short_run_in": 0, "quit_in_run_in": 0, "no_followup": 0}

    for log in logs.values():
        cutoff = log.t[0] + run_in_days * SECONDS_PER_DAY
        run_in = log.t[log.t < cutoff]
        if run_in.size < min_run_in_events:
            skipped["short_run_in"] += 1
            continue
        if log.t[-1] <= cutoff:  # already gone before follow-up begins
            skipped["quit_in_run_in"] += 1
            continue
        if dataset_end - cutoff < 30 * SECONDS_PER_DAY:
            skipped["no_followup"] += 1
            continue

        window = log.slice(t_to=cutoff)
        score = timing_consistency(window, model, now=cutoff)
        if not np.isfinite(score):
            skipped["short_run_in"] += 1
            continue

        # Frequency during the run-in, the confound to adjust for.
        rate = run_in.size / max(run_in_days, 1.0) * 7.0  # runs per week

        last = log.t[-1]
        gaps = np.diff(log.t)
        typical = float(np.median(gaps)) if gaps.size else np.nan
        silence = dataset_end - last
        # Stopped if silent for far longer than their own usual gap; otherwise
        # still going at the end of the data, and censored.
        threshold = max(4 * typical, quiet_days * SECONDS_PER_DAY) if np.isfinite(typical) \
            else quiet_days * SECONDS_PER_DAY
        stopped = silence > threshold
        duration = (last - cutoff) / SECONDS_PER_DAY if stopped else \
            (dataset_end - cutoff) / SECONDS_PER_DAY
        rows.append((score, rate, max(duration, 0.1), float(stopped)))

    return rows, skipped


def fit_and_report(rows, label: str, adjust: bool) -> None:
    s = np.array([r[0] for r in rows])
    rate = np.array([r[1] for r in rows])
    time_d = np.array([r[2] for r in rows])
    event = np.array([r[3] for r in rows])

    # Sign-flipped so the covariate reads as irregularity: a positive
    # coefficient then means "less regular, quits sooner".
    x = -(s - s.mean()) / s.std()
    names = ["irregularity"]
    X = x[:, None]
    if adjust:
        r = (np.log(rate + 0.1) - np.log(rate + 0.1).mean()) / np.log(rate + 0.1).std()
        X = np.column_stack([x, r])
        names = ["irregularity", "log run rate"]

    fit = cox_ph(X, time_d, event, names=names)
    print(f"\n{label}")
    print("  " + str(fit).replace("\n", "\n  "))
    lo, hi = fit.ci()[0]
    print(f"  irregularity HR {np.exp(fit.coef[0]):.3f} "
          f"(95% CI {np.exp(lo):.3f}-{np.exp(hi):.3f}) per SD")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="endomondoHR.json.gz")
    p.add_argument("--sport", default="run")
    p.add_argument("--run-in", type=float, default=90.0,
                   help="days of history used to score consistency (default 90)")
    p.add_argument("--min-run-in-events", type=int, default=10)
    p.add_argument("--quiet-days", type=float, default=180.0,
                   help="silence beyond this counts as having stopped (default 180)")
    p.add_argument("--min-events", type=int, default=20)
    p.add_argument("--min-days", type=float, default=120.0)
    args = p.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"No such file: {args.file}")
        return 2

    t0 = time.time()
    print(f"Loading {args.file} (sport={args.sport})")
    res = load_fitrec(args.file, sport=args.sport, min_events=args.min_events,
                      min_days=args.min_days)
    print("\n" + res.summary())

    dataset_end = max(v.t[-1] for v in res.logs.values())
    rows, skipped = build_cohort(res.logs, args.run_in, dataset_end,
                                 args.quiet_days, args.min_run_in_events)

    print(f"\nRun-in {args.run_in:.0f} days, then follow-up to the end of the data.")
    print(f"  {len(rows):,} analysable people")
    print(f"  excluded: {skipped['short_run_in']:,} too few runs in the run-in, "
          f"{skipped['quit_in_run_in']:,} already stopped during it, "
          f"{skipped['no_followup']:,} no follow-up left")
    if len(rows) < 50:
        print("\nToo few people to fit. Try --run-in 60 or --min-events 15.")
        return 1

    event = np.array([r[3] for r in rows])
    dur = np.array([r[2] for r in rows])
    print(f"  {event.mean():.0%} stopped, {1 - event.mean():.0%} censored; "
          f"median follow-up {np.median(dur):.0f} days")

    if event.sum() < 20:
        print(f"\nOnly {event.sum():.0f} people classed as stopped -- too few events to\n"
              "fit. Either almost everyone is still active at the end of the data, or\n"
              "--quiet-days is too long. Try a shorter --quiet-days.")
        return 1

    fit_and_report(rows, "UNADJUSTED -- consistency alone (frequency not held constant):",
                   adjust=False)
    fit_and_report(rows, "ADJUSTED -- consistency with run-in frequency held constant:",
                   adjust=True)

    print(
        "\nRead the adjusted model. The unadjusted one is shown only so the\n"
        "difference is visible: if the effect collapses once frequency enters,\n"
        "the score was a proxy for how often people ran, not for how regularly.\n"
        "\nAnd nothing here is causal -- regular runners differ from irregular ones\n"
        "in ways a workout log cannot record."
    )
    print(f"\nTotal time {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
