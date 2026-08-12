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
import math
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


def compare_groups(rows, quiet: bool = False) -> dict:
    """The plain comparison: people who stopped against people who stayed.

    The Cox model already contains this and more -- it uses *when* each person
    stopped, and it handles the fact that someone observed for 100 days had less
    opportunity to be seen quitting than someone observed for 1000. But a hazard
    ratio is hard to eyeball, and if the two groups' score distributions sit on
    top of each other, that is worth seeing directly.

    Read the follow-up row before the score rows. Unequal follow-up between the
    groups is the reason this comparison cannot stand on its own: it biases the
    raw contrast in whichever direction the imbalance runs, and correcting for
    it is precisely what the survival model is for.
    """
    s = np.array([r[0] for r in rows])
    rate = np.array([r[1] for r in rows])
    dur = np.array([r[2] for r in rows])
    stopped = np.array([r[3] for r in rows]) > 0.5

    a, b = s[stopped], s[~stopped]
    sd = math.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) if min(a.size, b.size) > 1 else np.nan
    d = (b.mean() - a.mean()) / sd if sd else np.nan

    quintiles = []
    edges = np.quantile(s, np.linspace(0, 1, 6))
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        m = (s >= lo) & (s <= hi if i == 4 else s < hi)
        if m.any():
            quintiles.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                              "stopped": float(stopped[m].mean())})
    stats = {"cohens_d": float(d), "n_stopped": int(a.size), "n_active": int(b.size),
             "quintiles": quintiles}
    if quiet:
        return stats

    print("\nStopped vs still active, compared directly:")
    print(f"{'':22s} {'stopped':>12s} {'still active':>13s}")
    print(f"  {'people':20s} {a.size:12d} {b.size:13d}")
    print(f"  {'consistency mean':20s} {a.mean():12.3f} {b.mean():13.3f}")
    print(f"  {'consistency SD':20s} {a.std(ddof=1):12.3f} {b.std(ddof=1):13.3f}")
    print(f"  {'runs/week mean':20s} {rate[stopped].mean():12.2f} "
          f"{rate[~stopped].mean():13.2f}")
    print(f"  {'median follow-up':20s} {np.median(dur[stopped]):11.0f}d "
          f"{np.median(dur[~stopped]):12.0f}d")
    print(f"\n  Cohen's d on consistency (positive = stayers more consistent): {d:+.3f}")
    if abs(d) < 0.2:
        print("  Negligible by any convention. The two distributions overlap almost"
              "\n  entirely -- there is no raw difference for a model to find.")

    # A linear hazard term would miss a U-shape or a tail-only effect, so look at
    # the dropout rate across the range rather than only at its slope.
    print("\nDropout rate by consistency quintile (lowest to highest):")
    for i, q in enumerate(quintiles):
        m = (s >= q["lo"]) & (s <= q["hi"] if i == len(quintiles) - 1 else s < q["hi"])
        print(f"  Q{i + 1}  consistency {q['lo']:.3f}-{q['hi']:.3f}  n={q['n']:3d}  "
              f"stopped {q['stopped']:4.0%}  "
              f"median follow-up {np.median(dur[m]):4.0f}d")
    print("  A monotone trend down this column would be the effect we are looking\n"
          "  for; a U-shape would be missed by the linear hazard term above.")
    return stats


def run_in_reliability(logs, run_in_days: float, min_run_in_events: int) -> float:
    """Split-half reliability of the *run-in* score, not the full-history one.

    This is the number that decides whether a null can be believed. The
    predictor is measured from 90 days, not from years, so it is noisier than
    the headline reliability, and measurement error in a covariate attenuates
    its coefficient toward zero. A null obtained with an unreliable predictor
    says nothing; one obtained with a reliable predictor is evidence.
    """
    from adherence.validate import split_alternate

    model = RoutineModel(bandwidth_min=30.0, half_life_days=365.0)
    a_scores, b_scores = [], []
    for log in logs.values():
        cutoff = log.t[0] + run_in_days * SECONDS_PER_DAY
        window = log.slice(t_to=cutoff)
        if len(window) < max(min_run_in_events, 6):
            continue
        a, b = split_alternate(window)
        sa = timing_consistency(a, model, now=cutoff)
        sb = timing_consistency(b, model, now=cutoff)
        if np.isfinite(sa) and np.isfinite(sb):
            a_scores.append(sa)
            b_scores.append(sb)
    if len(a_scores) < 10:
        return float("nan")
    r = float(np.corrcoef(a_scores, b_scores)[0, 1])
    return float(np.clip(2 * r / (1 + r), 0.0, 1.0)) if r > -1 else float("nan")


def report_precision(se: float, reliability: float) -> None:
    """Turn an estimate into a statement about what it rules out.

    A bare "p = 0.53" is not a finding. What makes a null informative is the
    pair of numbers underneath it: how large an effect the design could have
    caught, and how reliably the predictor was measured. Without both, a null
    is indistinguishable from having looked badly.
    """
    from scipy.stats import norm

    print("\nPrecision -- what this estimate can and cannot rule out:")
    for power, z in (("80%", 2.802), ("90%", 3.242)):
        print(f"  minimum detectable effect at {power} power: "
              f"HR {math.exp(z * se):.2f} per SD")
    for hr in (1.2, 1.3, 1.4):
        print(f"  power to have detected HR {hr}: "
              f"{norm.cdf(math.log(hr) / se - 1.96):.0%}")
    if np.isfinite(reliability) and reliability > 0:
        print(f"\n  Run-in score reliability {reliability:.2f}. Measurement error "
              f"attenuates\n  the coefficient by roughly that factor, so a true effect "
              f"of HR h would\n  show here as about HR {math.exp(math.log(1.3) * reliability):.2f} "
              "if h were 1.30.")
        if reliability < 0.5:
            print("  That is low enough that the null is weak evidence: a real effect\n"
                  "  could be hidden by noise in the predictor. Lengthen the run-in.")


def fit_and_report(rows, label: str, adjust: bool) -> float:
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
    return float(fit.se[0])


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
    p.add_argument("--sensitivity", action="store_true",
                   help="repeat across run-in lengths and silence thresholds, to check "
                        "the answer does not hinge on either")
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

    compare_groups(rows)

    fit_and_report(rows, "UNADJUSTED -- consistency alone (frequency not held constant):",
                   adjust=False)
    se = fit_and_report(rows, "ADJUSTED -- consistency with run-in frequency held constant:",
                        adjust=True)

    rel = run_in_reliability(res.logs, args.run_in, args.min_run_in_events)
    report_precision(se, rel)

    if args.sensitivity:
        print("\n\nSensitivity -- does the answer depend on the design choices?")
        print(f"{'run-in':>8s} {'quiet':>7s} {'n':>6s} {'events':>7s} "
              f"{'log HR':>8s} {'p':>8s}")
        for run_in in (60.0, 90.0, 180.0):
            for quiet in (90.0, 180.0, 365.0):
                r, _ = build_cohort(res.logs, run_in, dataset_end, quiet,
                                    args.min_run_in_events)
                ev = np.array([q[3] for q in r]) if r else np.zeros(0)
                if len(r) < 50 or ev.sum() < 20:
                    print(f"{run_in:7.0f}d {quiet:6.0f}d {len(r):6d} {ev.sum():7.0f}"
                          "   (too few)")
                    continue
                sc = np.array([q[0] for q in r])
                x = -(sc - sc.mean()) / sc.std()
                rate = np.array([q[1] for q in r])
                lr = (np.log(rate + 0.1) - np.log(rate + 0.1).mean()) / np.log(rate + 0.1).std()
                f = cox_ph(np.column_stack([x, lr]),
                           np.array([q[2] for q in r]), ev)
                print(f"{run_in:7.0f}d {quiet:6.0f}d {len(r):6d} {ev.sum():7.0f} "
                      f"{f.coef[0]:+8.3f} {f.p[0]:8.3f}")

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
