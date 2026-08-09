"""Does timing consistency behave like a real individual difference?

Run this against the Duolingo learning-traces release. One command:

    python examples/duolingo_check.py ~/Downloads/learning_traces.13m.csv.gz

Accepts .csv, .csv.gz or .zip. Takes a few minutes; prints a verdict at the end.

WHAT THIS CAN AND CANNOT SHOW

It can show whether people genuinely differ in how consistently they engage, and
whether the kernel score measures that more reliably than the established
indices. That is the assumption the whole concept rests on, and the cheapest one
to falsify.

It cannot say anything about dropout. The release spans roughly two weeks, which
is not long enough to observe anyone quitting. If the check passes, the next
dataset needs months, not weeks.

One caveat to keep in view: if this release is a *sample* of each person's
traces rather than all of them, missing sessions will look like irregularity and
every consistency estimate here is pessimistic. The reported sessions-per-person
is the clue -- if committed daily users show far fewer than one session per day,
suspect filtering.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Run straight from a clone, with or without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adherence import RoutineModel  # noqa: E402
from adherence.datasets import hour_of_day_histogram, load_duolingo  # noqa: E402
from adherence.validate import (  # noqa: E402
    OPTIONAL_SCORERS,
    SCORERS,
    reliability_report,
)


def print_hour_profile(logs) -> None:
    """Sanity check on the timestamps before trusting anything downstream."""
    h = hour_of_day_histogram(logs)
    peak = h.max()
    print("\nPooled engagement by hour (UTC, all people):")
    for i, v in enumerate(h):
        bar = "#" * int(round(40 * v / peak)) if peak > 0 else ""
        print(f"  {i:02d}:00 {v:6.3f} {bar}")
    flatness = h.min() / h.max() if h.max() > 0 else float("nan")
    print(f"  trough/peak ratio {flatness:.2f}", end="  ")
    if flatness > 0.6:
        print("-- flat: users span many timezones, or times were shifted per user.\n"
              "     Harmless for the scores (they are shift-invariant) but clock\n"
              "     labels on anchors are not interpretable.")
    else:
        print("-- a diurnal shape is present, so the timestamps carry real\n"
              "     time-of-day information.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?",
                   help="learning_traces.13m.csv.gz (or .zip / .csv)")
    p.add_argument("--self-test", action="store_true",
                   help="generate a small fake file in the same format and run on it. "
                        "Takes seconds -- use it to confirm the pipeline works before "
                        "starting the real multi-minute run")
    p.add_argument("--sample-pct", type=float, default=5.0,
                   help="percent of people to sample (default 5; a few thousand "
                        "people answer this as well as all of them)")
    p.add_argument("--gap-minutes", type=float, default=30.0,
                   help="merge records this close into one session (default 30)")
    p.add_argument("--min-events", type=int, default=8,
                   help="skip people with fewer sessions (default 8)")
    p.add_argument("--min-days", type=float, default=5.0,
                   help="skip people spanning fewer days (default 5)")
    p.add_argument("--bandwidth", type=float, default=45.0,
                   help="timing tolerance in minutes; FIXED across people so the "
                        "scores stay comparable (default 45)")
    p.add_argument("--with-anchor-precision", action="store_true",
                   help="also score anchor precision (slower: permutation null)")
    p.add_argument("--gap-sensitivity", action="store_true",
                   help="repeat at several session-merge gaps to check the choice "
                        "does not drive the answer")
    p.add_argument("--out", default=None, help="write per-person scores to this CSV")
    args = p.parse_args(argv)

    if args.self_test:
        import tempfile

        from adherence.datasets import write_synthetic_duolingo

        args.file = write_synthetic_duolingo(
            os.path.join(tempfile.mkdtemp(), "synthetic_traces.csv.gz"),
            n_users=300, days=14,
        )
        args.sample_pct = 100.0
        print("SELF-TEST: scoring simulated people in the Duolingo file format.")
        print("Their consistency differs by construction, so a healthy run reports")
        print("clear reliability and a PROCEED verdict. Nothing here is real data.\n")
    elif not args.file:
        p.error("provide a data file, or pass --self-test")
    elif not os.path.exists(args.file):
        print(f"No such file: {args.file}\n")
        _suggest_files()
        return 2

    t0 = time.time()
    print(f"Loading {args.file}")
    print("  (streaming; this reads the whole file once)")
    res = load_duolingo(
        args.file,
        sample_pct=args.sample_pct,
        gap_minutes=args.gap_minutes,
        min_events=args.min_events,
        min_days=args.min_days,
    )
    print("\n" + res.summary())
    print(f"  loaded in {time.time() - t0:.0f}s")

    if res.span_days < 25:
        print(f"\n  NOTE: the data spans {res.span_days:.0f} days. Long enough to ask "
              "whether\n        people differ; far too short to observe dropout.")

    if not res.logs:
        print("\nNo person met the inclusion thresholds. Try --min-events 5 --min-days 3.")
        return 1

    print_hour_profile(res.logs)

    scorers = dict(SCORERS)
    if args.with_anchor_precision:
        scorers.update(OPTIONAL_SCORERS)

    model = RoutineModel(bandwidth_min=args.bandwidth, half_life_days=28.0)
    print("\nScoring (fixed bandwidth "
          f"{args.bandwidth:.0f} min, so people stay comparable)...")
    report = reliability_report(res.logs, model, scorers)
    print("\n" + str(report))

    print("\n" + "=" * 72)
    print(report.verdict())
    print("=" * 72)

    if args.gap_sensitivity:
        print("\nSensitivity to the session-merge gap (a judgement call, so check it):")
        print(f"{'gap':>8s} {'people':>8s} {'mean':>8s} {'reliab.':>9s} {'true SD':>9s}")
        for gap in (0.0, 10.0, 30.0, 60.0, 120.0):
            r = load_duolingo(args.file, sample_pct=args.sample_pct, gap_minutes=gap,
                              min_events=args.min_events, min_days=args.min_days,
                              verbose=False)
            if not r.logs:
                continue
            rep = reliability_report(r.logs, model, {"timing_consistency":
                                                     SCORERS["timing_consistency"]},
                                     verbose=False)
            i = rep.get("timing_consistency")
            print(f"{gap:7.0f}m {len(r.logs):8d} {i.mean:8.3f} {i.reliability:9.3f} "
                  f"{i.reliable_sd:9.3f}")

    if args.out:
        _write_scores(args.out, res.logs, model, scorers)
        print(f"\nPer-person scores written to {args.out}")

    print(f"\nTotal time {time.time() - t0:.0f}s")
    return 0


def _suggest_files() -> None:
    """Point at plausible data files nearby rather than just failing."""
    roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    seen, found = set(), []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for pat in ("*.csv", "*.csv.gz", "*.zip", "*.gz"):
            found += [p for p in root.glob(pat) if p.stat().st_size > 1_000_000]
    if found:
        print("Data files found nearby -- did you mean one of these?")
        for p in sorted(set(found))[:8]:
            print(f"  python examples/duolingo_check.py \"{p}\"   ({p.stat().st_size / 1e6:.0f} MB)")
    else:
        print("Pass the path to learning_traces.13m.csv.gz (or the .zip), or use --self-test.")


def _write_scores(path, logs, model, scorers) -> None:
    import csv

    from adherence.validate import score_cohort

    scores = score_cohort(logs, model, scorers)
    names = list(scorers)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "n_sessions", "span_days"] + names)
        for i, (uid, log) in enumerate(logs.items()):
            span = (log.t[-1] - log.t[0]) / 86400.0 if len(log) else 0.0
            w.writerow([uid, len(log), f"{span:.2f}"]
                       + [f"{scores[n][i]:.6f}" for n in names])


if __name__ == "__main__":
    sys.exit(main())
