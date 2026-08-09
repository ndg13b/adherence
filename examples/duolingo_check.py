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

from adherence.datasets import load_duolingo  # noqa: E402
from adherence.screen import screen, write_scores  # noqa: E402


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
    p.add_argument("--bandwidth-scan", action="store_true",
                   help="score at a range of kernel widths to find this population's "
                        "timing tolerance. Cheap: reuses the single pass over the file")
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
    out = screen(
        res, bandwidth_min=args.bandwidth,
        with_anchor_precision=args.with_anchor_precision,
        do_bandwidth_scan=args.bandwidth_scan,
        short_span_days=25.0,
    )
    if not out:
        return 1

    if args.gap_sensitivity:
        print("\nSensitivity to the session-merge gap (a judgement call, so check it):")
        print(f"{'gap':>8s} {'people':>8s} {'mean':>8s} {'reliab.':>9s} {'true SD':>9s}")
        from adherence.validate import SCORERS, reliability_report
        for gap in (0.0, 10.0, 30.0, 60.0, 120.0):
            r = load_duolingo(args.file, sample_pct=args.sample_pct, gap_minutes=gap,
                              min_events=args.min_events, min_days=args.min_days,
                              verbose=False)
            if not r.logs:
                continue
            rep = reliability_report(r.logs, out["model"],
                                     {"timing_consistency": SCORERS["timing_consistency"]},
                                     verbose=False)
            i = rep.get("timing_consistency")
            print(f"{gap:7.0f}m {len(r.logs):8d} {i.mean:8.3f} {i.reliability:9.3f} "
                  f"{i.reliable_sd:9.3f}")

    if args.out:
        write_scores(args.out, res.logs, out["model"])
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


if __name__ == "__main__":
    sys.exit(main())
