"""Does an *exercise* population have habit-scale routines?

    python examples/fitrec_check.py --self-test              # dry run, seconds
    python examples/fitrec_check.py endomondoHR_proper.json --bandwidth-scan

WHY THIS DATASET

The Duolingo run showed the metric works and that population does not have the
phenomenon: reliability rose monotonically to a plateau, the signature of a
broad part-of-day preference rather than an anchored routine. Nobody schedules a
phone language app.

Exercise is different. It is genuinely scheduled -- the morning run, the gym
after work -- so it is the cheapest available test of whether habit-scale
anchors show up in real behaviour at all. And the FitRec release carries roughly
two years per person rather than two weeks, which is the first dataset here long
enough that disengagement can even be observed.

READ THE BANDWIDTH SCAN FIRST. If reliability peaks narrow and declines, this
population has routines and the retention question is worth asking. If it rises
and plateaus like Duolingo did, exercise loggers are no better a testbed and the
concept needs a domain where the activity is prescribed.

FORMAT NOTE

Each line is a Python dict literal -- single quotes, so not valid JSON. The
reference implementation parses it with eval(); this uses ast.literal_eval, and
only after a fast pattern scan that pulls the four fields needed without
decoding megabytes of per-second heart-rate and GPS arrays.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Run straight from a clone, with or without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adherence.datasets import load_fitrec  # noqa: E402
from adherence.screen import screen, write_scores  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="endomondoHR_proper.json (or .gz)")
    p.add_argument("--self-test", action="store_true",
                   help="generate a small fake file in the same format and run on it")
    p.add_argument("--sport", default=None,
                   help="restrict to one activity, e.g. 'bike' or 'run'. Someone who "
                        "runs at 07:00 and cycles at weekends has two routines")
    p.add_argument("--sample-pct", type=float, default=100.0)
    p.add_argument("--min-events", type=int, default=20,
                   help="skip people with fewer workouts (default 20)")
    p.add_argument("--min-days", type=float, default=60.0,
                   help="skip people spanning fewer days (default 60)")
    p.add_argument("--gap-minutes", type=float, default=30.0)
    p.add_argument("--bandwidth", type=float, default=45.0)
    p.add_argument("--no-localize", action="store_true",
                   help="skip the longitude-based local-time estimate")
    p.add_argument("--peek", action="store_true",
                   help="dump raw records and field statistics, then stop. Run this "
                        "first when a result looks impossible")
    p.add_argument("--min-year", type=int, default=2005,
                   help="drop workouts dated before this (default 2005)")
    p.add_argument("--max-year", type=int, default=2020,
                   help="drop workouts dated after this (default 2020)")
    p.add_argument("--window", choices=["person", "global"], default="person",
                   help="observation window per person (default person)")
    p.add_argument("--bandwidth-scan", action="store_true")
    p.add_argument("--with-anchor-precision", action="store_true")
    p.add_argument("--out", default=None, help="write per-person scores to this CSV")
    args = p.parse_args(argv)

    if args.self_test:
        import tempfile

        from adherence.datasets import write_synthetic_fitrec

        args.file = write_synthetic_fitrec(
            os.path.join(tempfile.mkdtemp(), "endo_synthetic.json"),
            n_users=150, days=365,
        )
        print("SELF-TEST: simulated exercisers in the FitRec file format, with tight")
        print("anchors on fixed weekdays. A healthy run reports an ANCHORED verdict")
        print("from the bandwidth scan. Nothing here is real data.\n")
    elif not args.file:
        p.error("provide a data file, or pass --self-test")
    elif not os.path.exists(args.file):
        print(f"No such file: {args.file}")
        print("Download from the FitRec project page (see the module docstring).")
        return 2

    if args.peek:
        from adherence.datasets import peek_fitrec

        peek_fitrec(args.file)
        return 0

    t0 = time.time()
    print(f"Loading {args.file}")
    print("  (streaming; one workout per line)")
    res = load_fitrec(
        args.file, sport=args.sport, sample_pct=args.sample_pct,
        gap_minutes=args.gap_minutes, min_events=args.min_events,
        min_days=args.min_days, localize=not args.no_localize,
        window=args.window, min_year=args.min_year, max_year=args.max_year,
    )
    print(f"  loaded in {time.time() - t0:.0f}s")

    out = screen(
        res, bandwidth_min=args.bandwidth,
        with_anchor_precision=args.with_anchor_precision,
        do_bandwidth_scan=args.bandwidth_scan or args.self_test,
    )
    if args.out and out:
        write_scores(args.out, res.logs, out["model"])
        print(f"\nPer-person scores written to {args.out}")

    print(f"\nTotal time {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
