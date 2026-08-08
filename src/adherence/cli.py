"""Command line interface.

    adherence score events.csv --tz Europe/London
    adherence predict events.csv --hours 24 --auto
    adherence compare events.csv
    adherence demo

Input is a CSV with a column of ISO-8601 timestamps (``--column``, default
``timestamp``). An optional ``--id-column`` splits a cohort file into one report
per participant.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

from .baselines import (
    HomogeneousModel,
    HourHistogramModel,
    LastTimeModel,
    interdaily_stability,
    sleep_regularity_index,
    social_rhythm_hit_rate,
    timing_entropy_bits,
)
from .events import EventLog
from .evaluate import compare as compare_models
from .model import RoutineModel
from .scores import consistency_report, outlook
from .tune import auto_model, select_bandwidth


def _read(path: str, column: str, id_column: str | None, tz: str, weight_column: str | None):
    """Read one CSV into ``{participant_id: EventLog}``."""
    rows = []
    with open(path, newline="") as fh:
        sniff = fh.read(4096)
        fh.seek(0)
        has_header = csv.Sniffer().has_header(sniff) if sniff.strip() else False
        if has_header:
            for row in csv.DictReader(fh):
                rows.append(row)
        else:
            for row in csv.reader(fh):
                if row:
                    rows.append({column: row[0]})

    if not rows:
        raise SystemExit(f"no rows found in {path}")
    if column not in rows[0]:
        raise SystemExit(
            f"column {column!r} not in {sorted(rows[0])}; pass --column"
        )

    grouped = defaultdict(list)
    weights = defaultdict(list)
    for row in rows:
        key = row.get(id_column, "") if id_column else ""
        grouped[key].append(row[column])
        if weight_column:
            weights[key].append(float(row[weight_column]))

    return {
        k: EventLog.from_records(v, tz=tz, weights=weights.get(k) or None)
        for k, v in grouped.items()
    }


def _fmt_time(ts: float, tz) -> str:
    return datetime.fromtimestamp(ts, tz).strftime("%a %d %b %H:%M")


def cmd_score(args) -> int:
    logs = _read(args.file, args.column, args.id_column, args.tz, args.weight_column)
    out = {}
    for pid, log in logs.items():
        model = RoutineModel(bandwidth_min=args.bandwidth, half_life_days=args.half_life)
        rep = consistency_report(log, model, prequential=not args.fast)
        out[pid] = rep.to_dict()
        if args.format == "text":
            if pid:
                print(f"\n--- {pid} ---")
            print(rep)
    if args.format == "json":
        json.dump(out if args.id_column else next(iter(out.values())), sys.stdout,
                  indent=2, default=_json_default)
        print()
    return 0


def cmd_predict(args) -> int:
    logs = _read(args.file, args.column, args.id_column, args.tz, args.weight_column)
    at = datetime.fromisoformat(args.at).timestamp() if args.at else None
    out = {}
    for pid, log in logs.items():
        if args.auto:
            model, sel = auto_model(log, fit=False)
            chosen = sel.best
        else:
            model = RoutineModel(bandwidth_min=args.bandwidth, half_life_days=args.half_life)
            chosen = {"bandwidth_min": args.bandwidth, "half_life_days": args.half_life}
        model.fit(log, now=at)
        ol = outlook(model, log, window_min=args.window)
        out[pid] = {**ol.to_dict(), "params": chosen}
        if args.format == "text":
            if pid:
                print(f"\n--- {pid} ---")
            print(f"  P(engage in next {args.hours:.0f}h)  {model.p_engage_next(args.hours):.3f}")
            print(f"  expected sessions, 7 days   {ol.expected_sessions_7d:.1f}")
            print(
                f"  best {args.window:.0f}-min window       "
                f"{_fmt_time(ol.best_window_start, log.tz)} "
                f"-> {_fmt_time(ol.best_window_end, log.tz)}  (p={ol.best_window_p:.2f})"
            )
            print(f"  days since last session     {ol.days_since_last:.1f} "
                  f"(usual gap {ol.typical_gap_days:.1f})")
            print(f"  parameters                  {chosen}")
    if args.format == "json":
        json.dump(out if args.id_column else next(iter(out.values())), sys.stdout,
                  indent=2, default=_json_default)
        print()
    return 0


def cmd_compare(args) -> int:
    logs = _read(args.file, args.column, args.id_column, args.tz, args.weight_column)
    for pid, log in logs.items():
        bw = select_bandwidth(log).best.get("bandwidth_min", args.bandwidth)
        res = compare_models(
            log,
            {
                "routine (auto)": RoutineModel(bandwidth_min=bw),
                f"routine ({args.bandwidth:.0f}m)": RoutineModel(bandwidth_min=args.bandwidth),
                "hour histogram": HourHistogramModel(),
                "last session": LastTimeModel(),
                "homogeneous": HomogeneousModel(),
            },
            bin_min=args.bin, warmup_days=args.warmup,
        )
        if pid:
            print(f"\n--- {pid} ---")
        print(f"{'forecaster':18s} {'log loss':>9s} {'skill':>7s} {'AUC':>6s} {'ECE':>7s}")
        for name, (m, sk) in res.items():
            print(f"{name:18s} {m.log_loss_bits:9.4f} {sk['log_loss_bits_skill']:+7.3f} "
                  f"{m.auc:6.3f} {m.ece:7.4f}")
        print(f"\nclassic indices: IS={interdaily_stability(log):.3f}  "
              f"SRI={sleep_regularity_index(log):.1f}  "
              f"SRM45={social_rhythm_hit_rate(log):.3f}  "
              f"entropy={timing_entropy_bits(log):.2f} bits")
    return 0


def cmd_demo(args) -> int:
    """Score the built-in archetypes: a one-command sanity check."""
    from .simulate import PRESETS, simulate_person

    start = datetime(2025, 1, 6)
    print(f"{'profile':17s} {'timing':>7s} {'anchor':>7s} {'weekday':>8s} "
          f"{'bits/ev':>8s} {'jitter':>7s} {'drift':>8s}")
    for name, prof in PRESETS.items():
        log, _ = simulate_person(prof, start, days=args.days, tz=args.tz, rng=args.seed,
                                 dropout=False)
        r = consistency_report(log)
        print(f"{name:17s} {r.timing_consistency:7.3f} {r.anchor_precision:7.3f} "
              f"{r.weekday_regularity:8.3f} {r.timing_bits:8.2f} "
              f"{r.jitter_min:6.0f}m {r.drift_min_per_week:+7.1f}m")
    print("\ntiming  = predictability of the next session's time (0 = uniform, 1 = exact)")
    print("anchor  = tightness around each slot, ignoring how many slots there are")
    print("weekday = how much the day of week tells you about whether they engage")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="adherence", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_file=True):
        if needs_file:
            sp.add_argument("file", help="CSV of engagement timestamps")
        sp.add_argument("--tz", default="UTC", help="local timezone, e.g. Europe/London")
        sp.add_argument("--column", default="timestamp", help="timestamp column name")
        sp.add_argument("--id-column", default=None, help="participant id column")
        sp.add_argument("--weight-column", default=None, help="per-event weight column")
        sp.add_argument("--bandwidth", type=float, default=45.0,
                        help="timing tolerance in minutes (default 45)")
        sp.add_argument("--half-life", type=float, default=28.0,
                        help="recency half-life in days (default 28)")
        sp.add_argument("--format", choices=["text", "json"], default="text")

    s = sub.add_parser("score", help="consistency report")
    common(s)
    s.add_argument("--fast", action="store_true", help="skip the out-of-sample score")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("predict", help="engagement probabilities and best nudge window")
    common(s)
    s.add_argument("--hours", type=float, default=24.0)
    s.add_argument("--window", type=float, default=60.0, help="nudge window width, minutes")
    s.add_argument("--auto", action="store_true", help="select bandwidth and half-life")
    s.add_argument("--at", default=None,
                   help="forecast from this ISO timestamp (default: end of the data)")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("compare", help="score forecasters against baselines")
    common(s)
    s.add_argument("--bin", type=float, default=30.0, help="evaluation bin, minutes")
    s.add_argument("--warmup", type=float, default=14.0, help="warm-up days")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("demo", help="score the built-in archetypes")
    s.add_argument("--days", type=int, default=120)
    s.add_argument("--tz", default="Europe/London")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
