"""Loading real engagement logs from public datasets.

The recurring problem with real data is that raw records are not engagement
events. A row is usually one *item* -- a word reviewed, a question answered --
and a person sitting down once produces a burst of them. Counting rows would
report a single Tuesday-evening sitting as thirty engagements, inflating the
rate and, worse, filling the kernel density with spurious mass at one instant.
So every loader here collapses bursts into occasions before returning anything.

Readers stream and sample rather than loading whole files: the point of these
checks is a distribution over people, and a few thousand people answer that as
well as a hundred thousand do, in a fraction of the memory.
"""

from __future__ import annotations

import ast
import csv
import gzip
import io
import re
import zipfile
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np

from .events import SECONDS_PER_DAY, EventLog

UTC = ZoneInfo("UTC")


def open_text(path: str) -> io.TextIOBase:
    """Open ``.csv``, ``.csv.gz`` or ``.zip`` (containing either) as text.

    Downloads arrive in whichever of these three shapes the host chose, and
    making the caller normalise that by hand is a pointless step to get wrong.
    """
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise ValueError(f"{path} is an empty archive")
        # Largest member: the data file, not a README or licence.
        name = max(names, key=lambda n: zf.getinfo(n).file_size)
        raw = zf.open(name)
        if name.endswith(".gz"):
            return io.TextIOWrapper(gzip.open(raw), encoding="utf-8", newline="")
        return io.TextIOWrapper(raw, encoding="utf-8", newline="")
    if path.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return open(path, mode="rt", encoding="utf-8", newline="")


def in_sample(user_id: str, pct: float) -> bool:
    """Deterministic user-level sampling.

    Hash the id rather than take the first N rows: files are often ordered by
    time or by user, so a prefix is a biased slice of people, while a hash is a
    random one and gives the same sample every run.
    """
    if pct >= 100.0:
        return True
    h = 2166136261
    for ch in user_id.encode("utf-8"):  # FNV-1a, stable across runs and machines
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return (h % 10000) < pct * 100.0


def merge_sessions(times: np.ndarray, gap_minutes: float = 30.0) -> np.ndarray:
    """Collapse a burst of records into one engagement occasion.

    Keeps the *first* timestamp of each burst, because the decision to start is
    what a time-of-day cue triggers; a completion stamp carries the session's
    duration as noise on top.
    """
    if times.size == 0:
        return times
    t = np.sort(times)
    if gap_minutes <= 0:
        return np.unique(t)
    keep = np.empty(t.size, dtype=bool)
    keep[0] = True
    np.greater(np.diff(t), gap_minutes * 60.0, out=keep[1:])
    return t[keep]


@dataclass
class CohortLoadResult:
    logs: dict[str, EventLog]
    n_rows: int
    n_users_seen: int
    n_users_sampled: int
    span_days: float
    t_start: float
    t_end: float
    gap_minutes: float
    n_out_of_range: int = 0

    def summary(self) -> str:
        ev = np.array([len(v) for v in self.logs.values()]) if self.logs else np.zeros(1)
        return (
            f"{self.n_rows:,} rows -> {len(self.logs):,} usable people "
            f"(sampled {self.n_users_sampled:,} of {self.n_users_seen:,} seen)\n"
            f"  data span {self.span_days:.1f} days\n"
            f"  sessions per person: median {np.median(ev):.0f}, "
            f"IQR {np.percentile(ev, 25):.0f}-{np.percentile(ev, 75):.0f}, "
            f"max {ev.max():.0f}\n"
            f"  sessions merged within {self.gap_minutes:.0f} min"
        )


def load_event_csv(
    path: str,
    user_column: str,
    time_column: str,
    time_format: str = "epoch",
    sample_pct: float = 100.0,
    gap_minutes: float = 30.0,
    min_events: int = 8,
    min_days: float = 5.0,
    tz: str = "UTC",
    window: str = "person",
    max_users: int | None = None,
    progress_every: int = 2_000_000,
    verbose: bool = True,
) -> CohortLoadResult:
    """Stream a CSV of ``(user, timestamp)`` rows into one EventLog per person.

    Columns are located **by header name**, so a dataset that adds or reorders
    fields still loads.

    ``time_format`` is ``"epoch"`` (seconds), ``"epoch_ms"``, or ``"iso"``.

    Note on timezone: these datasets rarely record one, and the default UTC is
    the honest choice. It costs nothing for the consistency scores, which are
    invariant to a constant time shift -- a person who trains at the same moment
    every day scores identically whichever meridian you label it from. What it
    does cost is the readable clock label on anchors, which will be wrong by
    each person's offset, and a possible one-hour step if their local zone
    changes over the window.
    """
    per_user: dict[str, list[float]] = {}
    n_rows = 0
    seen: set[str] = set()

    with open_text(path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = {name.strip(): i for i, name in enumerate(header)}
        for want in (user_column, time_column):
            if want not in cols:
                raise ValueError(
                    f"column {want!r} not found; available columns: {sorted(cols)}"
                )
        ui, ti = cols[user_column], cols[time_column]
        parse = _time_parser(time_format)
        width = len(header)

        for row in reader:
            n_rows += 1
            if verbose and progress_every and n_rows % progress_every == 0:
                print(f"    ...{n_rows:,} rows, {len(per_user):,} people kept", flush=True)
            if len(row) != width:
                continue
            uid = row[ui]
            seen.add(uid)
            if not in_sample(uid, sample_pct):
                continue
            if max_users is not None and uid not in per_user and len(per_user) >= max_users:
                continue
            try:
                per_user.setdefault(uid, []).append(parse(row[ti]))
            except (ValueError, TypeError):
                continue

    return _finalize(per_user, ZoneInfo(tz), gap_minutes, min_events, min_days,
                     n_rows, len(seen), window=window)


def _finalize(per_user, tzinfo, gap_minutes, min_events, min_days, n_rows, n_seen,
              window: str = "person"):
    """Merge bursts, apply inclusion rules, and build one EventLog per person.

    ``window`` sets each person's observation period, and the choice is not
    cosmetic. ``"global"`` spans the whole dataset, which is right for a trial
    where everyone is enrolled over the same period. ``"person"`` uses their own
    first-to-last event, which is right for an app people join and leave at
    different times -- and is the only safe default when the dataset spans
    years, because the binned indices (Interdaily Stability, SRI) build a
    day-by-bin grid over the window. Give those a decade-long grid for someone
    observed six months and it is ~99% empty, which drives SRI to 100 and IS to
    0 for everyone regardless of their actual routine.
    """
    merged = {u: merge_sessions(np.array(v, dtype=float), gap_minutes)
              for u, v in per_user.items()}
    all_t = [t for v in merged.values() for t in (v[0], v[-1]) if v.size]
    if not all_t:
        raise ValueError("no usable rows were parsed")
    t_start, t_end = float(min(all_t)), float(max(all_t))

    logs = {}
    for uid, t in merged.items():
        if t.size < min_events:
            continue
        if (t[-1] - t[0]) / SECONDS_PER_DAY < min_days:
            continue
        lo, hi = (float(t[0]), float(t[-1])) if window == "person" else (t_start, t_end)
        logs[uid] = EventLog.from_records(t, tz=tzinfo, t_start=lo, t_end=hi)

    return CohortLoadResult(
        logs=logs,
        n_rows=n_rows,
        n_users_seen=n_seen,
        n_users_sampled=len(per_user),
        span_days=(t_end - t_start) / SECONDS_PER_DAY,
        t_start=t_start,
        t_end=t_end,
        gap_minutes=gap_minutes,
    )


# --------------------------------------------------------------------- FitRec
# Each line is a Python dict literal (note: NOT valid JSON -- single quotes), one
# per workout, holding whole sensor sequences. We need four scalars from each, so
# the fields are picked out by pattern rather than by parsing megabytes of
# heart-rate and GPS arrays we would immediately discard.
_FITREC_USER = re.compile(r"'userId':\s*'?(\w+)'?")
_FITREC_TIME = re.compile(r"'timestamp':\s*\[\s*(-?[\d.eE+]+)")
_FITREC_LAT = re.compile(r"'latitude':\s*\[\s*(-?[\d.eE+]+)")
_FITREC_LON = re.compile(r"'longitude':\s*\[\s*(-?[\d.eE+]+)")
_FITREC_SPORT = re.compile(r"'sport':\s*'([^']*)'")


def _parse_fitrec_line(line: str):
    """Return ``(user, start_time, longitude, sport)`` or None."""
    mu, mt = _FITREC_USER.search(line), _FITREC_TIME.search(line)
    if mu and mt:
        lon = _FITREC_LON.search(line)
        sport = _FITREC_SPORT.search(line)
        return (mu.group(1), float(mt.group(1)),
                float(lon.group(1)) if lon else None,
                sport.group(1) if sport else None)
    # Fall back to a real parse for any line the patterns miss. literal_eval,
    # never eval: these files come off the internet and the reference code's
    # eval() would execute whatever a line contained.
    try:
        rec = ast.literal_eval(line)
        ts = rec.get("timestamp")
        if not ts:
            return None
        lons = rec.get("longitude") or [None]
        return (str(rec["userId"]), float(ts[0]), lons[0], rec.get("sport"))
    except (ValueError, SyntaxError, KeyError, TypeError, MemoryError):
        return None


def load_fitrec(
    path: str,
    sport: str | None = None,
    sample_pct: float = 100.0,
    gap_minutes: float = 30.0,
    min_events: int = 20,
    min_days: float = 30.0,
    localize: bool = True,
    window: str = "person",
    min_year: int | None = 2005,
    max_year: int | None = 2020,
    max_users: int | None = None,
    progress_every: int = 50_000,
    verbose: bool = True,
) -> CohortLoadResult:
    """Endomondo workout logs from the FitRec release (Ni, Muhlstein & McAuley).

    One workout per line; the event time is ``timestamp[0]``, the moment the
    workout *started*, which is the decision a time-of-day cue acts on.

    ``localize`` shifts each person's times by a whole-hour offset estimated from
    their median longitude. Their local clock is what a routine follows, and the
    dataset records none -- but it records GPS, so roughly where they are is
    recoverable. The offset is fixed per person and applied to every event, so
    it cannot affect the consistency scores (which are invariant to a constant
    shift); what it buys is anchor clock times that mean something, and a pooled
    hour-of-day profile that is interpretable rather than smeared across the
    world's timezones. It does not model daylight saving, so expect up to an
    hour of slippage across a transition.

    ``sport`` filters to one activity. Worth considering: someone who runs at
    07:00 and cycles at weekends has two routines, and pooling them looks like
    one incoherent routine.

    ``min_year``/``max_year`` drop implausible timestamps. Real logs contain
    them -- a handful of records dated 1970 or 2038 stretched the observed span
    of this dataset to nineteen years, which silently wrecks any index computed
    over a day-by-bin grid. Pass ``None`` to keep everything and see the damage.
    """
    from datetime import datetime, timezone

    t_lo = (datetime(min_year, 1, 1, tzinfo=timezone.utc).timestamp()
            if min_year else -np.inf)
    t_hi = (datetime(max_year, 1, 1, tzinfo=timezone.utc).timestamp()
            if max_year else np.inf)

    per_user: dict[str, list[float]] = {}
    lons: dict[str, list[float]] = {}
    n_rows = 0
    n_out_of_range = 0
    seen: set[str] = set()

    with open_text(path) as fh:
        for line in fh:
            n_rows += 1
            if verbose and progress_every and n_rows % progress_every == 0:
                print(f"    ...{n_rows:,} workouts, {len(per_user):,} people kept",
                      flush=True)
            parsed = _parse_fitrec_line(line)
            if parsed is None:
                continue
            uid, t, lon, sp = parsed
            if sport is not None and sp != sport:
                continue
            if not (t_lo <= t < t_hi):
                n_out_of_range += 1
                continue
            seen.add(uid)
            if not in_sample(uid, sample_pct):
                continue
            if max_users is not None and uid not in per_user and len(per_user) >= max_users:
                continue
            per_user.setdefault(uid, []).append(t)
            if lon is not None:
                lons.setdefault(uid, []).append(lon)

    if localize:
        for uid, ts in per_user.items():
            if lons.get(uid):
                # 15 degrees of longitude per hour, rounded to a whole hour.
                offset = round(float(np.median(lons[uid])) / 15.0) * 3600.0
                per_user[uid] = [t + offset for t in ts]

    if verbose and n_out_of_range:
        print(f"    dropped {n_out_of_range:,} workouts outside "
              f"{min_year}-{max_year}")
    res = _finalize(per_user, ZoneInfo("UTC"), gap_minutes, min_events, min_days,
                    n_rows, len(seen), window=window)
    res.n_out_of_range = n_out_of_range
    return res


def peek_fitrec(path: str, n_records: int = 2, scan: int = 100_000,
                truncate: int = 260) -> None:
    """Print what is actually in a FitRec file, before trusting any score.

    Written because the first real run produced two impossible readings -- a
    19-year span and a near-empty 06:00-09:00 -- and guessing at the cause from
    summary statistics is slower and less reliable than looking at the records.
    """
    from datetime import datetime, timezone

    print(f"First {n_records} raw records (truncated):\n")
    times, lons, sports, bad = [], [], {}, 0
    with open_text(path) as fh:
        for i, line in enumerate(fh):
            if i < n_records:
                print(f"  [{i}] {line[:truncate]}...\n")
            if i >= scan:
                break
            parsed = _parse_fitrec_line(line)
            if parsed is None:
                bad += 1
                continue
            _, t, lon, sp = parsed
            times.append(t)
            if lon is not None:
                lons.append(lon)
            sports[sp] = sports.get(sp, 0) + 1

    def when(x):
        try:
            return datetime.fromtimestamp(x, timezone.utc).strftime("%Y-%m-%d %H:%M")
        except (OSError, ValueError, OverflowError):
            return f"UNREPRESENTABLE ({x:.0f})"

    t = np.array(times)
    print(f"Scanned {min(scan, len(times) + bad):,} lines; {bad:,} unparseable\n")
    print("Workout start timestamps:")
    for label, q in [("min", 0), ("1%", 1), ("50%", 50), ("99%", 99), ("max", 100)]:
        print(f"  {label:>4s}  {when(np.percentile(t, q))}")
    lo, hi = np.percentile(t, [1, 99])
    print(f"  -> 1-99% span {(hi - lo) / SECONDS_PER_DAY:.0f} days, "
          f"full span {(t.max() - t.min()) / SECONDS_PER_DAY:.0f} days")
    if (t.max() - t.min()) > 1.5 * (hi - lo):
        print("  -> OUTLIERS: the tails stretch the span well beyond the bulk.\n"
              "     Use --min-year / --max-year to drop them.")

    if lons:
        lo_arr = np.array(lons)
        print(f"\nLongitude: {len(lons):,}/{len(times):,} records have one, "
              f"range {lo_arr.min():.1f} to {lo_arr.max():.1f}, "
              f"median {np.median(lo_arr):.1f}")
        if lo_arr.min() > -181 and lo_arr.max() < 181 and abs(lo_arr).max() > 5:
            print("  -> looks like real degrees; localisation should work")
        else:
            print("  -> NOT plausible degrees. Localisation will be wrong; "
                  "rerun with --no-localize")
    else:
        print("\nLongitude: absent -- times stay in UTC")

    print("\nSports: " + ", ".join(f"{k}={v:,}" for k, v in
                                   sorted(sports.items(), key=lambda kv: -kv[1])[:8]))


def load_duolingo(path: str, **kw) -> CohortLoadResult:
    """Duolingo learning traces (Settles & Meeder 2016).

    One row per word shown in a lesson, with ``user_id`` and a UNIX
    ``timestamp``; a lesson therefore spans many rows sharing a moment, and
    several lessons in one sitting are minutes apart. Both collapse into a
    single engagement occasion.

    The release covers roughly two weeks, which is enough to ask whether people
    differ in timing consistency and far too short to say anything about
    dropout. Check the reported span before trusting either.
    """
    kw.setdefault("min_events", 8)
    kw.setdefault("min_days", 5.0)
    return load_event_csv(path, user_column="user_id", time_column="timestamp",
                          time_format="epoch", **kw)


def _time_parser(fmt: str):
    if fmt == "epoch":
        return float
    if fmt == "epoch_ms":
        return lambda s: float(s) / 1000.0
    if fmt == "iso":
        from datetime import datetime

        def parse_iso(s: str) -> float:
            dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.timestamp()

        return parse_iso
    raise ValueError(f"unknown time_format {fmt!r}")


#: Column layout of the Duolingo learning-traces release.
DUOLINGO_COLUMNS = [
    "p_recall", "timestamp", "delta", "user_id", "learning_language", "ui_language",
    "lexeme_id", "lexeme_string", "history_seen", "history_correct",
    "session_seen", "session_correct",
]


def write_synthetic_duolingo(
    path: str,
    n_users: int = 400,
    days: int = 14,
    identical_people: bool = False,
    seed: int = 0,
) -> str:
    """Write a small file imitating the Duolingo release, for dry runs and tests.

    Same header, same one-row-per-word-per-lesson shape, same repeated timestamp
    within a lesson. Lets the whole pipeline be exercised in seconds before
    committing to a multi-minute pass over the real file, and lets the tests
    check the loader without shipping anyone's data.

    ``identical_people=True`` gives every simulated person the *same* true
    consistency. That is the negative control: any spread in the resulting
    scores is pure sampling noise, so a reliability estimate that does not
    collapse to zero would mean the analysis is fooling itself.
    """
    from datetime import datetime, timezone

    from .simulate import PersonProfile, Slot, simulate_person

    rng = np.random.default_rng(seed)
    start = datetime(2013, 3, 4, tzinfo=timezone.utc)
    rows = []
    for u in range(n_users):
        jitter = 25.0 if identical_people else float(rng.lognormal(np.log(25.0), 0.9))
        hour = float(rng.uniform(0, 24))
        prof = PersonProfile(
            slots=[Slot(hour=hour, jitter_min=min(jitter, 400.0), p=0.85)],
            off_schedule_per_day=0.15,
        )
        log, _ = simulate_person(prof, start, days=days, rng=int(rng.integers(1 << 30)),
                                 dropout=False)
        uid = f"u{u:05d}"
        for t in log.t:
            for k in range(int(rng.integers(2, 7))):  # several words per lesson
                rows.append([
                    f"{rng.uniform(0.5, 1.0):.1f}", f"{t:.1f}", f"{rng.integers(60, 9e5)}",
                    uid, "en", "es", f"lex{k}", f"word{k}/word{k}<v>",
                    f"{rng.integers(1, 40)}", f"{rng.integers(1, 40)}", "4", "3",
                ])
    rng.shuffle(rows)  # the real file is not grouped by user

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(DUOLINGO_COLUMNS)
        w.writerows(rows)
    return path


def hour_of_day_histogram(logs: dict[str, EventLog], bins: int = 24) -> np.ndarray:
    """Pooled distribution of engagement hour, as a data-sanity check.

    A real population shows a diurnal shape with an overnight trough. A flat
    histogram means either the timestamps were shifted per person, or the users
    span every timezone -- both survivable for per-person scoring, but you want
    to know which world you are in before reading any clock time as real.
    """
    counts = np.zeros(bins)
    for log in logs.values():
        idx = (log.daily_phase / (2 * np.pi) * bins).astype(int) % bins
        np.add.at(counts, idx, 1.0)
    return counts / max(counts.sum(), 1.0)


def write_synthetic_fitrec(
    path: str,
    n_users: int = 200,
    days: int = 365,
    workouts_per_week: float = 3.0,
    identical_people: bool = False,
    seed: int = 0,
) -> str:
    """Write a small file imitating the FitRec release, for dry runs and tests.

    Same one-dict-per-line layout, same single-quoted Python literals, same
    sensor sequences padded around the fields that matter. Exercise is simulated
    with tight anchors on a few fixed weekdays, which is the pattern the real
    data would need to show for the concept to be testable there.
    """
    from datetime import datetime, timezone

    from .simulate import PersonProfile, Slot, simulate_person

    rng = np.random.default_rng(seed)
    start = datetime(2014, 1, 6, tzinfo=timezone.utc)
    lines = []
    for u in range(n_users):
        jitter = 20.0 if identical_people else float(rng.lognormal(np.log(25.0), 0.9))
        hour = float(rng.uniform(5.5, 21.0))
        n_days_wk = int(np.clip(round(workouts_per_week), 1, 7))
        days_of_week = tuple(rng.choice(7, size=n_days_wk, replace=False).tolist())
        lon = float(rng.uniform(-125, 25))
        prof = PersonProfile(
            slots=[Slot(hour=hour, jitter_min=min(jitter, 300.0),
                        days=days_of_week, p=0.8)],
            name=f"u{u}",
        )
        log, _ = simulate_person(prof, start, days=days,
                                 rng=int(rng.integers(1 << 30)), dropout=False)
        # Undo the localisation the loader will apply, so the round trip lands
        # back on the simulated local hour.
        shift = round(lon / 15.0) * 3600.0
        for t in log.t:
            t0 = float(t) - shift
            ts = [int(t0 + k * 10) for k in range(5)]
            lat = [float(rng.uniform(30, 55))] * 5
            lons = [lon] * 5
            lines.append(
                "{'id': %d, 'userId': '%d', 'sport': 'bike', 'gender': 'male', "
                "'timestamp': %r, 'latitude': %r, 'longitude': %r, "
                "'altitude': %r, 'heart_rate': %r}"
                % (rng.integers(1 << 30), u, ts, lat, lons,
                   [100.0] * 5, [140] * 5)
            )
    rng.shuffle(lines)

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
