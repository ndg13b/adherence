"""Engagement events and their position in the daily / weekly cycle.

A routine is defined against *wall clock* time, not against elapsed seconds: a
person who trains at 08:00 keeps training at 08:00 across a daylight-saving
transition, and a model that works in UTC would record that as a one-hour jump
in their routine. Everything here therefore converts to a caller-supplied local
timezone before computing phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

TWO_PI = 2.0 * math.pi
SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 604800.0


def _to_utc_seconds(value) -> float:
    """Accept a datetime, ISO-8601 string or epoch number; return epoch seconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(
                f"naive datetime {value!r}: attach a timezone, or pass "
                "assume_tz= to EventLog.from_records"
            )
        return value.timestamp()
    raise TypeError(f"cannot interpret {value!r} as a timestamp")


@dataclass
class EventLog:
    """A person's engagement history.

    Attributes
    ----------
    t:
        Event times as UTC epoch seconds, sorted ascending.
    weight:
        Optional per-event importance (e.g. session length, or completion
        quality). Defaults to 1.0. These multiply the recency weights, so a
        30-second app open can be discounted relative to a full session.
    tz:
        Local timezone the routine lives in.
    t_start, t_end:
        Observation window. Exposure matters: two events in a week is a very
        different rate from two events in a year, and the window is the only
        way to know which one you are looking at. Defaults to the first and
        last event, which biases rates upward -- pass the real enrolment and
        censoring times when you have them.
    """

    t: np.ndarray
    tz: ZoneInfo
    weight: np.ndarray
    t_start: float
    t_end: float
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_records(
        cls,
        timestamps: Iterable,
        tz: str | ZoneInfo = "UTC",
        weights: Sequence[float] | None = None,
        t_start=None,
        t_end=None,
        assume_tz: bool = False,
        meta: dict | None = None,
    ) -> "EventLog":
        tzinfo = ZoneInfo(tz) if isinstance(tz, str) else tz
        raw = list(timestamps)
        if assume_tz:
            raw = [
                v.replace(tzinfo=tzinfo) if isinstance(v, datetime) and v.tzinfo is None else v
                for v in raw
            ]
        t = np.array([_to_utc_seconds(v) for v in raw], dtype=float)
        w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != t.shape:
            raise ValueError("weights must match timestamps in length")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

        order = np.argsort(t, kind="stable")
        t, w = t[order], w[order]

        start = _to_utc_seconds(t_start) if t_start is not None else (t[0] if len(t) else 0.0)
        end = _to_utc_seconds(t_end) if t_end is not None else (t[-1] if len(t) else 0.0)
        if end < start:
            raise ValueError("t_end precedes t_start")
        return cls(t=t, tz=tzinfo, weight=w, t_start=float(start), t_end=float(end),
                   meta=meta or {})

    # -------------------------------------------------------------- properties
    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def span_days(self) -> float:
        return (self.t_end - self.t_start) / SECONDS_PER_DAY

    @property
    def local_datetimes(self) -> list[datetime]:
        return [datetime.fromtimestamp(x, self.tz) for x in self.t]

    @property
    def daily_phase(self) -> np.ndarray:
        """Position within the day, in radians on ``[0, 2*pi)``."""
        return daily_phase(self.t, self.tz)

    @property
    def dow(self) -> np.ndarray:
        """Local day of week, Monday=0 ... Sunday=6."""
        return local_dow(self.t, self.tz)

    @property
    def local_day_index(self) -> np.ndarray:
        """Integer index of the local calendar day, for grouping events by day."""
        return local_day_index(self.t, self.tz)

    def slice(self, t_from: float | None = None, t_to: float | None = None) -> "EventLog":
        """Half-open ``[t_from, t_to)`` sub-log, keeping the observation window."""
        lo = -np.inf if t_from is None else t_from
        hi = np.inf if t_to is None else t_to
        m = (self.t >= lo) & (self.t < hi)
        return EventLog(
            t=self.t[m],
            tz=self.tz,
            weight=self.weight[m],
            t_start=max(self.t_start, lo if np.isfinite(lo) else self.t_start),
            t_end=min(self.t_end, hi if np.isfinite(hi) else self.t_end),
            meta=dict(self.meta),
        )


# ------------------------------------------------------------------ phase math
class _OffsetTable:
    """Piecewise-constant UTC offsets for a timezone over a time range.

    A zone changes offset at most a couple of times a year, so rather than
    calling ``datetime.fromtimestamp`` once per timestamp -- which dominates the
    runtime of every scoring loop -- we locate the transitions once (to the
    second, by bisection) and then assign offsets with a vectorised
    ``searchsorted``.
    """

    def __init__(self, tz: ZoneInfo, lo: float, hi: float):
        pad = 400 * SECONDS_PER_DAY
        self.lo = math.floor((lo - pad) / SECONDS_PER_DAY) * SECONDS_PER_DAY
        self.hi = math.ceil((hi + pad) / SECONDS_PER_DAY) * SECONDS_PER_DAY
        self.tz = tz

        probes = np.arange(self.lo, self.hi + SECONDS_PER_DAY, SECONDS_PER_DAY)
        offs = np.array([self._raw(p) for p in probes])

        breaks = [self.lo]
        values = [offs[0]]
        for i in np.nonzero(np.diff(offs) != 0)[0]:
            a, b = probes[i], probes[i + 1]
            off_b = offs[i + 1]
            while b - a > 1e-3:
                mid = 0.5 * (a + b)
                if self._raw(mid) == off_b:
                    b = mid
                else:
                    a = mid
            # `b` converges to the transition from above; real transitions fall on
            # whole seconds, so rounding lands exactly on it. Getting this wrong by
            # a fraction of a second would misplace the boundary instant itself.
            breaks.append(float(round(b)))
            values.append(off_b)
        self.breaks = np.array(breaks, dtype=float)
        self.values = np.array(values, dtype=float)

    def _raw(self, t: float) -> float:
        return datetime.fromtimestamp(float(t), self.tz).utcoffset().total_seconds()

    def covers(self, lo: float, hi: float) -> bool:
        return self.lo <= lo and hi <= self.hi

    def __call__(self, t: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(self.breaks, t, side="right") - 1, 0, len(self.values) - 1)
        return self.values[idx]


_OFFSET_CACHE: dict[str, _OffsetTable] = {}


def _offset_table(tz: ZoneInfo, lo: float, hi: float) -> _OffsetTable:
    key = str(getattr(tz, "key", tz))
    tbl = _OFFSET_CACHE.get(key)
    if tbl is None or not tbl.covers(lo, hi):
        tbl = _OffsetTable(tz, lo, hi)
        _OFFSET_CACHE[key] = tbl
    return tbl


def local_offsets(t, tz: ZoneInfo) -> np.ndarray:
    """UTC offset in seconds for each timestamp, honouring DST transitions."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if t.size == 0:
        return t
    return _offset_table(tz, float(t.min()), float(t.max()))(t)


def local_seconds(t, tz: ZoneInfo) -> np.ndarray:
    """Epoch seconds shifted so that arithmetic works in local wall-clock time."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return t + local_offsets(t, tz)


def local_midnight_epoch(day_index, tz: ZoneInfo) -> np.ndarray:
    """Epoch seconds of local midnight for each local day index.

    Solves ``t + offset(t) = day_index * 86400`` by fixed point; two passes are
    exact except inside the one ambiguous hour of a fall-back transition.
    """
    target = np.asarray(day_index, dtype=float) * SECONDS_PER_DAY
    t = target - local_offsets(target, tz)
    return target - local_offsets(t, tz)


def daily_phase(t, tz: ZoneInfo) -> np.ndarray:
    """Radians since local midnight."""
    ls = local_seconds(t, tz)
    return TWO_PI * (ls % SECONDS_PER_DAY) / SECONDS_PER_DAY


def weekly_phase(t, tz: ZoneInfo) -> np.ndarray:
    """Radians since local Monday 00:00.

    The epoch (1970-01-01) was a Thursday, so the week origin is shifted by four
    days to make Monday the start of the cycle.
    """
    ls = local_seconds(t, tz) + 4 * SECONDS_PER_DAY
    return TWO_PI * (ls % SECONDS_PER_WEEK) / SECONDS_PER_WEEK


def local_dow(t, tz: ZoneInfo) -> np.ndarray:
    ls = local_seconds(t, tz)
    return (((ls // SECONDS_PER_DAY).astype(np.int64) + 3) % 7).astype(np.int64)


def local_day_index(t, tz: ZoneInfo) -> np.ndarray:
    return (local_seconds(t, tz) // SECONDS_PER_DAY).astype(np.int64)


def phase_to_clock(phase: float) -> str:
    """Render a daily phase as ``HH:MM`` for reporting."""
    minutes = int(round((phase % TWO_PI) / TWO_PI * 1440.0)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def clock_to_phase(hh: float, mm: float = 0.0) -> float:
    return TWO_PI * ((hh * 60.0 + mm) % 1440.0) / 1440.0


def phase_to_minutes(phase) -> np.ndarray:
    return np.asarray(phase, dtype=float) / TWO_PI * 1440.0


def local_midnight_after(t: float, tz: ZoneInfo) -> float:
    """First local midnight strictly after ``t``, as epoch seconds."""
    dt = datetime.fromtimestamp(t, tz)
    nxt = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.timestamp()


def utc(*args) -> datetime:
    """Small helper for tests and examples."""
    return datetime(*args, tzinfo=timezone.utc)
