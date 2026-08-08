"""Synthetic engagement histories with known ground truth.

Two very different uses, and it matters not to confuse them:

1. **Measurement validation.** Generate a person whose true jitter is 15 minutes
   and check the score recovers 15 minutes, that it separates a clockwork user
   from a chaotic one, and that it does so from two weeks of data. This is a
   real test and :mod:`tests` relies on it.

2. **Power analysis.** :func:`simulate_linked_cohort` builds in a dependence
   between consistency and dropout, then asks whether an analysis could detect
   an effect of that size given ``n`` participants and ``k`` weeks of run-in.
   Recovering the effect here is *not* evidence that the effect exists -- it was
   assumed. The value is the sample-size answer, which is otherwise guesswork.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from .events import EventLog

MINUTES_PER_DAY = 1440.0


@dataclass
class Slot:
    """A recurring intention: "around 07:30 on weekdays, most of the time"."""

    hour: float
    jitter_min: float = 20.0
    days: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    p: float = 0.9  #: probability the slot is actually used on an eligible day
    drift_min_per_week: float = 0.0


@dataclass
class PersonProfile:
    slots: list[Slot]
    off_schedule_per_day: float = 0.0  #: rate of extra, unplanned sessions
    off_schedule_hours: tuple[float, float] = (7.0, 23.0)
    dropout_hazard: float = 0.0  #: daily probability of quitting permanently
    name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


#: Recognisable archetypes, useful as fixtures and as narrative examples.
PRESETS: dict[str, PersonProfile] = {
    # The routine the literature says survives: same slot, small jitter.
    "clockwork": PersonProfile(
        slots=[Slot(hour=7.5, jitter_min=8.0, p=0.95)],
        dropout_hazard=0.0005,
        name="clockwork",
    ),
    # Same intention, much looser execution.
    "loose": PersonProfile(
        slots=[Slot(hour=19.0, jitter_min=75.0, p=0.7)],
        off_schedule_per_day=0.05,
        dropout_hazard=0.004,
        name="loose",
    ),
    # No slot at all: engagement scattered through the waking day.
    "chaotic": PersonProfile(
        slots=[],
        off_schedule_per_day=0.9,
        off_schedule_hours=(7.0, 24.0),
        dropout_hazard=0.012,
        name="chaotic",
    ),
    # Twice a day, both slots tight -- regular, but invisible to a mean-time metric.
    "bimodal": PersonProfile(
        slots=[
            Slot(hour=7.0, jitter_min=12.0, p=0.9),
            Slot(hour=21.0, jitter_min=18.0, p=0.8),
        ],
        dropout_hazard=0.001,
        name="bimodal",
    ),
    # Tight on weekdays, late and loose at weekends: the day-of-week case.
    "weekend_shifter": PersonProfile(
        slots=[
            Slot(hour=6.75, jitter_min=12.0, days=(0, 1, 2, 3, 4), p=0.92),
            Slot(hour=10.5, jitter_min=70.0, days=(5, 6), p=0.6),
        ],
        dropout_hazard=0.002,
        name="weekend_shifter",
    ),
    # Mon/Wed/Fri only: low rate, high regularity. A pure-rate metric misreads this.
    "mwf": PersonProfile(
        slots=[Slot(hour=17.5, jitter_min=15.0, days=(0, 2, 4), p=0.95)],
        dropout_hazard=0.001,
        name="mwf",
    ),
    # Tight cluster that migrates ~25 min later each week -- consistent today,
    # unpredictable from last month. The drift/jitter split exists for this case.
    "drifter": PersonProfile(
        slots=[Slot(hour=8.0, jitter_min=15.0, p=0.9, drift_min_per_week=25.0)],
        dropout_hazard=0.003,
        name="drifter",
    ),
}


def simulate_person(
    profile: PersonProfile,
    start: datetime,
    days: int = 90,
    tz: str | ZoneInfo = "UTC",
    rng: np.random.Generator | int | None = None,
    dropout: bool = True,
) -> tuple[EventLog, dict]:
    """Generate one person's history. Returns the log and the ground truth.

    Set ``dropout=False`` to keep the person engaged for the full window, which
    is what you want when testing whether a score recovers a known timing
    parameter: otherwise a profile with a high hazard yields three events and
    the test measures the sample size, not the score.
    """
    rng = np.random.default_rng(rng)
    hazard = profile.dropout_hazard if dropout else 0.0
    tzinfo = ZoneInfo(tz) if isinstance(tz, str) else tz
    if start.tzinfo is None:
        start = start.replace(tzinfo=tzinfo)
    start_midnight = start.astimezone(tzinfo).replace(hour=0, minute=0, second=0, microsecond=0)

    times: list[float] = []
    dropout_day = None
    for d in range(days):
        if hazard > 0 and rng.random() < hazard:
            dropout_day = d
            break
        day0 = start_midnight + timedelta(days=d)
        dow = day0.weekday()

        for slot in profile.slots:
            if dow not in slot.days or rng.random() > slot.p:
                continue
            minute = slot.hour * 60.0
            minute += slot.drift_min_per_week * d / 7.0
            minute += rng.normal(0.0, slot.jitter_min)
            times.append(_epoch(day0, minute, tzinfo))

        if profile.off_schedule_per_day > 0:
            for _ in range(rng.poisson(profile.off_schedule_per_day)):
                lo, hi = profile.off_schedule_hours
                times.append(_epoch(day0, rng.uniform(lo, hi) * 60.0, tzinfo))

    t_start = start_midnight.timestamp()
    t_end = (start_midnight + timedelta(days=days)).timestamp()
    log = EventLog.from_records(
        sorted(times), tz=tzinfo, t_start=t_start, t_end=t_end, meta={"profile": profile.name}
    )
    truth = {
        "name": profile.name,
        "dropout_day": dropout_day,
        "survived_days": days if dropout_day is None else dropout_day,
        "n_slots": len(profile.slots),
        "mean_jitter_min": float(np.mean([s.jitter_min for s in profile.slots]))
        if profile.slots
        else float("nan"),
        "off_schedule_per_day": profile.off_schedule_per_day,
    }
    return log, truth


def _epoch(day0: datetime, minute_of_day: float, tzinfo: ZoneInfo) -> float:
    """Local wall-clock minute-of-day to epoch seconds, wrapping past midnight."""
    return (day0 + timedelta(minutes=float(minute_of_day))).timestamp()


def simulate_cohort(
    profiles: list[PersonProfile] | None = None,
    n_per_profile: int = 20,
    start: datetime | None = None,
    days: int = 90,
    tz: str | ZoneInfo = "UTC",
    rng: np.random.Generator | int | None = None,
) -> list[tuple[EventLog, dict]]:
    """A mixed cohort drawn from the presets, for end-to-end smoke testing."""
    rng = np.random.default_rng(rng)
    profiles = profiles if profiles is not None else list(PRESETS.values())
    start = start or datetime(2025, 1, 6)  # a Monday
    out = []
    for prof in profiles:
        for _ in range(n_per_profile):
            out.append(simulate_person(prof, start, days=days, tz=tz, rng=rng))
    return out


def simulate_linked_cohort(
    n: int = 300,
    days: int = 180,
    log_hazard_ratio: float = 0.8,
    jitter_lognorm: tuple[float, float] = (math.log(30.0), 0.8),
    base_hazard: float = 0.01,
    rate: float = 0.85,
    start: datetime | None = None,
    tz: str | ZoneInfo = "UTC",
    rng: np.random.Generator | int | None = None,
) -> list[tuple[EventLog, dict]]:
    """Cohort in which regularity *causes* retention, by construction.

    Each person gets a true jitter drawn from a log-normal; their daily dropout
    hazard is ``base_hazard * exp(log_hazard_ratio * z)``, where ``z`` is their
    standardised log-jitter -- that is, their irregularity in SD units. So
    ``log_hazard_ratio`` is the log hazard ratio per SD of irregularity, and the
    positive default means an irregular person quits sooner.

    The dependence is assumed, not discovered. What this supports is a power
    calculation: given an effect of this size, how many participants and how
    many run-in weeks does it take before the score -- estimated from short,
    noisy histories -- detects it?
    """
    rng = np.random.default_rng(rng)
    start = start or datetime(2025, 1, 6)
    mu, sigma = jitter_lognorm

    jitters = rng.lognormal(mu, sigma, size=n)
    z = (np.log(jitters) - mu) / sigma
    hazards = base_hazard * np.exp(log_hazard_ratio * z)
    hours = rng.uniform(6.0, 21.0, size=n)

    out = []
    for i in range(n):
        prof = PersonProfile(
            slots=[Slot(hour=float(hours[i]), jitter_min=float(jitters[i]), p=rate)],
            dropout_hazard=float(np.clip(hazards[i], 0.0, 0.5)),
            name=f"p{i:04d}",
        )
        log, truth = simulate_person(prof, start, days=days, tz=tz, rng=rng)
        truth.update(true_jitter_min=float(jitters[i]), true_hazard=float(hazards[i]))
        out.append((log, truth))
    return out
