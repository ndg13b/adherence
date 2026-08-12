"""Routine consistency scoring and engagement-timing prediction.

Quick start::

    from adherence import EventLog, RoutineModel, consistency_report

    log = EventLog.from_records(timestamps, tz="Europe/London")
    print(consistency_report(log))

    model = RoutineModel(bandwidth_min=45, half_life_days=28).fit(log)
    print(model.p_engage_next(hours=24))
    start, end, p = model.best_window(window_min=60)
"""

from .events import EventLog, clock_to_phase, daily_phase, phase_to_clock
from .kernels import make_kernel
from .model import RoutineModel
from .scores import (
    Anchor,
    ConsistencyReport,
    Outlook,
    anchor_precision,
    consistency_report,
    find_anchors,
    outlook,
    prequential_timing_bits,
    timing_consistency,
    timing_consistency_track,
    timing_consistency_track_permuted,
    weekday_regularity,
)

__version__ = "0.1.0"

__all__ = [
    "EventLog",
    "RoutineModel",
    "Anchor",
    "ConsistencyReport",
    "Outlook",
    "consistency_report",
    "timing_consistency",
    "timing_consistency_track",
    "timing_consistency_track_permuted",
    "anchor_precision",
    "weekday_regularity",
    "prequential_timing_bits",
    "find_anchors",
    "outlook",
    "make_kernel",
    "daily_phase",
    "phase_to_clock",
    "clock_to_phase",
    "__version__",
]
