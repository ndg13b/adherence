"""Choosing the bandwidth and half-life from the data.

The tolerance width is not a nuisance parameter to be guessed -- it is a
substantive claim about the person. Eight minutes of jitter and ninety minutes
of jitter are different routines, and a fixed 45-minute kernel over-smooths the
first while under-smoothing the second. Selection here maximises *prequential*
timing bits: each candidate is judged by how well it predicts events it has not
seen, so there is no leakage and no circularity.

Log-densities are comparable across bandwidths because every candidate is scored
against the same reference measure (uniform on the 24-hour circle), which is the
reason the bits formulation makes this selection well posed at all -- a
normalised ``[0, 1]`` score would not, since its ceiling moves with the
bandwidth.

**This creates a tension worth stating plainly.** A per-person bandwidth gives
the best forecast, but it makes the normalised consistency scores incomparable
between people, because each person's ceiling is then different. So:

* comparing people, or using consistency as a study variable -> fix the
  bandwidth for the whole cohort (:data:`DEFAULT_BANDWIDTH_MIN`), and report it;
* forecasting for one person -> select per person.

Do not mix the two and then rank participants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import EventLog
from .model import RoutineModel
from .scores import _model_params, prequential_timing_bits

#: Cohort-comparable default. 45 minutes matches the Social Rhythm Metric window,
#: which makes the score commensurable with that literature.
DEFAULT_BANDWIDTH_MIN = 45.0

BANDWIDTH_GRID = (5.0, 8.0, 12.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0)
HALF_LIFE_GRID = (7.0, 14.0, 28.0, 56.0, 112.0)


@dataclass
class Selection:
    """Result of a hyper-parameter search."""

    best: dict
    table: list = field(default_factory=list)
    criterion: str = "prequential_timing_bits"

    def __str__(self) -> str:
        rows = "\n".join(
            f"    {r['params']}  {r['bits']:+.3f} bits/event (n={r['n_scored']})"
            for r in self.table
        )
        return f"best: {self.best}\n{rows}"


def select_bandwidth(
    log: EventLog,
    candidates=BANDWIDTH_GRID,
    model: RoutineModel | None = None,
    stride: int = 1,
) -> Selection:
    """Pick the timing tolerance that best predicts this person's next session."""
    base = _model_params(model or RoutineModel())
    table = []
    for bw in candidates:
        params = {**base, "bandwidth_min": bw}
        bits, n = prequential_timing_bits(log, RoutineModel(**params), stride=stride)
        table.append({"params": {"bandwidth_min": bw}, "bits": bits, "n_scored": n})
    return _best(table, "bandwidth_min")


def select_half_life(
    log: EventLog,
    candidates=HALF_LIFE_GRID,
    model: RoutineModel | None = None,
    stride: int = 1,
) -> Selection:
    """Pick the recency half-life.

    A short half-life tracks a moving routine; a long one is stable but stale.
    Which is right is an empirical question about this person, and the answer is
    itself informative: a person best predicted by a 7-day memory is a person
    whose routine is still in flux.
    """
    base = _model_params(model or RoutineModel())
    table = []
    for hl in candidates:
        params = {**base, "half_life_days": hl}
        bits, n = prequential_timing_bits(log, RoutineModel(**params), stride=stride)
        table.append({"params": {"half_life_days": hl}, "bits": bits, "n_scored": n})
    return _best(table, "half_life_days")


def auto_model(
    log: EventLog,
    bandwidths=BANDWIDTH_GRID,
    half_lives=HALF_LIFE_GRID,
    stride: int = 1,
    fit: bool = True,
) -> tuple[RoutineModel, Selection]:
    """Joint search over bandwidth and half-life, then fit.

    Coordinate-wise rather than a full grid: bandwidth first (it matters more),
    then half-life given that bandwidth. Two passes over a small grid is enough
    when the surface is as smooth as this one, and it keeps selection affordable
    inside a per-participant loop.
    """
    bw_sel = select_bandwidth(log, bandwidths, stride=stride)
    bw = bw_sel.best.get("bandwidth_min", DEFAULT_BANDWIDTH_MIN)
    base = RoutineModel(bandwidth_min=bw)
    hl_sel = select_half_life(log, half_lives, model=base, stride=stride)
    hl = hl_sel.best.get("half_life_days", 28.0)

    chosen = {"bandwidth_min": bw, "half_life_days": hl}
    table = bw_sel.table + hl_sel.table
    model = RoutineModel(**chosen)
    if fit:
        model.fit(log)
    return model, Selection(best=chosen, table=table)


def _best(table: list, key: str) -> Selection:
    valid = [r for r in table if np.isfinite(r["bits"])]
    if not valid:
        return Selection(best={}, table=table)
    winner = max(valid, key=lambda r: r["bits"])
    return Selection(best=dict(winner["params"]), table=table)
