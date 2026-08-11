"""Is the "falling consistency" result real, or is it the one in twenty?

    python examples/fitrec_falling.py --self-test          # known answer, ~1 min
    python examples/fitrec_falling.py endomondoHR.json.gz
    python examples/fitrec_falling.py endomondoHR.json.gz --sport bike

WHAT THIS IS FOR

``fitrec_timevarying.py`` fitted roughly seventeen coefficients across two lags,
three model forms and five memory lengths. Sixteen were null. One was not:

    consistency falling   HR 1.166   se 0.062   p 0.014   (lag 30 days)

One result at p<0.05 out of seventeen is what chance produces -- the expected
count under a global null is 0.85, and Bonferroni puts that p at 0.24. So the
honest status of the number is "unexamined", and this script is the examination.
It adds no new specifications to the pile; it takes that single coefficient and
tries four ways to break it.

THE FOUR TESTS

1. **Is it just a falling run rate?** The score is estimated from a
   recency-weighted sample. When someone thins out that sample shrinks, and a
   leave-one-out density estimated from fewer neighbours drifts down on its own.
   Thinning out also precedes quitting. So "consistency fell" may be "they ran
   less", relabelled. The original model adjusted for the *level* of the run
   rate, which is the wrong control for a *change*; the change goes in here.

2. **A permutation null.** Each person's times of day are reordered among their
   own events. Every event time, every count and their overall time-of-day
   distribution survive untouched; the only thing destroyed is which time went
   with which event, and so any trend in the timing. A routine coming apart
   cannot survive that shuffle. A sample-size artefact can. Refitting under many
   shuffles gives the null distribution of the coefficient, and with it two
   things no Wald test can supply: a p-value that assumes nothing, and a check
   on whether the reported standard error is honest -- each person contributes
   about twenty-four correlated intervals, and a model treating them as
   independent will understate its own uncertainty.

3. **Split-half replication.** People split at random, each half fitted alone.
   An effect of the claimed size should land positive in both halves about 90%
   of the time; chance manages 25%.

4. **Does it survive the design choices?** Interval, lag and memory were all set
   by judgement. A real effect is dented by moving them; a fluke is one bright
   cell in the grid.

``--self-test`` runs the whole battery against a simulated cohort whose routines
demonstrably do come apart before they quit. That effect is built into the
generator, so finding it proves nothing about people -- what it proves is that
these four tests can find an effect of this size when one is there, which is the
only thing that makes a null on the real data worth reporting.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Run straight from a clone, with or without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adherence import (  # noqa: E402
    EventLog,
    RoutineModel,
    timing_consistency_track,
    timing_consistency_track_permuted,
)
from adherence.datasets import load_fitrec  # noqa: E402
from adherence.events import SECONDS_PER_DAY  # noqa: E402
from adherence.survival import cox_ph_time_varying  # noqa: E402
from fitrec_timevarying import cessation_time  # noqa: E402

DAY = SECONDS_PER_DAY

#: The one coefficient under examination, exactly as it was originally fitted.
TARGET = {"interval": 30.0, "lag": 30.0, "half_life": 28.0, "warmup": 90.0,
          "bandwidth": 30.0, "quiet": 180.0}


@dataclass
class Person:
    """One person's events, plus everything that does not depend on the settings."""

    log: EventLog
    origin: float
    end_t: float
    stopped: bool
    assess: np.ndarray = field(default_factory=lambda: np.zeros(0))


def prepare(logs, dataset_end: float, quiet_days: float, warmup_days: float,
            interval_days: float) -> list[Person]:
    """Per-person setup: when they stopped, and the grid of assessment moments.

    The grid depends on the interval but not on the lag -- a lag moves the window
    a score is asked to predict, never the moment it is measured -- so one grid
    serves every lag, and the scoring behind it is done once instead of three
    times.
    """
    people = []
    for log in logs.values():
        end_t, stopped = cessation_time(log, dataset_end, quiet_days)
        origin = float(log.t[0])
        first = origin + warmup_days * DAY
        n = int(math.floor(max(end_t - first, 0.0) / (interval_days * DAY))) + 1
        people.append(Person(
            log=log, origin=origin, end_t=end_t, stopped=stopped,
            assess=first + interval_days * DAY * np.arange(max(n, 0)),
        ))
    return people


def score_people(people: list[Person], half_life: float, bandwidth: float,
                 min_events: int) -> list[np.ndarray]:
    """Each person's consistency at each of their assessment moments."""
    model = RoutineModel(bandwidth_min=bandwidth, half_life_days=half_life)
    return [timing_consistency_track(p.log, p.assess, model, min_events=min_events)
            for p in people]


def permuted_scores(people: list[Person], half_life: float, bandwidth: float,
                    min_events: int, n_shuffles: int, seed: int = 1000):
    """``n_shuffles`` alternative score tracks per person, under reordered times.

    Computed person by person because reordering permutes each person's pairwise
    kernel weights rather than changing them, so the expensive part is shared
    across all their shuffles. Returned shuffle-major, which is how the fits want
    it.
    """
    model = RoutineModel(bandwidth_min=bandwidth, half_life_days=half_life)
    per_person = [
        timing_consistency_track_permuted(p.log, p.assess, n_shuffles, model,
                                          min_events=min_events, seed=seed + i)
        for i, p in enumerate(people)
    ]
    return [[per_person[i][r] for i in range(len(people))] for r in range(n_shuffles)]


def assemble(people: list[Person], scores, interval: float, lag: float,
             min_events: int, drop_first: bool = False) -> list[dict]:
    """One row per person per interval, with covariates knowable at its start."""
    rows = []
    for p, sc in zip(people, scores):
        t = p.log.t
        prev_score = prev_rate = None
        for moment, score in zip(p.assess, sc):
            risk_start = moment + lag * DAY
            risk_stop = risk_start + interval * DAY
            if risk_start >= p.end_t:
                break
            k = int(np.searchsorted(t, moment, side="left"))
            if k < min_events or not np.isfinite(score):
                continue

            lo = int(np.searchsorted(t, moment - interval * DAY, side="right"))
            rate = (k - lo) / interval * 7.0  # runs per week, strictly before now
            log_rate = math.log(rate + 0.1)
            first = prev_score is None
            delta = 0.0 if first else score - prev_score
            d_rate = 0.0 if first else log_rate - math.log(prev_rate + 0.1)
            prev_score, prev_rate = score, rate

            if drop_first and first:
                continue  # its change is a structural zero, not a measured one
            stop = min(risk_stop, p.end_t)
            if stop <= risk_start:
                continue
            rows.append({
                "person": id(p), "start": (risk_start - p.origin) / DAY,
                "stop": (stop - p.origin) / DAY,
                "event": 1.0 if (p.stopped and risk_start < p.end_t <= risk_stop) else 0.0,
                "score": score, "delta": delta, "log_rate": log_rate, "d_rate": d_rate,
            })
    return rows


# ------------------------------------------------------------------ model fits
def _z(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


#: name -> (row key, sign). The sign flips each covariate so that larger always
#: means more of the thing that is supposed to precede quitting.
TERMS = {
    "irregularity": ("score", -1.0),
    "consistency falling": ("delta", -1.0),
    "log run rate": ("log_rate", +1.0),
    "run rate falling": ("d_rate", -1.0),
}

TARGET_TERMS = ["irregularity", "consistency falling", "log run rate"]


def fit(rows: list[dict], terms: list[str]):
    cols = [sign * _z(np.array([r[key] for r in rows]))
            for key, sign in (TERMS[t] for t in terms)]
    return cox_ph_time_varying(
        np.column_stack(cols),
        np.array([r["start"] for r in rows]),
        np.array([r["stop"] for r in rows]),
        np.array([r["event"] for r in rows]),
        names=terms,
    )


def target_coef(rows: list[dict]) -> tuple[float, float, float]:
    """(coefficient, se, p) for `consistency falling` in the original model.

    All-``nan`` when the fit did not produce a usable estimate, so that a
    separated or non-converged fit drops out of a summary rather than entering
    it as a spectacular one.
    """
    f = fit(rows, TARGET_TERMS)
    if not f.usable:
        return float("nan"), float("nan"), float("nan")
    i = TARGET_TERMS.index("consistency falling")
    return float(f.coef[i]), float(f.se[i]), float(f.p[i])


def _enough(rows, min_events: int = 20) -> bool:
    return bool(rows) and sum(r["event"] for r in rows) >= min_events


# ---------------------------------------------------------------------- checks
def describe(rows: list[dict], people: list[Person]) -> None:
    """The raw comparison the hazard ratio is a summary of."""
    ev = np.array([r["event"] for r in rows]) > 0
    delta = np.array([r["delta"] for r in rows])
    score = np.array([r["score"] for r in rows])
    print("\n  Behind the hazard ratio, the plain comparison:")
    print(f"    {'':22s} {'quit next':>11s} {'kept going':>11s}")
    for label, v in (("median change in score", delta),
                     ("median score", score)):
        print(f"    {label:22s} {np.median(v[ev]):11.4f} {np.median(v[~ev]):11.4f}")
    print(f"    {'intervals':22s} {int(ev.sum()):11d} {int((~ev).sum()):11d}")

    # Disengagement is inferred from silence, so how much of the cohort ends up
    # classed as stopped says how much work the censoring is doing.
    stopped = sum(p.stopped for p in people)
    share = stopped / max(len(people), 1)
    print(f"\n  {stopped:,} of {len(people):,} people ({share:.0%}) are classed as "
          f"having stopped.")
    if share > 0.95:
        print("  Almost nobody is censored, which means 'disengaged' has collapsed")
        print("  into 'their data ended before the dataset did'. The comparison is")
        print("  then between people who left early and people who left late, not")
        print("  between leavers and stayers. Read any result here accordingly.")
    elif share < 0.05:
        print("  Almost nobody stopped, so there is very little outcome to predict.")


def check_rate_change(rows) -> float:
    print("\n" + "=" * 74)
    print("1. IS IT JUST A FALLING RUN RATE?")
    print("=" * 74)
    print("  The score is estimated from a shrinking sample when someone thins out,")
    print("  and thinning out precedes quitting. Adjusting for the *level* of the")
    print("  rate does not control a *change* in it.\n")
    kept = float("nan")
    for label, terms in (
        ("as originally fitted", TARGET_TERMS),
        ("adding the change in run rate", TARGET_TERMS + ["run rate falling"]),
    ):
        f = fit(rows, terms)
        print(f"  {label}")
        print("    " + str(f).replace("\n", "\n    ") + "\n")
        kept = float(f.coef[TARGET_TERMS.index("consistency falling")])
    return kept


def check_permutation(people, rows, args) -> float:
    print("\n" + "=" * 74)
    print("2. PERMUTATION NULL")
    print("=" * 74)
    print("  Each person's times of day reordered among their own events. Event")
    print("  times, counts and the marginal time-of-day distribution are held")
    print(f"  exactly; only the ordering goes. {args.shuffles} refits.\n")

    obs, _, obs_p = target_coef(rows)
    t0 = time.time()
    tracks = permuted_scores(people, TARGET["half_life"], TARGET["bandwidth"],
                             args.min_events, args.shuffles)
    print(f"  scored {len(people):,} people x {args.shuffles} shuffles "
          f"in {time.time() - t0:.0f}s")

    null, null_se = [], []
    for sc in tracks:
        r = assemble(people, sc, TARGET["interval"], TARGET["lag"], args.min_events)
        if not _enough(r):
            continue
        c, se, _ = target_coef(r)
        null.append(c)
        null_se.append(se)
    if len(null) < 10:
        print("  Too few usable shuffles to form a null.")
        return float("nan"), 0.0

    null = np.array(null)
    reported = float(np.mean(null_se))
    honest = float(null.std(ddof=1))
    # Two-sided and counting the observed value, so the p-value can never be 0.
    p_perm = (1 + int((np.abs(null - null.mean()) >= abs(obs - null.mean())).sum())) \
        / (len(null) + 1)

    print(f"\n  observed coefficient      {obs:+.4f}   (HR {math.exp(obs):.3f})")
    print(f"  null mean                 {null.mean():+.4f}")
    print(f"  null SD                   {honest:.4f}   <- the honest standard error")
    print(f"  model's own se            {reported:.4f}   <- what it reported")
    print(f"\n  permutation p             {p_perm:.3f}   ({len(null)} shuffles)")
    print(f"  Wald p                    {obs_p:.3f}")

    ratio = honest / max(reported, 1e-12)
    if ratio > 1.15:
        print(f"\n  The model understates its own uncertainty by {ratio:.2f}x. Each person")
        print("  contributes many correlated intervals and the fit treats them as")
        print("  independent, so every Wald p-value in the time-varying analysis is")
        print("  optimistic. Trust the permutation p, and use a cluster-robust")
        print("  variance for anything written up.")
    elif ratio < 0.87:
        print(f"\n  The model overstates its uncertainty by {1 / ratio:.2f}x -- conservative,")
        print("  so its p-values can be read as upper bounds.")
    else:
        print("\n  The reported standard error matches the permutation spread, so")
        print("  clustering is not inflating significance here.")

    centre = float(null.mean())
    if abs(centre) > 0.3 * honest:
        print(f"\n  The null does not sit at zero: shuffled data still gives {centre:+.4f}.")
        print("  So the covariate carries a built-in association with disengagement")
        print("  that has nothing to do with timing. A leave-one-out density is")
        print("  estimated from a recency-weighted sample; as that sample thins the")
        print("  estimate degrades, and thinning precedes quitting -- so the score")
        print("  drifts down before someone stops even when their times are random.")
        print(f"  The effect to explain is {obs - centre:+.4f}, not {obs:+.4f}.")
    return p_perm, centre


def check_split_half(people, args, centre: float = 0.0) -> float:
    print("\n" + "=" * 74)
    print("3. SPLIT-HALF REPLICATION")
    print("=" * 74)
    print("  People split at random, each half fitted alone. An effect of the")
    print("  claimed size lands beyond the null in both halves ~90% of the time;")
    print("  chance manages 25%.")
    if centre:
        # Counting "both positive" against a null that is not centred on zero
        # would score the artefact as a replication.
        print(f"\n  Judged against the permutation null's centre ({centre:+.4f}), not")
        print("  against zero -- otherwise the bias found above counts as a pass.")
    print()

    sc = score_people(people, TARGET["half_life"], TARGET["bandwidth"], args.min_events)
    both_beyond = both_sig = usable = 0
    print(f"  {'split':>6s} {'half A':>18s} {'half B':>18s}")
    for s in range(args.splits):
        rng = np.random.default_rng(2000 + s)
        mask = rng.random(len(people)) < 0.5
        pair = []
        for want in (True, False):
            idx = [i for i in range(len(people)) if mask[i] == want]
            r = assemble([people[i] for i in idx], [sc[i] for i in idx],
                         TARGET["interval"], TARGET["lag"], args.min_events)
            if not _enough(r, 15):
                pair = []
                break
            pair.append(target_coef(r))
        if len(pair) != 2 or not all(np.isfinite(c) for c, _, _ in pair):
            continue
        usable += 1
        both_beyond += pair[0][0] > centre and pair[1][0] > centre
        both_sig += pair[0][2] < 0.05 and pair[1][2] < 0.05
        if s < 8:
            print(f"  {s:6d} " + " ".join(f"{c:+8.3f} (p{p:5.3f})" for c, _, p in pair))
    if not usable:
        print("  No usable splits -- too few events per half.")
        return float("nan")
    label = "both halves beyond null" if centre else "both halves positive"
    print(f"\n  {label:26s} {both_beyond}/{usable} ({both_beyond / usable:.0%})")
    print(f"  {'both halves p<0.05':26s} {both_sig}/{usable} ({both_sig / usable:.0%})")
    if both_beyond / usable > 0.7 and both_sig == 0:
        print("\n  Consistent in direction but never significant in a half. Half a")
        print("  cohort is genuinely underpowered for an effect this small, so this")
        print("  is weak support at best -- it rules out an effect carried by a few")
        print("  people, and nothing more.")
    return both_beyond / usable


def check_specifications(logs, dataset_end, args) -> None:
    print("\n" + "=" * 74)
    print("4. DOES IT SURVIVE THE DESIGN CHOICES?")
    print("=" * 74)
    print("  Interval, lag and memory were judgement calls. The starred row is the")
    print("  original. A real effect is dented by moving them; a fluke is one")
    print("  bright cell.\n")
    print(f"  {'interval':>9s} {'lag':>5s} {'memory':>7s} {'rows':>7s} {'events':>7s} "
          f"{'log HR':>8s} {'se':>7s} {'p':>7s}")

    for interval in (20.0, 30.0, 45.0):
        people = prepare(logs, dataset_end, TARGET["quiet"], TARGET["warmup"], interval)
        for half_life in (14.0, 28.0, 90.0):
            sc = score_people(people, half_life, TARGET["bandwidth"], args.min_events)
            for lag in (0.0, 30.0, 60.0):
                r = assemble(people, sc, interval, lag, args.min_events)
                ne = sum(q["event"] for q in r)
                star = "*" if (interval, lag, half_life) == (
                    TARGET["interval"], TARGET["lag"], TARGET["half_life"]) else " "
                head = f" {star}{interval:8.0f} {lag:5.0f} {half_life:6.0f}d {len(r):7d} {ne:7.0f}"
                if not _enough(r):
                    print(f"{head}    (too few)")
                    continue
                c, se, p = target_coef(r)
                print(f"{head} {c:+8.3f} {se:7.3f} {p:7.3f}")

    people = prepare(logs, dataset_end, TARGET["quiet"], TARGET["warmup"],
                     TARGET["interval"])
    sc = score_people(people, TARGET["half_life"], TARGET["bandwidth"], args.min_events)
    r = assemble(people, sc, TARGET["interval"], TARGET["lag"], args.min_events,
                 drop_first=True)
    if _enough(r):
        c, se, p = target_coef(r)
        print("\n  Dropping each person's first interval, where the change is a")
        print(f"  structural zero rather than a measured one: {c:+.3f} "
              f"(se {se:.3f}, p {p:.3f})")


def verdict(adjusted, p_perm, replication, sport: str, self_test: bool) -> None:
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    marks = []
    if np.isfinite(adjusted):
        marks.append(("survives the change in run rate", adjusted > 0.05))
    if np.isfinite(p_perm):
        marks.append(("permutation p below 0.05", p_perm < 0.05))
    if np.isfinite(replication):
        marks.append(("holds beyond the null in both halves >70% of splits",
                      replication > 0.7))
    for label, ok in marks:
        print(f"    [{'PASS' if ok else 'FAIL'}]  {label}")

    passed = [ok for _, ok in marks]
    if marks and all(passed):
        print("\n  Survives every check run here. That makes it worth pursuing, not")
        print("  worth believing: it is still one coefficient in one cohort, on an")
        print("  outcome (silence in a workout log) that is inferred rather than")
        print("  observed.")
    elif marks and not any(passed):
        print("\n  Fails every check. Nothing here needs a subtler reading.")
    elif marks:
        print("\n  It does not survive. Passing some checks and failing others is not")
        print("  a partial result -- a coefficient has to clear all of them to be")
        print("  worth a second cohort, and the checks it failed are the ones that")
        print("  assume least.")

    if self_test:
        print("\n  This was the self-test: the effect was built into the generator, so")
        print("  passing shows the checks have power, and nothing about people.")
        return

    print("\n  Either way one thing was settled before this ran: with seventeen")
    print("  coefficients fitted, a single p of 0.014 is not evidence. Bonferroni")
    print("  puts it at 0.24. It earned a second look because it was the *lagged*")
    print("  estimate and larger than the unlagged one -- the opposite of what")
    print("  reverse causation produces -- not because it cleared a threshold.")

    other = "bike" if sport == "run" else "run"
    print(f"\n  Independent check: rerun with --sport {other}. A cohort that has not")
    print("  been fitted is the only place a result like this can be confirmed")
    print("  rather than defended.")


# ------------------------------------------------------------------- self-test
#: How long before quitting a decaying routine starts to loosen, in days. It has
#: to exceed lag + interval, or the fall happens inside the window it is meant
#: to predict and the design is measuring the slowdown rather than forecasting
#: it. It also must not exceed it by much: with a 28-day memory a routine that
#: loosened six months ago has already bottomed out, its score is flat again by
#: the time it quits, and the steepest falls land in intervals where nobody
#: quits -- which shows up as an effect in the *wrong direction*.
DECAY_LEAD_DAYS = 90.0


def synthetic_cohort(n_people: int = 400, seed: int = 0, span_days: int = 900):
    """A cohort whose routines mostly come apart before they quit, by construction.

    Four kinds of person, in deliberately messy proportions:

    - 40% loosen over ``DECAY_LEAD_DAYS`` and then stop -- the effect itself;
    - 15% loosen and then carry on anyway -- a routine coming apart is a warning,
      not a verdict;
    - 15% stop abruptly out of a tight routine -- injury, a new job, a move;
    - 30% hold their routine to the end of the data and are censored.

    The last three exist to stop the covariate predicting the outcome perfectly.
    A clean simulation is separable: the partial likelihood then has no finite
    maximum, and the fit reports an enormous coefficient with p=0 that means
    nothing. Real cohorts are never that tidy, and a self-test that was would be
    demonstrating a capability the analysis does not have.

    Nobody's *rate* of engagement changes, only their timing -- so the run-rate
    check has something to correctly find nothing in.

    The link here is built into the generator. Recovering it says nothing about
    people; it says these checks can find an effect of this size when one is
    present, which is what makes a null on real data worth reporting.
    """
    from datetime import datetime, timezone

    base = datetime(2014, 1, 6, tzinfo=timezone.utc).timestamp()
    rng = np.random.default_rng(seed)
    logs = {}
    for u in range(n_people):
        roll = rng.random()
        decays = roll < 0.55
        # Quit early enough to leave silence the loader will read as stopping;
        # anyone still going at the end of the span is censored instead.
        quits = roll < 0.40 or 0.55 <= roll < 0.70
        last_day = float(rng.integers(400, span_days - 250)) if quits else span_days
        loosen_from = last_day - DECAY_LEAD_DAYS if quits else rng.uniform(300, 600)
        depth = float(rng.uniform(120.0, 300.0))  # how far the routine loosens

        hour = float(rng.uniform(6, 20))
        ts = []
        for d in range(int(last_day)):
            if rng.random() > 0.55:  # engages on ~55% of days, throughout
                continue
            jitter = 20.0
            if decays and d >= loosen_from:
                jitter += depth * min((d - loosen_from) / DECAY_LEAD_DAYS, 1.0)
            ts.append(base + d * DAY + hour * 3600 + float(rng.normal(0, jitter * 60)))
        if len(ts) < 40:
            continue
        t = np.array(sorted(ts))
        logs[u] = EventLog.from_records(t, t_start=t[0], t_end=t[-1])
    return logs, base + span_days * DAY


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="endomondoHR.json.gz")
    p.add_argument("--self-test", action="store_true",
                   help="run the battery on a simulated cohort with a known, "
                        "built-in effect, to show the checks have power")
    p.add_argument("--sport", default="run",
                   help="'run' reproduces the original result; 'bike' is an "
                        "untouched cohort and the better test (default run)")
    p.add_argument("--shuffles", type=int, default=200,
                   help="permutation refits (default 200)")
    p.add_argument("--splits", type=int, default=40,
                   help="random split-half replications (default 40)")
    p.add_argument("--min-events", type=int, default=10)
    p.add_argument("--min-days", type=float, default=120.0)
    p.add_argument("--skip", default="",
                   help="comma-separated checks to skip: rate,permutation,split,grid")
    args = p.parse_args(argv)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    t0 = time.time()
    if args.self_test:
        print("SELF-TEST: a simulated cohort in which consistency genuinely decays")
        print("before people quit. The effect is built into the generator, so the")
        print("checks below should all PASS. Nothing here is real data.\n")
        logs, dataset_end = synthetic_cohort()
        print(f"  {len(logs):,} simulated people")
    elif not args.file:
        p.error("provide a data file, or pass --self-test")
    elif not os.path.exists(args.file):
        print(f"No such file: {args.file}")
        return 2
    else:
        print(f"Loading {args.file} (sport={args.sport})")
        res = load_fitrec(args.file, sport=args.sport, min_events=20,
                          min_days=args.min_days)
        print("\n" + res.summary())
        logs = res.logs
        dataset_end = max(v.t[-1] for v in logs.values())

    people = prepare(logs, dataset_end, TARGET["quiet"], TARGET["warmup"],
                     TARGET["interval"])
    scores = score_people(people, TARGET["half_life"], TARGET["bandwidth"],
                          args.min_events)
    rows = assemble(people, scores, TARGET["interval"], TARGET["lag"], args.min_events)
    n_ev = sum(r["event"] for r in rows)
    print(f"\nTarget specification: interval {TARGET['interval']:.0f}d, "
          f"lag {TARGET['lag']:.0f}d, memory {TARGET['half_life']:.0f}d")
    print(f"  {len(rows):,} intervals from {len({r['person'] for r in rows}):,} people, "
          f"{n_ev:.0f} disengagements")
    if not _enough(rows):
        print("  Too few events to fit anything.")
        return 1

    c, se, pv = target_coef(rows)
    print(f"  consistency falling: {c:+.4f} (HR {math.exp(c):.3f}, se {se:.3f}, "
          f"p {pv:.3f})")
    describe(rows, people)

    adjusted = check_rate_change(rows) if "rate" not in skip else float("nan")
    p_perm, centre = check_permutation(people, rows, args) \
        if "permutation" not in skip else (float("nan"), 0.0)
    replication = check_split_half(people, args, centre) if "split" not in skip \
        else float("nan")
    if "grid" not in skip:
        check_specifications(logs, dataset_end, args)
    verdict(adjusted, p_perm, replication, args.sport, args.self_test)

    print(f"\nTotal time {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
