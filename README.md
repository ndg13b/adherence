# adherence

Scoring how *regular* someone's engagement is, and turning that into a forecast
of when they will engage next.

The premise: people who engage at the same time of day, on the same days, keep
doing it. Regularity of schedule — not just volume of engagement — predicts
whether a behaviour survives. This package makes that measurable.

Everything is applied to engagement events, so it works the same for a cognitive
training programme, medication timing, exercise, or a habit someone is trying to
break.

```python
from adherence import EventLog, RoutineModel, consistency_report

log = EventLog.from_records(timestamps, tz="Europe/London")
print(consistency_report(log))
```

```
114 events over 119 days (0.93/day, active on 96% of days)
  timing consistency   0.961   (0 = times spread uniformly, 1 = same minute every time)
  anchor precision     0.961   (tightness around each slot, ignoring how many slots)
  weekday regularity   0.006
  out-of-sample timing +3.59 bits/event (normalised 0.987)
  typical jitter       +/-9 min, drift +0.1 min/week
  anchors:
    - 07:30 (+/-9 min, 100% of sessions, drift +0.1 min/week)
```

And the forecast the score is derived from:

```python
model = RoutineModel(bandwidth_min=45, half_life_days=28).fit(log)

model.p_engage_next(hours=24)           # 0.59
model.expected_events(t0, t0 + 7*86400) # 6.5 sessions expected this week
model.best_window(window_min=60)        # (07:21, 08:21, p=0.33) -- when to nudge
```

## Install

```bash
pip install -e .           # numpy + scipy
pip install -e ".[dev]"    # + pytest
pytest                     # 156 tests
```

## What it measures

Four numbers, kept separate because they answer different questions and
collapsing them hides the one you care about.

| | question | scale |
|---|---|---|
| `timing_consistency` | how predictable is the *time* of the next session? | 0 = uniform, 1 = same minute |
| `anchor_precision` | how tight is each slot, ignoring how many slots? | 0–1 |
| `weekday_regularity` | does the day of week tell you whether they engage? | 0–1 |
| `timing_bits` | out-of-sample bits per event vs. a chance forecast | negative = history is misleading |

Plus diagnostics: the recurring **anchors** ("07:28 ±7 min, 100% of sessions"),
day-to-day **jitter**, and schedule **drift** in minutes per week.

Two distinctions the package exists to make:

**Two tight slots ≠ one vague habit.** A person who trains at 07:00 and 21:00 is
genuinely less predictable per session — it is a coin flip which slot comes next
— but they are not less habitual. `timing_consistency` reports the first,
`anchor_precision` the second. Circular concentration (mean resultant length),
the textbook measure, collapses to ~0 for this person and calls them chaotic.

**Drifting ≠ dissolving.** A routine sliding 25 minutes later each week is tight
day to day. Every variance-based index charges the slide to noise. Here:

```
drifter          timing 0.289   jitter 13 min   drift +24.8 min/week
loose            timing 0.340   jitter 68 min   drift  +3.5 min/week
```

Same consistency score, opposite problems, different interventions.

## How it works

An inhomogeneous Poisson process whose intensity is periodic in wall-clock time:

```
lambda(t) = mu[day_of_week] * f(time_of_day) * (2*pi / 86400)
```

- `f` is a circular kernel density over past engagement times — each event
  contributes a bump with tails on either side, which is the "engagement tail"
  idea made normalisable. Bandwidth = the person's timing tolerance.
- `mu` is a weekday-specific rate, shrunk toward the pooled rate so one missed
  Tuesday does not convince the model Tuesdays are dead.
- Both are weighted by recency (exponential half-life), so the model follows a
  routine that moves.
- A uniform floor keeps a single 3 a.m. session from dominating the score.

Then `P(engage in [a,b]) = 1 - exp(-∫lambda)`.

Kernels: von Mises (default), wrapped normal, or compactly-supported
Epanechnikov/tricube if you want the tails to actually terminate. Asymmetric
tails are supported — `bandwidth_min=(10, 60)` gives a hard front edge and a
long back tail, which is what an evening routine tends to look like.

The scoring is **prequential**: for each event, the model is refit on strictly
earlier events and asked to predict it. Nothing is ever scored against data it
has seen. The test suite includes an explicit leakage test — two histories
identical up to day 60 and wildly different afterwards must produce
bit-identical forecasts at every origin up to day 60.

## Does it beat what already exists?

The Social Rhythm Metric (±45 min window), Interdaily Stability, and the Sleep
Regularity Index are implemented in `adherence/baselines.py` for direct
comparison. `python examples/discrimination.py` runs the sweep.

Averaged over enough people, every index orders groups correctly. The difference
is resolution for a *single person*. Separating two individuals one step apart
in jitter, below 20 minutes (Cohen's *d*):

| index | mean *d* |
|---|---|
| SRM ±45 min | 0.77 |
| Sleep Regularity Index | 0.68 |
| Interdaily Stability | 0.06 |
| hour-of-day entropy | 0.15 |
| **kernel score** | **4.46** |
| **prequential bits** | **6.50** |

Below ~20 minutes of jitter the established indices are at their resolution
limit — every session lands in the same box or the same epoch either way. That
band is where a habitual user separates from a merely willing one. Above 60
minutes every index agrees, because by then the routine has visibly gone.

Against forecasting baselines, on a simulated clockwork user (log-loss skill vs.
a rate-matched homogeneous Poisson):

```
routine (auto bandwidth)   +0.686
routine (fixed 45 min)     +0.520
hour-of-day histogram      +0.597
"same time as last"        +0.463
```

On a person with no routine at all, the model scores +0.03 — near zero, as it
must, or the metric would be flattering itself.

## Study design

`examples/cohort_power.py` answers the question that has to be settled before
collecting data: how long a run-in window, and how many participants?

```
Power at alpha=0.05, 28-day run-in, HR 1.4 per SD of irregularity:
 n enrolled   power
         50     37%
        100     70%
        200     90%
        400    100%
```

The cohort generator **builds in** the link between irregularity and dropout, so
recovering it is not evidence the effect is real — it was assumed. What is being
measured is the gap between the true effect and the detectable one, and that gap
is real: a score estimated from a short window is a noisy predictor, and noise
attenuates its coefficient. Powering a study on the true effect size leaves it
underpowered for the effect you can actually measure.

A minimal Cox proportional-hazards fit (Efron ties, validated against known
hazard ratios in `tests/test_survival.py`) is included so the example runs
without extra dependencies. For a real analysis, refit in `lifelines` or R.

## Validating on real data

Everything above is simulation. `examples/duolingo_check.py` runs the first real
check against the public [Duolingo learning traces](https://github.com/duolingo/halflife-regression)
(13M rows, no approval needed):

```bash
python examples/duolingo_check.py --self-test              # dry run, ~5 seconds
python examples/duolingo_check.py learning_traces.13m.csv.gz
```

The examples add `src/` to the path themselves, so they run from a fresh clone
whether or not you have run `pip install -e .`. Accepts `.csv`, `.csv.gz` or
`.zip` — no need to unpack. Streaming the full file takes under a minute;
scoring adds a minute or two.

It answers one question: **do people genuinely differ in timing consistency, or
is the apparent spread just noise?** That distinction is the whole ballgame, and
"look, the scores differ" cannot settle it — with two weeks of history each score
comes from a dozen events, and sampling noise spreads scores across a cohort of
*identical* people. In the built-in negative control, where every simulated
person has the same true consistency, the observed SD is still 0.13.

So the script reports **split-half reliability**: score each person twice from
interleaved halves of their own events and correlate. That gives the number that
matters,

```
reliable SD = observed SD × √reliability
```

the between-person spread with measurement error removed. On the negative control
reliability collapses to 0.00 and the verdict reads `NOT ESTABLISHED`; on a cohort
that genuinely differs it reads `PROCEED`.

Every classic index goes through the same procedure, so the comparison is like
for like, and two extra columns keep the analysis honest: `vs freq` flags any
index that is really measuring how *often* someone engages rather than how
regularly, and a correlation column asks whether the kernel score adds anything
over the indices that already exist.

The loader collapses bursts into occasions — a lesson is many rows sharing a
timestamp, and counting rows would report one Tuesday sitting as thirty
engagements. Use `--gap-sensitivity` to confirm the merge threshold isn't driving
the result. Timestamps are read as UTC, which costs nothing: consistency is
invariant to a constant time shift, so only the clock labels on anchors are
affected.

**First result.** On the Duolingo traces (419 usable people, 12 days) the score
came back reliable (0.575) and, uniquely among the indices tested, uncorrelated
with engagement frequency (−0.12, versus −0.94 for the Sleep Regularity Index,
which turns out to be measuring session count). But the bandwidth scan showed
reliability rising monotonically to a plateau at 240 min — the signature of a
population with no habit-scale anchor. These users engage "in the evening", not
"at 19:15". The metric works; that population does not have the phenomenon.
`docs/CONCEPT.md` has the full table and the curve-shape diagnostic.

What it cannot do is say anything about dropout — two weeks is too short to see
anyone quit. That needs a longer dataset; `adherence.datasets.load_event_csv`
takes any `(user, timestamp)` CSV by column name.

## Command line

```bash
adherence demo                                        # score the built-in archetypes
adherence score events.csv --tz Europe/London --id-column participant
adherence predict events.csv --auto --window 30       # when to send the nudge
adherence compare events.csv                          # vs. baselines and classic indices
```

CSV needs a column of ISO-8601 timestamps (`--column`, default `timestamp`);
`--id-column` splits a cohort file into one report per participant.

## Choosing parameters

`bandwidth_min` (default 45) is the timing tolerance — how far off their usual
time a session can land and still count as the same slot. `half_life_days`
(default 28) is how fast old behaviour is forgotten.

Both can be estimated instead of assumed:

```python
from adherence.tune import auto_model
model, selection = auto_model(log)   # selected by out-of-sample likelihood
```

**One caveat that matters.** A per-person bandwidth gives the best forecast but
makes the normalised 0–1 scores incomparable between people, because each
person's ceiling moves with it. For cohort comparisons or study variables, fix
the bandwidth and report it. For forecasting one person, select it.

## Caveats

Timestamps are not engagement; batch-syncing devices manufacture regularity. If
the intervention *assigns* a time, the score measures compliance with an
instruction rather than an endogenous habit. Irregular engagement may be
irregular life — shift work, caring, illness — so this is reasonable for
targeting support and not for judging people. Nothing here is causal. And
everything above is simulation: the package has not been validated on a real
dataset.

**New to this kind of model?** `docs/HOW-IT-WORKS.md` explains the mechanics in
plain terms — what is predicted, how the score moves over time, what every
parameter means, and how the out-of-sample checks work. `docs/CONCEPT.md` has the
prior art, what is and is not novel, the real-data results, and the failure
modes.

## Layout

```
src/adherence/
  events.py      EventLog, timezone/DST-correct phase
  kernels.py     circular kernels (von Mises, wrapped normal, compact, asymmetric)
  model.py       RoutineModel: recency-weighted periodic intensity + forecasts
  scores.py      consistency scores, anchors, jitter, drift
  baselines.py   SRM, IS, IV, SRI, entropy + forecasting baselines
  evaluate.py    rolling-origin evaluation, calibration, skill
  tune.py        bandwidth and half-life selection
  simulate.py    synthetic people with known ground truth
  survival.py    minimal Cox PH for study planning
  datasets.py    streaming loaders for public event logs (Duolingo, FitRec, CSV)
  validate.py    split-half reliability, bandwidth scan, anchor-scale diagnosis
  screen.py      one screening pass over a loaded cohort
  cli.py
```

MIT licensed.
