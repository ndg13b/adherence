# The concept, the prior art, and what is actually new

The full technical record. For the plain-language tour — what counts as an
engagement, which claims are established and which are open, what the whole thing
is useful for — see `docs/OVERVIEW.md`.

## The idea

People who engage with something at the same time of day, on the same days,
tend to keep doing it. The regularity of the schedule — not just the amount of
engagement — is what predicts whether the behaviour survives. This holds for
things people want to keep doing (cognitive training, medication, exercise) and,
symmetrically, for things they do not (a nightly drink at the same hour is a
harder habit to break than an erratic one).

The formalisation implemented here has four moving parts:

1. **Each engagement is an event with tails.** A session at 07:04 is evidence
   for a 07:00 routine, weaker evidence for 07:40, and almost none for 15:00.
   The taper is a kernel; its width is the person's timing tolerance.
2. **Recency matters.** A routine held three months ago says less about tomorrow
   than one held last week. Weights decay exponentially with a half-life.
3. **Multiple sessions per day are normal**, and a person with two tight slots
   is not a person with one vague one. Density, not a mean time.
4. **The output is a forecast, not a label.** The model predicts engagement in a
   given window, which is testable against what happened, and directly usable
   for timing a reminder.

## Is any of it novel?

Partly. Being specific about which parts, because the honest answer matters more
than a flattering one.

### What already exists

**Social Rhythm Metric** (Monk et al., 1990). Counts how many of 17 daily
activities occurred within ±45 minutes of their habitual time. This is the
direct ancestor of the idea, and it is 35 years old. The scoring here is a
generalisation of it: replace the hard ±45 minute box with a graded kernel.

**Interdaily Stability / Intradaily Variability** (Van Someren et al., 1999).
Actigraphy measures of how reproducible the 24-hour activity profile is.
Variance-ratio based, computed on binned counts.

**Sleep Regularity Index** (Phillips et al., 2017). Probability that a person is
in the same binary state at times 24 hours apart. Elegant, widely used, and
binary — it cannot weigh *how far* off a late session was.

**Entropy and predictability of human behaviour** (Song et al., 2010). Entropy
rate of a behavioural sequence bounds how predictable a person is. The
information-theoretic framing used below is a direct descendant.

**Context stability in habit theory** (Wood & Neal; Lally et al., 2010). Habits
form through consistent context–behaviour pairing, and time of day is one of the
strongest contexts. This is the *theory* the score operationalises.

**Temporal point processes** in mobile health. Poisson and Hawkes processes with
circadian intensity are standard for modelling engagement timing.

So: "regularity predicts adherence" is not new, kernel density estimation on a
circle is not new, and point processes for engagement are not new.

### What is new, or at least uncommon

**1. Consistency defined as self-predictability, in bits.** Rather than
inventing an index and arguing for it, define consistency as *how well a
person's own past predicts their next engagement*, measured prequentially:
refit on everything before event *i*, score the log density at event *i*,
compare against a uniform "any time of day" forecast. The unit is bits per
event. It cannot be inflated by fitting and evaluating on the same data, it is
comparable across people, and it can go **negative** — which is the diagnosis
that a routine has moved and the person's history is now actively misleading.
No index in the list above can express that.

**2. Regularity and forecast are the same object.** The existing indices are
descriptive: they score a history, and any predictive claim requires a separate
model. Here the score *is* a property of a forecaster, so "how consistent is
this person" and "will they engage between 7 and 9 tomorrow" come from one
fitted model and are validated by the same proper scoring rule.

**3. Splitting drift from jitter.** A person whose 08:00 slot slides 25 minutes
later each week has tight day-to-day behaviour and a migrating schedule. Every
variance-based index charges the slide to noise and reports them as irregular.
Separating the two changes what you would do about it — a drifting routine can
be predicted by following it; a dissolving one cannot.

**4. Splitting number-of-slots from tightness-of-slots.** A twice-daily user is
genuinely less predictable per session (a coin flip between slots) but is not
less habitual. Reporting `timing_consistency` alongside `anchor_precision`
distinguishes two crisp rituals from one vague habit, which a single number
provably cannot.

There is a trap in the second measure worth recording, because it is easy to
fall into and invisible once you have. Cutting the day into basins around
density peaks makes *any* set of times look concentrated within its own basin:
uniformly random engagement scored 0.48 on a 0–1 scale, above a real but sloppy
single-slot routine at 0.34. `anchor_precision` is therefore reported as the
excess over a permutation null — the same pipeline run on uniform phases with
the same event count and weights, and **matched on the number of slots**, since
a null free to choose its own *k* is not comparable. After correction, uniform
times score 0.04 and the ordering behaves.

**5. Bandwidth as an estimated quantity.** The tolerance width is a fact about
the person — 8 minutes and 90 minutes are different routines — and is selected
by out-of-sample likelihood rather than fixed by convention.

None of these is a revolution. Together they amount to a defensible measurement
instrument where the existing options are either coarse (SRM's box), insensitive
(SRI's binary state), or purely descriptive (IS/IV).

## What the implementation actually shows

From `examples/discrimination.py` — people differing *only* in timing jitter,
20 simulated people per row, 150 days each:

| true jitter | SRM ±45 | SRI | IS | entropy | kernel score | bits/event |
|---|---|---|---|---|---|---|
| 3 min | 1.000 | 95.8 | 0.479 | 1.00 | 0.995 | 3.63 |
| 5 min | 1.000 | 95.8 | 0.479 | 1.00 | 0.987 | 3.62 |
| 10 min | 1.000 | 95.8 | 0.479 | 1.00 | 0.951 | 3.57 |
| 15 min | 0.998 | 95.4 | 0.479 | 1.00 | 0.899 | 3.49 |
| 20 min | 0.977 | 94.8 | 0.477 | 1.01 | 0.837 | 3.39 |
| 30 min | 0.866 | 93.9 | 0.431 | 1.27 | 0.710 | 3.15 |
| 45 min | 0.684 | 93.2 | 0.322 | 1.72 | 0.549 | 2.77 |
| 60 min | 0.546 | 92.8 | 0.243 | 2.07 | 0.431 | 2.42 |
| 90 min | 0.387 | 92.5 | 0.154 | 2.62 | 0.283 | 1.85 |
| 150 min | 0.239 | 92.1 | 0.080 | 3.30 | 0.144 | 1.06 |

Averaged over enough people every index orders the groups correctly. The
difference is resolution for a *single* person. Ability to separate two
individuals one step apart, below 20 minutes of jitter (Cohen's *d*):

| index | mean *d* |
|---|---|
| SRM ±45 min | 0.77 |
| SRI | 0.68 |
| Interdaily Stability | 0.06 |
| hour entropy | 0.15 |
| **kernel score** | **4.46** |
| **prequential bits** | **6.50** |

Below about 20 minutes of jitter the established indices are at or near their
resolution limit — every session falls in the same box or the same epoch either
way. That band is exactly where a habitual user separates from a merely willing
one, and it is the entire reason to build something finer. Above 60 minutes all
the indices agree, because by then the routine has visibly gone.

## First contact with real data

Run against the public Duolingo learning traces (13M rows, 115k users; 419 with
at least 8 sessions over 5 days in a 5% sample), fixed 45-minute bandwidth:

| index | mean | reliability | reliable SD | vs. log(sessions) |
|---|---|---|---|---|
| **timing_consistency** | 0.109 | 0.575 | 0.091 | **−0.12** |
| srm_hit_rate | 0.284 | 0.322 | 0.118 | +0.05 |
| interdaily_stability | 0.121 | 0.515 | 0.044 | +0.19 |
| sleep_regularity_index | 91.9 | **0.985** | 3.585 | **−0.94** |
| timing_entropy_bits | 2.743 | 0.798 | 0.465 | +0.65 |
| resultant_length | 0.585 | 0.645 | 0.167 | −0.18 |

Two findings, and the second is the more important one.

**The established indices are contaminated by engagement frequency.** SRI posts
the highest reliability in the table by a distance — and correlates −0.94 with
log session count. It is reliably measuring *how often* people engage, not how
regularly; its reliability is the reliability of a session counter. Hour entropy
is partly the same (+0.65). The kernel score sits at −0.12: essentially
independent of volume. On simulated data SRI's confound registered only −0.45,
because the simulation held everyone's rate near-constant. Real session counts
vary enormously (IQR 9–15, max 66), and only real data exposed it.

**But this population does not have the phenomenon.** Calibrated against known
jitter at the same sample size, a score of 0.109 corresponds to roughly *three
hours* of scatter (uniform would be 0.009). Scanning the bandwidth, reliability
rises monotonically — 0.474 at 15 min to 0.639 at 240 min — and then plateaus.

That shape is diagnostic, and `diagnose_anchor_scale` now reads it. A population
with habit-scale anchors peaks at a narrow width and **declines** thereafter,
because a kernel wider than the anchors blurs real distinctions: a simulated
25-minute-anchor cohort peaked at 60 min and fell from 0.893 to 0.848 by 360
min. A population with only a broad part-of-day preference rises and plateaus,
because there is no characteristic timescale to find. Matching curve shapes
across bandwidths, the real cohort correlates **+0.99** with an anchorless
population confined to a personal ~6-hour window, and **−0.45** with an anchored
one.

So Duolingo users engage "in the evening", not "at 19:15". The metric works —
it discriminates reliably and cleanly — but the concept is *untested* here
rather than supported or refuted, because nobody schedules a phone language app.
Domains where the behaviour is actually scheduled (prescribed training with a
participant-chosen slot, medication, exercise) are where the hypothesis can be
tested at all.

A useful by-product: the bandwidth curve is a cheap screening test for whether a
candidate dataset contains routines *before* investing in a retention analysis.

## Second contact: exercise, and the first anchored population

The FitRec/Endomondo release (253k workouts, 1,104 people, 2011–2016). Restricted
to **running only**, 626 people with a median of 116 runs each:

| | Duolingo | FitRec, all sports | FitRec, running only |
|---|---|---|---|
| bandwidth optimum | 240 min | 360 min | **30 min** |
| verdict | no anchor | no anchor | **ANCHORED** |
| reliability | 0.575 | 0.593 | **0.928** |
| vs. log(sessions) | −0.12 | +0.04 | **−0.07** |

**Pooling sports destroyed the signal.** All sports together gave a 360-minute
optimum and no anchor; running alone peaks at 30 minutes and declines. Both
bandwidth scans used the same 28-day half-life, so the flip is attributable to
the sport filter alone. Someone who runs at 07:00 on weekdays and cycles on
Saturday afternoons has two routines, and pooling them looks like one incoherent
one. Any future analysis has to separate activities before scoring them.

**The 28-day half-life was discarding most of each history.** Reliability climbs
0.554 → 0.936 as the half-life lengthens to no decay, so these routines are
stable over *years*. The default is right for a two-week dataset and wasteful for
a multi-year one; there is now a `half_life_scan` to settle it per dataset.

**Clock labels here are meaningless, but the scores are not.** Neither the raw
UTC profile nor the longitude-localised one troughs overnight — both peak near
midnight and empty out at 08:00–09:00, which no exercise population does. So
these timestamps are not local wall-clock time. The consistency scores survive,
being invariant to a constant shift, but with one caveat worth stating: if the
offset varies *within* a person (upload rather than start times would do this),
the extra noise makes every consistency estimate here a **lower bound**. The
routines are at least as tight as measured, possibly tighter.

**The frequency confound, at its most extreme.** With 939 people the Sleep
Regularity Index correlated **−1.00** with log session count. It is a session
counter. `timing_consistency` sits at −0.07.

So exercise is the first real population carrying the phenomenon the concept is
about, and the retention question is finally askable on it. See
`examples/fitrec_retention.py`, which scores a run-in window, follows people to
their last run with the still-active censored, and — critically — adjusts for
baseline frequency, since people who run often keep running and an unadjusted
result would be trivial.

## The retention test: a clean null

Run on the FitRec running subset — 408 analysable people, 145 disengagements,
median follow-up 702 days, 90-day run-in:

| model | irregularity HR per SD | 95% CI | p |
|---|---|---|---|
| unadjusted | 0.952 | 0.824–1.101 | 0.51 |
| adjusted for run-in frequency | 0.955 | 0.827–1.103 | 0.53 |

**No effect, and the null is informative rather than merely non-significant.**
The confidence interval excludes anything above HR 1.10 per SD. The design had
99.6% power for HR 1.4 — the effect size `examples/cohort_power.py` was built
around — and 95% for HR 1.3. Run-in reliability was high, so attenuation does
not rescue it.

The pipeline is not broken: on synthetic data with a hazard ratio built in, the
identical code recovers HR 1.91 (p = 2e-19).

**The tell is that frequency does not predict retention either** (HR 0.922,
p = 0.33). Everyone expects people who run more often to keep running. When the
obvious predictor also shows nothing, suspect the outcome before the hypothesis.

Three reasons this dataset probably cannot answer the question:

1. **The outcome is platform abandonment, not behaviour cessation.** "Stopped
   logging on Endomondo" is not "stopped running", and the data ends in January
   2016, exactly as Strava was displacing Endomondo. Migration between apps has
   no reason to correlate with routine consistency, and it would dilute any real
   effect toward exactly what we observe.
2. **There is no true baseline.** A person's first workout in this file is not
   their first workout ever — it is where the sampling window opens. Habit
   formation happens at the start of a behaviour, and that period is invisible
   here, so the "run-in" is a window into an ongoing history rather than an
   enrolment period.
3. **Severe selection.** Inclusion required 20+ runs over 120+ days, so the
   cohort is people who had already persisted. Restriction of range on the
   predictor and the outcome together.

So the hypothesis is neither supported nor refuted. What *is* established, and
worth separating, is that the **forecasting** claim holds — calibrated
probabilities, beating rate-matched and histogram baselines — while the
**retention** claim remains untested. The practical use (placing a nudge where
someone is already likely to act) rests on the first; the research use rests on
the second.

Testing it properly needs a dataset with a real enrolment date and an outcome
that is the behaviour rather than the platform. That points back to
purpose-collected data, or to something like the Brighten trials.

## The time-varying test: a null, and one survivor

The frozen-baseline design above can only detect a *trait*. It scores each
person once and asks whether the regular ones lasted longer, which is blind to
the thing the concept actually describes — a routine coming apart. People rarely
quit out of a steady habit. So `examples/fitrec_timevarying.py` re-scores every
person every 30 days using only their history to that point, and asks whether
someone is at higher risk *while* their consistency is low or falling
(Andersen–Gill counting-process Cox).

618 usable running people, ~15,000 intervals, 207 disengagements.

**The level of irregularity is null at every memory length**, adjusted for run
rate, at lag 30:

| half-life | log HR | p |
|---|---|---|
| 7 d | +0.016 | 0.82 |
| 14 d | −0.008 | 0.91 |
| 28 d | −0.042 | 0.52 |
| 90 d | −0.013 | 0.84 |
| 365 d | +0.042 | 0.57 |

That sweep exists because the memory length decides whether the score *can* see
decay at all, and the setting that maximises reliability is the wrong one for
this question: a total 30-day collapse moves the score −0.79 with a 7-day memory
and only −0.11 with a 365-day one. Reliability rewards a stable description of a
person; detecting change needs the opposite. Nothing appears at either extreme
or anywhere between, so the null is not an artefact of that choice.

One coefficient was not null. Adding *whether consistency is falling*:

| | lag 0 | lag 30 |
|---|---|---|
| irregularity | 0.995 (p 0.94) | 0.920 (p 0.21) |
| **consistency falling** | 1.079 (p 0.27) | **1.166 (se 0.062, p 0.014)** |
| log run rate | 1.013 | 0.903 (p 0.14) |

**This is not evidence, and the arithmetic says so plainly.** Roughly seventeen
coefficients were fitted across two lags, three model forms and five memory
lengths. One at p < 0.05 is what chance produces — the expected count under a
global null is 0.85, and Bonferroni puts that p at 0.24.

What made it worth examining rather than discarding is the *pattern*, not the
threshold: it is larger in the lagged fit (1.166) than the unlagged one (1.079).
Reverse causation produces the opposite ordering. Someone already winding down
looks both irregular and about to quit, so a contaminated estimate is strongest
with no lag and decays as the gap widens. This one strengthens.

### Examining it: `examples/fitrec_falling.py`

Four ways to break a single coefficient, rather than seventeen more chances to
find one.

1. **Is it just a falling run rate?** The score is a leave-one-out density
   estimated from a recency-weighted sample. When someone thins out that sample
   shrinks and the estimate drifts down on its own — and thinning out precedes
   quitting. The original model adjusted for the *level* of the run rate, which
   is the wrong control for a *change*.
2. **A permutation null.** Reorder each person's times of day among their own
   events. Every event time, every count and their marginal time-of-day
   distribution survive; only the pairing between them goes, and with it any
   trend in the timing. A routine coming apart cannot survive that shuffle; a
   sample-size artefact can. This also answers a question the Wald test cannot:
   each person contributes ~24 correlated intervals, and a model treating them
   as independent understates its own uncertainty. The permutation SD *is* the
   honest standard error.
3. **Split-half replication.** An effect of the claimed size lands positive in
   both random halves ~90% of the time; chance manages 25%.
4. **The design grid.** Interval × lag × memory, 27 cells. A real effect is
   dented by moving them; a fluke is one bright cell.

`--self-test` runs the battery against a simulated cohort whose routines
demonstrably do come apart before they quit. It recovers HR 1.97 and passes all
four checks, which is the only thing that makes a null from it worth reporting.

### The answer: it does not survive

| check | running (618 people, 206 events) | cycling (527 people, 519 events) |
|---|---|---|
| reproduces the original | +0.1537, p 0.014 | +0.0448, p 0.295 |
| adjusted for change in run rate | +0.1535, p 0.014 — unchanged | +0.0449, p 0.29 |
| **permutation null** | **p 0.060**, null centred at **+0.0247** | **p 0.493**, null at +0.0137 |
| split-half | 40/40 same direction, **0/40 significant** | 25/40, 0/40 |
| design grid (27 cells) | present only at interval 30 / lag 30 | absent throughout |

The cycling cohort had never been fitted and carries 2.5× the events, and it is
null on every check. One caveat keeps it from being decisive on its own: **519
of its 521 people are classed as having stopped**, because that subset spans
5,019 days against running's 3,570, so the global end of data sits far past most
people's last ride. Censoring is then doing no work and the contrast is between
leaving early and leaving late, not between leavers and stayers. The script now
prints this fraction and says so.

So the cycling null is supporting evidence rather than the verdict. The verdict
comes from the running cohort's own diagnostics, and three of them are worth
keeping.

**The permutation null is not centred on zero.** Shuffled data still produces
+0.0247. The covariate therefore carries a built-in association with
disengagement that has nothing to do with timing, and the effect actually
needing explanation is +0.129, not +0.154. Against a properly centred null,
p = 0.060 rather than 0.014.

The mechanism is the one the battery was built to catch. `timing_consistency` is
a leave-one-out density estimated from a recency-weighted sample. As that sample
thins the estimate degrades and drifts downward — and thinning precedes quitting.
So the score falls before someone stops even when their times of day are random.

**The parametric control missed it entirely.** Adding the change in log run rate
moved the coefficient by 0.0002, and that covariate was itself dead null
(HR 0.994, p 0.94). The artefact is real and the obvious adjustment for it found
nothing, because a count over a fixed window and a recency-weighted effective
sample size are not the same quantity. The shuffle holds volume fixed *exactly*
and needs no model of it. That is the argument for permutation over adjustment,
and it is not hypothetical here.

**The grid shows one bright cell.** The effect exists at interval 30 with lag 30
(+0.191 at 14-day memory, +0.154 at 28-day) and nowhere else. At interval 20 and
45 it is gone. At lag 60 it *reverses* and is significantly negative (−0.181,
p 0.008; −0.190, p 0.0004). Significant results in both directions across 27
cells is what noise sliced 27 ways looks like.

One correction to the battery itself: the split-half check originally counted
"both halves positive", which assumed a null centred on zero. Once the
permutation showed it is not, that criterion scores the artefact as a
replication, so it now counts halves beyond the null's centre. It is the weakest
of the four in any case — two halves of one dataset are not independent tests, so
consistent direction is close to guaranteed whenever the full-sample estimate is
non-zero. It rules out an effect carried by a handful of people, and nothing
more.

### Failures found while building the self-test

- A first attempt had routines loosening over 150 days. With a 28-day memory the
  score has bottomed out well before the person quits, so the steepest falls
  land in intervals where *nobody* quits — and the coefficient came out strongly
  **negative**. The decay lead time has to exceed lag + interval without greatly
  exceeding the memory. This is a real constraint on what the design can detect,
  not a quirk of the simulation.
- A cleaner version was perfectly separable, and the Cox fit walked off to a
  coefficient of −19,217 with a standard error of 0.25 and p = 0 — a result that
  reads as overwhelming evidence and means the likelihood has no maximum.
  `adherence.survival` now does step-halving and flags separation explicitly,
  and `CoxResult.usable` gates anything built on a fit.

## Where this could break on real data

Stated plainly, because these are the things that would sink a study.

- **Timestamp ≠ engagement.** App-open times are not session times, and a device
  that syncs in batches will manufacture spurious regularity.
- **Prescription confounds routine.** If the intervention *tells* people to
  train at 09:00, consistency measures compliance with an instruction, not an
  endogenous habit. Chosen-time designs and assigned-time designs are not
  comparable.
- **Reverse causation.** People who are about to quit engage erratically first.
  A consistency score measured over a window that overlaps the run-up to
  dropout will predict it trivially and mean nothing. The run-in window must be
  strictly before the follow-up period — which `examples/cohort_power.py`
  enforces and most naive analyses do not.
- **Shift work, caring responsibilities, illness.** Irregular engagement may be
  irregular life, not weak motivation. A score used to target support is fine;
  used to judge people it is not.
- **The bandwidth/comparability tension.** Per-person bandwidths give better
  forecasts but make normalised scores incomparable across people. Fix the
  bandwidth for cohort comparisons. See `adherence/tune.py`.
- **Nothing here is causal.** A high score predicting retention does not mean
  that *making* someone regular would improve retention. That needs a trial that
  randomises the prompt, not an observational score.

## Things worth building next

- A drift-aware intensity (currently drift is *reported* but the forecaster does
  not follow it — the `drifter` archetype is predictable in principle and the
  model does not exploit it).
- Self-excitation: a Hawkes term for streaks and for the burst of activity that
  follows a lapse.
- Hierarchical pooling across a cohort, so a new participant's score borrows
  strength from the population instead of waiting weeks for data.
- Competing-risks survival, separating "quit" from "completed the programme".
- A cluster-robust (Lin–Wei sandwich) variance for the time-varying fit, so the
  standard errors are right by construction rather than checked by permutation
  after the fact.
- Validation on a dataset with a real enrolment date and an outcome that is the
  behaviour rather than the platform. Duolingo and FitRec establish that the
  score is reliable and that it measures something other than frequency; neither
  can test the retention claim, for the reasons above.

## References

- Monk TH, Flaherty JF, Frank E, Hoskinson K, Kupfer DJ (1990). The Social
  Rhythm Metric. *J Nerv Ment Dis* 178(2):120–126.
- Van Someren EJW et al. (1999). Bright light therapy: improved sensitivity to
  its effects on rest-activity rhythms. *Chronobiol Int* 16(4):505–518.
- Phillips AJK et al. (2017). Irregular sleep/wake patterns are associated with
  poorer academic performance. *Sci Rep* 7:3216.
- Song C, Qu Z, Blumm N, Barabási A-L (2010). Limits of predictability in human
  mobility. *Science* 327(5968):1018–1021.
- Lally P, van Jaarsveld CHM, Potts HWW, Wardle J (2010). How are habits formed.
  *Eur J Soc Psychol* 40(6):998–1009.
- Dawid AP (1984). Present position and potential developments: the prequential
  approach. *J R Stat Soc A* 147(2):278–292.
- Gneiting T, Raftery AE (2007). Strictly proper scoring rules, prediction, and
  estimation. *JASA* 102(477):359–378.
