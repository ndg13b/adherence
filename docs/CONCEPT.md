# The concept, the prior art, and what is actually new

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
- Validation on a real dataset. Everything above is simulation.

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
