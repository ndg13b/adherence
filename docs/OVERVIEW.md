# What this is, and what it's for

Written for someone who has not seen the project before. No statistics or
programming assumed. `docs/HOW-IT-WORKS.md` explains the mechanics;
`docs/CONCEPT.md` has the prior art and the full results.

## The one-paragraph version

Some people do a thing at the same time every day. Others do the same amount of
it, scattered anywhere across the day. Those are different behaviours, and the
difference is usually invisible in the data people collect — a training log
records *how many* sessions, rarely *how regular* they were. This package
measures the regularity of someone's schedule from nothing but a list of
timestamps, turns it into a forecast of when they will next engage, and provides
the tools to test whether that regularity predicts who keeps going. The
measurement works and is validated. The prediction of *when* works. Whether
regularity predicts who **sticks with it** is the question the project was built
around, and it is still open — this document is explicit about which is which.

## What counts as "engagement"?

An **engagement** is one occasion on which a person did the thing, with a
timestamp attached. That is the entire input. The package never sees what the
behaviour was, only when it happened, which is why the same code applies across
domains that otherwise have nothing in common:

| domain | one engagement is |
|---|---|
| cognitive training | starting a training session |
| exercise | starting a workout |
| medication | taking a dose |
| physiotherapy | doing the prescribed exercises |
| meditation or journaling | opening the session |
| language learning | a study sitting |
| CPAP or glucose monitoring | putting the device on / taking a reading |
| therapy homework | completing an assignment |
| a habit someone wants to *break* | each cigarette, drink, or episode |

Three things have to be true for the score to mean anything.

**1. The occasions are discrete and timestamped.** A start time is enough. This
package uses session *start* — for a 40-minute training session, the moment they
sat down. Start, finish, or the whole interval are all defensible choices and
nobody has established which predicts best; start is used because it is when the
decision to engage was made, and because it is the field most systems record
reliably.

**2. There are enough of them.** Roughly ten occasions gives a rough score,
thirty gives a good one, and below about eight the answer is mostly noise. The
software refuses rather than guesses when there is too little history.

**3. The person chose the time.** This is the one that matters most. If a
protocol *assigns* someone a 09:00 slot, then measuring how close they land to
09:00 is measuring compliance with an instruction. That is a legitimate thing to
measure, but it is a different construct from a self-organised habit, and the two
are not comparable across studies. Everything here is aimed at the case where the
participant picks their own time.

### Where it does not apply

- **Continuous states with no discrete occasion.** Sleep timing is better served
  by the Sleep Regularity Index, which was designed for it.
- **Batch-synced device data.** A wearable that uploads at 03:00 every night
  manufactures perfect regularity out of nothing. Check what the timestamp
  actually records before trusting any of this.
- **Behaviours that happen many times a day at no particular time.** Checking a
  phone has no routine to find.

## The idea being tested

The hypothesis, in one sentence: **people who engage at the same time of day, on
the same days, are more likely to still be doing it months later** — and this is
separate from how *often* they engage.

The intuition comes from habit research. A behaviour that runs off a stable cue —
after breakfast, before the school run, on getting home — costs less to maintain
than one requiring a fresh decision each day. Regularity of schedule is a visible
signature of that, and it is measurable from data almost everyone already
collects.

If it holds, it is useful in a specific and modest way: it would let a programme
identify who is drifting toward dropping out *before* they drop out, from
timestamps alone, with no extra questionnaire and no extra burden on the
participant.

## What the software produces

**A forecast.** Given someone's history, the probability that they will engage in
any window you ask about, and the hour of day when they are most likely to.

```
model.p_engage_next(hours=24)      # 0.59 -- chance they engage tomorrow
model.best_window(window_min=60)   # 07:21-08:21, p=0.33 -- when to send a nudge
```

**A description.** Four numbers, deliberately kept apart because collapsing them
hides the one you care about:

| | the question it answers |
|---|---|
| timing consistency | how predictable is the *time* of their next session? |
| anchor precision | how tight is each slot, ignoring how many slots there are? |
| weekday regularity | does knowing the day of week tell you whether they engage? |
| out-of-sample bits | how much better than chance did history actually predict? |

Plus the readable diagnostics: the recurring slots ("07:28 ±7 min, 100% of
sessions"), the day-to-day scatter, and whether the whole routine is *sliding*
later week by week.

Two distinctions the package exists to make, because standard measures get both
wrong:

**Two tight slots are not one vague habit.** Someone who trains at 07:00 and
21:00 is genuinely harder to predict per session — it is a coin flip which comes
next — but they are not less habitual. The textbook circular statistic scores
that person near zero and calls them chaotic.

**Drifting is not dissolving.** A routine sliding 25 minutes later each week is
tight day to day; a routine scattering across three hours is coming apart. Every
variance-based index charges both to noise. They are different problems and want
different responses.

## What has been established, and what has not

Three separate claims. They are usually bundled together, and they should not be.

| claim | status |
|---|---|
| **Measurement** — this measures schedule regularity, distinctly from frequency, reliably enough to use per person | **Established, on real data** |
| **Forecasting** — it predicts when someone will next engage, better than the obvious alternatives | **Established on simulated data**; the machinery is leakage-proof, but the head-to-head against baselines has not been rerun on a real cohort |
| **Retention** — regularity predicts who keeps going | **Open.** Two attempts, both on data that turned out unable to answer it |

### Measurement: established

On the FitRec running data (626 people, a median of 116 runs each, spanning
years), split-half reliability was **0.93** — score each person twice from
interleaved halves of their own sessions and the two agree. On the Duolingo
traces it was 0.575 from only twelve days of history each.

Resolution is where it beats what already exists. Separating two people one step
apart in timing scatter, in the band below 20 minutes (Cohen's *d*, higher is
better):

| measure | *d* |
|---|---|
| Social Rhythm Metric | 0.77 |
| Sleep Regularity Index | 0.68 |
| Interdaily Stability | 0.06 |
| hour-of-day entropy | 0.15 |
| **this package** | **4.46** |

Below 20 minutes the established indices are at their resolution limit — every
session lands in the same box either way. That band is exactly where a habitual
person separates from a merely willing one. Above an hour of scatter every
measure agrees, because by then the routine has visibly gone.

### Retention: open, and honestly so

Two designs, on the only public data long enough to try.

**Frozen at baseline** — score each person over a 90-day run-in, then follow them
to their last recorded run. 408 people, 145 disengagements, median follow-up 702
days. Result: hazard ratio **0.955** per standard deviation (95% CI 0.827–1.103).
Not merely non-significant: the interval excludes anything above 1.10, and the
design had 99.6% power for the effect size it was built to detect.

**Moving with the person** — re-score every 30 days using only prior history, and
ask whether someone is at higher risk while their regularity is low or falling.
618 people, 14,614 intervals, 206 disengagements. Null at every memory setting
from one week to one year.

One coefficient did come back at p = 0.014. It did not survive examination, and
how it failed is the most useful thing in the project — see below.

**But the tell is that engagement *frequency* did not predict retention either**
(HR 0.922, p = 0.33). Everyone expects people who run more often to keep running.
When the obvious predictor also shows nothing, suspect the outcome before the
hypothesis. Three reasons this dataset cannot answer the question:

1. **The outcome is leaving a platform, not stopping the behaviour.** The data
   ends as Strava was displacing Endomondo. Switching apps has no reason to
   correlate with routine regularity, and it dilutes any real effect.
2. **There is no true enrolment date.** A person's first workout in the file is
   where the sampling window opens, not where their behaviour began — and habit
   formation happens at the start, which is invisible here.
3. **Severe selection.** Inclusion required 20+ runs over 120+ days, so the
   cohort is people who had *already* persisted.

So the hypothesis is neither supported nor refuted. It has not been tested on
data capable of testing it.

## Findings that are useful regardless

The project produced several results that stand independently of whether the main
hypothesis is ever confirmed. These are arguably the most transferable part.

**A widely used regularity index is measuring session count.** The Sleep
Regularity Index posted the best reliability of any measure tested — and
correlated **−0.94** with log session count on Duolingo and **−1.00** on running
data. It is a session counter wearing a regularity label. On simulated data the
confound showed up as only −0.45, because simulations hold engagement rates too
steady; only real data exposed it. Anyone using SRI as a regularity covariate
outside sleep research should check this. This package's score sits at −0.07.

**"The scores differ between people" is not evidence that people differ.** In a
simulated cohort where every person had *identical* true regularity, the observed
spread of scores was still 0.13 — large enough to look like a finding. The fix,
borrowed from psychometrics, is split-half reliability, and it correctly
collapsed to 0.00 on that control. Every measure reported here goes through it.

**A permutation null beat the standard parametric control.** The one p = 0.014
result turned out to be an artefact: the score is estimated from a
recency-weighted sample of past sessions, so as someone's sessions thin out the
estimate drifts downward on its own — and thinning out precedes quitting. The
score therefore falls before someone stops *even when their times of day are pure
noise*. The obvious fix, adding "change in engagement rate" as a covariate, moved
the coefficient by 0.0002 and was itself entirely non-significant. Reshuffling
each person's times of day among their own events — which holds their volume
fixed exactly, and needs no model of it — caught the artefact immediately.
**When you can build a null by rearranging your own data, prefer it to a
covariate that approximates the same idea.**

**Pooling activity types destroys the signal.** All FitRec sports together showed
no routine at all. Running alone showed tight 30-minute anchors and reliability
of 0.93. Someone who runs at 07:00 on weekdays and cycles on Saturday afternoons
has two routines; pooling them looks like one incoherent one.

**There is a cheap screening test for whether a dataset has routines at all.**
Scan the timing tolerance and watch the reliability curve: a population with
real habit-scale anchors peaks at a narrow width and then declines, while a
population with only a broad part-of-day preference rises and plateaus. Duolingo
plateaued at 240 minutes — those users engage "in the evening", not "at 19:15".
This takes minutes and can save a retention study aimed at a population with no
phenomenon to find.

## What you could use this for now

Everything here rests on the established claims, not the open one.

- **Timing a prompt.** `best_window()` names the hour a given person is most
  likely to act. For any programme that sends reminders, this is a
  ready-to-use, per-person answer to a question usually settled by sending
  everything at 09:00.
- **Regularity as a trial outcome.** If an intervention is meant to help people
  build a routine, this measures whether it did — reliably, and without
  accidentally measuring how much they did instead.
- **Regularity as a covariate.** Reported alongside frequency, so the two are
  not confused.
- **Screening a candidate dataset** before committing to an analysis.
- **Describing a person to themselves.** "You train at 07:28, give or take seven
  minutes, and that has slid twenty minutes later over the past month" is
  something a participant can act on. A single 0–1 score is not.
- **Separating drift from dissolution** when deciding what support to offer.

And one speculative direction, flagged as untested: if a regular schedule
strengthens a habit, then deliberately *destabilising* the schedule — delaying an
unwanted behaviour until it no longer sits at its usual time — might weaken one.
The machinery to measure that already exists here. Nobody has tried it.

## What would settle the open question

A dataset with three properties none of the public ones have:

1. **A real enrolment date**, so the early period where habits form is visible.
2. **An outcome that is the behaviour**, not continued use of one particular app.
3. **Participant-chosen timing**, so the score measures a habit rather than
   compliance with an instruction.

Purpose-collected trial data has all three. `examples/cohort_power.py` answers
the design question directly: at 200 participants and a 28-day run-in, there is
90% power for a hazard ratio of 1.4 per standard deviation of irregularity. It
also makes a point worth heeding — powering a study on the *true* effect size
leaves it underpowered for the *measurable* one, because a score estimated from a
short window is a noisy predictor and noise shrinks the coefficient you can
actually detect.

The cheapest outstanding check does not need new data: rerunning the
forecasting comparison against baselines on the real running cohort, which would
move the second claim from "established on simulation" to "established on real
data".

## What this should not be used for

**Nothing here is causal.** Even if regularity predicted retention perfectly, it
would not follow that *making* someone regular keeps them engaged. People whose
routines hold differ from people whose routines come apart in ways no event log
records — injury, a new job, illness, caring responsibilities.

**Irregular engagement is often irregular life.** Shift work, childcare, chronic
illness and insecure housing all produce scattered timing from people doing their
best. Using a score like this to decide where to offer *support* is reasonable.
Using it to rank, judge, gate access, or infer motivation is not, and the
correlation with life circumstances means it would fall hardest on the people
already worst served.

**A timestamp is not a behaviour.** It records that something was logged. Whether
it was done, done properly, or done by the person whose account it is are all
outside what this can see.
