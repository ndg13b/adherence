# How it works, in plain terms

Written for someone comfortable with research and statistics but not with data
science. No prior knowledge of machine learning is assumed. `docs/CONCEPT.md`
covers the prior art and the empirical results; this covers the mechanics.

## 1. What the model is actually for

The thing the model produces is a **forecast of when**:

> What is the chance this person engages between 07:00 and 09:00 tomorrow?

Everything else — the consistency score, the anchors, the jitter — is a summary
of that forecast. If you only remember one thing: this is a model of *timing*,
not of quantity.

### How it builds the forecast

Give it a list of times when someone engaged. It does three things.

**Each past session becomes a bump.** A session at 07:04 is strong evidence for
"this person acts around 07:00", weaker evidence for 07:40, and almost none for
15:00. The bump encodes that fade-out. Its width is a parameter you set.

**The bumps are stacked into a daily profile.** Add all the bumps together and
you get a curve across the 24 hours showing where this person's sessions
concentrate. A sharp spike means a precise routine; a flat curve means none.

**The profile is scaled by how often they engage.** A curve says *when* within a
day. Multiply by how many sessions they do on that weekday and you have an
expected rate at every moment — which converts into a probability for any window
you ask about.

Those two pieces are kept separate on purpose. "Do they show up on Tuesdays at
all?" and "when on a Tuesday?" are different questions, they fail in different
ways, and a single blurred number would confuse a person who trains three times
a week like clockwork with one who trains daily at random hours.

## 2. The consistency score

The score asks how *peaked* that daily profile is, on a 0–1 scale:

- **0** means sessions are spread evenly across the 24 hours — no routine.
- **1** means every session lands at the same minute.

It is computed **leave-one-out**: each session is scored by the density that the
*other* sessions place at its time, so a lone session cannot vouch for itself.
Then it is rescaled against two fixed reference points — the flat "no routine"
curve and a perfect spike — which is what makes the number comparable between
people who engage at very different frequencies. A raw curve height would not
be: it grows with the number of sessions.

To convert a score into something concrete, here is what it corresponds to for
a person with about a dozen sessions over two weeks:

| score | typical scatter around their usual time |
|---|---|
| 0.01 | no routine at all |
| 0.11 | ~3 hours |
| 0.29 | ~90 minutes |
| 0.72 | ~30 minutes |
| 0.95 | ~10 minutes |

## 3. Is the score fixed, or does it move?

**It moves.** The score is always *as of a moment in time*, computed from every
session before that moment. Ask on day 60 and on day 100 and you get different
numbers.

Older sessions count less, controlled by a **half-life**. At the 28-day default,
a session from four weeks ago counts half as much as today's, one from eight
weeks ago a quarter, and so on.

Here is the same simulated person — tight at 07:00 for 60 days, then scattered —
scored every 10 days under three different memory settings:

| day | 7-day memory | 28-day memory | 365-day memory |
|---|---|---|---|
| 40 | 0.934 | 0.932 | 0.931 |
| 60 | 0.898 | 0.907 | 0.912 | ← routine breaks down here |
| 70 | 0.190 | 0.541 | 0.691 |
| 100 | 0.088 | 0.116 | 0.316 |

The three columns are the same data. The half-life is what sets how fast the
score reacts to a change — a short memory notices within days, a long one is
still half-convinced a month later. Neither is "correct"; it depends whether you
want an early alarm or a stable trait.

### Does one deviation get penalised?

Yes, and it fades:

```
after 60 tight days                  0.920
plus ONE session at 03:00            0.858    penalised
then 14 more normal days             0.883    partially recovers
```

The single odd session drags the score down, then ages out of the memory. That
is a design choice made explicit, not a fact about human behaviour.

## 4. Is later data used to check predictions made from earlier data?

Yes. This is the part worth scrutinising, because it is where measures of this
kind usually go wrong.

**Prequential scoring.** Walk through someone's sessions in order. At each one,
rebuild the model using *only the sessions before it*, then ask how well it
predicted this one. Nothing is ever scored against data it has already seen.

The result is reported in **bits per session**. One bit is one halving of
uncertainty. A typical tight routine scores about +3.5 bits, meaning the model
put roughly 11× more probability on the moment the session actually happened
than blind guessing would — like narrowing "sometime today" down to a two-hour
window.

Crucially it **can go negative**, which most regularity indices cannot express.
For a person whose schedule moves from 07:00 to 20:00, scored either side of the
change:

```
sessions before the move   +3.53 bits    history predicted them well
sessions after the move    +0.59 bits    history was actively misleading
```

**Rolling-origin forecasting.** Stand at a given day, predict the next 24 hours
in half-hour blocks, compare with what happened, advance a day, refit. The test
suite includes a leakage check: two histories that are identical up to day 60
and wildly different afterwards must produce *bit-identical* forecasts at every
origin up to day 60. If information ever flowed backwards through time, that
test fails.

## 5. The parameters

| parameter | default | what it means |
|---|---|---|
| `bandwidth_min` | 45 | **Timing tolerance.** How many minutes off their usual time still counts as the same slot. The most consequential setting. |
| `half_life_days` | 28 | **Memory.** How fast old sessions stop counting. Sets the score's reaction speed. |
| `weekday` | on | Whether to model Mon/Wed/Fri patterns separately from a daily habit. |
| `rate_shrinkage_days` | 3 | **Caution about weekdays.** Stops one missed Tuesday concluding "never trains on Tuesdays". |
| `timing_shrinkage_events` | 4 | The same caution applied to weekday-specific *times*. |
| `uniform_floor` | 0.02 | A 2% allowance that they might appear at any hour, so one 3 a.m. session cannot dominate. |
| `kernel` | von Mises | The shape of the bump. Alternatives cut the tails off hard, or make them asymmetric (late differs from early). |

In practice only the first two need thought, and both can be chosen from the
data rather than assumed:

- `--bandwidth-scan` tries a range of tolerances and reports which one measures
  people most **reliably** (see §7). Read it on reliability, not on the average
  score — a wider setting always raises the average, which proves nothing.
- `--half-life-scan` does the same for memory.

**One caveat that matters.** Choosing the bandwidth per person gives the best
forecast for that person but makes the 0–1 scores incomparable *between* people,
because each person's ceiling then differs. For comparing people, or for using
consistency as a study variable, fix the bandwidth across the cohort and report
it.

## 6. The other numbers in a report

| number | question it answers |
|---|---|
| `timing_consistency` | How predictable is the *time* of their next session? |
| `anchor_precision` | How tight is each slot, ignoring how many slots there are? |
| `weekday_regularity` | Does knowing the day of week tell you whether they engage? |
| `timing_bits` | Out-of-sample bits per session — the honest forecasting version. |
| **anchors** | The recurring slots: "07:28 ±7 min, 100% of sessions". |
| **jitter** | Day-to-day scatter in minutes, measured around the trend. |
| **drift** | Whether the whole routine is sliding, in minutes per week. |

Two distinctions these exist to make:

**Two tight slots are not one vague habit.** Someone who trains at 07:00 and
21:00 is genuinely less predictable per session — it is a coin flip which comes
next — but they are not less habitual. `timing_consistency` reports the first,
`anchor_precision` the second.

**Drifting is not dissolving.** A routine sliding 25 minutes later each week is
tight day to day. Reporting drift separately from jitter distinguishes a
schedule that is *moving* from one that is *coming apart* — different problems
with different remedies.

## 7. Reliability: how a score is checked

Borrowed from psychometrics, and the single most useful check in the package.

Score each person **twice**, from two interleaved halves of their own sessions,
and correlate the two. If the halves agree, the score is measuring something
about the person. If they do not, the apparent differences between people were
noise.

This matters more than it sounds. In a simulated cohort where **every person had
identical true consistency**, the observed spread of scores was still 0.13 — big
enough to look like a real finding. Reliability correctly collapsed to 0.00.

The correction that follows:

```
reliable spread = observed spread × √reliability
```

which strips out measurement error and leaves the between-person variation that
is actually there.

A second check runs alongside it: each index's correlation with **how often**
someone engages. An index that tracks session count is a frequency measure
wearing a regularity label. On real running data the Sleep Regularity Index
correlated **−1.00** with session count — it was a session counter. This
package's score sat at −0.07.

## 8. Two ways the score has been used as a predictor

**Frozen at baseline.** Score each person once from a run-in window, then follow
them forward and ask whether the score predicted disengagement. This is the
conventional design, and it is what `examples/fitrec_retention.py` does. On real
running data it found nothing (HR 0.955 per SD, CI 0.83–1.10).

**Moving with the person.** Re-score every 30 days using only history up to that
point, and ask whether someone is at higher risk *while* their consistency is
low or falling. That is `examples/fitrec_timevarying.py`.

The second design can see something the first cannot. A frozen baseline score
only detects a stable trait; it is blind to a routine that was fine at month
three and came apart at month nine. Since people rarely quit out of a steady
habit, that is arguably the more relevant question.

**The trap, and the guard.** Someone about to quit slows down first. Measure
their consistency in the days just before they stop and "prediction" becomes
near-circular — the score is detecting the beginning of the very event it claims
to forecast. So the model is fitted twice: once with no lag, and once with the
score measured a month *before* the interval it predicts. An effect that
survives the lag is a genuine early warning. One that appears only without a lag
was measuring the slowdown itself.

On the running data the moving score found nothing either: the *level* of
irregularity is null at every memory setting from a week to a year. One
coefficient was not null — whether consistency is *falling* — at p = 0.014.

## 9. Why that p = 0.014 is not a result

This is the most useful thing in the project to understand, and it has nothing
to do with routines.

About **seventeen** coefficients were fitted across the time-varying analysis:
two lags, three model forms, five memory lengths. If none of them were real, how
many would come out below p = 0.05 anyway? On average **0.85**. Getting exactly
one is not a signal; it is the expected outcome of looking seventeen times. The
crude correction for that (Bonferroni: multiply by the number of looks) turns
0.014 into 0.24.

This is the same logic as running seventeen t-tests and reporting the one that
worked, and it is worth stating flatly because a single p-value below 0.05, seen
in isolation at the bottom of a long output, reads as a discovery.

So why look at it further at all? Because of a *pattern* that is independent of
the threshold. The effect was **larger with the lag than without it** (1.17
versus 1.08). If the score were quietly detecting people already in the act of
quitting, the estimate would be strongest with no lag and would fade as the gap
widened. This one strengthens, which is the wrong shape for that particular
artefact.

That is a reason to examine a number, not to believe it. `examples/fitrec_falling.py`
does the examining — four ways to break the one coefficient, rather than
seventeen more chances to find another:

| test | what it would catch |
|---|---|
| adjust for the *change* in run rate | the score falls partly because the person is doing less; the original model controlled only the *level* |
| reshuffle each person's times of day | anything that survives when the timing structure is destroyed was never about timing |
| split the people at random, fit both halves | an effect that lives in a handful of people |
| vary interval, lag and memory (27 cells) | a result that only exists at one setting |

The second is the strongest of the four, and worth a sentence on its own. It
keeps every event time, every count and each person's overall distribution of
times of day exactly as they are, and destroys only *which time went with which
event*. A genuine "routine coming apart" cannot survive that. A statistical
artefact of a shrinking sample can. Refitting a few hundred times gives a
p-value that assumes nothing at all — and, as a bonus, an honest standard error:
each person contributes about two dozen intervals that are not independent of
each other, and a model that pretends otherwise reports more confidence than it
has earned.

## 10. What none of this establishes

Nothing here is causal. A score that predicts disengagement would be a useful
screening instrument; it would not show that *making* someone regular keeps them
engaged. People whose routines hold differ from people whose routines come apart
in ways no event log records — injury, job, illness, motivation.

And a practical warning: irregular engagement may be irregular life. Using this
to decide where support goes is reasonable. Using it to judge people is not.
