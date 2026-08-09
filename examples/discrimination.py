"""How the kernel score compares with the established regularity indices.

Simulates people who differ in exactly one respect -- how tightly their sessions
cluster around a single daily slot -- and asks each index to tell them apart.
This isolates measurement resolution: everything else (rate, hour, span,
timezone) is held fixed, so any index that fails to separate the rows is blind
to the thing being varied.

    python examples/discrimination.py
"""

import math
from datetime import datetime

import numpy as np

import sys
from pathlib import Path

# Run straight from a clone, with or without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adherence import timing_consistency
from adherence.baselines import (
    interdaily_stability,
    sleep_regularity_index,
    social_rhythm_hit_rate,
    timing_entropy_bits,
)
from adherence.scores import prequential_timing_bits
from adherence.simulate import PersonProfile, Slot, simulate_person

START = datetime(2025, 1, 6)
JITTERS = [3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 150.0]
INDICES = ("srm", "sri", "is", "entropy", "kernel", "bits")
N_REPEATS = 20


def run(days: int = 150) -> dict:
    """Returns ``{index: array(n_jitters, n_repeats)}`` plus the jitter axis."""
    out = {k: np.zeros((len(JITTERS), N_REPEATS)) for k in INDICES}
    for i, jitter in enumerate(JITTERS):
        for seed in range(N_REPEATS):
            log, _ = simulate_person(
                PersonProfile(slots=[Slot(hour=8.0, jitter_min=jitter, p=1.0)]),
                START, days=days, rng=seed, dropout=False,
            )
            out["srm"][i, seed] = social_rhythm_hit_rate(log)
            out["sri"][i, seed] = sleep_regularity_index(log)
            out["is"][i, seed] = interdaily_stability(log)
            out["entropy"][i, seed] = timing_entropy_bits(log)
            out["kernel"][i, seed] = timing_consistency(log)
            out["bits"][i, seed] = prequential_timing_bits(log)[0]
    out["jitter"] = np.array(JITTERS)
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised separation between two groups of individual scores.

    A saturated index gives both groups the identical score with zero variance;
    that is zero discrimination, not infinite, so the degenerate case is
    resolved by the numerator.
    """
    diff = abs(a.mean() - b.mean())
    sd = math.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    if sd == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / sd


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    r = run()
    m = {k: r[k].mean(axis=1) for k in INDICES}

    print(f"Single daily slot; only the jitter varies. Means over {N_REPEATS} "
          "simulated people, 150 days each.\n")
    print(f"{'true jitter':>11s} {'SRM +/-45':>10s} {'SRI':>7s} {'IS':>7s} "
          f"{'entropy':>8s} {'kernel':>7s} {'bits/ev':>8s}")
    for i, j in enumerate(r["jitter"]):
        print(f"{j:9.0f}m  {m['srm'][i]:10.3f} {m['sri'][i]:7.1f} {m['is'][i]:7.3f} "
              f"{m['entropy'][i]:8.2f} {m['kernel'][i]:7.3f} {m['bits'][i]:8.2f}")

    print("\nRank correlation with true jitter, on the group means:")
    print("  " + "  ".join(f"{k}={spearman(r['jitter'], m[k]):+.2f}" for k in INDICES))
    print("  -- every index orders the groups correctly, so averaged over enough")
    print("     people they all 'work'. The differences show up per person.\n")

    print("Separation of two INDIVIDUALS one step apart in jitter (Cohen's d).")
    print("d < 0.5 means the index mostly cannot tell these two people apart:\n")
    header = "  ".join(f"{k:>7s}" for k in INDICES)
    print(f"{'comparison':>16s}  {header}")
    for i in range(len(JITTERS) - 1):
        lo, hi = r["jitter"][i], r["jitter"][i + 1]
        ds = [cohens_d(r[k][i], r[k][i + 1]) for k in INDICES]
        print(f"{f'{lo:.0f}m vs {hi:.0f}m':>16s}  " + "  ".join(f"{d:7.2f}" for d in ds))

    fine = [i for i, j in enumerate(JITTERS) if j <= 20.0][:-1]
    print("\nMean d over the fine-grained comparisons (jitter <= 20 min):")
    for k in INDICES:
        d = np.mean([cohens_d(r[k][i], r[k][i + 1]) for i in fine])
        print(f"  {k:8s} {d:5.2f}")

    print(
        "\nBelow ~20 minutes of jitter -- the range that separates a genuinely\n"
        "habitual user from a merely willing one -- the hard +/-45 minute window\n"
        "and the binary 30-minute-epoch indices have almost no resolution left,\n"
        "because every session lands in the same box either way. The graded\n"
        "kernel keeps separating people there. That range is the whole reason to\n"
        "bother: at 90 minutes of jitter every index agrees the routine is gone."
    )


if __name__ == "__main__":
    main()
