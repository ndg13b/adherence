"""Circular kernels used to give each engagement event a pair of "tails" in time.

An event that happened at 07:00 does not only support the hypothesis "this person
trains at 07:00" -- it also lends some support to 06:45 and 07:20, and less to
09:00. A kernel makes that decay explicit and, crucially, normalised: every
kernel here integrates to 1 over the circle, so a set of events becomes a proper
probability density over time-of-day (or time-of-week).

All kernels operate on a circle of circumference ``2*pi`` radians. Bandwidths are
supplied in *minutes* and converted with the period of the cycle being modelled
(1440 minutes for a daily cycle, 10080 for a weekly one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import i0e

TWO_PI = 2.0 * math.pi
DAY_MINUTES = 1440.0
WEEK_MINUTES = 10080.0

#: Log density of the uniform distribution on the circle -- the "no routine at
#: all" reference against which every score in this package is normalised.
LOG_UNIFORM = -math.log(TWO_PI)


def wrap_to_pi(delta: np.ndarray | float) -> np.ndarray:
    """Map an angular difference into ``[-pi, pi)``."""
    return (np.asarray(delta, dtype=float) + math.pi) % TWO_PI - math.pi


class CircularKernel:
    """Base class: a unimodal density on the circle, centred at 0."""

    #: Nominal spread in radians, used for reporting and for peak normalisation.
    sigma: float

    def log_pdf(self, delta: np.ndarray | float) -> np.ndarray:
        raise NotImplementedError

    def pdf(self, delta: np.ndarray | float) -> np.ndarray:
        return np.exp(self.log_pdf(delta))

    @property
    def peak_log_pdf(self) -> float:
        """Log density at the centre -- the best any single event can score."""
        return float(self.log_pdf(0.0))

    @property
    def bits_ceiling(self) -> float:
        """Maximum bits of timing information a perfectly on-time event can earn.

        This is the ceiling used to rescale information-based scores onto
        ``[0, 1]``: it depends only on the bandwidth, not on the data.
        """
        return (self.peak_log_pdf - LOG_UNIFORM) / math.log(2.0)


@dataclass
class VonMises(CircularKernel):
    """The natural Gaussian analogue on a circle. Default choice.

    Parameterised by ``sigma`` (radians); the concentration is ``1 / sigma**2``.
    Evaluated in log space via the exponentially scaled Bessel function so that
    tight bandwidths (large concentration) do not overflow.
    """

    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        self.kappa = 1.0 / (self.sigma**2)
        # log(1 / (2*pi*I0(kappa))) with I0 factored as exp(kappa) * i0e(kappa)
        self._log_norm = -(math.log(TWO_PI) + math.log(float(i0e(self.kappa))) + self.kappa)

    def log_pdf(self, delta):
        d = np.asarray(delta, dtype=float)
        return self.kappa * np.cos(d) + self._log_norm


@dataclass
class WrappedNormal(CircularKernel):
    """A Gaussian wrapped around the circle, summed over a few windings."""

    sigma: float
    n_wraps: int = 3

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def log_pdf(self, delta):
        d = wrap_to_pi(delta)[..., None]
        wraps = np.arange(-self.n_wraps, self.n_wraps + 1) * TWO_PI
        z = (d + wraps) / self.sigma
        log_terms = -0.5 * z**2 - math.log(self.sigma * math.sqrt(TWO_PI))
        m = log_terms.max(axis=-1)
        return m + np.log(np.exp(log_terms - m[..., None]).sum(axis=-1))


@dataclass
class TwoPieceWrappedNormal(CircularKernel):
    """Asymmetric tails: different tolerance for being early vs. being late.

    Running 20 minutes late is not the same event as starting 20 minutes early --
    an evening routine, for instance, usually has a hard front edge (you cannot
    start before you get home) and a long back tail. ``sigma_before`` governs
    times *earlier* than the anchor, ``sigma_after`` times later.
    """

    sigma_before: float
    sigma_after: float
    n_wraps: int = 3

    def __post_init__(self) -> None:
        if self.sigma_before <= 0 or self.sigma_after <= 0:
            raise ValueError("sigmas must be positive")
        self.sigma = 0.5 * (self.sigma_before + self.sigma_after)
        # Two-piece normal: f(x) = 2/(s1+s2) * phi(x/s_side), continuous at 0
        # and integrating to s1/(s1+s2) + s2/(s1+s2) = 1.
        self._log_scale = (
            math.log(2.0)
            - math.log(self.sigma_before + self.sigma_after)
            - 0.5 * math.log(TWO_PI)
        )

    def log_pdf(self, delta):
        d = wrap_to_pi(delta)[..., None]
        wraps = np.arange(-self.n_wraps, self.n_wraps + 1) * TWO_PI
        x = d + wraps
        sig = np.where(x < 0, self.sigma_before, self.sigma_after)
        log_terms = -0.5 * (x / sig) ** 2 + self._log_scale
        m = log_terms.max(axis=-1)
        return m + np.log(np.exp(log_terms - m[..., None]).sum(axis=-1))


@dataclass
class Epanechnikov(CircularKernel):
    """A bump that terminates: zero support beyond ``halfwidth`` radians.

    Use this when the model should say "an 03:00 session tells us literally
    nothing about a 09:00 routine" rather than assigning it a vanishing but
    non-zero amount of evidence.
    """

    halfwidth: float

    def __post_init__(self) -> None:
        if not 0 < self.halfwidth <= math.pi:
            raise ValueError("halfwidth must be in (0, pi]")
        self.sigma = self.halfwidth / math.sqrt(5.0)
        self._peak = 0.75 / self.halfwidth

    def log_pdf(self, delta):
        u = wrap_to_pi(delta) / self.halfwidth
        inside = np.abs(u) < 1.0
        dens = np.where(inside, self._peak * (1.0 - u**2), 0.0)
        with np.errstate(divide="ignore"):
            return np.log(dens)


@dataclass
class Tricube(CircularKernel):
    """Compact support like :class:`Epanechnikov` but with softer shoulders."""

    halfwidth: float

    def __post_init__(self) -> None:
        if not 0 < self.halfwidth <= math.pi:
            raise ValueError("halfwidth must be in (0, pi]")
        self.sigma = self.halfwidth * math.sqrt(35.0 / 243.0)
        self._peak = (70.0 / 81.0) / self.halfwidth

    def log_pdf(self, delta):
        u = np.abs(wrap_to_pi(delta)) / self.halfwidth
        inside = u < 1.0
        dens = np.where(inside, self._peak * (1.0 - np.where(inside, u, 0.0) ** 3) ** 3, 0.0)
        with np.errstate(divide="ignore"):
            return np.log(dens)


_COMPACT = {"epanechnikov", "tricube"}


def make_kernel(
    name: str = "vonmises",
    bandwidth_min: float | tuple[float, float] = 45.0,
    period_min: float = DAY_MINUTES,
) -> CircularKernel:
    """Build a kernel from a human-scale bandwidth in minutes.

    Parameters
    ----------
    name:
        ``"vonmises"`` (default), ``"wrapped_normal"``, ``"epanechnikov"`` or
        ``"tricube"``. Pass a 2-tuple bandwidth to get the asymmetric
        two-piece wrapped normal.
    bandwidth_min:
        For the smooth kernels this is the standard deviation in minutes -- read
        it as "how many minutes off their usual time is still recognisably the
        same session". For the compact kernels it is the hard cut-off half-width.
    period_min:
        1440 for a daily cycle, 10080 for a weekly one.
    """
    name = name.lower()
    to_rad = lambda m: TWO_PI * float(m) / float(period_min)  # noqa: E731

    if isinstance(bandwidth_min, (tuple, list)):
        if name not in ("vonmises", "wrapped_normal", "two_piece"):
            raise ValueError("asymmetric bandwidth requires a wrapped-normal kernel")
        before, after = bandwidth_min
        return TwoPieceWrappedNormal(to_rad(before), to_rad(after))

    if name == "vonmises":
        return VonMises(to_rad(bandwidth_min))
    if name == "wrapped_normal":
        return WrappedNormal(to_rad(bandwidth_min))
    if name == "epanechnikov":
        return Epanechnikov(min(to_rad(bandwidth_min), math.pi))
    if name == "tricube":
        return Tricube(min(to_rad(bandwidth_min), math.pi))
    raise ValueError(f"unknown kernel {name!r}")
