import math

import numpy as np
import pytest

from adherence.kernels import (
    TWO_PI,
    Epanechnikov,
    Tricube,
    TwoPieceWrappedNormal,
    VonMises,
    WrappedNormal,
    make_kernel,
    wrap_to_pi,
)

ALL = [
    VonMises(0.2),
    WrappedNormal(0.2),
    Epanechnikov(0.6),
    Tricube(0.6),
    TwoPieceWrappedNormal(0.1, 0.4),
]


@pytest.mark.parametrize("k", ALL, ids=lambda k: type(k).__name__)
def test_integrates_to_one(k):
    g = np.linspace(-math.pi, math.pi, 200_001)
    assert np.trapezoid(k.pdf(g), g) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("k", ALL, ids=lambda k: type(k).__name__)
def test_peak_is_at_centre(k):
    g = np.linspace(-math.pi, math.pi, 4001)
    assert k.peak_log_pdf >= k.log_pdf(g).max() - 1e-9


@pytest.mark.parametrize("k", ALL, ids=lambda k: type(k).__name__)
def test_periodic(k):
    g = np.linspace(-math.pi, math.pi, 101)
    np.testing.assert_allclose(k.pdf(g), k.pdf(g + TWO_PI), rtol=1e-8, atol=1e-10)


def test_tight_bandwidth_does_not_overflow():
    # 1-minute bandwidth => concentration ~1.2e7; the naive Bessel form overflows.
    k = make_kernel("vonmises", bandwidth_min=1.0)
    assert np.isfinite(k.peak_log_pdf)
    assert np.isfinite(k.log_pdf(np.array([0.0, 0.5, math.pi]))).all()


def test_compact_kernel_has_zero_tail():
    k = make_kernel("epanechnikov", bandwidth_min=60.0)  # +/-1 hour hard cut-off
    far = TWO_PI * 120.0 / 1440.0
    assert k.pdf(np.array([far]))[0] == 0.0
    assert k.pdf(np.array([0.0]))[0] > 0.0


def test_asymmetric_tails_favour_the_wide_side():
    k = make_kernel("wrapped_normal", bandwidth_min=(10.0, 60.0))
    early = TWO_PI * -30.0 / 1440.0
    late = TWO_PI * 30.0 / 1440.0
    assert k.pdf(np.array([late]))[0] > k.pdf(np.array([early]))[0]


def test_bits_ceiling_grows_as_bandwidth_tightens():
    wide = make_kernel("vonmises", bandwidth_min=120.0).bits_ceiling
    tight = make_kernel("vonmises", bandwidth_min=10.0).bits_ceiling
    assert tight > wide > 0


def test_wrap_to_pi():
    np.testing.assert_allclose(wrap_to_pi(np.array([0.0, TWO_PI, -TWO_PI])), 0.0, atol=1e-12)
    assert wrap_to_pi(np.array([math.pi + 0.1]))[0] < 0
