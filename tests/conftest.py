from datetime import datetime

import pytest

from adherence.simulate import PRESETS, simulate_person

START = datetime(2025, 1, 6)  # a Monday


def make(name: str, days: int = 120, seed: int = 11, tz: str = "UTC", dropout: bool = False):
    """A preset person with dropout switched off, so tests measure the score."""
    return simulate_person(PRESETS[name], START, days=days, tz=tz, rng=seed, dropout=dropout)


@pytest.fixture
def clockwork():
    return make("clockwork")[0]


@pytest.fixture
def loose():
    return make("loose")[0]


@pytest.fixture
def chaotic():
    return make("chaotic")[0]


@pytest.fixture
def bimodal():
    return make("bimodal")[0]


@pytest.fixture
def mwf():
    return make("mwf")[0]


@pytest.fixture
def drifter():
    return make("drifter")[0]
