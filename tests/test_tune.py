import pytest

from adherence import RoutineModel
from adherence.evaluate import compare
from adherence.baselines import HomogeneousModel
from adherence.simulate import PersonProfile, Slot, simulate_person
from adherence.tune import auto_model, select_bandwidth, select_half_life
from conftest import START


@pytest.mark.parametrize("jitter,lo,hi", [(8.0, 3.0, 20.0), (30.0, 12.0, 60.0), (75.0, 30.0, 180.0)])
def test_selected_bandwidth_tracks_true_jitter(jitter, lo, hi):
    log, _ = simulate_person(
        PersonProfile(slots=[Slot(hour=9.0, jitter_min=jitter, p=0.95)]),
        START, days=150, rng=17, dropout=False,
    )
    bw = select_bandwidth(log).best["bandwidth_min"]
    assert lo <= bw <= hi


def test_selection_is_monotone_across_people():
    picks = []
    for jitter in (8.0, 30.0, 75.0):
        log, _ = simulate_person(
            PersonProfile(slots=[Slot(hour=9.0, jitter_min=jitter, p=0.95)]),
            START, days=150, rng=17, dropout=False,
        )
        picks.append(select_bandwidth(log).best["bandwidth_min"])
    assert picks == sorted(picks)


def test_noise_selects_the_widest_kernel(chaotic):
    """With nothing to find, the honest choice is to smooth everything away."""
    sel = select_bandwidth(chaotic)
    assert sel.best["bandwidth_min"] >= 90.0
    narrow = next(r["bits"] for r in sel.table if r["params"]["bandwidth_min"] == 5.0)
    assert narrow < 0  # a tight kernel on noise is worse than saying "any time"


def test_tuning_improves_the_forecast(clockwork):
    """Selection has to pay for itself out of sample, not just fit better."""
    bw = select_bandwidth(clockwork).best["bandwidth_min"]
    res = compare(
        clockwork,
        {
            "auto": RoutineModel(bandwidth_min=bw),
            "default": RoutineModel(),
            "homogeneous": HomogeneousModel(),
        },
        bin_min=30, warmup_days=21,
    )
    assert res["auto"][1]["log_loss_bits_skill"] > res["default"][1]["log_loss_bits_skill"]


def test_half_life_selection_prefers_short_memory_for_a_drifting_routine(drifter, clockwork):
    hl_drift = select_half_life(drifter, model=RoutineModel(bandwidth_min=20)).best["half_life_days"]
    hl_fixed = select_half_life(clockwork, model=RoutineModel(bandwidth_min=20)).best["half_life_days"]
    assert hl_drift <= hl_fixed


def test_auto_model_returns_a_fitted_model(clockwork):
    m, sel = auto_model(clockwork)
    assert m.fitted
    assert set(sel.best) == {"bandwidth_min", "half_life_days"}
    assert 0.0 <= m.p_engage_next(24.0) <= 1.0
    assert "bandwidth_min" in str(sel)
