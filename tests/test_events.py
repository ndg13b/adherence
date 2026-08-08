from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from adherence.events import (
    EventLog,
    clock_to_phase,
    daily_phase,
    local_dow,
    local_midnight_epoch,
    phase_to_clock,
)

LONDON = ZoneInfo("Europe/London")


def test_phase_matches_local_clock():
    t = datetime(2025, 3, 4, 7, 30, tzinfo=LONDON).timestamp()
    assert daily_phase(np.array([t]), LONDON)[0] == pytest.approx(clock_to_phase(7, 30))


def test_routine_is_wall_clock_not_utc_across_dst():
    """08:00 local before and after the spring transition must be the same phase.

    In UTC these two events are an hour apart; a UTC-based model would record a
    one-hour jump in the routine that the person never experienced.
    """
    before = datetime(2025, 3, 20, 8, 0, tzinfo=LONDON).timestamp()
    after = datetime(2025, 4, 10, 8, 0, tzinfo=LONDON).timestamp()
    ph = daily_phase(np.array([before, after]), LONDON)
    assert ph[0] == pytest.approx(ph[1], abs=1e-9)
    # ...and they really are on opposite sides of the DST switch
    assert (after - before) % 86400 == pytest.approx(82800.0)


def test_dow_monday_is_zero():
    monday = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc).timestamp()
    assert local_dow(np.array([monday]), ZoneInfo("UTC"))[0] == 0
    week = monday + np.arange(7) * 86400.0
    np.testing.assert_array_equal(local_dow(week, ZoneInfo("UTC")), np.arange(7))


def test_local_midnight_epoch_round_trips():
    days = np.arange(20180, 20200)
    mids = local_midnight_epoch(days, LONDON)
    for m in mids:
        dt = datetime.fromtimestamp(m, LONDON)
        assert (dt.hour, dt.minute, dt.second) == (0, 0, 0)


def test_offset_table_agrees_with_datetime():
    """The vectorised offset lookup must match zoneinfo exactly, DST included."""
    base = datetime(2025, 10, 26, tzinfo=timezone.utc).timestamp()
    ts = base + np.arange(-48, 48) * 1800.0
    got = daily_phase(ts, LONDON)
    want = np.array(
        [
            (
                datetime.fromtimestamp(t, LONDON).hour * 3600
                + datetime.fromtimestamp(t, LONDON).minute * 60
                + datetime.fromtimestamp(t, LONDON).second
            )
            / 86400.0
            * 2
            * np.pi
            for t in ts
        ]
    )
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_naive_datetime_rejected():
    with pytest.raises(ValueError, match="naive datetime"):
        EventLog.from_records([datetime(2025, 1, 1, 8, 0)])


def test_assume_tz_accepts_naive():
    log = EventLog.from_records([datetime(2025, 1, 1, 8, 0)], tz="Europe/London", assume_tz=True)
    assert len(log) == 1


def test_events_are_sorted_and_sliced():
    t0 = datetime(2025, 1, 6, tzinfo=timezone.utc)
    ts = [t0 + timedelta(days=d, hours=8) for d in [3, 1, 2, 0]]
    log = EventLog.from_records(ts)
    assert np.all(np.diff(log.t) > 0)
    assert len(log.slice(t_to=log.t[2])) == 2


def test_phase_to_clock_round_trip():
    assert phase_to_clock(clock_to_phase(7, 45)) == "07:45"
    assert phase_to_clock(clock_to_phase(0, 0)) == "00:00"
