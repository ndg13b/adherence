"""Loader tests, run against a synthetic file in the real Duolingo layout.

The loader is the part that cannot be checked by reading it: a wrong column
index or a broken session merge produces plausible-looking numbers rather than
an error. So the format is reproduced exactly and the parsed result is compared
against timestamps that are known by construction.
"""

import csv
import gzip
import io
import zipfile

import numpy as np
import pytest

from adherence.datasets import (
    DUOLINGO_COLUMNS,
    hour_of_day_histogram,
    in_sample,
    load_duolingo,
    load_event_csv,
    merge_sessions,
    open_text,
    write_synthetic_duolingo,
)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return write_synthetic_duolingo(str(d / "traces.csv.gz"), n_users=60, days=14, seed=1)


# ------------------------------------------------------------------- file types
def test_open_text_handles_csv_gz_and_zip(tmp_path):
    body = "a,b\n1,2\n"
    plain = tmp_path / "x.csv"
    plain.write_text(body)
    gz = tmp_path / "x.csv.gz"
    with gzip.open(gz, "wt") as fh:
        fh.write(body)
    zp = tmp_path / "x.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("x.csv", body)
    zgz = tmp_path / "xg.zip"
    with zipfile.ZipFile(zgz, "w") as zf:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as g:
            g.write(body.encode())
        zf.writestr("x.csv.gz", buf.getvalue())

    for p in (plain, gz, zp, zgz):
        with open_text(str(p)) as fh:
            assert fh.read() == body, p


def test_zip_picks_the_data_file_not_the_readme(tmp_path):
    zp = tmp_path / "d.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("README.txt", "notes")
        zf.writestr("data.csv", "a,b\n" + "1,2\n" * 500)
    with open_text(str(zp)) as fh:
        assert fh.readline().strip() == "a,b"


# ---------------------------------------------------------------- session merge
def test_merge_sessions_collapses_a_burst():
    """Twelve words in one lesson are one engagement, not twelve."""
    t = np.array([1000.0] * 12)
    assert merge_sessions(t, 30.0).size == 1


def test_merge_sessions_keeps_the_first_of_each_burst():
    t = np.array([0.0, 60.0, 120.0, 10_000.0, 10_060.0])
    np.testing.assert_allclose(merge_sessions(t, 30.0), [0.0, 10_000.0])


def test_merge_sessions_separates_distant_events():
    t = np.array([0.0, 3600.0, 7200.0])
    assert merge_sessions(t, 30.0).size == 3
    assert merge_sessions(t, 120.0).size == 1


def test_merge_sessions_zero_gap_only_dedupes_exact():
    t = np.array([5.0, 5.0, 65.0])
    np.testing.assert_allclose(merge_sessions(t, 0.0), [5.0, 65.0])


def test_merge_sessions_handles_empty_and_unsorted():
    assert merge_sessions(np.zeros(0)).size == 0
    np.testing.assert_allclose(merge_sessions(np.array([100.0, 0.0]), 0.5), [0.0, 100.0])


# --------------------------------------------------------------------- sampling
def test_in_sample_is_deterministic():
    ids = [f"u{i}" for i in range(200)]
    assert [in_sample(i, 30) for i in ids] == [in_sample(i, 30) for i in ids]


def test_in_sample_hits_roughly_the_requested_share():
    ids = [f"user-{i}" for i in range(20_000)]
    frac = np.mean([in_sample(i, 10.0) for i in ids])
    assert 0.08 < frac < 0.12


def test_in_sample_full_keeps_everyone():
    assert all(in_sample(f"u{i}", 100.0) for i in range(50))


# ----------------------------------------------------------------------- loader
def test_loads_synthetic_duolingo(synthetic):
    res = load_duolingo(synthetic, sample_pct=100.0, min_events=5, min_days=3,
                        verbose=False)
    assert res.n_rows > 1000
    assert 10 < len(res.logs) <= 60
    assert 10 < res.span_days < 16
    assert "people" in res.summary()


def test_rows_within_a_lesson_become_one_event(synthetic):
    """The decisive loader test: many rows per lesson must not inflate the rate."""
    with open_text(synthetic) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ui, ti = header.index("user_id"), header.index("timestamp")
        raw = {}
        for row in reader:
            raw.setdefault(row[ui], []).append(float(row[ti]))

    res = load_duolingo(synthetic, sample_pct=100.0, min_events=5, min_days=3,
                        gap_minutes=30.0, verbose=False)
    uid, log = next(iter(res.logs.items()))
    expected = merge_sessions(np.array(raw[uid]), 30.0)
    np.testing.assert_allclose(log.t, expected)
    assert len(log) < len(raw[uid])  # the burst really was collapsed


def test_columns_are_found_by_name_not_position(tmp_path):
    """A reordered or extended file must still load."""
    p = tmp_path / "reordered.csv"
    cols = list(reversed(DUOLINGO_COLUMNS)) + ["extra"]
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(40):
            row = {c: "0" for c in cols}
            row["user_id"] = "u1"
            row["timestamp"] = str(1_000_000 + i * 86_400)
            w.writerow([row[c] for c in cols])
    res = load_event_csv(str(p), "user_id", "timestamp", min_events=5, min_days=3,
                         verbose=False)
    assert len(res.logs) == 1


def test_missing_column_names_the_available_ones(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("who,when\nu1,100\n")
    with pytest.raises(ValueError, match="user_id.*not found|not found.*available"):
        load_event_csv(str(p), "user_id", "when", verbose=False)


def test_inclusion_thresholds_are_applied(synthetic):
    loose = load_duolingo(synthetic, sample_pct=100.0, min_events=2, min_days=1,
                          verbose=False)
    strict = load_duolingo(synthetic, sample_pct=100.0, min_events=25, min_days=12,
                           verbose=False)
    assert len(strict.logs) < len(loose.logs)
    assert all(len(v) >= 25 for v in strict.logs.values())


def test_all_people_share_one_observation_window(synthetic):
    """Rates are only comparable if exposure is measured on a common window."""
    res = load_duolingo(synthetic, sample_pct=100.0, min_events=5, min_days=3,
                        verbose=False)
    starts = {v.t_start for v in res.logs.values()}
    ends = {v.t_end for v in res.logs.values()}
    assert len(starts) == 1 and len(ends) == 1


def test_epoch_ms_and_iso_time_formats(tmp_path):
    for fmt, render in [("epoch_ms", lambda t: str(int(t * 1000))),
                        ("iso", lambda t: __import__("datetime").datetime
                         .utcfromtimestamp(t).isoformat() + "+00:00")]:
        p = tmp_path / f"{fmt}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["user_id", "ts"])
            for i in range(30):
                w.writerow(["u1", render(1_600_000_000 + i * 86_400)])
        res = load_event_csv(str(p), "user_id", "ts", time_format=fmt,
                             min_events=5, min_days=3, verbose=False)
        assert len(res.logs) == 1
        assert res.span_days == pytest.approx(29.0, abs=0.1)


def test_malformed_rows_are_skipped_not_fatal(tmp_path):
    p = tmp_path / "messy.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "ts"])
        for i in range(30):
            w.writerow(["u1", str(1_600_000_000 + i * 86_400)])
        w.writerow(["u1", "not-a-number"])
        w.writerow(["u1"])  # short row
    res = load_event_csv(str(p), "user_id", "ts", min_events=5, min_days=3,
                         verbose=False)
    assert len(res.logs["u1"]) == 30


def test_hour_histogram_sums_to_one(synthetic):
    res = load_duolingo(synthetic, sample_pct=100.0, min_events=5, min_days=3,
                        verbose=False)
    h = hour_of_day_histogram(res.logs)
    assert h.shape == (24,)
    assert h.sum() == pytest.approx(1.0)
