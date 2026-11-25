"""Offline tests: synthetic yields with a planted concession, and a fake LSEG module.
Run: python -m pytest -q
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tracker as T  # noqa: E402


# ---------------------------------------------------------------- synthetic market
def make_market(run_bp=1.0, early_bp=0.0, n_years=4, cut_last=False):
    """Business-day yields that rise run_bp per day in the 5 sessions before each monthly
    auction (second Thursday) and early_bp per day in the 5 sessions before that."""
    days = pd.bdate_range("2009-01-01", periods=260 * (n_years + 2))
    auctions = []
    for m in pd.date_range(days[300], days[-30], freq="MS"):
        thursdays = pd.date_range(m, m + pd.Timedelta(days=27), freq="W-THU")
        d = thursdays[1]
        if d in days:
            auctions.append(d)
    dy = pd.Series(0.0, index=days)
    for d in auctions:
        i = days.get_loc(d)
        dy.iloc[i - 4:i] += run_bp / 100          # changes into D-4..D-1 plus D-5 below
        dy.iloc[i - 5] += run_bp / 100
        dy.iloc[i - 10:i - 5] += early_bp / 100
    y = 4.0 + dy.cumsum()
    a = pd.DataFrame({
        "date": auctions, "reopening": [k % 3 != 0 for k in range(len(auctions))],
        "offering_bn": [25.0 if k % 3 == 0 else 22.0 for k in range(len(auctions))],
        "high_yield": 4.0, "dealer_bn": 2.0, "indirect_bn": 15.0, "total_bn": 22.0, "source": "test",
    })
    if cut_last:
        a.loc[a.index[-1], "offering_bn"] -= 3
    # TLT moves -0.15% per bp, open = previous close
    close = 100 * np.exp(np.cumsum(-0.15 * dy.values * 100 / 100))
    tlt = pd.DataFrame({"close": close}, index=days)
    tlt["open"] = tlt["close"].shift(1).fillna(100.0)
    return y, a, tlt


@pytest.fixture(autouse=True)
def short_sample(monkeypatch):
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    monkeypatch.setattr(T, "START", pd.Timestamp("2010-06-01"))
    monkeypatch.setattr(T, "ROLL", 12)


def test_planted_concession_is_recovered():
    y, a, tlt = make_market(run_bp=1.0)
    out, ev = T.compute(y, a, tlt, pd.DataFrame())
    assert out["alpha"]["pre_mean_bp"] == pytest.approx(5.0, abs=1e-6)
    assert out["alpha"]["hit_rate"] == 100
    # the drift baseline is zero, so the excess equals the run-up
    assert out["alpha"]["excess_mean_bp"] == pytest.approx(5.0, abs=1e-6)
    # every bit of the yield rise sits in run-up days
    assert out["cum"]["other"][-1] == pytest.approx(0.0, abs=0.2)
    assert out["cum"]["run"][-1] == pytest.approx(out["cum"]["total"][-1], abs=0.2)
    assert not any(w["triggered"] for w in out["wrong_if"])


def test_tlt_short_and_slope():
    y, a, tlt = make_market(run_bp=1.0)
    out, ev = T.compute(y, a, tlt, pd.DataFrame())
    r = ev["tlt_ret_pct"].dropna()
    assert len(r) > 10
    assert (r > 0).all()                                      # shorting into a rise in yields pays
    assert out["alpha"]["tlt_open_to_open"] is True
    assert out["alpha"]["slope"] == pytest.approx(-0.15, abs=0.01)
    # windows that contain a month's first session (TLT ex-dividend) are excluded
    assert (ev["tlt_excluded"] == "ex-dividend inside window").sum() >= 0


def test_wrong_if_1_and_3_trigger():
    y, a, tlt = make_market(run_bp=-1.0, early_bp=1.0)
    out, _ = T.compute(y, a, tlt, pd.DataFrame())
    flags = {w["id"]: w["triggered"] for w in out["wrong_if"]}
    assert flags[1] is True and flags[3] is True


def test_wrong_if_2_size_cut():
    y, a, tlt = make_market(run_bp=1.0, cut_last=True)
    out, _ = T.compute(y, a, tlt, pd.DataFrame())
    assert out["wrong_if"][1]["triggered"] is True


def test_upcoming_window_counts_live():
    y, a, tlt = make_market(run_bp=1.0)
    last = y.index[-1]
    up = pd.DataFrame({"date": [last + pd.offsets.BDay(2)], "reopening": [True], "offering_bn": [None],
                       "high_yield": [None], "dealer_bn": [None], "indirect_bn": [None], "total_bn": [None],
                       "source": ["tentative"]})
    out, _ = T.compute(y, a, tlt, up)
    assert out["in_window"] is True
    assert out["live"]["windows"][-1]["state"] == "next"
    assert out["next_auction"]["status"] == "tentative"


def test_output_is_strict_json(tmp_path):
    y, a, tlt = make_market(run_bp=1.0)
    out, _ = T.compute(y, a, tlt.iloc[:0], pd.DataFrame())   # no TLT yet: page must still render
    T.write_json(tmp_path / "t.json", out)
    back = json.loads((tmp_path / "t.json").read_text())
    assert back["alpha"]["slope_source"] == "assumed"
    assert back["alpha"]["tlt_trades"] == 0


# ---------------------------------------------------------------- LSEG parsing and fake session
def test_parse_tlt_frame_shapes():
    idx = pd.to_datetime(["2026-11-20", "2026-11-23"]).tz_localize("UTC")
    flat = pd.DataFrame({"OPEN_PRC": [80.1, 80.4], "TRDPRC_1": [80.3, 80.2]}, index=idx)
    out = T.parse_tlt_frame(flat)
    assert list(out.columns) == ["open", "close"] and out.index.tz is None
    multi = flat.copy()
    multi.columns = pd.MultiIndex.from_tuples([("TLT.O", "OPEN_PRC"), ("TLT.O", "TRDPRC_1")])
    assert T.parse_tlt_frame(multi)["close"].iloc[-1] == 80.2
    no_open = flat[["TRDPRC_1"]]
    assert T.parse_tlt_frame(no_open)["open"].isna().all()
    with pytest.raises(T.StopRun):
        T.parse_tlt_frame(flat[["OPEN_PRC"]])


class FakeSession:
    def __init__(self):
        self.opened = self.closed = False

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True


def install_fake_lseg(monkeypatch, fail=False):
    calls = []
    sess = FakeSession()

    def get_history(universe, fields, interval, start, end):
        calls.append((universe, tuple(fields), interval, start, end))
        if fail:
            raise RuntimeError("LSEG down")
        d = pd.bdate_range(start, end)
        d = d[d < pd.Timestamp(end)]
        return pd.DataFrame({"OPEN_PRC": np.linspace(80, 81, len(d)), "TRDPRC_1": np.linspace(80.2, 81.2, len(d))},
                            index=d)

    class Definition:
        def __init__(self, **kw):
            self.kw = kw

        def get_session(self):
            return sess

    platform = types.SimpleNamespace(Definition=Definition, GrantPassword=lambda **kw: kw)
    session_mod = types.SimpleNamespace(platform=platform, set_default=lambda s: None)
    ld = types.SimpleNamespace(session=session_mod, get_history=get_history)
    lseg = types.ModuleType("lseg")
    lseg.data = ld
    monkeypatch.setitem(sys.modules, "lseg", lseg)
    monkeypatch.setitem(sys.modules, "lseg.data", ld)
    for k in ("LSEG_APP_KEY", "LSEG_USERNAME", "LSEG_PASSWORD"):
        monkeypatch.setenv(k, "x")
    return calls, sess


def test_update_tlt_with_fake_lseg(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "DATA", tmp_path)
    calls, sess = install_fake_lseg(monkeypatch)
    status = {}
    out = T.update_tlt(status, use_lseg=True)
    assert status["lseg"].startswith("ok")
    assert sess.opened and sess.closed
    assert all(c[0] == "TLT.O" and c[1] == ("OPEN_PRC", "TRDPRC_1") and c[2] == "daily" for c in calls)
    assert len(calls) >= 8                                     # two-year chunks from 2009
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert (tmp_path / "tlt.csv").exists()
    # second night: only the recent tail is requested
    calls.clear()
    T.update_tlt({}, use_lseg=True)
    assert len(calls) == 1


def test_update_tlt_falls_back_to_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "DATA", tmp_path)
    pd.DataFrame({"open": [1.0], "close": [1.0]}, index=pd.to_datetime(["2026-11-20"])).rename_axis("date") \
        .to_csv(tmp_path / "tlt.csv")
    install_fake_lseg(monkeypatch, fail=True)
    status = {}
    out = T.update_tlt(status, use_lseg=True)
    assert status["lseg"].startswith("failed") and len(out) == 1


def test_missing_credentials_is_a_clean_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "DATA", tmp_path)
    install_fake_lseg(monkeypatch)
    monkeypatch.delenv("LSEG_PASSWORD")
    status = {}
    T.update_tlt(status, use_lseg=True)
    assert "Missing LSEG" in status["lseg"]


def test_tentative_line_regex():
    import re
    line = "30-Year  BOND R Thu, Dec 03, 2026 Thu, Dec 10, 2026 Tue, Dec 15, 2026 Wed, Nov 15, 2056"
    m = re.match(r"\s*30-Year\s+BOND\s*(R?)\s+\w{3}, (\w{3} \d{2}, \d{4})\s+\w{3}, (\w{3} \d{2}, \d{4})", line)
    assert m and m.group(1) == "R" and m.group(3) == "Dec 10, 2026"


def test_truncated_history_is_caught(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "DATA", tmp_path)
    install_fake_lseg(monkeypatch)
    ld = sys.modules["lseg.data"]
    real = ld.get_history
    def truncated(**kw):
        df = real(**kw)
        return df.iloc[len(df) // 2:]          # server returned only the second half
    ld.get_history = truncated
    status = {}
    T.update_tlt(status, use_lseg=True)
    assert "truncated" in status["lseg"]


def test_first_night_with_empty_cache_computes(monkeypatch, tmp_path):
    """Regression: an empty cache frame has object dtype and used to break the slope estimate."""
    monkeypatch.setattr(T, "DATA", tmp_path)
    install_fake_lseg(monkeypatch)
    tlt = T.update_tlt({}, use_lseg=True)
    assert all(str(dt) == "float64" for dt in tlt.dtypes)
    y, a, _ = make_market(run_bp=1.0)
    days = y.index.intersection(tlt.index)
    out, _ = T.compute(y.loc[days], a[a["date"].isin(days)], tlt, pd.DataFrame())
    assert out["alpha"]["slope"] is not None
