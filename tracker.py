"""
Nightly tracker for the 30-year auction concession thesis.

Belief: the 30-year Treasury yield rises in the five sessions before each
30-year bond auction, because buyers have to be paid to absorb a large,
pre-announced block of long-duration supply.

Every night this script:
  1. pulls 30-year auction dates and sizes (Treasury Fiscal Data + TreasuryDirect),
  2. pulls the daily 30-year yield (FRED DGS30, extended with Treasury's own par-yield file),
  3. pulls TLT daily open and close from LSEG (the instrument the strategy will trade),
  4. writes data/tracker.json for index.html, plus CSVs anyone can download.

If a source fails, the last good copy in data/ is used and the failure is logged.

Run:   python tracker.py              normal nightly run
       python tracker.py --no-lseg    skip LSEG (local runs without credentials)
       python tracker.py --diagnose   test the LSEG connection only
"""

import argparse
import io
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"

START = pd.Timestamp("2010-01-01")   # sample start for every statistic
WINDOW = 5          # sessions before the auction that the short covers
ROLL = 24           # auctions in the rolling wrong-if statistics (two years)
BASE_DAYS = 252     # trailing sessions used for the drift baseline
LIVE_DAYS = 120     # sessions shown in the live chart
TLT_RIC = "TLT.O"
TRADES_PER_YEAR = 12

# Cost assumptions for the alpha note (per round trip, in % of notional)
SPREAD_COST = 0.02      # TLT's median bid/ask is about 0.01%; paid going in and coming out
BORROW_FEE = 0.50       # annual fee to borrow TLT, % (general collateral; conservative)
HOLD_DAYS = 7           # calendar days from the open five sessions before to the open on auction day
FALLBACK_SLOPE = -0.147 # TLT % move per 1bp: minus its 14.70-year effective duration (iShares, Nov 17 2026).
                        # Used only until LSEG history lets the script estimate it.

FISCAL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
TD = "https://www.treasurydirect.gov/TA_WS/securities"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS30"
UST_PAR = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           "daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
TENTATIVE = "https://home.treasury.gov/system/files/221/TentativeAuctionScheduleQ{q}{y}.pdf"
HEADERS = {"User-Agent": "535Final nightly tracker (github.com/apinyanadal/535Final)"}


class StopRun(Exception):
    pass


# ================================================================ helpers
def http_get(url, params=None, tries=3, timeout=60):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:  # network hiccups are common on shared runners
            last = e
            time.sleep(2 * (k + 1))
    raise last


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def r1(x, n=2):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, allow_nan=False, default=str)


def is_30y(term):
    t = str(term or "")
    return t.startswith("30") or t.startswith("29")


# ================================================================ auctions
def _norm(date_, term, reopening, offering, high_yield, tips, dealer, indirect, total, source):
    return {
        "date": pd.Timestamp(str(date_)[:10]),
        "term": str(term),
        "reopening": str(reopening).strip().lower() == "yes",
        "offering_bn": None if num(offering) is None else num(offering) / 1e9,
        "high_yield": num(high_yield),
        "tips": str(tips).strip().lower() == "yes",
        "dealer_bn": None if num(dealer) is None else num(dealer) / 1e9,
        "indirect_bn": None if num(indirect) is None else num(indirect) / 1e9,
        "total_bn": None if num(total) is None else num(total) / 1e9,
        "source": source,
    }


def fetch_fiscal_history():
    """Full history of bond auctions from Treasury Fiscal Data. Lags about a month."""
    rows, page = [], 1
    fields = ("auction_date,original_security_term,reopening,offering_amt,high_yield,"
              "inflation_index_security,primary_dealer_accepted,indirect_bidder_accepted,total_accepted")
    while True:
        js = http_get(FISCAL, params={
            "fields": fields, "filter": "security_type:eq:Bond,auction_date:gte:2005-01-01",
            "sort": "auction_date", "page[size]": 1000, "page[number]": page}).json()
        for r in js["data"]:
            rows.append(_norm(r["auction_date"], r["original_security_term"], r["reopening"], r["offering_amt"],
                              r["high_yield"], r["inflation_index_security"], r["primary_dealer_accepted"],
                              r["indirect_bidder_accepted"], r["total_accepted"], "fiscaldata"))
        if page >= js["meta"]["total-pages"]:
            break
        page += 1
    return pd.DataFrame(rows)


def _td_rows(records, source):
    out = []
    for r in records:
        d = r.get("competitiveAuctionCloseDate") or r.get("auctionDate")
        if not d:
            continue
        out.append(_norm(d, r.get("originalSecurityTerm") or r.get("securityTerm"), r.get("reopening"),
                         r.get("offeringAmount"), r.get("highYield"), r.get("inflationProtected"),
                         r.get("primaryDealersCompetitiveParAwarded"), r.get("indirectBiddersCompetitiveParAwarded"),
                         r.get("totalParAwardedInsideOfOfferingAmount"), source))
    return out


def fetch_treasurydirect():
    """Most recent ~250 bond auctions (current to the day) and any announced, upcoming ones."""
    done = http_get(f"{TD}/auctioned", params={"format": "json", "type": "Bond", "pagesize": 250}).json()
    upcoming = http_get(f"{TD}/upcoming", params={"format": "json"}).json()
    upcoming = [u for u in upcoming if (u.get("securityType") or u.get("type")) == "Bond"]
    return pd.DataFrame(_td_rows(done, "treasurydirect")), pd.DataFrame(_td_rows(upcoming, "announced"))


def fetch_tentative(status):
    """30-year auctions from Treasury's quarterly tentative schedule (published at each refunding,
    months ahead). This is how the dates are known before the formal announcement a week out."""
    import re
    from pypdf import PdfReader
    today = pd.Timestamp(date.today())
    tried = []
    for back in range(0, 3):
        t = today - pd.DateOffset(months=3 * back)
        url = TENTATIVE.format(q=(t.month - 1) // 3 + 1, y=t.year)
        tried.append(url)
        try:
            raw = http_get(url, tries=1).content
        except Exception:
            continue
        text = "\n".join(pg.extract_text() or "" for pg in PdfReader(io.BytesIO(raw)).pages)
        rows = []
        for line in text.splitlines():
            m = re.match(r"\s*30-Year\s+BOND\s*(R?)\s+\w{3}, (\w{3} \d{2}, \d{4})\s+\w{3}, (\w{3} \d{2}, \d{4})", line)
            if m:
                rows.append(_norm(pd.to_datetime(m.group(3), format="%b %d, %Y"), "30-Year",
                                  "Yes" if m.group(1) else "No", None, None, "No", None, None, None, "tentative"))
        if rows:
            status["tentative_schedule"] = f"ok ({url.rsplit('/', 1)[-1]})"
            return pd.DataFrame(rows)
    status["tentative_schedule"] = "not found"
    return pd.DataFrame()


def build_auctions(status):
    """30-year nominal auctions, history merged with the current feed. Falls back to the cached CSV."""
    cache = DATA / "auctions_30y.csv"
    frames, upcoming = [], pd.DataFrame()
    try:
        frames.append(fetch_fiscal_history())
        status["fiscaldata"] = "ok"
    except Exception as e:
        status["fiscaldata"] = f"failed: {e.__class__.__name__}"
    try:
        recent, upcoming = fetch_treasurydirect()
        frames.append(recent)
        status["treasurydirect"] = "ok"
    except Exception as e:
        status["treasurydirect"] = f"failed: {e.__class__.__name__}"
    if cache.exists():
        old = pd.read_csv(cache, parse_dates=["date"])
        old["source"] = "cache"
        frames.append(old)
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise StopRun("No auction data from any source and no cache")
    a = pd.concat(frames, ignore_index=True)
    a = a[a["term"].map(is_30y) & ~a["tips"].astype(bool)].copy()
    rank = {"treasurydirect": 0, "fiscaldata": 1, "cache": 2}
    a["_r"] = a["source"].map(rank).fillna(3)
    a = a.sort_values(["date", "_r"]).drop_duplicates("date", keep="first").drop(columns="_r")
    a = a.sort_values("date").reset_index(drop=True)
    keep = ["date", "term", "tips", "reopening", "offering_bn", "high_yield", "dealer_bn", "indirect_bn",
            "total_bn", "source"]
    a[keep].to_csv(cache, index=False)

    # Upcoming: formally announced auctions (with size) first, then the tentative schedule
    ups = []
    if not upcoming.empty:
        ups.append(upcoming[upcoming["term"].map(is_30y) & ~upcoming["tips"].astype(bool)])
    try:
        tent = fetch_tentative(status)
        if not tent.empty:
            ups.append(tent)
    except Exception as e:
        status["tentative_schedule"] = f"failed: {e.__class__.__name__}"
    up = pd.DataFrame(columns=keep)
    if ups:
        up = pd.concat(ups, ignore_index=True)
        up["_r"] = up["source"].map({"announced": 0, "tentative": 1})
        up = up.sort_values(["date", "_r"]).drop_duplicates("date").drop(columns="_r")
        up = up[up["date"] > a["date"].max()].sort_values("date").reset_index(drop=True)
    return a[keep], up


# ================================================================ yields
def fetch_yields(status):
    """Daily 30-year constant-maturity yield, % . FRED for history, Treasury's par-yield file for the latest days."""
    cache = DATA / "dgs30.csv"
    parts = []
    try:
        df = pd.read_csv(io.StringIO(http_get(FRED).text))
        s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                      index=pd.to_datetime(df.iloc[:, 0])).dropna()
        parts.append(s)
        status["fred"] = "ok"
    except Exception as e:
        status["fred"] = f"failed: {e.__class__.__name__}"
    try:
        today = date.today()
        years = [today.year - 1, today.year] if today.month == 1 else [today.year]
        for y in years:
            df = pd.read_csv(io.StringIO(http_get(UST_PAR.format(y=y)).text))
            col = [c for c in df.columns if c.strip().lower() in ("30 yr", "30 yr.")][0]
            s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                          index=pd.to_datetime(df["Date"], format="%m/%d/%Y")).dropna()
            parts.append(s)
        status["treasury_par"] = "ok"
    except Exception as e:
        status["treasury_par"] = f"failed: {e.__class__.__name__}"
    if cache.exists():
        old = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        parts.insert(0, old)
    if not parts:
        raise StopRun("No yield data from any source and no cache")
    y = parts[0]
    for p in parts[1:]:  # later sources win: Treasury's own file overrides FRED for the latest days
        y = p.combine_first(y)
    y = y.sort_index()
    y = y[~y.index.duplicated(keep="last")]
    y.index.name = "date"
    y.rename("dgs30").to_csv(cache)
    return y


# ================================================================ LSEG
def open_lseg_session():
    """Same platform session as the class nightly_refresh.py and 535Wheel; creds come from GitHub secrets."""
    import lseg.data as ld
    app_key = os.environ.get("LSEG_APP_KEY")
    username = os.environ.get("LSEG_USERNAME")
    password = os.environ.get("LSEG_PASSWORD")
    if not all([app_key, username, password]):
        raise StopRun("Missing LSEG_APP_KEY, LSEG_USERNAME, or LSEG_PASSWORD")
    session = ld.session.platform.Definition(
        app_key=app_key,
        grant=ld.session.platform.GrantPassword(username=username, password=password),
        signon_control=True,
    ).get_session()
    session.open()
    ld.session.set_default(session)
    print("LSEG platform session opened")
    return ld, session


def _pick_col(cols, word):
    for c in cols:
        name = " ".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
        if word in name.lower():
            return c
    return None


def parse_tlt_frame(df):
    """LSEG history frame -> DataFrame(open, close) indexed by date. Open may be missing."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open", "close"])
    c_close = _pick_col(df.columns, "trdprc_1")
    c_open = _pick_col(df.columns, "open_prc")
    if c_close is None:
        raise StopRun(f"LSEG frame has no TRDPRC_1 column: {list(df.columns)}")
    out = pd.DataFrame({
        "open": pd.to_numeric(df[c_open], errors="coerce") if c_open is not None else np.nan,
        "close": pd.to_numeric(df[c_close], errors="coerce"),
    })
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out.dropna(subset=["close"])


def fetch_tlt(ld, start, end):
    """Daily TLT open/close in two-year chunks (~500 rows; LSEG caps interday history near 3,000 rows
    and truncates silently, so every chunk is checked), paced well under 5 requests a second."""
    frames = []
    a = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while a <= end:
        b = min(a + pd.DateOffset(years=2), end)
        df = ld.get_history(universe=TLT_RIC, fields=["OPEN_PRC", "TRDPRC_1"], interval="daily",
                            start=a.date().isoformat(), end=(b + pd.Timedelta(days=1)).date().isoformat())
        chunk = parse_tlt_frame(df)
        if (b - a).days > 20 and (chunk.empty or chunk.index.min() > a + pd.Timedelta(days=10)):
            raise StopRun(f"LSEG history for {TLT_RIC} looks truncated: asked from {a.date()}, "
                          f"got from {None if chunk.empty else chunk.index.min().date()}")
        frames.append(chunk)
        a = b + pd.Timedelta(days=1)
        time.sleep(0.4)
    out = pd.concat(frames) if frames else pd.DataFrame(columns=["open", "close"])
    return out[~out.index.duplicated(keep="last")].sort_index()


def update_tlt(status, use_lseg=True):
    cache = DATA / "tlt.csv"
    old = pd.read_csv(cache, index_col=0, parse_dates=True) if cache.exists() else pd.DataFrame(columns=["open", "close"])
    if not use_lseg:
        status["lseg"] = "skipped"
        return old
    try:
        ld, session = open_lseg_session()
        try:
            start = (old.index.max() - pd.Timedelta(days=14)) if len(old) else START - pd.Timedelta(days=60)
            new = fetch_tlt(ld, start, pd.Timestamp(date.today()))
        finally:
            try:
                session.close()
                print("LSEG session closed")
            except Exception:
                pass
        if new.empty:
            raise StopRun("LSEG returned no TLT rows")
        tlt = pd.concat([f for f in (old, new) if len(f)]).astype(float)
        tlt = tlt[~tlt.index.duplicated(keep="last")].sort_index()
        tlt.index.name = "date"
        tlt.to_csv(cache)
        status["lseg"] = f"ok ({len(new)} rows, opens {'yes' if new['open'].notna().any() else 'missing'})"
        return tlt
    except Exception as e:
        status["lseg"] = f"failed: {e}"
        print("LSEG failed, using cache:", e)
        return old


# ================================================================ calendar
def bond_days_before(auction_date, n):
    """The n bond-market sessions before an announced auction (weekdays less federal holidays).
    Used only to place tonight's open window; every statistic uses actual published yield dates."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    lo = auction_date - pd.Timedelta(days=30)
    hol = USFederalHolidayCalendar().holidays(lo, auction_date)
    days = pd.bdate_range(lo, auction_date, freq="C", holidays=hol)
    return days[days < auction_date][-n:]


# ================================================================ statistics
def compute(y, auctions, tlt, upcoming, now=None):
    now = now or datetime.now(timezone.utc)
    y = y.sort_index()
    idx = y.index
    dy = y.diff() * 100.0                                   # daily change, bp
    last_date = idx[-1]

    # Days that belong to a run-up window: the five daily changes into D-5 ... D-1.
    # Every auction in the history is masked, including those before START, so the drift
    # baseline never contains another auction's run-up.
    every = auctions[auctions["date"].isin(idx)]
    run = np.zeros(len(idx), dtype=bool)
    for d in every["date"]:
        i = idx.get_loc(d)
        run[max(0, i - WINDOW):i] = True

    # Completed auctions inside the sample
    a = every[every["date"] >= START].reset_index(drop=True)
    pos = [idx.get_loc(d) for d in a["date"]]
    # Auctions held but not yet in the yield file count as upcoming for one night
    pending = auctions[auctions["date"] > last_date]
    if len(pending):
        upcoming = pd.concat([pending.assign(source="held, yield not yet published"), upcoming], ignore_index=True)
    # The next window (announced or tentative) is shaded and, once it opens, counts live
    live_window = None
    if upcoming is not None and not upcoming.empty:
        nxt = upcoming.iloc[0]
        win = bond_days_before(nxt["date"], WINDOW + 1)     # D-6 ... D-1; the changes into D-5 ... D-1 count
        live_window = {"auction": nxt["date"], "start": win[0], "entry": win[1], "end": win[-1]}
        run |= np.asarray((idx > win[0]) & (idx <= win[-1]))

    rows = []
    for i, (_, ev) in zip(pos, a.iterrows()):
        if i < BASE_DAYS + 11:
            continue
        lo = i - BASE_DAYS
        base_days = dy.iloc[lo:i - WINDOW][~run[lo:i - WINDOW]]
        baseline = base_days.mean() * WINDOW
        pre = (y.iloc[i - 1] - y.iloc[i - 1 - WINDOW]) * 100
        early = (y.iloc[i - 1 - WINDOW] - y.iloc[i - 1 - 2 * WINDOW]) * 100
        path = {k: ((y.iloc[i + k] - y.iloc[i - 11]) * 100 if i + k < len(idx) else None) for k in range(-10, 6)}
        rows.append({
            "date": ev["date"], "reopening": bool(ev["reopening"]), "offering_bn": ev["offering_bn"],
            "high_yield": ev["high_yield"], "dealer_bn": ev["dealer_bn"], "indirect_bn": ev["indirect_bn"],
            "total_bn": ev["total_bn"],
            "y_start": y.iloc[i - 1 - WINDOW], "y_before": y.iloc[i - 1],
            "pre_bp": pre, "baseline_bp": baseline, "excess_bp": pre - baseline, "early_bp": early,
            "path": path,
        })
    ev = pd.DataFrame(rows)
    if ev.empty:
        raise StopRun("No completed 30-year auctions in the sample")
    ev["roll_excess"] = ev["excess_bp"].rolling(ROLL).mean()
    ev["roll_pre"] = ev["pre_bp"].rolling(ROLL).mean()
    ev["roll_early"] = ev["early_bp"].rolling(ROLL).mean()

    # --- TLT: short at the open five sessions before, cover at the open on auction day
    tlt_ok = tlt is not None and len(tlt) > 0
    have_open = tlt_ok and tlt["open"].notna().sum() > 0.9 * len(tlt)
    ev["tlt_ret_pct"] = np.nan
    ev["tlt_excluded"] = ""
    slope = None
    if tlt_ok:
        t = tlt.sort_index().astype(float)
        tidx = t.index
        month_start = set(pd.Series(tidx).groupby([tidx.year, tidx.month]).min())
        for k, r in ev.iterrows():
            d = r["date"]
            if d not in tidx:
                ev.at[k, "tlt_excluded"] = "no TLT bar"
                continue
            j = tidx.get_loc(d)
            if j < WINDOW + 1:
                continue
            if have_open:
                p0, p1 = t["open"].iloc[j - WINDOW], t["open"].iloc[j]
                span = tidx[j - WINDOW + 1: j + 1]
            else:  # close-to-close fallback: close of D-6 to close of D-1
                p0, p1 = t["close"].iloc[j - WINDOW - 1], t["close"].iloc[j - 1]
                span = tidx[j - WINDOW: j]
            if any(s in month_start for s in span):
                ev.at[k, "tlt_excluded"] = "ex-dividend inside window"
                continue
            if p0 and p1 and p0 > 0:
                ev.at[k, "tlt_ret_pct"] = -(p1 / p0 - 1) * 100
        # Sensitivity of TLT to the 30-year yield over the last two years
        both = pd.concat([t["close"].pct_change() * 100, dy], axis=1, join="inner").dropna()
        both = both[both.index >= both.index.max() - pd.Timedelta(days=730)]
        both.columns = ["tlt", "dy"]
        both = both[~both.index.isin(list(month_start))]
        if len(both) > 120 and both["dy"].var() > 0:
            xv, yv = both["dy"].to_numpy(dtype=float), both["tlt"].to_numpy(dtype=float)
            slope = float(np.cov(yv, xv)[0, 1] / np.var(xv, ddof=1))

    # --- cumulative split since START
    m = idx >= START
    d_run = np.where(run, dy.fillna(0).values, 0.0)[m]
    d_oth = np.where(~run, dy.fillna(0).values, 0.0)[m]
    cum = {
        "dates": [d.strftime("%Y-%m-%d") for d in idx[m]],
        "run": np.round(np.cumsum(d_run), 1).tolist(),
        "other": np.round(np.cumsum(d_oth), 1).tolist(),
        "total": np.round((y[m] - y[idx < START].iloc[-1]).values * 100, 1).tolist(),
        "run_days": int(run[m].sum()), "other_days": int((~run[m]).sum()),
    }

    # --- average path around the auction
    ks = list(range(-10, 6))
    def avg_path(sub):
        P = pd.DataFrame([r for r in sub["path"]]).astype(float)
        return [r1(P[k].mean()) for k in ks], int(P[5].notna().sum())
    p_all, n_all = avg_path(ev)
    p_rec, n_rec = avg_path(ev.tail(ROLL))

    # --- live chart: last LIVE_DAYS sessions
    lv = y.iloc[-LIVE_DAYS:]
    windows = []
    for i, d in zip(pos, a["date"]):
        if i >= len(idx) - LIVE_DAYS - WINDOW:
            windows.append({"start": idx[max(0, i - WINDOW - 1)].strftime("%Y-%m-%d"),
                            "end": idx[i - 1].strftime("%Y-%m-%d"), "auction": d.strftime("%Y-%m-%d"),
                            "pre_bp": r1((y.iloc[i - 1] - y.iloc[i - 1 - WINDOW]) * 100), "state": "done"})
    if live_window is not None:
        windows.append({"start": live_window["start"].strftime("%Y-%m-%d"),
                        "end": live_window["end"].strftime("%Y-%m-%d"),
                        "auction": live_window["auction"].strftime("%Y-%m-%d"), "pre_bp": None, "state": "next"})

    # --- wrong-if conditions
    last = ev.iloc[-1]
    sizes = auctions[["date", "reopening", "offering_bn"]]
    if upcoming is not None and not upcoming.empty:         # an announced size counts the day it is announced
        sizes = pd.concat([sizes, upcoming[["date", "reopening", "offering_bn"]]])
    sizes = sizes.dropna(subset=["offering_bn"]).sort_values("date")
    new_issues = sizes[~sizes["reopening"].astype(bool)]
    reopens = sizes[sizes["reopening"].astype(bool)]
    def size_change(df):
        if len(df) < 2:
            return None
        return float(df["offering_bn"].iloc[-1] - df["offering_bn"].iloc[-2])
    d_new, d_reo = size_change(new_issues), size_change(reopens)
    cut = any(x is not None and x < 0 for x in (d_new, d_reo))
    wrong = [
        {"id": 1, "label": "The concession disappears",
         "rule": f"Average excess run-up over the last {ROLL} auctions falls to 0 bp or below",
         "current": f"{last['roll_excess']:+.1f} bp", "triggered": bool(last["roll_excess"] <= 0)},
        {"id": 2, "label": "Treasury shrinks the supply",
         "rule": "Treasury cuts the size of the 30-year new issue or reopening",
         "current": (f"new issue ${new_issues['offering_bn'].iloc[-1]:.0f}bn "
                     f"({'n/a' if d_new is None else f'{d_new:+.0f}'} vs last), "
                     f"reopening ${reopens['offering_bn'].iloc[-1]:.0f}bn "
                     f"({'n/a' if d_reo is None else f'{d_reo:+.0f}'} vs last)"),
         "triggered": bool(cut)},
        {"id": 3, "label": "Traders get in earlier",
         "rule": f"Over the last {ROLL} auctions, the yield rises more in sessions D-10 to D-6 than in D-5 to D-1 (D = auction day)",
         "current": f"earlier {last['roll_early']:+.1f} bp vs window {last['roll_pre']:+.1f} bp",
         "triggered": bool(last["roll_early"] > last["roll_pre"])},
    ]

    # --- alpha note
    pre = ev["pre_bp"]
    sl = slope if slope is not None else FALLBACK_SLOPE
    y_now = float(y.iloc[-1])
    carry = y_now * HOLD_DAYS / 365.0           # the short pays the bond's income while it holds
    borrow = BORROW_FEE * HOLD_DAYS / 365.0
    cost = SPREAD_COST + carry + borrow
    gross_yield_based = -sl * pre.mean()
    tr = ev["tlt_ret_pct"].dropna()
    use_tlt = len(tr) >= 60
    gross = float(tr.mean()) if use_tlt else gross_yield_based
    sd = float(tr.std()) if use_tlt else float(-sl * pre.std())
    net = gross - cost
    alpha = {
        "n": int(len(ev)), "since": START.strftime("%Y"),
        "pre_mean_bp": r1(pre.mean()), "pre_median_bp": r1(pre.median()), "pre_sd_bp": r1(pre.std()),
        "hit_rate": r1((pre > 0).mean() * 100, 0),
        "excess_mean_bp": r1(ev["excess_bp"].mean()),
        "other_week_bp": r1(np.mean([r["baseline_bp"] for _, r in ev.iterrows()])),
        "t_stat": r1(pre.mean() / (pre.std() / math.sqrt(len(pre))), 1) if pre.std() > 0 else None,
        "years_positive": int((ev.groupby(ev["date"].dt.year)["excess_bp"].mean() > 0).sum()),
        "years": int(ev["date"].dt.year.nunique()),
        "slope": r1(sl, 3), "slope_source": "LSEG TLT vs DGS30, last 2 years" if slope is not None else "assumed",
        "gross_pct": r1(gross, 2), "gross_source": f"TLT from LSEG, {len(tr)} trades" if use_tlt else "average run-up x TLT move per bp",
        "tlt_trades": int(len(tr)), "tlt_hit_rate": r1((tr > 0).mean() * 100, 0) if use_tlt else None,
        "tlt_open_to_open": bool(have_open),
        "spread_pct": SPREAD_COST, "carry_pct": r1(carry, 2), "borrow_pct": r1(borrow, 3), "cost_pct": r1(cost, 2),
        "net_pct": r1(net, 2), "sd_pct": r1(sd, 2), "trades_per_year": TRADES_PER_YEAR,
        "net_year_pct": r1(net * TRADES_PER_YEAR, 1),
        "sharpe": r1(net / sd * math.sqrt(TRADES_PER_YEAR), 2) if sd else None,
        "y_now": r1(y_now, 2),
    }

    # --- by year
    by_year = (ev.groupby(ev["date"].dt.year)
                 .agg(n=("pre_bp", "size"), pre=("pre_bp", "mean"), excess=("excess_bp", "mean"))
                 .reset_index().rename(columns={"date": "year"}))

    nxt = None
    if upcoming is not None and not upcoming.empty:
        u = upcoming.iloc[0]
        nxt = {"date": u["date"].strftime("%Y-%m-%d"), "reopening": bool(u["reopening"]),
               "offering_bn": r1(num(u["offering_bn"]), 0), "status": u["source"],
               "entry": live_window["entry"].strftime("%Y-%m-%d") if live_window else None}
    in_window = bool(live_window and live_window["start"] <= last_date <= live_window["end"])

    events_out = []
    for _, r in ev.iterrows():
        events_out.append({
            "date": r["date"].strftime("%Y-%m-%d"), "type": "reopening" if r["reopening"] else "new issue",
            "size_bn": r1(r["offering_bn"], 0), "high_yield": r1(r["high_yield"], 3),
            "y_start": r1(r["y_start"]), "y_before": r1(r["y_before"]),
            "pre_bp": r1(r["pre_bp"], 1), "baseline_bp": r1(r["baseline_bp"], 1), "excess_bp": r1(r["excess_bp"], 1),
            "early_bp": r1(r["early_bp"], 1), "roll_excess": r1(r["roll_excess"], 2),
            "tlt_ret_pct": r1(r["tlt_ret_pct"], 3), "tlt_note": r["tlt_excluded"],
        })

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M UTC"),
        "last_yield_date": last_date.strftime("%Y-%m-%d"),
        "y_now": r1(y_now),
        "next_auction": nxt, "in_window": in_window,
        "last_auction": events_out[-1],
        "cum": cum,
        "path": {"k": ks, "all": p_all, "recent": p_rec, "n_all": n_all, "n_recent": n_rec},
        "live": {"dates": [d.strftime("%Y-%m-%d") for d in lv.index], "y": [r1(v) for v in lv.values],
                 "windows": windows},
        "wrong_if": wrong,
        "alpha": alpha,
        "by_year": [{"year": int(r["year"]), "n": int(r["n"]), "pre": r1(r["pre"], 1), "excess": r1(r["excess"], 1)}
                    for _, r in by_year.iterrows()],
        "events": events_out,
    }, ev


# ================================================================ main
def run(use_lseg=True):
    status = {}
    auctions, upcoming = build_auctions(status)
    y = fetch_yields(status)
    tlt = update_tlt(status, use_lseg=use_lseg)
    out, ev = compute(y, auctions, tlt, upcoming)
    out["sources"] = status
    write_json(DATA / "tracker.json", out)
    cols = ["date", "type", "size_bn", "high_yield", "y_start", "y_before", "pre_bp", "baseline_bp",
            "excess_bp", "early_bp", "roll_excess", "tlt_ret_pct", "tlt_note"]
    pd.DataFrame(out["events"])[cols].to_csv(DATA / "events_30y.csv", index=False)
    log = {"run_at": out["generated_at"], "last_yield_date": out["last_yield_date"], "sources": status,
           "next_auction": out["next_auction"], "wrong_if_triggered": [w["id"] for w in out["wrong_if"] if w["triggered"]],
           "rolling_excess_bp": out["wrong_if"][0]["current"]}
    write_json(LOGS / "latest.json", log)
    print(json.dumps(log, indent=1))
    for k, v in status.items():          # surface any failed source as a warning on the Actions run page
        if str(v).startswith("failed") or v == "not found":
            print(f"::warning::{k}: {v}")
    return out


def diagnose():
    ld, session = open_lseg_session()
    try:
        end = pd.Timestamp(date.today())
        df = fetch_tlt(ld, end - pd.Timedelta(days=10), end)
        print(df.tail())
    finally:
        session.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-lseg", action="store_true")
    p.add_argument("--diagnose", action="store_true")
    a = p.parse_args()
    try:
        if a.diagnose:
            diagnose()
        else:
            run(use_lseg=not a.no_lseg)
    except StopRun as e:
        print("STOP:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
