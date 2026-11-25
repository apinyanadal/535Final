# Paying for Supply: the 30-year auction concession

FINTECH 535 final project, part 1 (investment thesis). Live site: https://apinyanadal.github.io/535Final/

**Thesis.** Buyers have to be paid to absorb each month's pre-scheduled 30-year bond auction, so the 30-year
Treasury yield rises in the five sessions before it, and a short position in long bonds over those five sessions
collects that payment.

## How it runs
`tracker.py` runs every night at 10pm ET (`.github/workflows/nightly.yml`) and writes `data/tracker.json`,
which `index.html` reads. `pages.yml` redeploys the site when the nightly finishes.

| Data | Source |
|---|---|
| 30-year auction dates, sizes, results | TreasuryDirect, Treasury Fiscal Data |
| Upcoming auction dates | Treasury tentative auction schedule (PDF, quarterly) |
| 30-year yield | FRED `DGS30`, extended with Treasury's daily par yield file |
| TLT open and close | LSEG (`TLT.O`), cached in the Actions cache, not committed |

If a source fails, the last good copy in `data/` is used and the failure shows in `logs/latest.json`,
in the page footer and as a warning on the Actions run.

## Setup
1. Repository secrets: `LSEG_APP_KEY`, `LSEG_USERNAME`, `LSEG_PASSWORD` (same values as 535Wheel).
2. Settings > Pages > Source: GitHub Actions.
3. Actions > Nightly tracker > Run workflow (`run`). The first run pulls TLT history from LSEG in two-year chunks.

## Local
```
pip install -r requirements.txt
python tracker.py --no-lseg     # Treasury and FRED only
python -m pytest -q             # offline tests, LSEG mocked
```
