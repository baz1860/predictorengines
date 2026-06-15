"""Suite-level shared bankroll + ledger.

ONE bankroll and ONE ledger for every engine's bets. Replaces the per-engine
bankroll.json / ledger.csv files. Each ledger row is tagged with `engine` and
`sport` so the Bankroll view can filter, but staking and compounding work off
the single pooled balance.

Files (at the project root, shared by all engines):
    data/suite_bankroll.json   {"bankroll": float, "peak": float, "start": float}
    data/suite_ledger.csv      one row per bet, columns = COLS

Settlement is engine-agnostic: open bets are grouped by engine and handed to
that engine's adapter (`grade_open_bets`), which knows how to grade its own
markets against its own results. The store applies P&L and compounds the pooled
bankroll in chronological (ledger) order.
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE = DATA / "suite_bankroll.json"
LEDGER = DATA / "suite_ledger.csv"
START_BANKROLL = 100.0
MIN_STAKE = 0.10

# Original ledger columns (V2). Never reorder these — old ledgers are read by
# position-independent name, but keeping the prefix stable is friendlier.
_CORE_COLS = ["placed_on", "engine", "sport", "match_date", "home", "away", "side",
              "bet", "odds", "stake", "status", "pnl", "bankroll_after"]
# V3 M4 optional provenance/settlement columns, appended backward-compatibly.
# Old ledgers without these load fine (load_ledger backfills them as "").
_V3_COLS = ["event_id", "market", "line", "source", "model", "closing_odds"]
COLS = _CORE_COLS + _V3_COLS


# ── state ────────────────────────────────────────────────────────────────────
def _load_state() -> dict:
    if STATE.exists():
        try:
            d = json.loads(STATE.read_text())
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


def current_bankroll() -> float:
    return float(_load_state().get("bankroll", START_BANKROLL))


def current_peak() -> float:
    d = _load_state()
    return float(d.get("peak", max(START_BANKROLL, d.get("bankroll", START_BANKROLL))))


def start_bankroll() -> float:
    return float(_load_state().get("start", START_BANKROLL))


def _save_state(bankroll: float, peak: float | None = None, start: float | None = None) -> None:
    d = _load_state()
    d["bankroll"] = round(bankroll, 2)
    d["peak"] = round(peak if peak is not None else max(bankroll, d.get("peak", bankroll)), 2)
    if start is not None:
        d["start"] = round(start, 2)
    d.setdefault("start", START_BANKROLL)
    DATA.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(d))


# ── ledger ───────────────────────────────────────────────────────────────────
def load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        df = pd.read_csv(LEDGER)
        for c in COLS:
            if c not in df.columns:
                df[c] = ""        # backfill any missing (incl. all V3 optional) cols
        return df
    return pd.DataFrame(columns=COLS)


def _save_ledger(df: pd.DataFrame) -> None:
    DATA.mkdir(exist_ok=True)
    df.to_csv(LEDGER, index=False)


def _event_id_for(r, engine_id: str) -> str:
    """Stable event id for a candidate row: prefer an explicit event_id, else a
    fixture key (golf carries its participant in `home`)."""
    eid = getattr(r, "event_id", "") if hasattr(r, "event_id") else ""
    if isinstance(eid, str) and eid.strip():
        return eid.strip()
    from .engines.contracts import fixture_key
    return fixture_key(getattr(r, "match_date", ""), getattr(r, "home", ""),
                       getattr(r, "away", ""))


def place_bets(engine_id: str, sport: str, candidates: pd.DataFrame,
               peak: float | None = None) -> pd.DataFrame:
    """Record new bets. `candidates` needs: match_date, home, away, side, bet,
    odds, and either `stake` (currency) or `kelly_stake` (fraction of bankroll).
    Optional: event_id, market, line, source, model.

    Dedupes against existing open bets on (engine, event_id, side); the shared
    suite caps (app.portfolio) clamp single-event / correlated / daily / drawdown
    exposure before the pooled-funds clamp."""
    from .portfolio import apply_caps
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=COLS)
    ledger = load_ledger()
    bankroll = current_bankroll()
    peak = peak if peak is not None else current_peak()
    open_rows = ledger[ledger["status"] == "open"].copy()
    open_rows["stake_n"] = pd.to_numeric(open_rows["stake"], errors="coerce").fillna(0.0)
    available = bankroll - float(open_rows["stake_n"].sum())

    # prior exposure so caps see the whole open book, not just this batch
    today = str(date.today())
    prior_day = float(open_rows.loc[open_rows["placed_on"].astype(str) == today, "stake_n"].sum())
    prior_event = open_rows.groupby(open_rows["event_id"].fillna(""))["stake_n"].sum().to_dict()
    prior_engine = open_rows.groupby(open_rows["engine"].fillna(""))["stake_n"].sum().to_dict()

    existing = set(zip(ledger["engine"], ledger["event_id"].fillna(""), ledger["side"]))
    legacy = set(zip(ledger["engine"], ledger["home"], ledger["away"], ledger["side"]))

    cand = []
    for r in candidates.itertuples(index=False):
        event_id = _event_id_for(r, engine_id)
        side = getattr(r, "side")
        if (engine_id, event_id, side) in existing:
            continue
        if (engine_id, getattr(r, "home"), getattr(r, "away"), side) in legacy:
            continue
        if hasattr(r, "stake") and pd.notna(getattr(r, "stake")):
            stake = float(getattr(r, "stake"))
        else:
            stake = round(float(getattr(r, "kelly_stake")) * bankroll, 2)
        cand.append({
            "engine": engine_id, "event_id": event_id, "side": side,
            "stake": round(stake, 2),
            "match_date": getattr(r, "match_date"), "home": getattr(r, "home"),
            "away": getattr(r, "away"), "bet": getattr(r, "bet"),
            "odds": float(getattr(r, "odds")),
            "market": getattr(r, "market", "") if hasattr(r, "market") else "",
            "line": getattr(r, "line", "") if hasattr(r, "line") else "",
            "source": getattr(r, "source", "") if hasattr(r, "source") else "",
            "model": getattr(r, "model", "") if hasattr(r, "model") else ""})

    capped = apply_caps(cand, bankroll=bankroll, peak=peak,
                        prior_event_stake=prior_event,
                        prior_engine_stake=prior_engine, prior_day_stake=prior_day)

    new_rows = []
    for c in capped:
        stake = min(round(c["stake"], 2), round(max(available, 0.0), 2))
        if stake < MIN_STAKE:
            continue
        available -= stake
        new_rows.append({
            "placed_on": today, "engine": engine_id, "sport": sport,
            "match_date": c["match_date"], "home": c["home"], "away": c["away"],
            "side": c["side"], "bet": c["bet"], "odds": c["odds"],
            "stake": stake, "status": "open", "pnl": 0.0, "bankroll_after": "",
            "event_id": c["event_id"], "market": c["market"], "line": c["line"],
            "source": c["source"], "model": c["model"], "closing_odds": ""})
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        _save_ledger(ledger)
    return pd.DataFrame(new_rows, columns=COLS)


def settle(registry, verbose: bool = False, dry_run: bool = False) -> dict:
    """Settle open bets across all engines. Returns a summary dict.

    With dry_run=True nothing is written: the return value previews exactly what
    *would* settle (per-bet status/pnl and the resulting bankroll), so the user
    can review before committing the ledger."""
    ledger = load_ledger()
    open_mask = ledger["status"] == "open"
    if not open_mask.any():
        return {"settled": 0, "still_open": 0, "bankroll": current_bankroll(),
                "dry_run": dry_run, "preview": []}

    # Ask each engine to grade its own open rows: idx -> (won: bool|None, score:str)
    graded: dict[int, tuple] = {}
    for engine_id in ledger.loc[open_mask, "engine"].dropna().unique():
        rows = ledger[open_mask & (ledger["engine"] == engine_id)]
        try:
            adapter = registry.get(engine_id)
        except KeyError:
            continue
        if hasattr(adapter, "grade_open_bets"):
            graded.update(adapter.grade_open_bets(rows))

    bankroll, peak = current_bankroll(), current_peak()
    settled = 0
    preview = []
    for i in ledger.index[open_mask]:
        if i not in graded:
            continue
        status, detail = graded[i]
        if status not in ("won", "lost", "push"):
            continue  # None / unknown — leave open (e.g. outright not yet played)
        stake, odds = float(ledger.at[i, "stake"]), float(ledger.at[i, "odds"])
        pnl = round(stake * (odds - 1), 2) if status == "won" else (0.0 if status == "push" else -stake)
        bankroll = round(bankroll + pnl, 2)
        peak = max(peak, bankroll)
        preview.append({
            "engine": ledger.at[i, "engine"], "match_date": ledger.at[i, "match_date"],
            "home": ledger.at[i, "home"], "away": ledger.at[i, "away"],
            "bet": ledger.at[i, "bet"], "status": status, "detail": detail,
            "pnl": pnl, "bankroll_after": bankroll})
        if not dry_run:
            ledger.at[i, "status"] = status
            ledger.at[i, "pnl"] = pnl
            ledger.at[i, "bankroll_after"] = bankroll
        settled += 1

    if not dry_run:
        _save_ledger(ledger)
        _save_state(bankroll, peak)
    still_open = int((ledger["status"] == "open").sum()) + (settled if dry_run else 0)
    return {"settled": settled, "still_open": still_open,
            "bankroll": bankroll, "dry_run": dry_run, "preview": preview}


def reset(amount: float) -> None:
    """Back up the ledger and start fresh at `amount`."""
    if LEDGER.exists():
        shutil.copy(LEDGER, LEDGER.with_suffix(".csv.bak"))
        LEDGER.unlink()
    _save_state(amount, peak=amount, start=amount)


def status_summary() -> dict:
    """Everything the suite-level Bankroll tab needs."""
    ledger = load_ledger()
    bankroll = current_bankroll()
    out = {
        "bankroll": round(bankroll, 2),
        "peak": round(current_peak(), 2),
        "start": round(start_bankroll(), 2),
        "open": [], "settled": [],
        "by_sport": [],
        "totals": {"open_count": 0, "open_stake": 0.0,
                   "settled_count": 0, "won": 0, "net_pnl": 0.0},
    }
    if ledger.empty:
        return out

    ledger["stake_n"] = pd.to_numeric(ledger["stake"], errors="coerce").fillna(0.0)
    ledger["pnl_n"] = pd.to_numeric(ledger["pnl"], errors="coerce").fillna(0.0)
    open_bets = ledger[ledger["status"] == "open"]
    closed = ledger[ledger["status"].isin(["won", "lost", "push"])]

    def _match(r):
        home, away = str(r.home or ""), str(r.away or "")
        if not away or "OUTRIGHT" in away.upper():
            return home or "—"          # outrights/specials have no opponent
        return f"{home} v {away}"

    def _rows(df):
        return [
            {"sport": r.sport or "", "match_date": r.match_date,
             "match": _match(r), "bet": r.bet,
             "odds": r.odds, "stake": round(float(r.stake_n), 2),
             "status": r.status, "pnl": round(float(r.pnl_n), 2)}
            for r in df.itertuples(index=False)
        ]

    out["open"] = _rows(open_bets)
    out["settled"] = _rows(closed.tail(25).iloc[::-1])
    out["totals"] = {
        "open_count": int(len(open_bets)),
        "open_stake": round(float(open_bets["stake_n"].sum()), 2),
        "settled_count": int(len(closed)),
        "won": int((closed["status"] == "won").sum()),
        "net_pnl": round(float(closed["pnl_n"].sum()), 2),
    }
    for sport, grp in ledger.groupby(ledger["sport"].fillna("")):
        c = grp[grp["status"].isin(["won", "lost"])]
        out["by_sport"].append({
            "sport": sport or "—",
            "open": int((grp["status"] == "open").sum()),
            "settled": int(len(c)),
            "net_pnl": round(float(c["pnl_n"].sum()), 2),
        })
    return out
