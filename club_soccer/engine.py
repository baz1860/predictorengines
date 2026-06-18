"""In-process command API for the Club Soccer engine (refactor Phase 4).

The command logic that used to live in app/engines/runners/club_soccer_runner.py,
now imported and called directly by the adapter (no subprocess). Functions take a
params dict and return a JSON-able dict; errors are plain exceptions that the adapter
dispatches through app.engines._inproc.run_inprocess (allowlist + redaction + finite-JSON).
"""
from __future__ import annotations

from . import competitions as C
from . import edge as E
from . import model as M


def cmd_schema(_p: dict | None = None) -> dict:
    df = M.load_fixtures()
    return {"kind": "match", "names": M.team_names(df),
            "models": ["ensemble", "goals", "elo"],
            "supports_home": False, "neutral_toggle": True,
            "team_label": "Club",
            "filters": [
                {"id": "competition", "label": "Competition",
                 "options": [""] + C.names()},
                {"id": "season", "label": "Season",
                 "options": [""] + sorted([str(x) for x in df["season"].dropna().unique()], reverse=True)},
                {"id": "date_from", "label": "From", "type": "date"},
                {"id": "date_to", "label": "To", "type": "date"},
            ]}


def cmd_predict(p: dict) -> dict:
    home = (p.get("team1") or "").strip()
    away = (p.get("team2") or "").strip()
    if not home or not away:
        raise ValueError("Pick two clubs.")
    comp = (p.get("competition") or "").strip()
    neutral = bool(p.get("neutral", False))
    model_name = p.get("model") or "ensemble"
    pred = M.predict(home, away, comp, model_name, neutral)
    probs = pred["probs"]
    venue = "neutral venue" if neutral else f"{home} home"
    return {
        "competitors": [{"name": home, "sub": comp or "club soccer"},
                        {"name": away, "sub": model_name}],
        "headline": f"{pred['xg_home']:.2f} - {pred['xg_away']:.2f} expected goals · {venue}",
        "outcomes": [
            {"label": f"{home} win", "prob": probs["home"], "kind": "win"},
            {"label": "Draw", "prob": probs["draw"], "kind": "draw"},
            {"label": f"{away} win", "prob": probs["away"], "kind": "loss"}],
        "stats": [
            {"label": "Over 2.5", "value": f"{probs['over25']:.1%}"},
            {"label": "BTTS", "value": f"{probs['btts_yes']:.1%}"},
            {"label": "Model", "value": model_name}],
        "table": {"title": "Most likely scorelines",
                  "columns": [{"key": "score", "label": "Score", "fmt": "text"},
                              {"key": "prob", "label": "Prob", "fmt": "pct"}],
                  "rows": pred["scorelines"], "bar": "prob"}}


def _columns() -> list[dict]:
    return [
        {"key": "date", "label": "Date", "fmt": "text"},
        {"key": "competition", "label": "Competition", "fmt": "text"},
        {"key": "match", "label": "Match", "fmt": "text"},
        {"key": "bet", "label": "Bet", "fmt": "text"},
        {"key": "odds", "label": "Odds", "fmt": "num"},
        {"key": "p_model", "label": "Model", "fmt": "pct"},
        {"key": "p_book", "label": "Book", "fmt": "pct"},
        {"key": "edge", "label": "Edge", "fmt": "signed_pct"},
        {"key": "ev_per_unit", "label": "EV", "fmt": "num"},
        {"key": "stake_gbp", "label": "Stake", "fmt": "gbp"},
    ]


def cmd_edge(p: dict) -> dict:
    bankroll = float(p.get("bankroll", 100.0))
    model_name = p.get("model") or "ensemble"
    odds_source = p.get("odds_source") or "manual"
    if odds_source == "api":
        odds = E.fetch_api_odds()
        note = "API-Football odds"
    elif odds_source == "the-odds-api":
        odds = E.fetch_the_odds_api()
        note = "The Odds API odds"
    else:
        odds = E.load_odds()
        note = "Manual odds from club_soccer/data/odds.csv"
    rows = E.rows_from_odds(odds, model_name, bankroll)
    comp = (p.get("competition") or "").strip()
    if comp:
        rows = [r for r in rows if r.get("competition") == comp]
    season = (p.get("season") or "").strip()
    if season:
        rows = [r for r in rows if str(r.get("date", ""))[:4] == season]
    date_from = (p.get("date_from") or "").strip()
    if date_from:
        rows = [r for r in rows if str(r.get("date", "")) >= date_from]
    date_to = (p.get("date_to") or "").strip()
    if date_to:
        rows = [r for r in rows if str(r.get("date", "")) <= date_to]
    return {"note": f"{note} · {len(rows)} priced outcome(s)",
            "columns": _columns(), "rows": rows}


def cmd_edge_template(_p: dict | None = None) -> dict:
    E.write_template()
    return {"path": "club_soccer/data/odds.csv"}


COMMANDS = {"schema": lambda p: cmd_schema(), "predict": cmd_predict,
            "edge": cmd_edge, "edge_template": cmd_edge_template}
