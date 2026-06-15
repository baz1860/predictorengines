#!/usr/bin/env python3
"""Isolated runner for the Club Soccer engine."""
from __future__ import annotations

import json
import sys

import pandas as pd

import competitions as C
import edge as E
import model as M


def _params():
    raw = sys.stdin.read().strip()
    return json.loads(raw) if raw else {}


def cmd_schema(_p=None):
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


def cmd_predict(p):
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


def _columns():
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


def cmd_edge(p):
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


def cmd_edge_template(_p=None):
    E.write_template()
    return {"path": "club_soccer/data/odds.csv"}


COMMANDS = {"schema": lambda p: cmd_schema(), "predict": cmd_predict,
            "edge": cmd_edge, "edge_template": cmd_edge_template}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "schema"
    try:
        print(json.dumps(COMMANDS[cmd](_params())))
    except ValueError as e:
        print(json.dumps({"error": str(e)})); sys.exit(2)
    except Exception as e:  # noqa
        print(json.dumps({"error": f"{type(e).__name__}: {e}"})); sys.exit(1)


if __name__ == "__main__":
    main()
