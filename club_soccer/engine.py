"""In-process command API for the Club Soccer engine (refactor Phase 4).

The command logic that used to live in app/engines/runners/club_soccer_runner.py,
now imported and called directly by the adapter (no subprocess). Functions take a
params dict and return a JSON-able dict; errors are plain exceptions that the adapter
dispatches through app.engines._inproc.run_inprocess (allowlist + redaction + finite-JSON).
"""
from __future__ import annotations

from . import edge as E
from . import model as M
from . import prediction_scope as PS


def cmd_schema(_p: dict | None = None) -> dict:
    df = M.load_fixtures()
    visible = df[df["competition"].astype(str).isin(PS.SURFACED_COMPETITION_SET)]
    return {"kind": "match", "names": M.team_names(visible),
            "models": ["ensemble", "goals", "elo", "xg"],
            "supports_home": False, "neutral_toggle": True,
            "team_label": "Club",
            "filters": [
                {"id": "competition", "label": "Competition",
                 "options": list(PS.SURFACED_COMPETITIONS)},
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
    if not PS.is_surfaced(comp):
        raise ValueError(
            "Choose one of the surfaced competitions: "
            + ", ".join(PS.SURFACED_COMPETITIONS)
        )
    neutral = bool(p.get("neutral", False))
    model_name = p.get("model") or "ensemble"
    match_date = (p.get("match_date") or p.get("date") or "").strip()
    fixture_id = p.get("fixture_id")
    player_adj = p.get("player_adj") if isinstance(p.get("player_adj"), dict) else None
    if match_date:
        pred = M.predict_match(home, away, comp, match_date, model_name, neutral,
                               player_adj=player_adj, fixture_id=fixture_id)
    else:
        pred = M.predict(home, away, comp, model_name, neutral, player_adj=player_adj)
    from .calibrate import load_active_maps, apply as apply_calibration
    calib_maps = load_active_maps()
    if calib_maps is not None:
        ph, pdr, pa = apply_calibration(
            pred["probs"]["home"], pred["probs"]["draw"],
            pred["probs"]["away"], calib_maps)
        pred["probs"].update({"home": ph, "draw": pdr, "away": pa})
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
    if odds_source == "bsd":
        odds = E.fetch_bsd_odds()
        note = "BSD odds"
    elif odds_source == "the-odds-api":
        odds = E.fetch_the_odds_api()
        note = "The Odds API odds"
    else:
        odds = E.load_odds()
        note = "Manual odds from club_soccer/data/odds.csv"
    # Same gate as season.py: no pricing of settled/past fixtures, no stale
    # manual file. Without this the app could recommend (and record!) bets
    # on matches that finished weeks ago.
    odds, quote_issues = E.validate_quotes(
        odds, source="manual" if odds_source == "manual" else "live")
    for msg in quote_issues:
        note += f" · {msg}"
    blend = p.get("market_blend") if "market_blend" in p else None
    rows = E.rows_from_odds(
        odds, model_name, bankroll, market_blend=blend
    ) if not odds.empty else []
    rows = PS.filter_rows(rows)
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
    result = {"note": f"{note} · {len(rows)} priced outcome(s)",
              "columns": _columns(), "rows": rows}
    blend_weights = sorted({
        float(r["market_blend_w"]) for r in rows if "market_blend_w" in r
    })
    if blend_weights:
        from app.market_blend import is_default_on
        result["market_blend"] = {
            "applied": True, "w": blend_weights[0],
            "experimental": not is_default_on("club_soccer"),
        }
        result["note"] += f" · market-blended (w={blend_weights[0]:.2f})"
    return result


def cmd_edge_template(_p: dict | None = None) -> dict:
    E.write_template()
    return {"path": "club_soccer/data/odds.csv"}


COMMANDS = {"schema": lambda p: cmd_schema(), "predict": cmd_predict,
            "edge": cmd_edge, "edge_template": cmd_edge_template}
