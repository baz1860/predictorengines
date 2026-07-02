"""
golf/edge.py  –  Betting edge calculator for the golf engine.

Markets supported:
  - Outright winner       (odds_win)
  - Top 5                 (odds_top5)
  - Top 10                (odds_top10)
  - Top 20                (odds_top20)
  - Make / miss cut       (odds_cut / odds_nocut)
  - Head-to-head matchups (--h2h / --h2h-all)

De-vig method: multiplicative (default) or power.

Output: data/edge_report.csv

Usage:
  python -m golf.edge [--min-edge 3.0] [--h2h-all] [--h2h "Rory McIlroy" "Scottie Scheffler"]
                 [--kelly 0.25] [--no-bet] [--market win|top5|top10|top20|cut|all]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PARENT_DATA = Path(__file__).parent.parent / "data"

DEFAULT_API_KEY = ""  # The Odds API key (optional)
DEFAULT_MIN_EDGE = 3.0
DEFAULT_KELLY = 0.25
EXPOSURE_CAP = 0.20    # max fraction of bankroll on any single bet
DEFAULT_SIGMA = 2.85   # field round-to-round σ, fallback for legacy h2h approx


# ─────────────────────────────────────────────
# Bankroll
# ─────────────────────────────────────────────

def load_bankroll() -> float:
    """Load current bankroll from shared bankroll.json."""
    for p in [PARENT_DATA / "bankroll.json", DATA_DIR / "bankroll.json"]:
        if p.exists():
            with open(p) as f:
                return float(json.load(f).get("balance", 100.0))
    return 100.0


# ─────────────────────────────────────────────
# Odds loading
# ─────────────────────────────────────────────

def load_predictions(path: Path | None = None) -> dict[str, dict]:
    """
    Load predictions CSV → dict keyed by lowercase player name.
    Defaults to predictions_inplay.csv if it exists, otherwise predictions.csv.
    Override with --predictions flag.
    """
    if path is None:
        inplay = DATA_DIR / "predictions_inplay.csv"
        path = inplay if inplay.exists() else DATA_DIR / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"No predictions at {path}. Run simulate.py or simulate_inplay.py first.")

    preds = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if name:
                preds[name.lower()] = {
                    "name":    name,
                    "win":     float(row.get("win_pct",  0)) / 100,
                    "top5":    float(row.get("top5_pct", 0)) / 100,
                    "top10":   float(row.get("top10_pct",0)) / 100,
                    "top20":   float(row.get("top20_pct",0)) / 100,
                    "cut":     float(row.get("cut_pct",  0)) / 100,
                    "rating":  float(row.get("rating",   0)),
                    "sigma":   float(row.get("sigma", 0) or 0),
                }
    return preds


def load_odds_csv(path: Path | None = None) -> dict[str, dict]:
    """
    Load manual odds.csv → dict keyed by lowercase player name.
    Columns: name, odds_win, odds_top5, odds_top10, odds_top20, odds_cut, odds_nocut
    """
    path = path or DATA_DIR / "odds.csv"
    if not path.exists():
        return {}

    odds = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("name", row.get("player", "")).strip()
            if not name:
                continue
            entry = {}
            for col in ("odds_win", "odds_top5", "odds_top10", "odds_top20", "odds_cut", "odds_nocut"):
                val = row.get(col, "")
                if val:
                    try:
                        entry[col] = float(val)
                    except ValueError:
                        pass
            if entry:
                odds[name.lower()] = {"name": name, **entry}
    return odds


# ─────────────────────────────────────────────
# De-vig
# ─────────────────────────────────────────────

def devig_multiplicative(fair_odds_list: list[float]) -> list[float]:
    """
    Multiplicative de-vig: scale each implied probability so they sum to 1.
    Returns fair decimal odds.
    """
    implied = [1.0 / o for o in fair_odds_list]
    overround = sum(implied)
    fair_probs = [p / overround for p in implied]
    return [1.0 / p for p in fair_probs]


def devig_market(book_odds: float, other_book_odds: list[float] | None = None) -> float:
    """
    De-vig a single side given the book odds and (optionally) other sides.
    For a two-way market (win/lose): other_book_odds = [odds_of_other_side].
    For outright, we simply normalise by overround.
    """
    if other_book_odds:
        all_odds = [book_odds] + other_book_odds
        fair = devig_multiplicative(all_odds)
        return fair[0]
    # Single side — can't de-vig without market, return as-is
    return book_odds


# ─────────────────────────────────────────────
# EV and Kelly
# ─────────────────────────────────────────────

def ev(model_prob: float, fair_decimal: float) -> float:
    """Expected value per unit staked (e.g. 0.05 = +5%)."""
    return model_prob * (fair_decimal - 1.0) - (1.0 - model_prob)


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Full Kelly fraction."""
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    return (b * model_prob - q) / b if b > 0 else 0.0


def stake(model_prob: float, decimal_odds: float, bankroll: float, kelly_mult: float = 0.25) -> float:
    """Quarter-Kelly stake in £, capped at EXPOSURE_CAP * bankroll."""
    kf = kelly_fraction(model_prob, decimal_odds) * kelly_mult
    kf = max(0.0, kf)
    raw = kf * bankroll
    cap = EXPOSURE_CAP * bankroll
    return round(min(raw, cap), 2)


# ─────────────────────────────────────────────
# Market evaluation
# ─────────────────────────────────────────────

MARKET_MAP = {
    "win":   ("odds_win",   "win"),
    "top5":  ("odds_top5",  "top5"),
    "top10": ("odds_top10", "top10"),
    "top20": ("odds_top20", "top20"),
    "cut":   ("odds_cut",   "cut"),
}


def evaluate_markets(
    preds: dict[str, dict],
    odds_data: dict[str, dict],
    markets: list[str],
    min_edge: float,
    bankroll: float,
    kelly_mult: float,
) -> list[dict]:
    """
    Compare model probabilities to available odds across chosen markets.
    Returns list of bets with positive EV above min_edge%.
    """
    bets = []

    for name_lower, pred in preds.items():
        if name_lower not in odds_data:
            continue
        od = odds_data[name_lower]

        for mkt in markets:
            if mkt not in MARKET_MAP:
                continue
            odds_col, prob_col = MARKET_MAP[mkt]
            book_odds = od.get(odds_col)
            model_prob = pred.get(prob_col, 0.0)

            if not book_odds or not model_prob:
                continue

            ev_pct = ev(model_prob, book_odds) * 100
            if ev_pct < min_edge:
                continue

            st = stake(model_prob, book_odds, bankroll, kelly_mult)
            bets.append({
                "player":     pred["name"],
                "market":     mkt,
                "model_prob": f"{model_prob*100:.2f}%",
                "book_odds":  f"{book_odds:.2f}",
                "ev_pct":     f"{ev_pct:.1f}%",
                "kelly_stake":f"£{st:.2f}",
                "rating":     pred.get("rating", 0.0),
            })

    bets.sort(key=lambda b: float(b["ev_pct"].rstrip("%")), reverse=True)
    return bets


# ─────────────────────────────────────────────
# Head-to-head
# ─────────────────────────────────────────────

def h2h_prob_from_predictions(
    preds: dict[str, dict],
    player_a: str,
    player_b: str,
    n_sims: int = 50_000,
) -> float | None:
    """
    Compute P(A beats B | both make cut) using their rating difference.

    Uses the normal distribution approximation:
      P(A < B in 4-round score) where scores ~ Normal(-rating, sigma)

    This is quick but ignores actual cut simulation.
    For higher accuracy, load predictions from simulate.py directly.
    """
    import math

    key_a = player_a.lower()
    key_b = player_b.lower()

    if key_a not in preds or key_b not in preds:
        return None

    r_a = preds[key_a]["rating"]
    r_b = preds[key_b]["rating"]

    # Difference in expected 72-hole score: delta ~ Normal(r_a − r_b, sigma_diff).
    # Over 4 independent rounds variance adds (×4); the difference of two players
    # adds their variances:  Var(delta) = 4·(σ_a² + σ_b²).  Use each player's
    # fitted per-round σ (predictions.csv); fall back to the field default only
    # when σ is missing. NB: this is the legacy closed-form approximation and
    # ignores the cut — the joint simulation (price_all) is the accurate path.
    s_a = preds[key_a].get("sigma") or DEFAULT_SIGMA
    s_b = preds[key_b].get("sigma") or DEFAULT_SIGMA
    sigma_diff = math.sqrt(4.0 * (s_a ** 2 + s_b ** 2))

    # P(A beats B) = P(score_A < score_B) = Phi((r_a - r_b) / sigma_diff)
    z = (r_a - r_b) / sigma_diff
    p_a_wins = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return p_a_wins


def evaluate_h2h(
    preds: dict[str, dict],
    odds_data: dict[str, dict],
    pairs: list[tuple[str, str]] | None = None,
    all_pairs: bool = False,
    min_edge: float = 3.0,
    bankroll: float = 100.0,
    kelly_mult: float = 0.25,
) -> list[dict]:
    """
    Evaluate head-to-head matchups.
    odds_data entries need keys: 'h2h_a_odds', 'h2h_b_odds' — or
    the odds.csv can list individual player lines for head-to-head.
    """
    bets = []

    if all_pairs:
        field_keys = list(preds.keys())
        pairs = list(combinations(field_keys, 2))

    if not pairs:
        return bets

    for key_a, key_b in pairs:
        p_a = h2h_prob_from_predictions(preds, key_a, key_b)
        if p_a is None:
            continue
        p_b = 1.0 - p_a

        name_a = preds[key_a]["name"]
        name_b = preds[key_b]["name"]

        # Look for odds in odds_data (h2h market column)
        # Odds.csv h2h format: player, market=h2h, opponent, odds
        # Fall back: scan for matching h2h entries
        for key, od in odds_data.items():
            h2h_odds_a = od.get("h2h_a_odds")
            h2h_odds_b = od.get("h2h_b_odds")
            if not h2h_odds_a or not h2h_odds_b:
                continue

            ev_a = ev(p_a, h2h_odds_a) * 100
            ev_b = ev(p_b, h2h_odds_b) * 100

            if ev_a >= min_edge:
                st = stake(p_a, h2h_odds_a, bankroll, kelly_mult)
                bets.append({
                    "player":      name_a,
                    "market":      f"H2H vs {name_b}",
                    "model_prob":  f"{p_a*100:.1f}%",
                    "book_odds":   f"{h2h_odds_a:.2f}",
                    "ev_pct":      f"{ev_a:.1f}%",
                    "kelly_stake": f"£{st:.2f}",
                    "rating":      preds[key_a]["rating"],
                })
            if ev_b >= min_edge:
                st = stake(p_b, h2h_odds_b, bankroll, kelly_mult)
                bets.append({
                    "player":      name_b,
                    "market":      f"H2H vs {name_a}",
                    "model_prob":  f"{p_b*100:.1f}%",
                    "book_odds":   f"{h2h_odds_b:.2f}",
                    "ev_pct":      f"{ev_b:.1f}%",
                    "kelly_stake": f"£{st:.2f}",
                    "rating":      preds[key_b]["rating"],
                })

    bets.sort(key=lambda b: float(b["ev_pct"].rstrip("%")), reverse=True)
    return bets


def evaluate_h2h_no_odds(preds: dict, top_n: int = 20) -> None:
    """Print matchup ratings even without book odds (for manual odds lookup)."""
    print(f"\n{'Player A':<28} {'P(A)':>6}  {'Player B':<28} {'P(B)':>6}  {'Rating diff':>12}")
    print("-" * 90)
    field_keys = list(preds.keys())
    pairs = combinations(field_keys, 2)
    rows = []
    for ka, kb in pairs:
        p_a = h2h_prob_from_predictions(preds, ka, kb)
        if p_a is None:
            continue
        r_diff = abs(preds[ka]["rating"] - preds[kb]["rating"])
        rows.append((ka, kb, p_a, 1 - p_a, r_diff))

    rows.sort(key=lambda x: x[4], reverse=True)
    for ka, kb, p_a, p_b, r_diff in rows[:top_n]:
        na = preds[ka]["name"]
        nb = preds[kb]["name"]
        print(f"{na:<28} {p_a*100:>5.1f}%  {nb:<28} {p_b*100:>5.1f}%  {r_diff:>+12.3f}")


# ═════════════════════════════════════════════════════════════════════════
# v2 unified pricing: calibration + market-blend across all markets, incl.
# joint-sim matchups & 3-balls.  Pure function consumed by both the CLI and
# the app runner.
# ═════════════════════════════════════════════════════════════════════════

from . import market  # noqa: E402
from . import calibrate  # noqa: E402

PLACE_MARKETS = ["win", "top5", "top10", "top20", "cut"]
PLACE_COL = {"win": "odds_win", "top5": "odds_top5", "top10": "odds_top10",
             "top20": "odds_top20", "cut": "odds_cut"}

# No real betting market is a certainty. A model probability at/above this is a
# simulation artifact — almost always make-cut/top-N in a field that is no larger
# than the cut rule (limited-field/no-cut events or an off-week stub field),
# where the cut never binds so everyone "survives". Such markets are skipped so
# they can never be flagged as +EV and Kelly-staked.
CERTAINTY = 0.99
MARKET_LABEL = {"win": "Win outright", "top5": "Top 5", "top10": "Top 10",
                "top20": "Top 20", "cut": "Make cut"}


def load_matchup_odds(path: Path | None = None) -> dict[tuple[str, str], dict]:
    """matchups.csv: player_a, player_b, odds_a, odds_b → {(a,b): {a,b odds}}."""
    path = path or DATA_DIR / "matchups.csv"
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            a, b = r.get("player_a", "").strip(), r.get("player_b", "").strip()
            try:
                oa, ob = float(r["odds_a"]), float(r["odds_b"])
            except (KeyError, ValueError):
                continue
            if a and b:
                out[(a, b)] = {"a": oa, "b": ob}
    return out


def load_threeball_odds(path: Path | None = None) -> dict[tuple[str, str, str], dict]:
    """threeballs.csv: player_a/b/c, odds_a/b/c → {(a,b,c): {a,b,c odds}}."""
    path = path or DATA_DIR / "threeballs.csv"
    out: dict[tuple[str, str, str], dict] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            a, b, c = (r.get(f"player_{x}", "").strip() for x in "abc")
            try:
                oa, ob, oc = (float(r[f"odds_{x}"]) for x in "abc")
            except (KeyError, ValueError):
                continue
            if a and b and c:
                out[(a, b, c)] = {"a": oa, "b": ob, "c": oc}
    return out


def _bet_row(player, market_label, side, book, p_model, p_final, bankroll, kelly):
    e = ev(p_final, book)
    return {
        "player": player, "market": market_label,
        "bet": f"{market_label} — {player}", "side": side,
        "odds": round(float(book), 2),
        "p_model": round(float(p_final), 4),
        "ev_per_unit": round(float(e), 3),
        "kelly_frac": round(max(0.0, kelly_fraction(p_final, book) * kelly), 4),
        "stake_gbp": stake(p_final, book, bankroll, kelly),
        "_ev": e,
    }


def price_all(rated, results, odds_data, matchup_odds, threeball_odds,
              bankroll, kelly=DEFAULT_KELLY, calibrated=True, blended=True,
              min_edge=0.0) -> list[dict]:
    """Price every market from one simulation, applying calibration (per-player,
    nested) and market blend (log-odds). Returns bet rows sorted by EV."""
    calib_maps = calibrate.load_maps() if calibrated else None
    bw = market.blend_weights() if blended else {}
    rows: list[dict] = []

    # If the 36-hole cut never binds (field ≤ cut rule), make-cut and the wider
    # top-N bands are degenerate (~1.0) and must not be priced as markets.
    cut_binds = results.get("__cut_binds__", True)

    # outright board → fair win probs (robust to partial boards)
    win_board = {od["name"]: od["odds_win"]
                 for od in odds_data.values() if od.get("odds_win")}
    win_fair = market.devig_outright(win_board) if win_board else {}

    # ── outright / place / cut ──
    for od in odds_data.values():
        name = od["name"]
        if name not in results:
            continue
        r = results[name]
        raw = {"win": r["win"], "top5": r["top5"], "top10": r["top10"],
               "top20": r["top20"], "cut": r["made_cut"]}
        probs = calibrate.apply_row(raw, calib_maps) if calibrated else raw
        for mkt in PLACE_MARKETS:
            book = od.get(PLACE_COL[mkt])
            if not book:
                continue
            # No make-cut market when the cut doesn't bind; never price a
            # certainty (sim artifact) regardless of market.
            if mkt == "cut" and not cut_binds:
                continue
            p_model = probs[mkt]
            if p_model >= CERTAINTY:
                continue
            p_mkt = win_fair.get(name) if mkt == "win" else market.devig_line(book, mkt)
            p_final = market.blend(p_model, p_mkt, bw.get(mkt, 0.0)) \
                if (blended and p_mkt) else p_model
            row = _bet_row(name, MARKET_LABEL[mkt], mkt, book, p_model, p_final,
                           bankroll, kelly)
            row["p_market"] = round(p_mkt, 4) if p_mkt else ""
            rows.append(row)

    # ── matchups (joint-sim) ──
    mres = results.get("__matchups__", {})
    for (a, b), o in matchup_odds.items():
        d = mres.get((a, b))
        if d is None and (b, a) in mres:
            dd = mres[(b, a)]
            d = {a: dd[a], b: dd[b], "tie": dd["tie"]}
        if d is None:
            continue
        fair = market.devig([o["a"], o["b"]], method="multiplicative")
        for side, player, opp, book, pm, pf in (
                ("a", a, b, o["a"], d[a], fair[0]), ("b", b, a, o["b"], d[b], fair[1])):
            p_final = market.blend(pm, pf, bw.get("matchup", 0.0)) if blended else pm
            row = _bet_row(player, f"Matchup vs {opp}", f"matchup:{player}|{opp}",
                           book, pm, p_final, bankroll, kelly)
            row["p_market"] = round(pf, 4)
            rows.append(row)

    # ── 3-balls (joint-sim) ──
    tres = results.get("__threeballs__", {})
    for trio, o in threeball_odds.items():
        d = tres.get(trio)
        if d is None:
            continue
        fair = market.devig([o["a"], o["b"], o["c"]], method="multiplicative")
        for side, player, book, pm, pf in (
                ("a", trio[0], o["a"], d[trio[0]], fair[0]),
                ("b", trio[1], o["b"], d[trio[1]], fair[1]),
                ("c", trio[2], o["c"], d[trio[2]], fair[2])):
            p_final = market.blend(pm, pf, bw.get("3ball", 0.0)) if blended else pm
            others = "/".join(p for p in trio if p != player)
            # side encodes the player first, then the trio, so it can be graded
            ordered = player + "|" + "|".join(p for p in trio if p != player)
            row = _bet_row(player, f"3-ball vs {others}", f"3ball:{ordered}",
                           book, pm, p_final, bankroll, kelly)
            row["p_market"] = round(pf, 4)
            rows.append(row)

    rows = [r for r in rows if r["_ev"] * 100 >= min_edge]
    rows.sort(key=lambda r: -r["_ev"])
    for r in rows:
        r.pop("_ev", None)
    return rows


# ─────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────

def write_edge_report(bets: list[dict], path: Path | None = None) -> Path:
    path = path or DATA_DIR / "edge_report.csv"
    if not bets:
        print("  No bets with sufficient edge.")
        return path

    cols = ["player", "market", "side", "odds", "p_model", "p_market",
            "ev_per_unit", "stake_gbp", "recommended"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(bets)
    print(f"  {len(bets)} priced markets → {path}")
    return path


def print_edge_report(bets: list[dict]) -> None:
    if not bets:
        print("  No edges found at current threshold.")
        return
    print(f"\n{'Player':<28} {'Market':<14} {'Model%':>8} {'Odds':>6} {'EV%':>6} {'Stake':>8}")
    print("-" * 78)
    for b in bets:
        print(
            f"{b['player']:<28} {b['market']:<14} {b['model_prob']:>8} "
            f"{b['book_odds']:>6} {b['ev_pct']:>6} {b['kelly_stake']:>8}"
        )


# ─────────────────────────────────────────────
# v2 standalone field builder + report
# ─────────────────────────────────────────────

def build_rated_field(course: str = "", major: bool = False):
    """Rated Player objects from the fitted model (legacy players.csv fallback)."""
    from . import model as M
    from .model import load_field, load_players
    field = load_field(players=load_players())
    params = M.load_params()
    if params:
        return M.predict_field(field, params, course=course, is_major=major), True
    from .model import compute_ratings, load_course_history, load_recent_form
    ch = load_course_history(course) if course else {}
    return compute_ratings(field, course=course, is_major=major,
                           course_history=ch, recent_form=load_recent_form()), False


def print_priced(rows: list[dict]) -> None:
    if not rows:
        print("  No priced markets.")
        return
    print(f"\n{'Player':<22}{'Market':<20}{'Odds':>6}{'Model':>7}{'Mkt':>7}"
          f"{'EV':>7}{'Stake':>8}{'':>3}")
    print("-" * 80)
    for r in rows:
        pm = f"{r['p_market']*100:.1f}%" if r.get("p_market") not in ("", None) else "—"
        flag = "✓" if r.get("recommended") else ""
        print(f"{r['player']:<22}{r['market']:<20}{r['odds']:>6.2f}"
              f"{r['p_model']*100:>6.1f}%{pm:>7}{r['ev_per_unit']*100:>6.1f}%"
              f"£{r['stake_gbp']:>6.2f}{flag:>3}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    from . import simulate as GSIM
    from . import portfolio as GPORT
    import numpy as np

    ap = argparse.ArgumentParser(description="Golf betting edge calculator (v2)")
    ap.add_argument("--sims", type=int, default=50000)
    ap.add_argument("--course", default="")
    ap.add_argument("--major", action="store_true")
    ap.add_argument("--cut-rule", type=int, default=65)
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="Minimum EV%% to report (default %(default)s)")
    ap.add_argument("--kelly", type=float, default=DEFAULT_KELLY)
    ap.add_argument("--raw", action="store_true",
                    help="Disable calibration and market blend (raw sim probs)")
    ap.add_argument("--no-bet", action="store_true", help="Print only; don't write report")
    args = ap.parse_args()

    rated, fitted = build_rated_field(args.course, args.major)
    odds_data = load_odds_csv()
    matchup_odds = load_matchup_odds()
    threeball_odds = load_threeball_odds()
    if not (odds_data or matchup_odds or threeball_odds):
        print("No odds. Add data/odds.csv (name, odds_win, odds_top5, odds_top10, "
              "odds_top20, odds_cut) and/or matchups.csv / threeballs.csv.")
        sys.exit(1)
    bankroll = load_bankroll()

    print(f"Golf edge ({'fitted' if fitted else 'legacy'} model) | bankroll £{bankroll:.2f}"
          f" | {len(odds_data)} outright/place, {len(matchup_odds)} matchup, "
          f"{len(threeball_odds)} 3-ball boards")

    results = GSIM.simulate_tournament(
        rated, n_sims=args.sims, cut_rule=args.cut_rule,
        rng=np.random.default_rng(0),
        matchups=list(matchup_odds), threeballs=list(threeball_odds))

    if not results.get("__cut_binds__", True):
        print(f"  ⚠ field ({len(rated)}) ≤ cut rule ({args.cut_rule}): cut does "
              "not bind — make-cut/top-N suppressed (no-cut or stub field).")

    rows = price_all(rated, results, odds_data, matchup_odds, threeball_odds,
                     bankroll=bankroll, kelly=args.kelly,
                     calibrated=not args.raw, blended=not args.raw,
                     min_edge=args.min_edge)
    staked = GPORT.apply_portfolio([r for r in rows if r["ev_per_unit"] > 0],
                                   bankroll=bankroll)
    stake_keys = {(r["player"], r["side"]) for r in staked}
    stake_by = {(r["player"], r["side"]): r["stake_gbp"] for r in staked}
    for r in rows:
        r["stake_gbp"] = stake_by.get((r["player"], r["side"]), 0.0)
        r["recommended"] = (r["player"], r["side"]) in stake_keys

    print_priced(rows)
    print(f"\n{GPORT.summary(staked, bankroll)}")
    if not args.no_bet:
        write_edge_report(rows)


if __name__ == "__main__":
    main()
