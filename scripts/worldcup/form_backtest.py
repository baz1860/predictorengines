#!/usr/bin/env python3
"""Leak-free A/B backtest of the player-form layer on WC 2026 (66 played matches).

The form store is a single current snapshot, so feeding it into the walk-forward
harness would be look-ahead. Instead we build each player's PRE-TOURNAMENT form —
their per-match history with the WC event-id block (8287-8352) removed — and apply
it as team multipliers. Because the baseline (Elo+Poisson lambdas) is identical in
both arms, the comparison isolates the marginal value of the form layer with zero
WC-result leakage.

Arms, scored against actual BSD scorelines:
  BASELINE : lam from expected_goals(elo_h, elo_a, beta)
  +FORM    : lam scaled by pre-tournament attack/defence multipliers

Metrics: 3-way accuracy, Brier, log-loss (lower Brier/logloss = better).

Usage:
    python3 -m scripts.worldcup.form_backtest               # all 66 matches
    python3 -m scripts.worldcup.form_backtest --max-matches 20
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scripts.worldcup.player_form import (  # noqa: E402
    POS_BASELINE, RATING_BASELINE, compute_form, fetch_player_stats, _client,
)
from scripts.worldcup.player_form_multipliers import (  # noqa: E402
    W_ATT, W_DEF, G_ATT, G_DEF, _norm_pos, _clamp,
)

LINEUPS = HERE / "data" / "worldcup" / "lineups.csv"
PRETOURN_CACHE = HERE / "data" / "worldcup" / "pretourn_form_cache.json"
EVENTS_CACHE = HERE / "data" / "worldcup" / "wc_events_cache.json"
WC_LO, WC_HI = 8287, 8352          # WC 2026 event-id block (excluded from form)

# BSD team name -> predictor (results.csv) name, for the handful that differ
ALIAS = {
    "Côte d'Ivoire": "Ivory Coast", "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "Cabo Verde": "Cape Verde",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina", "USA": "United States",
}


# ── pre-tournament form (leak-free) ───────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def prefetch_forms(pid_pos: dict[str, str], get, key, workers: int) -> dict:
    """Concurrently build pre-tournament form for all needed players, resuming
    from (and flushing to) the disk cache."""
    cache = _load_json(PRETOURN_CACHE)
    missing = [(p, pos) for p, pos in pid_pos.items()
               if p not in cache and p.isdigit()]
    if not missing:
        return cache
    print(f"  prefetching {len(missing)} player histories "
          f"({workers} workers)...", file=sys.stderr)

    def work(item):
        pid, pos = item
        rows = fetch_player_stats(get, key, int(pid))
        pre = [r for r in rows
               if not (WC_LO <= int(r.get("event_id", 0) or 0) <= WC_HI)]
        return pid, (compute_form(pre, pos) if pre else None)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pid, form in ex.map(work, missing):
            cache[pid] = form
            done += 1
            if done % 100 == 0:
                PRETOURN_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"    ...{done}/{len(missing)}", file=sys.stderr)
    PRETOURN_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    return cache


def prefetch_events(eids: list[int], get, key, workers: int) -> dict:
    """Concurrently fetch + cache the minimal event fields we score on."""
    cache = _load_json(EVENTS_CACHE)
    missing = [e for e in eids if str(e) not in cache]
    if missing:
        from bsd_client import get_event

        def work(eid):
            try:
                ev = get_event(key, eid)
            except Exception:  # noqa: BLE001
                return eid, None
            return eid, {
                "home_team": ev.get("home_team"), "away_team": ev.get("away_team"),
                "home_score": ev.get("home_score"), "away_score": ev.get("away_score"),
                "neutral": ev.get("is_neutral_ground", True),
            }
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for eid, rec in ex.map(work, missing):
                if rec:
                    cache[str(eid)] = rec
        EVENTS_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    return cache


# ── lineups (per fixture) ─────────────────────────────────────────────────────

def load_xi_by_fixture() -> dict[str, dict[str, list[dict]]]:
    """fixture_id -> {team_name: [starter dicts]}."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with LINEUPS.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("starter")).strip().lower() != "true":
                continue
            out[str(r["provider_fixture_id"])][r["team"]].append({
                "player_id": (r.get("provider_player_id") or "").strip(),
                "pos": _norm_pos(r.get("position")),
            })
    return out


def team_mult(xi: list[dict], cache) -> tuple[float, float]:
    na = da = nd = dd = 0.0
    for pl in xi:
        f = cache.get(pl["player_id"])
        if not f:
            continue
        pos = pl["pos"]
        bxg, bxa = POS_BASELINE.get(pos, POS_BASELINE["MF"])
        rt = f["rating"] - RATING_BASELINE
        att = rt + 1.5 * ((f["xg90"] + f["xa90"]) - (bxg + bxa))
        na += W_ATT[pos] * att; da += W_ATT[pos]
        nd += W_DEF[pos] * rt;  dd += W_DEF[pos]
    att_d = na / da if da else 0.0
    def_d = nd / dd if dd else 0.0
    return _clamp(1 + G_ATT * att_d), _clamp(1 - G_DEF * def_d)


# ── outcome probabilities ─────────────────────────────────────────────────────

def hda(lam1: float, lam2: float) -> np.ndarray:
    from engines.worldcup.predictor import score_matrix
    M = score_matrix(lam1, lam2)
    return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])


def score(probs: np.ndarray, actual: int) -> tuple[float, float, int]:
    oneh = np.zeros(3); oneh[actual] = 1.0
    brier = float(((probs - oneh) ** 2).sum())
    logloss = float(-math.log(max(probs[actual], 1e-12)))
    correct = int(np.argmax(probs) == actual)
    return brier, logloss, correct


# ── run ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Leak-free form-layer backtest")
    ap.add_argument("--max-matches", type=int)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    from engines.worldcup.predictor import (load_matches, compute_elo,
                                            fit_goal_model, expected_goals,
                                            HOME_ADV)
    played, _ = load_matches()
    ratings, played = compute_elo(played)
    beta = fit_goal_model(played)

    get, key = _client()
    xi_by_fx = load_xi_by_fixture()

    eids = list(range(WC_LO, WC_HI + 1))
    if args.max_matches:
        eids = eids[:args.max_matches]

    # ── concurrent prefetch (resumable, disk-cached) ──
    pid_pos: dict[str, str] = {}
    for e in eids:
        for xi in xi_by_fx.get(str(e), {}).values():
            for pl in xi:
                if pl["player_id"]:
                    pid_pos.setdefault(pl["player_id"], pl["pos"])
    form_cache = prefetch_forms(pid_pos, get, key, args.workers)
    ev_cache = prefetch_events(eids, get, key, args.workers)

    def rget(name):  # ratings lookup with alias
        return ratings.get(ALIAS.get(name, name))

    agg = {"base": [0.0, 0.0, 0], "form": [0.0, 0.0, 0]}
    n = 0
    skipped = 0
    for eid in eids:
        ev = ev_cache.get(str(eid))
        if not ev:
            continue
        hn, an = ev.get("home_team"), ev.get("away_team")
        hs, as_ = ev.get("home_score"), ev.get("away_score")
        if hs is None or as_ is None:
            continue
        rh, ra = rget(hn), rget(an)
        if rh is None or ra is None:
            skipped += 1
            print(f"  skip {eid}: no Elo for "
                  f"{hn if rh is None else an}", file=sys.stderr)
            continue
        xis = xi_by_fx.get(str(eid), {})
        xi_h = xis.get(hn, [])
        xi_a = xis.get(an, [])
        if not xi_h or not xi_a:
            skipped += 1
            continue

        adv = 0.0 if ev.get("neutral", True) else HOME_ADV
        lam1, lam2 = expected_goals(rh, ra, beta, adv)

        am_h, dm_h = team_mult(xi_h, form_cache)
        am_a, dm_a = team_mult(xi_a, form_cache)
        lam1f = lam1 * am_h * dm_a       # home scoring: home attack + away leaking
        lam2f = lam2 * am_a * dm_h

        actual = 0 if hs > as_ else (1 if hs == as_ else 2)
        for arm, (l1, l2) in (("base", (lam1, lam2)), ("form", (lam1f, lam2f))):
            b, ll, c = score(hda(l1, l2), actual)
            agg[arm][0] += b; agg[arm][1] += ll; agg[arm][2] += c
        n += 1

    if not n:
        sys.exit("no scorable matches")

    print(f"\nLeak-free form backtest — {n} WC matches "
          f"({skipped} skipped)\n")
    print(f"{'arm':10} {'accuracy':>9} {'Brier':>8} {'log-loss':>9}")
    print("─" * 40)
    for arm, label in (("base", "baseline"), ("form", "+form")):
        b, ll, c = agg[arm]
        print(f"{label:10} {c/n:>8.1%} {b/n:>8.4f} {ll/n:>9.4f}")
    db = (agg["form"][0] - agg["base"][0]) / n
    dll = (agg["form"][1] - agg["base"][1]) / n
    dacc = (agg["form"][2] - agg["base"][2]) / n
    print("─" * 40)
    print(f"{'delta':10} {dacc:>+8.1%} {db:>+8.4f} {dll:>+9.4f}")
    better = db < -1e-4 and dll < -1e-4
    worse = db > 1e-4 or dll > 1e-4
    verdict = ("FORM HELPS — lower Brier & log-loss" if better
               else "FORM HURTS — regresses Brier/log-loss" if worse
               else "NEUTRAL — within noise")
    print(f"\nverdict: {verdict}")
    print("(lower Brier/log-loss = better; baseline identical in both arms, so this\n"
          " isolates the form layer. Pre-tournament form only — no WC leakage.)")


if __name__ == "__main__":
    main()
