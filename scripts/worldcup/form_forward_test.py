#!/usr/bin/env python3
"""Forward test — the fully-clean check the backtest can't be.

Freezes predictions for UPCOMING WC fixtures NOW, before any result exists, then
scores them after they're played. Because the predictions are written ahead of
kickoff, there is zero possibility of leakage — the decisive test of whether the
form layer helps.

  --predict : for each not-yet-played WC fixture, write baseline + form H/D/A
              probabilities to forward_predictions.csv (idempotent: never
              overwrites a fixture already frozen).
  --score   : for frozen fixtures that have since finished, pull the result and
              report cumulative baseline vs +form (Brier, log-loss, accuracy).

Lineups for upcoming games aren't published until ~1h pre-kickoff, so we use each
team's most-recent confirmed XI as the lineup proxy (re-run --predict once real
projected lineups drop for a sharper test). Form source: the frozen pre-tournament
cache (leak-free), consistent with the backtest.

Usage:
    python3 -m scripts.worldcup.form_forward_test --predict
    python3 -m scripts.worldcup.form_forward_test --score
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scripts.worldcup.form_config import (  # noqa: E402
    team_deltas, multipliers, load_params, load_player_club,
)
from scripts.worldcup.player_form_multipliers import load_xis  # noqa: E402
from scripts.worldcup.form_backtest import ALIAS, PRETOURN_CACHE  # noqa: E402

PRED_CSV = HERE / "data" / "worldcup" / "forward_predictions.csv"
WC_LEAGUE_ID = 27
COLS = ["frozen_at", "event_id", "date", "home", "away", "lineup_source",
        "base_H", "base_D", "base_A", "form_H", "form_D", "form_A",
        "status", "home_score", "away_score"]


def _hda(lam1, lam2):
    from engines.worldcup.predictor import score_matrix
    M = score_matrix(lam1, lam2)
    return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])


def _client():
    from api_keys import get_key
    from bsd_client import _get
    return _get, get_key("bsd", env="BSD_API_KEY")


def _upcoming(get, key):
    evs = get(f"/api/events/?league={WC_LEAGUE_ID}&limit=200", key).get("results") or []
    return [e for e in evs if str(e.get("status")) == "notstarted"]


def cmd_predict() -> None:
    get, key = _client()
    from engines.worldcup.predictor import (load_matches, compute_elo,
                                            fit_goal_model, expected_goals,
                                            HOME_ADV)
    played, _ = load_matches()
    ratings, played = compute_elo(played)
    beta = fit_goal_model(played)
    g_att, g_def = load_params()
    club = load_player_club()
    form_cache = json.loads(PRETOURN_CACHE.read_text()) if PRETOURN_CACHE.exists() else {}
    xis = load_xis()                                   # team -> most-recent XI

    existing = set()
    rows_out = []
    if PRED_CSV.exists():
        for r in csv.DictReader(PRED_CSV.open()):
            existing.add(r["event_id"]); rows_out.append(r)

    def rget(n):
        return ratings.get(ALIAS.get(n, n))

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    added = 0
    for e in _upcoming(get, key):
        eid = str(e.get("id"))
        if eid in existing:
            continue
        hn, an = e.get("home_team"), e.get("away_team")
        rh, ra = rget(hn), rget(an)
        xi_h, xi_a = xis.get(hn, []), xis.get(an, [])
        if rh is None or ra is None or not xi_h or not xi_a:
            continue
        adv = 0.0 if e.get("is_neutral_ground", True) else HOME_ADV
        lam1, lam2 = expected_goals(rh, ra, beta, adv)
        base = _hda(lam1, lam2)

        ah, dh, _ = team_deltas(xi_h, form_cache.get, club)
        aa, da, _ = team_deltas(xi_a, form_cache.get, club)
        am_h, dm_h = multipliers(ah, dh, g_att, g_def)
        am_a, dm_a = multipliers(aa, da, g_att, g_def)
        form = _hda(lam1 * am_h * dm_a, lam2 * am_a * dm_h)

        rows_out.append({
            "frozen_at": now, "event_id": eid,
            "date": str(e.get("event_date"))[:10], "home": hn, "away": an,
            "lineup_source": "latest_confirmed_xi",
            "base_H": round(float(base[0]), 4), "base_D": round(float(base[1]), 4),
            "base_A": round(float(base[2]), 4),
            "form_H": round(float(form[0]), 4), "form_D": round(float(form[1]), 4),
            "form_A": round(float(form[2]), 4),
            "status": "pending", "home_score": "", "away_score": "",
        })
        added += 1

    PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PRED_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows_out)
    print(f"froze {added} new prediction(s); {len(rows_out)} total in {PRED_CSV.name}")
    if added:
        print("\n  fixture                         base P(H/D/A)     +form P(H/D/A)")
        for r in rows_out[-added:]:
            print(f"  {r['home'][:12]:12} v {r['away'][:12]:12}  "
                  f"{r['base_H']:.2f}/{r['base_D']:.2f}/{r['base_A']:.2f}   "
                  f"{r['form_H']:.2f}/{r['form_D']:.2f}/{r['form_A']:.2f}")


def cmd_score() -> None:
    if not PRED_CSV.exists():
        sys.exit("no predictions yet — run --predict first.")
    get, key = _client()
    from bsd_client import get_event
    rows = list(csv.DictReader(PRED_CSV.open()))

    agg = {"base": [0.0, 0.0, 0], "form": [0.0, 0.0, 0]}
    scored = pending = 0
    changed = False
    for r in rows:
        if r["status"] == "scored":
            hs, as_ = int(r["home_score"]), int(r["away_score"])
        else:
            ev = get_event(key, int(r["event_id"]))
            if str(ev.get("status")) != "finished" or ev.get("home_score") is None:
                pending += 1
                continue
            hs, as_ = int(ev["home_score"]), int(ev["away_score"])
            r["status"] = "scored"; r["home_score"] = hs; r["away_score"] = as_
            changed = True
        actual = 0 if hs > as_ else (1 if hs == as_ else 2)
        for arm in ("base", "form"):
            p = np.array([float(r[f"{arm}_H"]), float(r[f"{arm}_D"]), float(r[f"{arm}_A"])])
            oneh = np.zeros(3); oneh[actual] = 1
            agg[arm][0] += float(((p - oneh) ** 2).sum())
            agg[arm][1] += float(-math.log(max(p[actual], 1e-12)))
            agg[arm][2] += int(np.argmax(p) == actual)
        scored += 1

    if changed:
        with PRED_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)

    print(f"\nForward test — {scored} scored, {pending} still pending\n")
    if not scored:
        print("No frozen fixtures have finished yet. Re-run --score after the next round.")
        return
    print(f"{'arm':10} {'accuracy':>9} {'Brier':>8} {'log-loss':>9}")
    print("─" * 40)
    for arm, lbl in (("base", "baseline"), ("form", "+form")):
        b, ll, c = agg[arm]
        print(f"{lbl:10} {c/scored:>8.1%} {b/scored:>8.4f} {ll/scored:>9.4f}")
    db = (agg["form"][0] - agg["base"][0]) / scored
    dll = (agg["form"][1] - agg["base"][1]) / scored
    print("─" * 40)
    print(f"{'delta':10} {'':>9} {db:>+8.4f} {dll:>+9.4f}")
    print("\n(predictions were frozen before kickoff — zero leakage.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Form-layer forward test")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.predict:
        cmd_predict()
    elif args.score:
        cmd_score()
    else:
        ap.error("pass --predict or --score")


if __name__ == "__main__":
    main()
