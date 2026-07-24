#!/usr/bin/env python3
"""Club-soccer forward test — the leak-free successor to the WC form test.

Freezes predictions for UPCOMING club fixtures NOW, before any result exists,
then scores them once played. Because predictions are written ahead of kickoff
there is zero possibility of leakage.

Three arms, each differing from its comparator by exactly ONE thing, so every
delta answers a single question:

    base  = ensemble with the xgf component zeroed and weights renormalised
            (no recency-form, no availability)
    xgf   = production ensemble (xgf weight 0.20 restored), no availability
    padj  = production ensemble + PlayerFeatureStore availability multipliers

    xgf  vs base  -> is the recency-SoT form component earning its weight?
    padj vs xgf   -> should --player-adj be switched ON in production?

Ensemble weights are pinned to DEFAULT_ENSEMBLE_W explicitly rather than read
from ensemble_weights.json. That artifact is currently deactivated, and pinning
means a weight change mid-test cannot silently redefine an arm after some rows
are already frozen.

Scope: BSD upcoming events whose league resolves to the club_soccer competition
registry AND whose teams both exist in the fitted params. Preseason friendlies
are excluded by construction (not in the registry). Note this also excludes MLS,
Brazil and Liga MX — the model's team universe is European.

    python3 -m scripts.club_soccer.form_forward_test --predict
    python3 -m scripts.club_soccer.form_forward_test --score
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from club_soccer import model as cs_model  # noqa: E402
from club_soccer.competitions import comp_from_bsd_league  # noqa: E402

PRED_CSV = HERE / "club_soccer" / "data" / "forward_predictions_club.csv"

ARMS = ("base", "xgf", "padj")
ARM_LABEL = {"base": "base (no form)", "xgf": "+xgf form", "padj": "+xgf +player-adj"}

COLS = (["frozen_at", "event_id", "date", "comp", "home", "away", "neutral",
         "n_missing_home", "n_missing_away", "lineup_confidence"]
        + [f"{a}_{o}" for a in ARMS for o in ("H", "D", "A")]
        + ["status", "home_score", "away_score"])


def _weights() -> tuple[dict, dict]:
    """(base_weights, production_weights) — base drops xgf and renormalises."""
    prod = dict(cs_model.DEFAULT_ENSEMBLE_W)
    base = {k: (0.0 if k == "xgf" else v) for k, v in prod.items()}
    s = sum(base.values())
    base = {k: v / s for k, v in base.items()}
    return base, prod


def _client():
    from api_keys import get_key
    key = get_key("bsd", env="BSD_API_KEY")
    if not key:
        sys.exit("BSD_API_KEY missing — cannot fetch fixtures. Stopping.")
    return key


def _hda(pred: dict) -> tuple[float, float, float]:
    p = pred["probs"]
    return float(p["home"]), float(p["draw"]), float(p["away"])


def cmd_predict() -> None:
    key = _client()
    from bsd_client import get_all_events, league_name as bsd_league_name
    from club_soccer.player_features import PlayerFeatureStore

    params = cs_model.load_params()
    teams = set(params["teams"])
    w_base, w_prod = _weights()

    store = PlayerFeatureStore()
    store.load()
    if not store._player_records():
        n = store.refresh_from_cache()
        if n:
            print(f"  player_adj: built player stats from {n} cached events.")

    existing: set[str] = set()
    rows_out: list[dict] = []
    if PRED_CSV.exists():
        for r in csv.DictReader(PRED_CSV.open()):
            existing.add(r["event_id"])
            rows_out.append(r)

    try:
        events = get_all_events(key, status="notstarted")
    except Exception as exc:
        sys.exit(f"BSD fetch failed ({exc}) — stopping, no fallback source.")

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    added = 0
    skipped_comp = skipped_team = 0

    for ev in events:
        eid = str(ev.get("id"))
        if eid in existing:
            continue
        comp = comp_from_bsd_league(bsd_league_name(ev))
        if comp is None:
            skipped_comp += 1
            continue
        home = str(ev.get("home_team") or "")
        away = str(ev.get("away_team") or "")
        if home not in teams or away not in teams or home == away:
            skipped_team += 1
            continue

        neutral = bool(ev.get("is_neutral_ground", False))
        mdate = str(ev.get("event_date") or "")[:10]

        adj = store.adjustments_for_match(ev)
        conf = ""
        try:
            from club_soccer.availability import match_availability, match_confidence
            conf = round(float(match_confidence(match_availability(store, ev))), 3)
        except Exception:
            pass

        kw = dict(competition=comp.name, model="ensemble", neutral=neutral,
                  params=params, match_date=mdate)
        preds = {
            "base": cs_model.predict(home, away, ensemble_weights=w_base, **kw),
            "xgf": cs_model.predict(home, away, ensemble_weights=w_prod, **kw),
            "padj": cs_model.predict(home, away, ensemble_weights=w_prod,
                                     player_adj=adj, **kw),
        }

        row = {
            "frozen_at": now, "event_id": eid, "date": mdate, "comp": comp.name,
            "home": home, "away": away, "neutral": int(neutral),
            "n_missing_home": int((adj.get("home") or {}).get("n_missing", 0)),
            "n_missing_away": int((adj.get("away") or {}).get("n_missing", 0)),
            "lineup_confidence": conf,
            "status": "pending", "home_score": "", "away_score": "",
        }
        for arm in ARMS:
            h, d, a = _hda(preds[arm])
            row[f"{arm}_H"] = round(h, 4)
            row[f"{arm}_D"] = round(d, 4)
            row[f"{arm}_A"] = round(a, 4)
        rows_out.append(row)
        added += 1

    PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PRED_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows_out)

    print(f"froze {added} new prediction(s); {len(rows_out)} total in {PRED_CSV.name}")
    print(f"  skipped: {skipped_comp} outside competition registry "
          f"(incl. friendlies), {skipped_team} with a team not in fitted params")
    if added:
        print(f"\n  {'fixture':34} {'base H/D/A':17} {'+xgf':17} {'+padj':17} miss")
        for r in rows_out[-added:]:
            fx = f"{r['home'][:15]} v {r['away'][:15]}"
            cells = " ".join(
                f"{float(r[f'{a}_H']):.2f}/{float(r[f'{a}_D']):.2f}/{float(r[f'{a}_A']):.2f}   "
                for a in ARMS)
            print(f"  {fx:34} {cells}{r['n_missing_home']}/{r['n_missing_away']}")


def cmd_score() -> None:
    if not PRED_CSV.exists():
        sys.exit("no predictions yet — run --predict first.")
    key = _client()
    from bsd_client import get_event

    rows = list(csv.DictReader(PRED_CSV.open()))
    agg = {a: [0.0, 0.0, 0] for a in ARMS}   # brier, logloss, hits
    scored = pending = 0
    changed = False

    for r in rows:
        if r["status"] == "scored":
            hs, as_ = int(r["home_score"]), int(r["away_score"])
        else:
            try:
                ev = get_event(key, int(r["event_id"]))
            except Exception as exc:
                sys.exit(f"BSD fetch failed ({exc}) — stopping.")
            if str(ev.get("status")) != "finished" or ev.get("home_score") is None:
                pending += 1
                continue
            hs, as_ = int(ev["home_score"]), int(ev["away_score"])
            r["status"] = "scored"
            r["home_score"] = hs
            r["away_score"] = as_
            changed = True
        actual = 0 if hs > as_ else (1 if hs == as_ else 2)
        for arm in ARMS:
            p = np.array([float(r[f"{arm}_H"]), float(r[f"{arm}_D"]), float(r[f"{arm}_A"])])
            oneh = np.zeros(3)
            oneh[actual] = 1
            agg[arm][0] += float(((p - oneh) ** 2).sum())
            agg[arm][1] += float(-math.log(max(p[actual], 1e-12)))
            agg[arm][2] += int(np.argmax(p) == actual)
        scored += 1

    if changed:
        with PRED_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)

    print(f"\nClub forward test — {scored} scored, {pending} still pending\n")
    if not scored:
        print("No frozen fixtures have finished yet. Re-run --score after the next round.")
        return

    print(f"{'arm':18} {'accuracy':>9} {'Brier':>8} {'log-loss':>9}")
    print("─" * 48)
    for arm in ARMS:
        b, ll, c = agg[arm]
        print(f"{ARM_LABEL[arm]:18} {c/scored:>8.1%} {b/scored:>8.4f} {ll/scored:>9.4f}")
    print("─" * 48)

    def delta(a: str, b: str, lbl: str) -> None:
        db = (agg[a][0] - agg[b][0]) / scored
        dll = (agg[a][1] - agg[b][1]) / scored
        print(f"{lbl:18} {'':>9} {db:>+8.4f} {dll:>+9.4f}")

    delta("xgf", "base", "xgf - base")
    delta("padj", "xgf", "padj - xgf")
    print("\n(negative = better. Predictions frozen before kickoff — zero leakage.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Club-soccer form/availability forward test")
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
