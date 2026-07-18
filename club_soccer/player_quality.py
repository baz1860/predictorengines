"""Point-in-time player-quality features and a strict validation gate.

The metric-stability paper is useful here as a feature-selection prior, not as
evidence that player data will improve match forecasts automatically.  This
module therefore does three things explicitly:

* builds team snapshots using only player appearances before the match;
* shrinks xG/90 and pass completion toward positional/league-like priors;
* keeps the resulting match correction inactive unless fixed walk-forward
  splits improve both Brier score and log-loss.

The cache is intentionally treated as an observation store.  A player who
transfers contributes only the appearances for the queried team, and a
missing/low-coverage team returns a neutral correction rather than an
invented squad strength.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLAYER_CACHE = DATA / "player_stats_cache.json"
ARTIFACT = DATA / "player_quality.json"
VALIDATION_PREDICTIONS = DATA / "validation_predictions.csv"

SPLITS = ("2025-01-01", "2025-07-01", "2025-12-01")
LOOKBACK_DAYS = 365
HALF_LIFE_DAYS = 180.0
MIN_PLAYER_MINUTES = 450.0
MIN_TEAM_PLAYERS = 6
PRIOR_XG90 = {"GK": 0.00, "DF": 0.03, "MF": 0.07, "FW": 0.22}
PRIOR_PASS_PCT = 0.78
PRIOR_PASS_COUNT = 200.0
PRIOR_XG_MINUTES = 450.0


def _norm_team(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _asof(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _metrics(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    value = row.get("metrics", {}) if hasattr(row, "get") else {}
    return value if isinstance(value, dict) else {}


class PlayerQualityStore:
    """Query recency-weighted player observations as-of a match date."""

    def __init__(self, cache_path: Path = PLAYER_CACHE):
        self.cache_path = Path(cache_path)
        self._by_team_player: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._loaded = False

    def load(self) -> "PlayerQualityStore":
        try:
            raw = json.loads(self.cache_path.read_text())
        except Exception:
            raw = {}
        for key, rec in raw.items():
            if key == "v" or not isinstance(rec, dict):
                continue
            player_key = str(rec.get("player_id") or key)
            pos = str(rec.get("pos") or "MF").upper()
            for app in rec.get("apps", []):
                if not isinstance(app, dict) or not app.get("team") or not app.get("date"):
                    continue
                team_key = _norm_team(app.get("team"))
                if not team_key:
                    continue
                bucket = self._by_team_player[team_key].setdefault(
                    player_key, {"pos": pos, "apps": []})
                bucket["pos"] = pos if pos != "MF" else bucket.get("pos", "MF")
                bucket["apps"].append(app)
        for players in self._by_team_player.values():
            for rec in players.values():
                rec["apps"].sort(key=lambda a: str(a.get("date", "")))
        self._loaded = True
        return self

    @property
    def app_count(self) -> int:
        return sum(len(rec["apps"])
                   for players in self._by_team_player.values()
                   for rec in players.values())

    @property
    def team_count(self) -> int:
        return len(self._by_team_player)

    def team_quality(self, team: str, as_of: Any,
                     lookback_days: int = LOOKBACK_DAYS) -> dict[str, Any]:
        """Return a strictly pre-match team quality snapshot.

        The top 11 are selected by recency-weighted minutes.  xG/90 receives
        a positional prior equivalent to 450 minutes; pass completion receives
        a conservative 200-pass prior.  These shrinkage choices are deliberate
        guardrails against one short cameo dominating a forecast.
        """
        if not self._loaded:
            self.load()
        anchor = _asof(as_of)
        start = anchor - pd.Timedelta(days=int(lookback_days))
        rows: list[dict[str, Any]] = []
        for player, rec in self._by_team_player.get(_norm_team(team), {}).items():
            eligible: list[dict[str, Any]] = []
            for app in rec["apps"]:
                try:
                    date = _asof(app.get("date"))
                except Exception:
                    continue
                if not (start <= date < anchor):
                    continue
                mins = max(0.0, float(app.get("mins", 0.0) or 0.0))
                if mins <= 0:
                    continue
                age = max(0, int((anchor - date).days))
                wt = math.exp(-math.log(2.0) * age / HALF_LIFE_DAYS)
                eligible.append({"app": app, "mins": mins, "wt": wt})
            raw_minutes = sum(r["mins"] for r in eligible)
            if raw_minutes < MIN_PLAYER_MINUTES:
                continue

            weighted_minutes = sum(r["mins"] * r["wt"] for r in eligible)
            weighted_xg = sum(float(r["app"].get("xg", 0.0) or 0.0) * r["wt"]
                              for r in eligible)
            pos = rec.get("pos", "MF")
            pos_prior = PRIOR_XG90.get(pos, PRIOR_XG90["MF"])
            xg90 = ((weighted_xg + pos_prior * PRIOR_XG_MINUTES / 90.0) /
                    max(1e-9, (weighted_minutes + PRIOR_XG_MINUTES) / 90.0))

            pass_total = 0.0
            pass_accurate = 0.0
            for item in eligible:
                stats = _metrics(item["app"])
                total = stats.get("total_pass")
                accurate = stats.get("accurate_pass")
                if total is None or accurate is None:
                    continue
                try:
                    pass_total += max(0.0, float(total)) * item["wt"]
                    pass_accurate += max(0.0, float(accurate)) * item["wt"]
                except (TypeError, ValueError):
                    continue
            pass_pct = ((pass_accurate + PRIOR_PASS_PCT * PRIOR_PASS_COUNT) /
                        (pass_total + PRIOR_PASS_COUNT)) if pass_total > 0 else np.nan
            rows.append({
                "player": player,
                "pos": pos,
                "raw_minutes": raw_minutes,
                "weighted_minutes": weighted_minutes,
                "xg90": float(xg90),
                "pass_pct": float(pass_pct) if np.isfinite(pass_pct) else np.nan,
                "pass_total": pass_total,
            })

        rows.sort(key=lambda r: r["weighted_minutes"], reverse=True)
        xi = rows[:11]
        xi_minutes = sum(r["weighted_minutes"] for r in xi)
        if not xi:
            return {"team": str(team), "as_of": str(anchor.date()), "n_players": 0,
                    "n_pass_players": 0, "attack_xg90": np.nan,
                    "pass_pct": np.nan, "coverage": 0.0, "uncertainty": 1.0}
        attack_xg90 = sum(r["xg90"] for r in xi)
        pass_rows = [r for r in xi if np.isfinite(r["pass_pct"])]
        pass_weight = sum(r["weighted_minutes"] for r in pass_rows)
        pass_pct = (sum(r["pass_pct"] * r["weighted_minutes"] for r in pass_rows) /
                    pass_weight) if pass_weight else np.nan
        coverage = min(1.0, len(xi) / 11.0)
        # Uncertainty is for downstream stake haircuts, not a hidden point
        # estimate.  It rises with low XI coverage and missing pass data.
        uncertainty = min(1.0, 0.5 * (1.0 - coverage) +
                          0.5 * (1.0 - len(pass_rows) / max(1, len(xi))))
        return {
            "team": str(team), "as_of": str(anchor.date()),
            "n_players": len(xi), "n_pass_players": len(pass_rows),
            "attack_xg90": float(attack_xg90),
            "pass_pct": float(pass_pct) if np.isfinite(pass_pct) else np.nan,
            "coverage": float(coverage), "uncertainty": float(uncertainty),
            "minutes": float(xi_minutes),
        }

    def match_features(self, home: str, away: str, as_of: Any) -> dict[str, Any]:
        """Return home-minus-away quality features and coverage diagnostics."""
        h = self.team_quality(home, as_of)
        a = self.team_quality(away, as_of)
        qh, qa = h.get("attack_xg90"), a.get("attack_xg90")
        ph, pa = h.get("pass_pct"), a.get("pass_pct")
        return {
            "home": h, "away": a,
            "attack_xg90_diff": float(qh - qa) if np.isfinite(qh) and np.isfinite(qa) else np.nan,
            "pass_pct_diff": float(ph - pa) if np.isfinite(ph) and np.isfinite(pa) else np.nan,
            "coverage": float(min(h.get("coverage", 0.0), a.get("coverage", 0.0))),
            "usable": bool(h.get("n_players", 0) >= MIN_TEAM_PLAYERS and
                           a.get("n_players", 0) >= MIN_TEAM_PLAYERS and
                           np.isfinite(qh) and np.isfinite(qa)),
        }


def quality_probs(base: np.ndarray, attack_diff: float, pass_diff: float,
                  coefficients: dict[str, float], max_shift: float = 0.20) -> np.ndarray:
    """Apply a small home/away logit shift to an existing 1X2 forecast."""
    p = np.asarray(base, dtype=float)
    logits = np.log(np.clip(p, 1e-9, 1.0))
    q = float(attack_diff) if np.isfinite(attack_diff) else 0.0
    pp = float(pass_diff) if np.isfinite(pass_diff) else 0.0
    shift = (float(coefficients.get("attack_xg90", 0.0)) * q +
             float(coefficients.get("pass_pct", 0.0)) * pp)
    shift = float(np.clip(shift, -max_shift, max_shift))
    logits[0] += shift
    logits[2] -= shift
    logits -= logits.max()
    out = np.exp(logits)
    return out / out.sum()


def _scores(P: np.ndarray, A: np.ndarray) -> dict[str, float]:
    if len(A) == 0:
        return {"n": 0, "brier": None, "log_loss": None}
    one = np.eye(3)[A]
    return {"n": int(len(A)),
            "brier": float(((P - one) ** 2).sum(1).mean()),
            "log_loss": float((-np.log(np.clip(P[np.arange(len(A)), A], 1e-12, 1.0))).mean())}


def _candidate_frame(store: PlayerQualityStore) -> pd.DataFrame:
    if not VALIDATION_PREDICTIONS.exists():
        return pd.DataFrame()
    df = pd.read_csv(VALIDATION_PREDICTIONS)
    df["date"] = pd.to_datetime(df["date"])
    # The player cache starts in 2025-07; earlier rows are retained as
    # explicit no-feature rows so the coverage shortfall is visible.
    df["attack_xg90_diff"] = np.nan
    df["pass_pct_diff"] = np.nan
    df["coverage"] = 0.0
    df["usable"] = False
    cache: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    for i, row in df.iterrows():
        key = (str(row["date"]), pd.Timestamp(row["date"]).normalize())
        # The first element keeps the cache key unambiguous in debug output;
        # the actual lookup is still keyed by both teams and date.
        pair = (str(row["home"]), str(row["away"]), key[1])
        if pair not in cache:
            cache[pair] = store.match_features(row["home"], row["away"], key[1])
        feat = cache[pair]
        df.at[i, "attack_xg90_diff"] = feat["attack_xg90_diff"]
        df.at[i, "pass_pct_diff"] = feat["pass_pct_diff"]
        df.at[i, "coverage"] = feat["coverage"]
        df.at[i, "usable"] = feat["usable"]
    return df


def _fit_coefficients(df: pd.DataFrame) -> dict[str, float]:
    """Small, regularised grid fit; deterministic and dependency-light."""
    if len(df) < 50:
        return {"attack_xg90": 0.0, "pass_pct": 0.0}
    base = df[["p_home", "p_draw", "p_away"]].to_numpy(float)
    actual = df["actual"].to_numpy(int)
    q = df["attack_xg90_diff"].to_numpy(float)
    pp = df["pass_pct_diff"].to_numpy(float)
    best = (float("inf"), 0.0, 0.0)
    for bx in np.linspace(-1.5, 1.5, 31):
        for by in np.linspace(-8.0, 8.0, 33):
            pred = np.array([quality_probs(p, x, y,
                              {"attack_xg90": bx, "pass_pct": by})
                             for p, x, y in zip(base, q, pp)])
            ll = float((-np.log(np.clip(pred[np.arange(len(actual)), actual], 1e-12, 1.0))).mean())
            objective = ll + 0.01 * (bx * bx + (by / 8.0) ** 2)
            if objective < best[0]:
                best = (objective, float(bx), float(by))
    return {"attack_xg90": best[1], "pass_pct": best[2]}


def validate(verbose: bool = True, write: bool = True) -> dict[str, Any]:
    store = PlayerQualityStore().load()
    df = _candidate_frame(store)
    if df.empty:
        raise SystemExit("No validation predictions available.")
    usable = df[df["usable"] & df["attack_xg90_diff"].notna() &
                df["pass_pct_diff"].notna()].copy()
    split_results: list[dict[str, Any]] = []
    for split_name in SPLITS:
        split = pd.Timestamp(split_name)
        train = usable[usable["date"] < split]
        test = usable[usable["date"] >= split]
        row: dict[str, Any] = {"split": split_name, "train_n": int(len(train)),
                               "test_n": int(len(test)),
                               "eligible": bool(len(train) >= 50 and len(test) >= 100)}
        if row["eligible"]:
            coef = _fit_coefficients(train)
            base = test[["p_home", "p_draw", "p_away"]].to_numpy(float)
            cand = np.array([quality_probs(p, x, y, coef)
                             for p, x, y in zip(base, test["attack_xg90_diff"],
                                                test["pass_pct_diff"])])
            actual = test["actual"].to_numpy(int)
            raw = _scores(base, actual)
            adjusted = _scores(cand, actual)
            row.update({"coefficients": coef, "raw": raw, "candidate": adjusted,
                        "delta_brier": adjusted["brier"] - raw["brier"],
                        "delta_log_loss": adjusted["log_loss"] - raw["log_loss"]})
        split_results.append(row)
        if verbose:
            if row["eligible"]:
                print(f"  split {split_name}: train={row['train_n']} test={row['test_n']} "
                      f"ΔBrier {row['delta_brier']:+.6f} "
                      f"Δlog-loss {row['delta_log_loss']:+.6f}")
            else:
                print(f"  split {split_name}: ineligible "
                      f"(train={row['train_n']}, test={row['test_n']})")

    promotes = bool(split_results) and all(
        r["eligible"] and r["candidate"]["brier"] < r["raw"]["brier"] and
        r["candidate"]["log_loss"] < r["raw"]["log_loss"]
        for r in split_results
    )
    all_coef = _fit_coefficients(usable)
    all_base = usable[["p_home", "p_draw", "p_away"]].to_numpy(float)
    all_cand = np.array([quality_probs(p, x, y, all_coef)
                         for p, x, y in zip(all_base, usable["attack_xg90_diff"],
                                            usable["pass_pct_diff"])])
    all_actual = usable["actual"].to_numpy(int)
    payload = {
        "active": bool(promotes),
        "coefficients": all_coef,
        "lookback_days": LOOKBACK_DAYS,
        "half_life_days": HALF_LIFE_DAYS,
        "min_player_minutes": MIN_PLAYER_MINUTES,
        "min_team_players": MIN_TEAM_PLAYERS,
        "coverage": {"validation_rows": int(len(df)),
                      "usable_rows": int(len(usable)),
                      "usable_rate": float(len(usable) / max(1, len(df))),
                      "cache_apps": store.app_count, "cache_teams": store.team_count},
        "heldout": {"splits": split_results, "promote": bool(promotes),
                    "all_data": {"raw": _scores(all_base, all_actual),
                                 "candidate": _scores(all_cand, all_actual)}},
        "note": "Inactive unless every fixed split has at least 50 training and "
                "100 test rows and both Brier and log-loss improve. Features are "
                "point-in-time and neutral on low coverage.",
    }
    if write:
        ARTIFACT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    if verbose:
        print(f"  player-quality coverage: {len(usable)}/{len(df)} "
              f"({len(usable) / max(1, len(df)):.1%})")
        print(f"  gate: {'PROMOTE' if promotes else 'keep inactive'}")
    return payload


def load_artifact(path: Path = ARTIFACT) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {"active": False, "coefficients": {}}
    return payload if isinstance(payload, dict) else {"active": False, "coefficients": {}}


def adjustment_for_match(home: str, away: str, match_date: Any,
                         store: PlayerQualityStore | None = None,
                         artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a gated, model-ready quality adjustment or a neutral result."""
    artifact = load_artifact() if artifact is None else artifact
    store = PlayerQualityStore().load() if store is None else store
    features = store.match_features(home, away, match_date)
    if not artifact.get("active") or not features.get("usable"):
        return {"active": False, "coverage": features.get("coverage", 0.0),
                "features": features}
    coef = artifact.get("coefficients") or {}
    shift = (float(coef.get("attack_xg90", 0.0)) *
             (features["attack_xg90_diff"] if np.isfinite(features["attack_xg90_diff"]) else 0.0) +
             float(coef.get("pass_pct", 0.0)) *
             (features["pass_pct_diff"] if np.isfinite(features["pass_pct_diff"]) else 0.0))
    shift = float(np.clip(shift, -0.20, 0.20))
    return {"active": True, "shift": shift, "coverage": features["coverage"],
            "features": features, "coefficients": coef}


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate point-in-time player quality.")
    ap.add_argument("--validate", action="store_true", help="run fixed-split gate")
    ap.add_argument("--no-write", action="store_true", help="do not write artifact")
    args = ap.parse_args()
    if args.validate or not args.no_write:
        validate(write=not args.no_write)


if __name__ == "__main__":
    main()
