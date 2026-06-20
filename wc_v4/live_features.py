"""Attach report-only live World Cup feed features to V4 feature rows."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup import squads as SQ  # noqa: E402
from wc_v4 import availability as AV  # noqa: E402

DATA_DIR = ROOT / "data" / "worldcup"
AVAILABILITY_CSV = DATA_DIR / "player_availability.csv"
LINEUPS_CSV = DATA_DIR / "lineups.csv"
MARKET_SNAPSHOTS_CSV = DATA_DIR / "market_snapshots.csv"
SQUAD_RATINGS = ROOT / "data" / "squad_ratings.csv"


def _as_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _read_asof(path: Path, asof: Any, time_col: str = "fetched_at") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or time_col not in df.columns:
        return df.iloc[0:0]
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    return df[ts.notna() & (ts <= _as_utc(asof))].copy()


def enrich(df: pd.DataFrame, asof: Any) -> pd.DataFrame:
    """Fill live report-only columns where canonical feed data exists."""
    out = df.copy()
    for col in ("avail_adj_h", "avail_adj_a", "lineup_conf_h", "lineup_conf_a",
                "confirmed_xi_power_h", "confirmed_xi_power_a",
                "bench_power_h", "bench_power_a",
                "formation_known_h", "formation_known_a",
                "market_dispersion_h", "market_dispersion_d",
                "market_dispersion_a"):
        if col not in out.columns:
            out[col] = np.nan

    availability = _availability_features(asof)
    lineups = _lineup_features(asof)
    market = _market_features(asof)

    for i, r in out.iterrows():
        home, away = str(r.get("home", "")), str(r.get("away", ""))
        eid = r.get("event_id", "")
        for side, team in (("h", home), ("a", away)):
            av = availability.get(team, {})
            out.at[i, f"avail_adj_{side}"] = av.get("avail_adj", np.nan)
            out.at[i, f"lineup_conf_{side}"] = av.get("lineup_conf", np.nan)
            lu = lineups.get((eid, team), {})
            if lu:
                out.at[i, f"confirmed_xi_power_{side}"] = lu.get("confirmed_xi_power")
                out.at[i, f"bench_power_{side}"] = lu.get("bench_power")
                out.at[i, f"formation_known_{side}"] = lu.get("formation_known")
                out.at[i, f"lineup_conf_{side}"] = 1.0
        md = market.get(eid, {})
        for col in ("market_dispersion_h", "market_dispersion_d",
                    "market_dispersion_a"):
            if col in md:
                out.at[i, col] = md[col]
    return out


def _availability_features(asof: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if SQUAD_RATINGS.exists():
        try:
            sr = pd.read_csv(SQUAD_RATINGS)
            for r in sr.itertuples(index=False):
                out[str(r.team)] = {
                    "avail_adj": float(getattr(r, "elo_adj", np.nan)),
                    "lineup_conf": np.nan,
                }
        except Exception:
            pass

    feed = _read_asof(AVAILABILITY_CSV, asof)
    if not feed.empty:
        feed = feed.drop_duplicates(subset=["team", "player"], keep="last")
        for team, grp in feed.groupby("team"):
            uncertain = grp[
                (grp.get("certainty", "").astype(str).str.lower() != "certain")
                | grp.get("status", "").astype(str).str.lower().isin(
                    ["doubtful", "limited_training"])
            ]
            conf = float(np.clip(1.0 - 0.18 * len(uncertain), 0.25, 1.0))
            out.setdefault(str(team), {"avail_adj": np.nan})["lineup_conf"] = conf

    # Manual/current absence fallback keeps the feature useful before API data lands.
    try:
        absences = AV._absences_df()
        for team in absences["team"].unique():
            c = AV.lineup_confidence(team, absences)
            out.setdefault(str(team), {"avail_adj": np.nan})
            if np.isnan(out[str(team)].get("lineup_conf", np.nan)):
                out[str(team)]["lineup_conf"] = float(c["confidence"])
    except Exception:
        pass
    return out


def _lineup_features(asof: Any) -> dict[tuple[str, str], dict[str, float]]:
    df = _read_asof(LINEUPS_CSV, asof)
    if df.empty:
        return {}
    if "published_at" in df.columns:
        pts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df[pts.notna() & (pts <= _as_utc(asof))].copy()
    if df.empty:
        return {}
    ea = None
    out: dict[tuple[str, str], dict[str, float]] = {}
    for (eid, team), grp in df.groupby(["event_id", "team"], dropna=False):
        if ea is None:
            try:
                ea = SQ.load_ea()
            except Exception:
                ea = pd.DataFrame()
        starters = grp[grp["starter"].astype(str).str.lower().isin(["true", "1"])]
        bench = grp[grp["role"].astype(str).str.lower() == "bench"]
        out[(str(eid), str(team))] = {
            "confirmed_xi_power": _mean_overall(str(team), starters["player"], ea),
            "bench_power": _mean_overall(str(team), bench["player"], ea),
            "formation_known": float(grp["formation"].fillna("").astype(str).str.len().gt(0).any()),
        }
    return out


def _mean_overall(team: str, players: pd.Series, ea: pd.DataFrame) -> float:
    vals = []
    if ea.empty or "nat" not in ea.columns:
        return np.nan
    pool = ea[ea["nat"] == team]
    if pool.empty:
        return np.nan
    cand = [(set(SQ.norm(r.long_name)) | set(SQ.norm(r.short_name)),
             float(r.overall)) for r in pool.itertuples()]
    for player in players.dropna().astype(str):
        toks = set(SQ.norm(player))
        best = None
        for ctoks, overall in cand:
            shared = len(toks & ctoks)
            if shared == 0:
                continue
            if shared < 2 and min(len(toks), len(ctoks)) > 1:
                continue
            if best is None or (shared, overall) > best:
                best = (shared, overall)
        if best is not None:
            vals.append(best[1])
    return float(np.mean(vals)) if vals else np.nan


def _market_features(asof: Any) -> dict[str, dict[str, float]]:
    df = _read_asof(MARKET_SNAPSHOTS_CSV, asof, "snapshot_time")
    if df.empty:
        return {}
    try:
        from scripts.worldcup.live_data import summarize_wide_market
    except Exception:
        return {}
    # Keep the latest known row per bookmaker/market/side/line before summarising.
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True,
                                         errors="coerce")
    df = (df.dropna(subset=["snapshot_time"])
            .sort_values("snapshot_time")
            .drop_duplicates(
                subset=["event_id", "bookmaker", "market", "side", "line"],
                keep="last"))
    wide = summarize_wide_market(df)
    out = {}
    for r in wide.itertuples(index=False):
        out[str(r.event_id)] = {
            "market_dispersion_h": getattr(r, "market_dispersion_h", np.nan),
            "market_dispersion_d": getattr(r, "market_dispersion_d", np.nan),
            "market_dispersion_a": getattr(r, "market_dispersion_a", np.nan),
        }
    return out
