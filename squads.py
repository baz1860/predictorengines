#!/usr/bin/env python3
"""Squad power ratings from EA FC 26 player data, with availability adjustment.

Quantifies what match-results models can't see: who is actually available.
Squad quality itself is already priced into Elo/Dixon-Coles ratings, so the
adjustment applied to predictions is driven only by the *gap* between a
team's full-strength squad and its currently available squad (injuries,
suspensions, withdrawals listed in data/absences.csv).

    power      = mean overall rating of the squad's best 18 players
    elo_adj    = slope * (power_available - power_full)        [<= 0]

where slope (Elo points per rating point) is calibrated by regressing
current Elo on squad power across well-covered teams.

Teams whose squads can't be matched to enough EA players (n < MIN_MATCHED)
get no adjustment - predictions fall back to the unadjusted model.

Usage:
  python3 squads.py                  # refresh data/squad_ratings.csv
  python3 squads.py --report         # team table: power, coverage, adjustment
  python3 squads.py --match "Canada" "Bosnia and Herzegovina" [--home]
  python3 squads.py --match ... --without "Alphonso Davies"   # what-if

data/squads.csv    team,pos,player,source  (fifa | wiki | ea_proxy | ea_topup)
data/absences.csv  team,player,note        ('#' lines are comments)
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EA_CSV = HERE / "data" / "ea_players.csv"
SQUADS_CSV = HERE / "data" / "squads.csv"
ABSENCES_CSV = HERE / "data" / "absences.csv"
OUT_CSV = HERE / "data" / "squad_ratings.csv"

TOP_N = 18          # squad power uses the best TOP_N players
MIN_MATCHED = 15    # below this, no adjustment (insufficient EA coverage)
# starter-weighting (M5): likely-minutes proxy by squad rank (best XI, then bench)
STARTER_W = [1.0] * 11 + [0.5] * 7      # ranks 1-11 full, 12-18 half, rest 0
# position -> share of an absence's impact that lands on DEFENCE (rest on attack):
# a missing GK/DF mostly raises the opponent's goals; a missing FW mostly lowers
# the team's own goals; midfielders split evenly.
POS_DEF_SHARE = {"GK": 0.75, "DF": 0.75, "MF": 0.5, "FW": 0.25}

EA_NAT_ALIAS = {
    "Korea Republic": "South Korea", "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde", "Curacao": "Curaçao", "Congo DR": "DR Congo",
    "Türkiye": "Turkey", "IR Iran": "Iran", "Czechia": "Czech Republic",
}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", " ", s.lower()).split()


def load_ea():
    ea = pd.read_csv(EA_CSV, low_memory=False)
    ea["nat"] = ea["nationality_name"].map(lambda n: EA_NAT_ALIAS.get(n, n))
    return ea


def match_squads(squads, ea, verbose=False):
    """Match each squad player to an EA player of the same nationality.

    Token-overlap scoring between the squad-list name and EA long/short
    names; ties broken by higher overall (squad members are usually the
    better-known player). Returns DataFrame: team, player, overall, pos.
    """
    out = []
    for team, grp in squads.groupby("team"):
        pool = ea[ea["nat"] == team]
        if pool.empty:
            continue
        cand = [(set(norm(r.long_name)) | set(norm(r.short_name)),
                 r.overall, r.player_id) for r in pool.itertuples()]
        used, rows = set(), []
        for sq in grp.itertuples():
            toks = set(norm(sq.player))
            best = None
            for ctoks, overall, pid in cand:
                shared = len(toks & ctoks)
                if shared == 0:
                    continue
                # require 2+ shared tokens unless one name is a single token
                if shared < 2 and min(len(toks), len(ctoks)) > 1:
                    continue
                key = (shared, overall)
                if best is None or key > best[0]:
                    best = (key, pid, overall)
            if best and best[1] not in used:
                used.add(best[1])
                rows.append({"team": team, "player": sq.player,
                             "pos": sq.pos, "overall": best[2]})
            elif verbose and not best:
                print(f"  unmatched: {team}: {sq.player}")
        out.extend(rows)
    return pd.DataFrame(out)


def squad_power(overalls, top_n=TOP_N):
    """Starter-weighted squad power: the best XI counts full, ranks 12-18 half,
    the rest not at all (a proxy for likely minutes). Removing a starter promotes
    a weaker backup into the XI, so availability gaps are captured naturally."""
    top = sorted((float(o) for o in overalls), reverse=True)[:top_n]
    if not top:
        return np.nan
    w = STARTER_W[:len(top)]
    return float(np.dot(top, w) / sum(w))


def load_absences():
    """Manual absences (data/absences.csv) + API pulls (data/absences_api.csv)."""
    frames = []
    for path in (ABSENCES_CSV, ABSENCES_CSV.with_name("absences_api.csv")):
        if path.exists():
            frames.append(pd.read_csv(path, comment="#"))
    if not frames:
        return pd.DataFrame(columns=["team", "player", "note"])
    return (pd.concat(frames, ignore_index=True)
            .dropna(subset=["player"])
            .drop_duplicates(subset=["team", "player"]))


def current_elo():
    from predictor import load_matches, compute_elo
    played, _ = load_matches()
    ratings, _ = compute_elo(played)
    return ratings


def refresh(verbose=False, absences=None, save=True):
    squads = pd.read_csv(SQUADS_CSV)
    ea = load_ea()
    matched = match_squads(squads, ea, verbose=verbose)
    if absences is None:
        absences = load_absences()
    ratings = current_elo()

    rows = []
    for team, grp in matched.groupby("team"):
        n_squad = (squads["team"] == team).sum()
        full = squad_power(grp["overall"])
        absent = absences[absences["team"] == team]["player"].tolist()
        if absent:
            absent_toks = [set(norm(a)) for a in absent]
            is_out = grp["player"].map(
                lambda p: any(len(set(norm(p)) & at) >= min(2, len(at))
                              for at in absent_toks))
        else:
            is_out = pd.Series(False, index=grp.index)
        avail, gone = grp[~is_out], grp[is_out]
        adj = squad_power(avail["overall"])
        # split the absence impact into defence vs attack, weighted by overall
        if len(gone) and gone["overall"].sum() > 0:
            def_frac = float(sum(POS_DEF_SHARE.get(p, 0.5) * o
                                 for p, o in zip(gone["pos"], gone["overall"]))
                             / gone["overall"].sum())
        else:
            def_frac = 0.5
        rows.append({"team": team, "n_squad": n_squad, "n_matched": len(grp),
                     "n_out": len(gone),
                     "power_full": round(full, 2), "power_avail": round(adj, 2),
                     "def_frac": round(def_frac, 3),
                     "elo": round(ratings.get(team, np.nan))})
    df = pd.DataFrame(rows)

    # calibrate Elo-per-rating-point on well-covered teams, full-strength power
    ok = df[df["n_matched"] >= MIN_MATCHED].dropna(subset=["elo"])
    slope = np.polyfit(ok["power_full"], ok["elo"], 1)[0]
    df["elo_adj"] = np.where(
        df["n_matched"] >= MIN_MATCHED,
        (slope * (df["power_avail"] - df["power_full"])).round(1), 0.0)
    # position-aware split (att_adj + def_adj == elo_adj)
    df["att_adj"] = (df["elo_adj"] * (1.0 - df["def_frac"])).round(2)
    df["def_adj"] = (df["elo_adj"] * df["def_frac"]).round(2)
    df = df.sort_values("power_full", ascending=False)
    if save:
        df.to_csv(OUT_CSV, index=False)
        print(f"Calibration: {slope:.1f} Elo points per rating point "
              f"(fit on {len(ok)} teams with >= {MIN_MATCHED} matched players)")
        print(f"Saved -> {OUT_CSV.relative_to(HERE)}")
    return df, slope


def load_adjustments():
    """team -> Elo adjustment (<= 0). Empty dict if never refreshed."""
    if not OUT_CSV.exists():
        return {}
    df = pd.read_csv(OUT_CSV)
    return dict(zip(df["team"], df["elo_adj"]))


load_adj = load_adjustments   # alias used by edge.py combined --conf-adj --squad-adj path


def load_adj_split():
    """team -> (att_adj, def_adj). Falls back to a 50/50 split of elo_adj for a
    csv written before M5. Empty dicts if never refreshed."""
    if not OUT_CSV.exists():
        return {}, {}
    df = pd.read_csv(OUT_CSV)
    if "att_adj" in df.columns and "def_adj" in df.columns:
        return dict(zip(df["team"], df["att_adj"])), dict(zip(df["team"], df["def_adj"]))
    half = df["elo_adj"] / 2.0
    return dict(zip(df["team"], half)), dict(zip(df["team"], half))


def adjusted_sources(model="blend", extra_out=None):
    """build_sources(), with each lambda pair scaled asymmetrically by availability.

    A team's attack adjustment lowers its OWN goals; its defence adjustment raises
    the OPPONENT's goals (M5). The 2k factor makes a 50/50 split reduce exactly to
    the old symmetric Elo-gap model. extra_out: optional [(team, player), ...]
    what-if absences.
    """
    from dixoncoles import build_sources
    from predictor import load_matches, compute_elo, fit_goal_model
    played, _ = load_matches()
    _, played = compute_elo(played)
    k = fit_goal_model(played)[1] / 400.0   # goals-model slope per Elo point

    if extra_out:
        df, _ = refresh_with_extra(extra_out)
        attA = dict(zip(df["team"], df["att_adj"]))
        defD = dict(zip(df["team"], df["def_adj"]))
        adj = dict(zip(df["team"], df["elo_adj"]))
    else:
        attA, defD = load_adj_split()
        adj = load_adjustments()

    sources, ratings = build_sources(model)
    wrapped = []
    for fn, rho in sources:
        def make(fn):
            def f(t1, t2, h1=0.0, h2=0.0):
                l1, l2 = fn(t1, t2, h1, h2)
                a1, d1 = attA.get(t1, 0.0), defD.get(t1, 0.0)
                a2, d2 = attA.get(t2, 0.0), defD.get(t2, 0.0)
                return (l1 * np.exp(2 * k * (a1 - d2)),
                        l2 * np.exp(2 * k * (a2 - d1)))
            return f
        wrapped.append((make(fn), rho))
    return wrapped, ratings, adj


def refresh_with_extra(extra_out):
    """Recompute ratings with extra hypothetical absences (not persisted)."""
    absences = load_absences()
    extra = pd.DataFrame([{"team": t, "player": p, "note": "what-if"}
                          for t, p in extra_out])
    combined = pd.concat([absences, extra], ignore_index=True)
    return refresh(absences=combined, save=False)


def cmd_match(team1, team2, home=False, without=None):
    from dixoncoles import outcome_probs, build_sources
    extra = None
    if without:
        squads = pd.read_csv(SQUADS_CSV)
        extra = []
        for name in without:
            hit = squads[squads["player"].str.contains(name.split()[-1],
                                                       case=False, na=False)]
            for t in hit["team"].unique():
                if t in (team1, team2):
                    extra.append((t, name))
        if not extra:
            sys.exit(f"No player matching {without} found in either squad.")

    h1 = 1.0 if home else 0.0
    raw_sources, _ = build_sources("blend")
    adj_sources, _, adj = adjusted_sources("blend", extra_out=extra)

    def blend_probs(sources):
        ps = []
        for fn, rho in sources:
            l1, l2 = fn(team1, team2, h1, 0.0)
            w, d, l, _ = outcome_probs(l1, l2, rho)
            ps.append((w, d, l, l1, l2))
        m = np.mean(ps, axis=0)
        return m

    raw, new = blend_probs(raw_sources), blend_probs(adj_sources)
    print(f"\n{team1} vs {team2}" + ("  [home: " + team1 + "]" if home else "  [neutral]"))
    if extra:
        print("What-if absences: " + ", ".join(f"{p} ({t})" for t, p in extra))
    print(f"{'':14s}{'win':>8s}{'draw':>8s}{'loss':>8s}{'xG':>12s}")
    print(f"{'unadjusted':14s}{raw[0]:8.1%}{raw[1]:8.1%}{raw[2]:8.1%}"
          f"{raw[3]:7.2f}-{raw[4]:.2f}")
    print(f"{'adjusted':14s}{new[0]:8.1%}{new[1]:8.1%}{new[2]:8.1%}"
          f"{new[3]:7.2f}-{new[4]:.2f}")
    print(f"Adjustments in play: "
          f"{team1} {adj.get(team1, 0):+.1f} Elo, {team2} {adj.get(team2, 0):+.1f} Elo")


def cmd_report():
    df, _ = refresh()
    absences = load_absences()
    df["flag"] = np.where(df["n_matched"] < MIN_MATCHED, "low-coverage", "")
    print(df.to_string(index=False))
    if len(absences):
        print("\nListed absences:")
        print(absences.to_string(index=False))
    else:
        print("\nNo absences listed in data/absences.csv.")


def main():
    ap = argparse.ArgumentParser(description="EA-based squad power ratings")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--match", nargs=2, metavar=("TEAM1", "TEAM2"))
    ap.add_argument("--home", action="store_true", help="team1 at home")
    ap.add_argument("--without", action="append", metavar="PLAYER",
                    help="what-if absence (repeatable), used with --match")
    ap.add_argument("--verbose", action="store_true", help="show unmatched players")
    args = ap.parse_args()
    if args.match:
        cmd_match(args.match[0], args.match[1], args.home, args.without)
    elif args.report:
        cmd_report()
    else:
        refresh(verbose=args.verbose)


if __name__ == "__main__":
    main()
