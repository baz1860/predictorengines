#!/usr/bin/env python3
"""Monte Carlo simulator for the 2026 World Cup (48-team format).

Uses the Elo + Poisson match model from predictor.py. Simulates the group
stage (already-played matches are taken as fixed results, so re-run it as
the tournament progresses after refreshing data/results.csv), ranks the
twelve groups, picks the eight best third-placed teams, and plays out the
official FIFA bracket (Round of 32 slots per the tournament regulations;
third-place teams assigned to slots by constraint matching).

Usage:
  python simulate.py            # 10,000 simulations
  python simulate.py -n 50000   # more precision
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .predictor import load_matches, score_matrix, MAX_GOALS
from .dixoncoles import build_sources

HOSTS = {"United States", "Mexico", "Canada"}

# Official FIFA Annex C third-place allocation table (optional, v2 M4).
# FIFA 2026 regulations Annex C predefine, for each of the C(12,8)=495 possible
# combinations of which eight groups supply a qualifying third-placed team, the
# exact Round-of-32 slot each of those teams takes. The constraints alone do NOT
# determine it (every combination admits 3-214 valid matchings), so the official
# mapping has to be supplied as data. If data/annexc_thirds.json is present it is
# used; otherwise simulate.py falls back to a constraint-valid allocation (see
# allocate_thirds). Schema: {"ABCDEFGH": {"T74":"A","T77":"C", ...}, ...} where
# the key is the sorted 8-letter combination and the value maps each third-place
# slot to the source group. Regenerate/verify against the FIFA regulations PDF:
#   https://publications.fifa.com/  (FIFA World Cup 2026 Regulations, Annex C)
ANNEXC_FILE = Path(__file__).resolve().parents[2] / "data" / "annexc_thirds.json"

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Norway", "Senegal", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
TEAM_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}

# Round of 32: (match, slot1, slot2). "1A"=winner A, "2A"=runner-up A,
# "T74"=third-placed team assigned to match 74.
R32 = [
    ("M73", "2A", "2B"), ("M74", "1E", "T74"), ("M75", "1F", "2C"),
    ("M76", "1C", "2F"), ("M77", "1I", "T77"), ("M78", "2E", "2I"),
    ("M79", "1A", "T79"), ("M80", "1L", "T80"), ("M81", "1D", "T81"),
    ("M82", "1G", "T82"), ("M83", "2K", "2L"), ("M84", "1H", "2J"),
    ("M85", "1B", "T85"), ("M86", "1J", "2H"), ("M87", "1K", "T87"),
    ("M88", "2D", "2G"),
]
# Allowed source groups for each third-place slot (FIFA regulations).
THIRD_SLOTS = {"T74": "ABCDF", "T77": "CDFGH", "T79": "CEFHI", "T80": "EHIJK",
               "T81": "BEFIJ", "T82": "AEHIJ", "T85": "EFGIJ", "T87": "DEIJL"}

R16 = [("M89", "M74", "M77"), ("M90", "M73", "M75"), ("M91", "M76", "M78"),
       ("M92", "M79", "M80"), ("M93", "M83", "M84"), ("M94", "M81", "M82"),
       ("M95", "M86", "M88"), ("M96", "M85", "M87")]
QF = [("M97", "M89", "M90"), ("M98", "M93", "M94"),
      ("M99", "M91", "M92"), ("M100", "M95", "M96")]
SF = [("M101", "M97", "M98"), ("M102", "M99", "M100")]


class MatchModel:
    """Caches score distributions per (team1, team2, home flags).

    Averages the scoreline matrices of one or more lambda sources
    (Elo+Poisson, Dixon-Coles, or the blend of both)."""

    def __init__(self, sources):
        self.sources = sources
        self.cache = {}

    def _dist(self, t1, t2, h1, h2=0.0):
        key = (t1, t2, h1, h2)
        if key not in self.cache:
            Ms, l1s, l2s = [], [], []
            for fn, rho in self.sources:
                lam1, lam2 = fn(t1, t2, h1, h2)
                Ms.append(score_matrix(lam1, lam2, rho))
                l1s.append(lam1); l2s.append(lam2)
            M = np.mean(Ms, axis=0)
            self.cache[key] = (np.cumsum(M.ravel()),
                               float(np.mean(l1s)), float(np.mean(l2s)))
        return self.cache[key]

    def sample(self, t1, t2, h1, h2, rng):
        cum, _, _ = self._dist(t1, t2, h1, h2)
        idx = int(np.searchsorted(cum, rng.random()))
        return divmod(idx, MAX_GOALS + 1)

    def knockout_winner(self, t1, t2, rng):
        """90 min -> extra time (1/3 intensity) -> penalties (50/50)."""
        h1 = 1.0 if (t1 in HOSTS and t2 not in HOSTS) else 0.0
        h2 = 1.0 if (t2 in HOSTS and t1 not in HOSTS) else 0.0
        g1, g2 = self.sample(t1, t2, h1, h2, rng)
        if g1 != g2:
            return t1 if g1 > g2 else t2
        _, lam1, lam2 = self._dist(t1, t2, h1, h2)
        e1, e2 = rng.poisson(lam1 / 3), rng.poisson(lam2 / 3)
        if e1 != e2:
            return t1 if e1 > e2 else t2
        return t1 if rng.random() < 0.5 else t2


def load_group_matches():
    """All 72 WC 2026 group matches; played ones carry fixed scores."""
    played, upcoming = load_matches()
    wc_p = played[(played["tournament"] == "FIFA World Cup") &
                  (played["date"] >= "2026-06-01")]
    wc_u = upcoming[upcoming["tournament"] == "FIFA World Cup"]
    matches = []
    for df, fixed in ((wc_p, True), (wc_u, False)):
        for r in df.itertuples(index=False):
            g1, g2 = TEAM_GROUP.get(r.home_team), TEAM_GROUP.get(r.away_team)
            if g1 is None or g2 is None or g1 != g2:
                continue  # knockout match or unknown team
            matches.append({
                "group": g1, "home": r.home_team, "away": r.away_team,
                "neutral": bool(r.neutral), "fixed": fixed,
                "hs": int(r.home_score) if fixed else None,
                "as": int(r.away_score) if fixed else None})
    assert len(matches) == 72, f"expected 72 group matches, got {len(matches)}"
    return matches


def rank_group(teams, results, rng):
    """results: list of (home, away, hs, as). Returns teams ranked 1-4.
    Points > GD > GF > head-to-head > random (proxy for drawing of lots)."""
    pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
    h2h = {}
    for h, a, hs, as_ in results:
        gd[h] += hs - as_; gd[a] += as_ - hs; gf[h] += hs; gf[a] += as_
        if hs > as_: pts[h] += 3; h2h[(h, a)] = h
        elif hs < as_: pts[a] += 3; h2h[(h, a)] = a
        else: pts[h] += 1; pts[a] += 1; h2h[(h, a)] = None

    def h2h_bonus(t, tied):
        return sum(1 for (x, y), w in h2h.items()
                   if w == t and x in tied and y in tied)

    order = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], rng.random()),
                   reverse=True)
    # refine exact ties with head-to-head among the tied set
    i = 0
    while i < 3:
        tied = [t for t in order if (pts[t], gd[t], gf[t]) ==
                (pts[order[i]], gd[order[i]], gf[order[i]])]
        if len(tied) > 1:
            tied_sorted = sorted(tied, key=lambda t: (h2h_bonus(t, set(tied)),
                                                      rng.random()), reverse=True)
            j = order.index(tied[0])
            order[j:j + len(tied)] = tied_sorted
            i = j + len(tied)
        else:
            i += 1
    return order, pts, gd, gf


def _load_annexc():
    """Load the official Annex C table if committed, else None. Validates each
    entry is a genuine perfect matching against THIRD_SLOTS so a malformed table
    fails loudly rather than silently mis-slotting teams."""
    if not ANNEXC_FILE.exists():
        return None
    table = json.loads(ANNEXC_FILE.read_text())
    for combo, asg in table.items():
        groups = set(combo)
        if len(combo) != 8 or len(groups) != 8:
            raise ValueError(f"Annex C combo {combo!r} is not 8 distinct groups")
        if set(asg) != set(THIRD_SLOTS):
            raise ValueError(f"Annex C combo {combo!r} does not fill all 8 slots")
        if set(asg.values()) != groups:
            raise ValueError(f"Annex C combo {combo!r} slot groups != combo groups")
        for slot, g in asg.items():
            if g not in THIRD_SLOTS[slot]:
                raise ValueError(f"Annex C {combo!r}: group {g} illegal for {slot}")
    return table


_ANNEXC = _load_annexc()


def allocate_thirds(thirds_by_group, rng):
    """Assign the 8 qualified third-placed teams to Round-of-32 slots.

    Uses the official FIFA Annex C table when data/annexc_thirds.json is present
    (deterministic, matches the real bracket); otherwise falls back to a
    constraint-valid allocation via backtracking on FIFA's allowed-group rules
    (the table only pins down which of the several valid matchings FIFA uses)."""
    qualified = set(thirds_by_group)
    if _ANNEXC is not None:
        combo = "".join(sorted(qualified))
        asg = _ANNEXC.get(combo)
        if asg is not None:
            return {slot: thirds_by_group[g] for slot, g in asg.items()}
    slots = sorted(THIRD_SLOTS, key=lambda s: len(set(THIRD_SLOTS[s]) & qualified))
    assign = {}

    def bt(i, used):
        if i == len(slots):
            return True
        s = slots[i]
        opts = [g for g in THIRD_SLOTS[s] if g in qualified and g not in used]
        rng.shuffle(opts)
        for g in opts:
            assign[s] = g
            if bt(i + 1, used | {g}):
                return True
            del assign[s]
        return False

    if not bt(0, set()):
        raise RuntimeError(f"no valid third-place allocation for {qualified}")
    return {s: thirds_by_group[g] for s, g in assign.items()}


def simulate_once(model, group_matches, rng):
    # --- group stage ---
    by_group = defaultdict(list)
    for m in group_matches:
        if m["fixed"]:
            hs, as_ = m["hs"], m["as"]
        else:
            h1 = 0.0 if m["neutral"] else 1.0
            hs, as_ = model.sample(m["home"], m["away"], h1, 0.0, rng)
        by_group[m["group"]].append((m["home"], m["away"], hs, as_))

    slot_team = {}
    third_stats = []
    for g, teams in GROUPS.items():
        order, pts, gd, gf = rank_group(teams, by_group[g], rng)
        slot_team["1" + g] = order[0]
        slot_team["2" + g] = order[1]
        t3 = order[2]
        third_stats.append((pts[t3], gd[t3], gf[t3], rng.random(), g, t3))

    third_stats.sort(reverse=True)
    qualified_thirds = {g: t for _, _, _, _, g, t in third_stats[:8]}
    slot_team.update(allocate_thirds(qualified_thirds, rng))

    advanced = set(slot_team.values())

    # --- knockout ---
    winners = {}
    for rnd in (R32, R16, QF, SF):
        for match, s1, s2 in rnd:
            t1 = slot_team.get(s1) or winners[s1]
            t2 = slot_team.get(s2) or winners[s2]
            winners[match] = model.knockout_winner(t1, t2, rng)
    finalists = (winners["M101"], winners["M102"])
    champion = model.knockout_winner(*finalists, rng)

    r16 = {winners[m] for m, _, _ in R32}
    qf = {winners[m] for m, _, _ in R16}
    sf = {winners[m] for m, _, _ in QF}
    group_winners = {slot_team["1" + g] for g in GROUPS}
    return group_winners, advanced, r16, qf, sf, set(finalists), champion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["elo", "dc", "blend"], default="blend",
                    help="lambda source (default: blend, best in backtest)")
    args = ap.parse_args()

    sources, ratings = build_sources(args.model)
    model = MatchModel(sources)
    group_matches = load_group_matches()
    rng = np.random.default_rng(args.seed)

    stages = ["win_group", "reach_R32", "reach_R16", "reach_QF",
              "reach_SF", "reach_final", "champion"]
    counts = {t: dict.fromkeys(stages, 0) for ts in GROUPS.values() for t in ts}

    for _ in range(args.sims):
        gw, adv, r16, qf, sf, fin, champ = simulate_once(model, group_matches, rng)
        for t in gw: counts[t]["win_group"] += 1
        for t in adv: counts[t]["reach_R32"] += 1
        for t in r16: counts[t]["reach_R16"] += 1
        for t in qf: counts[t]["reach_QF"] += 1
        for t in sf: counts[t]["reach_SF"] += 1
        for t in fin: counts[t]["reach_final"] += 1
        counts[champ]["champion"] += 1

    rows = [{"team": t, "group": TEAM_GROUP[t], "elo": round(ratings[t]),
             **{s: round(c[s] / args.sims, 4) for s, c in
                ((s, counts[t]) for s in stages)}}
            for t in counts]
    df = pd.DataFrame(rows).sort_values("champion", ascending=False)
    dest = "tournament_odds.csv"
    df.to_csv(dest, index=False)

    pd.set_option("display.width", 140)
    print(f"\n2026 World Cup — {args.sims:,} simulations\n")
    show = df.head(20).copy()
    for s in stages:
        show[s] = (show[s] * 100).map("{:.1f}%".format)
    print(show.to_string(index=False))
    print(f"\nFull table (48 teams) saved -> {dest}")


if __name__ == "__main__":
    main()
