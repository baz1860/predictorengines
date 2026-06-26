"""tennis/simulate.py — Markov-chain match math + tournament Monte Carlo.

Two layers:

  1. Match level (exact, no simulation). From point-on-serve probabilities the
     game → set → match win probability is computed by a recursive Markov chain,
     including the deuce limit and the 7-point tiebreak. Sub-markets (set
     handicap, first-set winner, total games) read off the same chain.

  2. Tournament level (Monte Carlo). Given an ordered bracket of first-round
     pairings, draw each match via the fitted model and advance winners to get
     per-player win / final / SF / QF reach probabilities for outright markets.

The model's match-winner probability comes from the Bradley-Terry fit
(`model.predict_match`); `point_edge_for_target` inverts the Markov chain so the
set/game sub-markets are *consistent* with that headline probability.
"""
from __future__ import annotations

import math
from functools import lru_cache

# ATP/WTA average point-win-on-serve baseline; sub-markets are derived as a
# symmetric edge around this so they stay consistent with the BT match prob.
BASE_SERVE = 0.64


# ─────────────────────────────────────────────
# Game / tiebreak / set / match win probabilities
# ─────────────────────────────────────────────

def prob_win_game(p: float) -> float:
    """P(server wins a standard ad game) given point-win prob `p` on serve."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    q = 1.0 - p
    pre_deuce = p ** 4 * (1 + 4 * q + 10 * q * q)
    deuce = 20 * p ** 3 * q ** 3 * (p * p / (p * p + q * q))
    return pre_deuce + deuce


def prob_win_tiebreak(ps_a: float, ps_b: float, target: int = 7) -> float:
    """P(A wins a first-to-`target` (win-by-2) tiebreak), A serving point 1.

    Serve order is the standard 1, 2-2, 2-2, … alternation. `ps_a`/`ps_b` are
    each player's point-win prob *on their own serve*.
    """
    pa_serve = min(max(ps_a, 1e-6), 1 - 1e-6)
    pb_serve = min(max(ps_b, 1e-6), 1 - 1e-6)

    def server_is_a(points_played: int) -> bool:
        # A serves point 0; thereafter serves swap every 2 points.
        return ((points_played + 1) // 2) % 2 == 0

    # From (target−1, target−1) it is win-by-2 in a serve-alternating sequence.
    # Resolve that tail with the standard deuce formula on A's average per-point
    # win rate (across own and opponent serve) — a tiny approximation confined
    # to the rare tiebreak-deuce tail, and it keeps the recursion finite.
    pbar = 0.5 * (pa_serve + (1.0 - pb_serve))
    deuce_a = pbar * pbar / (pbar * pbar + (1.0 - pbar) ** 2)

    @lru_cache(maxsize=None)
    def rec(a: int, b: int) -> float:
        if a == target - 1 and b == target - 1:
            return deuce_a
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        a_serving = server_is_a(a + b)
        p_a_point = pa_serve if a_serving else (1.0 - pb_serve)
        return p_a_point * rec(a + 1, b) + (1.0 - p_a_point) * rec(a, b + 1)

    val = rec(0, 0)
    rec.cache_clear()
    return val


def _set_dp(ps_a: float, ps_b: float, a_serves_first: bool):
    """(P(A wins set), E[games in set]) via a game-level Markov chain with serve
    alternation and a 7-point tiebreak at 6-6."""
    g_a_on_a = prob_win_game(ps_a)            # A wins a game on A's serve
    g_a_on_b = 1.0 - prob_win_game(ps_b)      # A wins a game on B's serve
    tb = prob_win_tiebreak(ps_a, ps_b)

    @lru_cache(maxsize=None)
    def rec(a: int, b: int):
        # returns (P(A wins set from here), E[remaining games])
        if a == 6 and b <= 4:
            return (1.0, 0.0)
        if b == 6 and a <= 4:
            return (0.0, 0.0)
        if a == 7:        # 7-5
            return (1.0, 0.0)
        if b == 7:
            return (0.0, 0.0)
        if a == 6 and b == 6:
            return (tb, 1.0)   # tiebreak counts as one (13th) game
        games_played = a + b
        a_serving = (games_played % 2 == 0) == a_serves_first
        p_a_game = g_a_on_a if a_serving else g_a_on_b
        win_p, win_g = rec(a + 1, b)
        lose_p, lose_g = rec(a, b + 1)
        prob = p_a_game * win_p + (1.0 - p_a_game) * lose_p
        egames = 1.0 + p_a_game * win_g + (1.0 - p_a_game) * lose_g
        return (prob, egames)

    out = rec(0, 0)
    rec.cache_clear()
    return out


def set_win_prob(ps_a: float, ps_b: float) -> float:
    """P(A wins a set), averaged over which player serves first."""
    p1, _ = _set_dp(ps_a, ps_b, True)
    p2, _ = _set_dp(ps_a, ps_b, False)
    return 0.5 * (p1 + p2)


def expected_games_per_set(ps_a: float, ps_b: float) -> float:
    _, g1 = _set_dp(ps_a, ps_b, True)
    _, g2 = _set_dp(ps_a, ps_b, False)
    return 0.5 * (g1 + g2)


def _first_to(n: int, s: float) -> float:
    """P(A wins a first-to-`n`-sets match) from iid per-set prob `s`."""
    total = 0.0
    for k in range(n):                      # k = sets the loser takes
        total += math.comb(n - 1 + k, k) * s ** n * (1 - s) ** k
    return total


def match_win_prob(ps_a: float, ps_b: float, best_of: int = 3) -> float:
    s = set_win_prob(ps_a, ps_b)
    return _first_to((best_of + 1) // 2, s)


# ─────────────────────────────────────────────
# Inversion: point edge that reproduces a target match probability
# ─────────────────────────────────────────────

def point_edge_for_target(target_p: float, best_of: int = 3,
                          base: float = BASE_SERVE) -> tuple[float, float]:
    """Find symmetric serve point probs (ps_a, ps_b) = (base+δ, base−δ) whose
    Markov match prob equals `target_p`. Monotonic in δ → bisection."""
    target_p = min(max(target_p, 1e-4), 1 - 1e-4)
    lo, hi = 0.0, min(base, 1 - base) - 1e-3
    # δ=0 → 0.5; increasing δ raises A's match prob monotonically.
    if target_p <= 0.5:
        lo, hi = -(min(base, 1 - base) - 1e-3), 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = match_win_prob(base + mid, base - mid, best_of)
        if p < target_p:
            lo = mid
        else:
            hi = mid
    delta = 0.5 * (lo + hi)
    return (base + delta, base - delta)


def match_markets(target_p: float, best_of: int = 3,
                  base: float = BASE_SERVE, games_cal: float = 1.0) -> dict:
    """Set/game sub-markets consistent with a headline match prob `target_p`.

    `base` is the matchup serve level (see model.serve_base); `games_cal` is the
    fitted multiplicative correction that de-biases the expected total games
    (params["games_cal"], default 1.0). Returns p_match, set win prob, first-set
    winner, set-handicap (−1.5/+1.5), and the (calibrated) expected total games.
    """
    ps_a, ps_b = point_edge_for_target(target_p, best_of, base)
    s = set_win_prob(ps_a, ps_b)
    sets_to_win = (best_of + 1) // 2
    p_match = _first_to(sets_to_win, s)
    # A wins -1.5 sets ⇒ A wins in straight sets (loser takes 0).
    p_a_straight = s ** sets_to_win
    p_b_straight = (1 - s) ** sets_to_win
    egames = expected_games_per_set(ps_a, ps_b) * _expected_sets(s, best_of) * float(games_cal)
    return {
        "p_match": p_match,
        "p_set": s,
        "p_first_set": s,
        "p_a_minus_1_5_sets": p_a_straight,        # A covers −1.5 sets
        "p_a_plus_1_5_sets": 1.0 - p_b_straight,   # A covers +1.5 sets
        "exp_total_games": egames,
        "ps_a": ps_a,
        "ps_b": ps_b,
    }


def _expected_sets(s: float, best_of: int) -> float:
    """Expected number of sets played given per-set prob `s`."""
    n = (best_of + 1) // 2
    # P(match lasts exactly t sets) for t = n .. best_of, summed t·P.
    exp = 0.0
    for t in range(n, best_of + 1):
        k = t - n  # sets the eventual loser won
        # last set won by the winner; loser won k of the first t−1.
        p_a = math.comb(t - 1, k) * s ** n * (1 - s) ** k
        p_b = math.comb(t - 1, k) * (1 - s) ** n * s ** k
        exp += t * (p_a + p_b)
    return exp


# ─────────────────────────────────────────────
# Tournament Monte Carlo
# ─────────────────────────────────────────────

def simulate_draw(pairings: list[tuple[str, str]], params, surface: str,
                  best_of: int = 3, n_sims: int = 50000, rng=None,
                  h2h_fn=None) -> dict:
    """Simulate a single-elimination bracket.

    `pairings` is the ordered list of first-round matches (each consecutive pair
    feeds one second-round slot). Returns per-player reach frequencies:
    {player: {"win", "final", "sf", "qf"}}.
    """
    import numpy as np
    from . import model as M

    rng = rng or np.random.default_rng(0)
    if not pairings:
        return {}
    n_first = len(pairings)
    if n_first & (n_first - 1) != 0:
        raise ValueError(f"bracket needs a power-of-two number of first-round "
                         f"matches, got {n_first}")
    rounds = n_first.bit_length()           # matches → rounds after them
    players = [p for pair in pairings for p in pair]

    # Pre-compute base P(A beats B) per ordered pair we may encounter. Cheap to
    # just memoise on demand inside the loop.
    cache: dict[tuple[str, str], float] = {}

    def p_beats(a: str, b: str) -> float:
        key = (a, b)
        if key not in cache:
            h2h = h2h_fn(a, b, surface) if h2h_fn else 0.0
            cache[key] = M.predict_match(a, b, surface, params, h2h_log_odds=h2h)["p_a"]
            cache[(b, a)] = 1.0 - cache[key]
        return cache[key]

    counts = {p: {"win": 0, "final": 0, "sf": 0, "qf": 0} for p in players}
    # A player reaching a stage of field-size 2^k must have won (rounds−k)
    # matches: last-8 (QF) ⇒ won ≥ rounds−3, last-4 ⇒ ≥ rounds−2, etc. Negative
    # thresholds (draws already ≤ that stage) credit everyone, which is correct.
    qf_t, sf_t, final_t = rounds - 3, rounds - 2, rounds - 1

    for _ in range(n_sims):
        alive = list(players)
        wins = {p: 0 for p in players}
        for _r in range(rounds):
            winners = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                w = a if rng.random() < p_beats(a, b) else b
                wins[w] += 1
                winners.append(w)
            alive = winners
        for p, n in wins.items():
            if n >= qf_t:
                counts[p]["qf"] += 1
            if n >= sf_t:
                counts[p]["sf"] += 1
            if n >= final_t:
                counts[p]["final"] += 1
        counts[alive[0]]["win"] += 1

    return {p: {k: v / n_sims for k, v in d.items()} for p, d in counts.items()}


# Round ranks: shallower (closer to the title) = smaller. Used by the bracket
# simulator to credit how far each entrant reached.
ROUND_RANK = {"F": 0, "SF": 1, "QF": 2, "R16": 3, "R32": 4, "R64": 5,
              "R128": 6, "R256": 7}

ROUND_ORDER = ["Q1", "Q2", "QF-Q", "R256", "R128", "R64", "R32", "R16",
               "R1", "R2", "R3", "R4", "QF", "SF", "F"]
FIELD_ROUND = {2: "F", 4: "SF", 8: "QF", 16: "R16", 32: "R32",
               64: "R64", 128: "R128", 256: "R256"}


def _is_tbd(name: str) -> bool:
    return not str(name or "").strip() or str(name).strip().upper() in {"TBD", "BYE"}


def _round_sort_key(round_name: str, first_idx: int) -> tuple[int, int]:
    try:
        return (ROUND_ORDER.index(str(round_name or "")), first_idx)
    except ValueError:
        return (len(ROUND_ORDER), first_idx)


def _round_rank(round_name: str, field_size: int) -> int:
    """Reach-rank for a round. Generic feed labels such as R1/R2 are inferred
    from the number of players in that round when possible."""
    r = str(round_name or "")
    if r in {"Q1", "Q2", "QF-Q"}:
        return 99
    if r in ROUND_RANK:
        return ROUND_RANK[r]
    inferred = FIELD_ROUND.get(field_size)
    return ROUND_RANK.get(inferred, 99)


def _credit_reach(counts: dict, best_rank: dict[str, int], player: str, rank: int) -> None:
    if _is_tbd(player):
        return
    counts.setdefault(player, {"win": 0, "final": 0, "sf": 0, "qf": 0})
    if rank < best_rank.get(player, 99):
        best_rank[player] = rank


def simulate_draw_rows(draw_rows: list[dict], params, surface: str,
                       best_of: int = 3, n_sims: int = 50000, rng=None,
                       h2h_fn=None) -> dict:
    """Simulate a tournament from draw rows that may include known results.

    Rows are expected to carry at least ``round/player_a/player_b`` and may also
    include ``state`` and ``winner``. Completed rows lock their winner; unresolved
    rows are simulated. Later rows with ``TBD`` slots are filled from the previous
    round's winners in row order. If the feed stops before listing the final, the
    remaining rounds are synthesized from the surviving players.
    """
    import numpy as np
    from . import model as M

    rng = rng or np.random.default_rng(0)
    rows = []
    for i, raw in enumerate(draw_rows):
        a = str(raw.get("player_a") or "").strip()
        b = str(raw.get("player_b") or "").strip()
        if _is_tbd(a) and _is_tbd(b):
            continue
        rows.append({
            "idx": i,
            "round": str(raw.get("round") or ""),
            "player_a": a,
            "player_b": b,
            "state": str(raw.get("state") or "").lower(),
            "winner": str(raw.get("winner") or "").strip(),
        })
    if not rows:
        return {}

    players = {
        n for r in rows for n in (r["player_a"], r["player_b"], r["winner"])
        if not _is_tbd(n)
    }
    counts = {p: {"win": 0, "final": 0, "sf": 0, "qf": 0} for p in players}
    cache: dict[tuple[str, str], float] = {}

    def p_beats(a: str, b: str) -> float:
        key = (a, b)
        if key not in cache:
            h2h = h2h_fn(a, b, surface) if h2h_fn else 0.0
            cache[key] = M.predict_match(a, b, surface, params, h2h_log_odds=h2h)["p_a"]
            cache[(b, a)] = 1.0 - cache[key]
        return cache[key]

    grouped: dict[str, list[dict]] = {}
    first_idx: dict[str, int] = {}
    for r in rows:
        grouped.setdefault(r["round"], []).append(r)
        first_idx.setdefault(r["round"], r["idx"])
    round_keys = sorted(grouped, key=lambda r: _round_sort_key(r, first_idx[r]))

    qf_r, sf_r, final_r = ROUND_RANK["QF"], ROUND_RANK["SF"], ROUND_RANK["F"]

    def resolve_slot(name: str, prev: list[str], cursor: list[int]) -> str:
        if not _is_tbd(name):
            return name
        if cursor[0] < len(prev):
            out = prev[cursor[0]]
            cursor[0] += 1
            return out
        return ""

    def play_match(a: str, b: str, fixed_winner: str = "") -> str:
        if fixed_winner and not _is_tbd(fixed_winner):
            return fixed_winner
        return a if rng.random() < p_beats(a, b) else b

    for _ in range(n_sims):
        prev_winners: list[str] = []
        best_rank: dict[str, int] = {}

        for rnd in round_keys:
            winners: list[str] = []
            cursor = [0]
            resolved_pairs: list[tuple[str, str]] = []

            for row in grouped[rnd]:
                a = resolve_slot(row["player_a"], prev_winners, cursor)
                b = resolve_slot(row["player_b"], prev_winners, cursor)
                if _is_tbd(a) or _is_tbd(b):
                    continue
                rank = _round_rank(rnd, 2 * len(grouped[rnd]))
                _credit_reach(counts, best_rank, a, rank)
                _credit_reach(counts, best_rank, b, rank)
                fixed = row["winner"] if row["state"] == "post" else ""
                winners.append(play_match(a, b, fixed))
                resolved_pairs.append((a, b))

            if resolved_pairs:
                prev_winners = winners

        alive = list(prev_winners)
        while len(alive) > 1:
            if len(alive) % 2:
                raise ValueError("cannot synthesize remaining draw rounds from "
                                 f"{len(alive)} surviving players")
            rank = _round_rank(FIELD_ROUND.get(len(alive), ""), len(alive))
            winners = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                _credit_reach(counts, best_rank, a, rank)
                _credit_reach(counts, best_rank, b, rank)
                winners.append(play_match(a, b))
            alive = winners

        for p, rk in best_rank.items():
            if rk <= qf_r:
                counts[p]["qf"] += 1
            if rk <= sf_r:
                counts[p]["sf"] += 1
            if rk <= final_r:
                counts[p]["final"] += 1
        if alive:
            counts.setdefault(alive[0], {"win": 0, "final": 0, "sf": 0, "qf": 0})
            counts[alive[0]]["win"] += 1

    return {p: {k: v / n_sims for k, v in d.items()} for p, d in counts.items()}


def _bracket_leaves(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    return _bracket_leaves(node["a"]) + _bracket_leaves(node["b"])


def simulate_bracket(root, params, surface: str, best_of: int = 3,
                     n_sims: int = 20000, rng=None, h2h_fn=None) -> dict:
    """Monte-Carlo a reconstructed tournament tree.

    `root` is a nested node dict ``{"round": label, "a": child, "b": child}``
    where each child is another node or a leaf player name (a first-round entry
    or a bye). Returns per-player reach frequencies
    {player: {"win", "final", "sf", "qf"}} — an entrant of a round-R node has
    "reached" round R, so reach-QF means appearing in a QF (or shallower) node.
    """
    import numpy as np
    from . import model as M

    rng = rng or np.random.default_rng(0)
    players = _bracket_leaves(root)
    counts = {p: {"win": 0, "final": 0, "sf": 0, "qf": 0} for p in players}

    cache: dict[tuple[str, str], float] = {}

    def p_beats(a: str, b: str) -> float:
        key = (a, b)
        if key not in cache:
            h2h = h2h_fn(a, b, surface) if h2h_fn else 0.0
            cache[key] = M.predict_match(a, b, surface, params, h2h_log_odds=h2h)["p_a"]
            cache[(b, a)] = 1.0 - cache[key]
        return cache[key]

    qf_r, sf_r, final_r = ROUND_RANK["QF"], ROUND_RANK["SF"], ROUND_RANK["F"]

    for _ in range(n_sims):
        best_rank: dict[str, int] = {}

        def sim(node):
            if isinstance(node, str):
                return node
            a = sim(node["a"])
            b = sim(node["b"])
            rank = ROUND_RANK.get(node["round"], 99)
            for pl in (a, b):
                if rank < best_rank.get(pl, 99):
                    best_rank[pl] = rank
            return a if rng.random() < p_beats(a, b) else b

        champ = sim(root)
        for pl, rk in best_rank.items():
            if rk <= qf_r:
                counts[pl]["qf"] += 1
            if rk <= sf_r:
                counts[pl]["sf"] += 1
            if rk <= final_r:
                counts[pl]["final"] += 1
        counts[champ]["win"] += 1

    return {p: {k: v / n_sims for k, v in d.items()} for p, d in counts.items()}
