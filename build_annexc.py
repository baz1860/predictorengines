#!/usr/bin/env python3
"""Build data/annexc_thirds.json from the FIFA Annex C third-place table.

Source: FIFA World Cup 2026 Regulations, Annex C (the 495 round-of-32 third-place
combination scenarios). Transcribed into data/annexc_raw.txt from Wikipedia's
"2026 FIFA World Cup knockout stage" rendering of that table:
  https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
  (Combinations of matches in the round of 32)

Raw row format (one scenario per line):
  <No.> <8 group letters whose 3rd-placed team advanced> <8 tokens "3X">
The 8 result columns correspond, in order, to the group WINNERS that play a
third-placed team. From the R32 schedule those winners and their bracket slots are:
  column order : 1A   1B   1D   1E   1G   1I   1K   1L
  slot (Txx)   : T79  T85  T81  T74  T82  T77  T87  T80
(e.g. Match 79 = Winner Group A vs 3rd-place, so the 1A column is slot T79.)

Output: data/annexc_thirds.json = {"<sorted 8-letter combo>": {"T74":"A", ...}}.
Every row is validated as a genuine perfect matching against simulate.THIRD_SLOTS;
all 495 combinations must be present and distinct, or the build aborts.
"""
import json
from itertools import combinations
from pathlib import Path

from engines.worldcup.simulate import THIRD_SLOTS

HERE = Path(__file__).parent
RAW = HERE / "data" / "annexc_raw.txt"
OUT = HERE / "data" / "annexc_thirds.json"

# result columns (group winners), in the order they appear on each raw row
WINNER_COLS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]
SLOT_FOR_WINNER = {"1A": "T79", "1B": "T85", "1D": "T81", "1E": "T74",
                   "1G": "T82", "1I": "T77", "1K": "T87", "1L": "T80"}


def build():
    table = {}
    for ln, line in enumerate(RAW.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        tok = line.split()
        # tok[0] = index; tok[1:9] = 8 group letters; tok[9:17] = 8 "3X" results
        if len(tok) != 17:
            raise ValueError(f"line {ln}: expected 17 tokens, got {len(tok)}: {line!r}")
        combo_groups = tok[1:9]
        results = tok[9:17]
        if any(len(g) != 1 or not g.isalpha() for g in combo_groups):
            raise ValueError(f"line {ln}: bad group letters {combo_groups}")
        asg = {}
        for col, res in zip(WINNER_COLS, results):
            if not (res.startswith("3") and len(res) == 2):
                raise ValueError(f"line {ln}: bad result token {res!r}")
            asg[SLOT_FOR_WINNER[col]] = res[1]
        combo = "".join(sorted(combo_groups))

        # validate this scenario is a genuine perfect matching
        if len(set(combo_groups)) != 8:
            raise ValueError(f"line {ln}: combo {combo} not 8 distinct groups")
        if set(asg) != set(THIRD_SLOTS):
            raise ValueError(f"line {ln}: {combo} does not fill all 8 slots")
        if set(asg.values()) != set(combo_groups):
            raise ValueError(f"line {ln}: {combo} assigned groups != combo groups")
        for slot, g in asg.items():
            if g not in THIRD_SLOTS[slot]:
                raise ValueError(f"line {ln}: {combo}: group {g} illegal for {slot} "
                                 f"(allowed {THIRD_SLOTS[slot]})")
        if combo in table:
            raise ValueError(f"line {ln}: duplicate combo {combo}")
        table[combo] = asg

    # completeness: exactly the C(12,8)=495 combinations, all present
    all_combos = {"".join(c) for c in combinations("ABCDEFGHIJKL", 8)}
    missing = all_combos - set(table)
    extra = set(table) - all_combos
    if missing or extra:
        raise ValueError(f"coverage error: {len(missing)} missing, {len(extra)} extra")
    assert len(table) == 495, len(table)

    OUT.write_text(json.dumps(table, sort_keys=True, separators=(",", ":")))
    print(f"Wrote {len(table)} validated scenarios -> {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    build()
