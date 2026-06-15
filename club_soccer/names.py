#!/usr/bin/env python3
"""Shared team-name reconciliation for the Club Soccer seeders.

The league base (seed_footballdata.py) uses football-data.co.uk's canonical
names. UEFA data (openfootball) and cup/UEFA data (API-Football) spell clubs
differently, so both must be mapped back onto that canon or they create phantom
duplicate teams. `make_canon(league_teams)` returns a mapper: OVERRIDES first,
then exact match, then accent/suffix-stripped fuzzy match (difflib, cutoff 0.90).
"""
from __future__ import annotations

import difflib
import re
import unicodedata

# Source spelling (country code already stripped) -> football-data canon name.
# Covers both openfootball and API-Football variants of the same clubs.
OVERRIDES = {
    # openfootball spellings
    "FC Bayern München": "Bayern Munich", "FC Internazionale Milano": "Inter",
    "Bayer 04 Leverkusen": "Bayer Leverkusen", "Club Atlético de Madrid": "Atletico Madrid",
    "Atlético de Madrid": "Atletico Madrid", "Real Madrid CF": "Real Madrid",
    "FC Barcelona": "Barcelona", "Paris Saint-Germain FC": "Paris Saint-Germain",
    "Aston Villa FC": "Aston Villa", "Atalanta BC": "Atalanta", "Bologna FC 1909": "Bologna",
    "Juventus FC": "Juventus", "SSC Napoli": "Napoli", "AS Roma": "Roma", "SS Lazio": "Lazio",
    "Lazio Roma": "Lazio", "VfB Stuttgart": "Stuttgart", "Girona FC": "Girona",
    "Villarreal CF": "Villarreal", "AS Monaco FC": "Monaco", "Lille OSC": "Lille",
    "Olympique de Marseille": "Marseille", "Stade Brestois 29": "Brest",
    "Olympique Lyonnais": "Lyon", "Newcastle United FC": "Newcastle United",
    "Celtic FC": "Celtic", "Rangers FC": "Rangers", "Real Betis Balompié": "Betis",
    "Real Betis": "Betis", "Tottenham Hotspur": "Tottenham", "Tottenham Hotspur FC": "Tottenham",
    "Real Sociedad de Fútbol": "Real Sociedad", "Athletic Club": "Athletic Bilbao",
    # API-Football spellings that differ from the above / from canon
    "Bayern München": "Bayern Munich", "Paris Saint Germain": "Paris Saint-Germain",
    "Newcastle": "Newcastle United", "Wolves": "Wolverhampton", "Nottingham Forest": "Nottingham Forest",
    "Manchester United": "Manchester United", "Manchester City": "Manchester City",
    "Sheffield Utd": "Sheffield United", "Atletico Madrid": "Atletico Madrid",
    "Inter": "Inter", "Milan": "AC Milan", "Borussia Monchengladbach": "Borussia Monchengladbach",
}

STOP = {"fc", "cf", "afc", "acf", "ac", "bc", "sc", "sk", "fk", "nk", "gnk", "rc", "rb",
        "ss", "ssc", "as", "ogc", "ofc", "cd", "ud", "club", "calcio", "de", "futbol",
        "hotspur", "1846", "1899", "1909", "04", "05", "1907", "kv", "bsc", "vfb",
        "vfl", "tsg", "if"}


def simplify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace("munchen", "munich").replace("internazionale", "inter")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in STOP]
    return " ".join(toks).strip()


def make_canon(league_teams):
    league_teams = set(league_teams)
    simp_map = {simplify(t): t for t in league_teams}
    cache: dict[str, str] = {}

    def canon(raw: str) -> str:
        c = re.sub(r"\s*\([A-Z]{3}\)\s*$", "", str(raw)).strip()
        if c in cache:
            return cache[c]
        if c in OVERRIDES:
            out = OVERRIDES[c]
        elif c in league_teams:
            out = c
        else:
            sc = simplify(c)
            if sc in simp_map:
                out = simp_map[sc]
            else:
                m = difflib.get_close_matches(sc, list(simp_map), n=1, cutoff=0.90)
                out = simp_map[m[0]] if m else c
        cache[c] = out
        return out
    return canon
