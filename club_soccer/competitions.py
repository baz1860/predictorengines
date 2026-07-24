"""Competition registry for the Club Soccer engine."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_COMP_STRENGTH_FILE = Path(__file__).resolve().parent / "data" / "comp_strength.json"
_comp_strength_cache: dict | None = None


@dataclass(frozen=True)
class Competition:
    name: str
    country: str
    kind: str
    tier: int
    api_id: int        # api-football league id (kept for fixtures.csv backwards compat)
    strength: float
    fdorg_code: str    # football-data.org competition code ("PL", "BL1", …); "" = not covered free
    bsd_league: str    # BSD league name substring for filtering; "" = use self.name
    # P2 expansion. fd.co.uk is the primary source for the new leagues: BSD's
    # catalogue turned out to hold only 18 UEFA domestic competitions and does
    # NOT include Austria. See uefa_registry.py for the verified source map.
    fdcouk_code: str = ""      # /mmz4281/{season}/{code}.csv division code
    fdcouk_new: str = ""       # /new/{CODE}.csv country file
    fdcouk_league: str = ""    # League column value inside a /new/ file
    uefa_country: str = ""     # association key into uefa_registry (default: country)
    # Nordic/Irish leagues run March-November, so their season IS the calendar
    # year. The default Aug-May rule splits one real season across two labels
    # at the July boundary, merging two different squads into one "season" —
    # Eliteserien showed 19-20 teams for a 16-team league. That corrupts
    # standings, promoted/relegated detection and per-league-season scoring
    # rates for every affected competition.
    calendar_season: bool = False
    # League geometry — parameterises P4.3 motivation bands. Not legally
    # perfect (e.g. playoff spots folded into a single count); documented
    # approximation per the plan. Zero for cups/Europe (not a table).
    teams_n: int = 0
    releg_spots: int = 0
    promo_spots: int = 0
    euro_spots: int = 0


COMPETITIONS = [
    Competition("Premier League",       "England",  "league", 1,  39,  1.00, "PL",  "",
               teams_n=20, releg_spots=3, promo_spots=0, euro_spots=7),
    Competition("Championship",         "England",  "league", 2,  40,  0.72, "ELC", "Championship",
               teams_n=24, releg_spots=3, promo_spots=3, euro_spots=0),
    Competition("League One",           "England",  "league", 3,  41,  0.50, "EL1", "League One",
               teams_n=24, releg_spots=4, promo_spots=3, euro_spots=0),
    Competition("League Two",           "England",  "league", 4,  42,  0.35, "EL2", "League Two",
               teams_n=24, releg_spots=2, promo_spots=4, euro_spots=0),
    Competition("FA Cup",               "England",  "cup",    0,  45,  0.75, "FAC", "FA Cup"),
    Competition("EFL Cup",              "England",  "cup",    0,  48,  0.70, "",    "EFL Cup"),
    Competition("Scottish Premiership", "Scotland", "league", 1, 179,  0.58, "",    "Scottish Premiership",
               teams_n=12, releg_spots=1, promo_spots=1, euro_spots=4),
    Competition("Scottish Championship","Scotland", "league", 2, 180,  0.38, "",    "Scottish Championship",
               teams_n=10, releg_spots=1, promo_spots=1, euro_spots=0),
    Competition("Scottish League One",  "Scotland", "league", 3, 181,  0.28, "",    "Scottish League One",
               teams_n=10, releg_spots=1, promo_spots=1, euro_spots=0),
    Competition("Scottish League Two",  "Scotland", "league", 4, 182,  0.22, "",    "Scottish League Two",
               teams_n=10, releg_spots=1, promo_spots=1, euro_spots=0),
    Competition("Scottish Cup",         "Scotland", "cup",    0, 183,  0.48, "",    "Scottish Cup"),
    Competition("Scottish League Cup",  "Scotland", "cup",    0, 184,  0.45, "",    "Scottish League Cup"),
    Competition("Bundesliga",           "Germany",  "league", 1,  78,  0.93, "BL1", "Bundesliga",
               teams_n=18, releg_spots=3, promo_spots=0, euro_spots=7),
    Competition("DFB-Pokal",            "Germany",  "cup",    0,  81,  0.72, "DFB", "DFB-Pokal"),
    Competition("Serie A",              "Italy",    "league", 1, 135,  0.91, "SA",  "Serie A",
               teams_n=20, releg_spots=3, promo_spots=0, euro_spots=8),
    Competition("Coppa Italia",         "Italy",    "cup",    0, 137,  0.70, "",    "Coppa Italia"),
    Competition("Ligue 1",              "France",   "league", 1,  61,  0.86, "FL1", "Ligue 1",
               teams_n=18, releg_spots=3, promo_spots=0, euro_spots=7),
    Competition("Coupe de France",      "France",   "cup",    0,  66,  0.64, "",    "Coupe de France"),
    Competition("La Liga",              "Spain",    "league", 1, 140,  0.92, "PD",  "La Liga",
               teams_n=20, releg_spots=3, promo_spots=0, euro_spots=8),
    Competition("Copa del Rey",         "Spain",    "cup",    0, 143,  0.70, "",    "Copa del Rey"),
    Competition("Champions League",     "Europe",   "europe", 0,   2,  1.08, "CL",  "Champions League"),
    Competition("Europa League",        "Europe",   "europe", 0,   3,  0.88, "EL",  "Europa League"),
    Competition("Conference League",    "Europe",   "europe", 0, 848,  0.70, "",    "Conference League"),
    Competition("UEFA Super Cup",       "Europe",   "europe", 0, 531,  1.00, "",    "UEFA Super Cup"),

    # ── P2: UEFA expansion ───────────────────────────────────────────────
    # api_id 9000+ is a synthetic local range: these leagues are not sourced
    # from api-football, but fixtures.csv carries competition_id for every
    # row, so they need stable non-colliding ids.
    #
    # `strength` values come from uefa_registry.strength_prior(), which is
    # anchored on the existing hand-set scale (Premier League 1.00 at
    # coefficient 89.16, Scottish Premiership 0.58 at 27.50) and linear in the
    # UEFA coefficient between them. They are PRIORS, not fitted values — P4
    # replaces them, and comp_strength.json stays gated off meanwhile.
    #
    # teams_n is from live 2025/26 files where observed. releg/promo/euro
    # spots are left at 0 rather than guessed: they only parameterise the
    # P4.3 motivation bands, and 0 already means "not a table" for cups, so
    # the effect is that motivation simply does not apply to these leagues
    # yet. Populating them is a P3 follow-up.
    Competition("Serie B",              "Italy",    "league", 2, 9001, 0.6521, "", "",
                fdcouk_code="I2", teams_n=20, releg_spots=4, promo_spots=3, euro_spots=0),
    Competition("Segunda División",     "Spain",    "league", 2, 9002, 0.6277, "", "Segunda División",
                fdcouk_code="SP2", teams_n=22, releg_spots=4, promo_spots=3, euro_spots=0),
    Competition("2. Bundesliga",        "Germany",  "league", 2, 9003, 0.6166, "", "",
                fdcouk_code="D2", teams_n=18, releg_spots=3, promo_spots=3, euro_spots=0),
    Competition("Ligue 2",              "France",   "league", 2, 9004, 0.5502, "", "",
                fdcouk_code="F2", teams_n=18, releg_spots=3, promo_spots=3, euro_spots=0),
    Competition("National League",      "England",  "league", 5, 9005, 0.26,   "", "",
                fdcouk_code="EC", teams_n=24, releg_spots=4, promo_spots=2, euro_spots=0),
    Competition("Eredivisie",           "Netherlands", "league", 1, 9006, 0.7598, "", "Eredivisie",
                fdcouk_code="N1", teams_n=18, releg_spots=3, promo_spots=0, euro_spots=6),
    Competition("Liga Portugal",        "Portugal", "league", 1, 9007, 0.7225, "", "Liga Portugal Betclic",
                fdcouk_code="P1", teams_n=18, releg_spots=3, promo_spots=0, euro_spots=5),
    Competition("Liga 3",               "Portugal", "league", 3, 9008, 0.3612, "", "Liga 3",
                teams_n=20, releg_spots=4, promo_spots=4, euro_spots=0),
    Competition("Belgian Pro League",   "Belgium",  "league", 1, 9009, 0.6815, "", "Pro League",
                fdcouk_code="B1", teams_n=16, releg_spots=2, promo_spots=0, euro_spots=5),
    Competition("Super Lig",            "Turkey",   "league", 1, 9010, 0.6216, "", "Trendyol Super Lig",
                fdcouk_code="T1", teams_n=18, releg_spots=3, promo_spots=0, euro_spots=5),
    Competition("Austrian Bundesliga",  "Austria",  "league", 1, 9011, 0.5834, "", "",
                fdcouk_new="AUT", fdcouk_league="Bundesliga",
                teams_n=12, releg_spots=1, promo_spots=1, euro_spots=5),
    Competition("Eliteserien",          "Norway",   "league", 1, 9012, 0.5826, "", "Eliteserien",
                fdcouk_new="NOR", fdcouk_league="Eliteserien", calendar_season=True,
                teams_n=16, releg_spots=3, promo_spots=2, euro_spots=5),
    Competition("Greek Super League",   "Greece",   "league", 1, 9013, 0.574,  "", "Stoiximan Super League",
                fdcouk_code="G1", teams_n=14, releg_spots=2, promo_spots=0, euro_spots=5),
    Competition("Swiss Super League",   "Switzerland", "league", 1, 9014, 0.5737, "", "Super League",
                fdcouk_new="SWZ", fdcouk_league="Super League",
                teams_n=12, releg_spots=2, promo_spots=2, euro_spots=4),
    Competition("Danish Superliga",     "Denmark",  "league", 1, 9015, 0.572,  "", "",
                fdcouk_new="DNK", fdcouk_league="Superliga",
                teams_n=12, releg_spots=2, promo_spots=2, euro_spots=4),
    Competition("Ekstraklasa",          "Poland",   "league", 1, 9016, 0.5528, "", "Ekstraklasa",
                fdcouk_new="POL", fdcouk_league="Ekstraklasa",
                teams_n=18, releg_spots=3, promo_spots=3, euro_spots=4),
    Competition("Parva Liga",           "Bulgaria", "league", 1, 9017, 0.5068, "", "Parva Liga",
                teams_n=16, releg_spots=2, promo_spots=2, euro_spots=4),
    Competition("Romanian Superliga",   "Romania",  "league", 1, 9018, 0.5034, "", "Superliga",
                fdcouk_new="ROU", fdcouk_league="Superliga",
                teams_n=16, releg_spots=3, promo_spots=3, euro_spots=4),
    Competition("Allsvenskan",          "Sweden",   "league", 1, 9019, 0.5013, "", "Allsvenskan",
                fdcouk_new="SWE", fdcouk_league="Allsvenskan", calendar_season=True,
                teams_n=16, releg_spots=3, promo_spots=2, euro_spots=4),
    Competition("League of Ireland",    "Ireland",  "league", 1, 9020, 0.4620, "", "",
                fdcouk_new="IRL", fdcouk_league="Premier Division", calendar_season=True,
                teams_n=10, releg_spots=2, promo_spots=1, euro_spots=4),
    Competition("Veikkausliiga",        "Finland",  "league", 1, 9021, 0.4586, "", "Veikkausliiga",
                fdcouk_new="FIN", fdcouk_league="Veikkausliiga", calendar_season=True,
                teams_n=12, releg_spots=2, promo_spots=1, euro_spots=4),
    # Russia is UEFA-suspended: its clubs do not enter European competition, so
    # these matches add no cross-league links and cannot inform relative league
    # strength. Registered for completeness; P3 ingest is optional.
    Competition("Russian Premier League", "Russia", "league", 1, 9022, 0.5173, "", "",
                fdcouk_new="RUS", fdcouk_league="Premier League",
                teams_n=16, releg_spots=4, promo_spots=4, euro_spots=4),

    # ── Non-UEFA club competitions (BSD) ─────────────────────────────────
    # Registered for BETTING these leagues, not for global comparability.
    #
    # Read this before using their ratings across confederations. The UEFA
    # expansion worked because European competition links those leagues — the
    # Premier League alone has 274 inter-league matches, which is what puts
    # every European league on one Elo scale. These leagues have essentially
    # ZERO competitive matches against the fitted European set; the only
    # bridge is pre-season Club Friendlies, which are worthless for rating
    # (rotated squads, no competitive intensity). And unlike the P4
    # zero-connectivity leagues, there is no UEFA coefficient to anchor them.
    #
    # So: within-league ranking is sound. "Is an MLS side better than a
    # Bundesliga side" is NOT answerable from this data, and `strength` below
    # must not be read as commensurable with the European values. They are
    # placeholders on a separate, unanchored scale — which is also why
    # `pool` marks them as a distinct rating population.
    #
    # Data quality is good: 95-100% xG coverage on BSD, better than the ten
    # fd.co.uk leagues that carry no shot data at all.
    Competition("MLS",                  "USA",       "league", 1, 9101, 0.55, "", "MLS",
                calendar_season=True, teams_n=30,
                # No relegation in MLS — conferences and playoffs instead. 0 is
                # correct here rather than unknown; motivation bands built on a
                # relegation battle do not apply.
                releg_spots=0, promo_spots=0, euro_spots=0),
    Competition("USL Championship",     "USA",       "league", 2, 9102, 0.38, "", "USL Championship",
                calendar_season=True, teams_n=24,
                releg_spots=0, promo_spots=0, euro_spots=0),
    Competition("J1 League",            "Japan",     "league", 1, 9103, 0.55, "", "J1 League",
                calendar_season=True, teams_n=20,
                releg_spots=3, promo_spots=0, euro_spots=0),
    Competition("K League 1",           "South Korea", "league", 1, 9104, 0.55, "", "K League 1",
                calendar_season=True, teams_n=12,
                releg_spots=2, promo_spots=0, euro_spots=0),
    Competition("Brasileirao Serie A",  "Brazil",    "league", 1, 9105, 0.62, "", "Brasileirão Serie A",
                calendar_season=True, teams_n=20,
                releg_spots=4, promo_spots=0, euro_spots=0),
    Competition("Brasileirao Serie B",  "Brazil",    "league", 2, 9106, 0.44, "", "Brasileirão Serie B",
                calendar_season=True, teams_n=20,
                releg_spots=4, promo_spots=4, euro_spots=0),
    Competition("Categoria Primera A",  "Colombia",  "league", 1, 9107, 0.50, "", "Categoría Primera A",
                calendar_season=True, teams_n=20,
                releg_spots=2, promo_spots=0, euro_spots=0),
    Competition("Saudi Pro League",     "Saudi Arabia", "league", 1, 9108, 0.55, "", "Saudi Pro League",
                teams_n=18, releg_spots=3, promo_spots=0, euro_spots=0),
    Competition("Chinese Super League", "China",     "league", 1, 9109, 0.45, "", "Chinese Super League",
                calendar_season=True, teams_n=16,
                releg_spots=2, promo_spots=0, euro_spots=0),
    Competition("Botola Pro",           "Morocco",   "league", 1, 9110, 0.42, "", "Botola Pro",
                teams_n=16, releg_spots=2, promo_spots=0, euro_spots=0),
    # Liga MX plays TWO championships per calendar year (Apertura Jul-Dec,
    # Clausura Jan-May) with the same clubs. They are registered separately
    # because they are separate tournaments, and calendar_season keeps each
    # inside its own year rather than straddling the July boundary — but note
    # the season model still assumes one competition-season per year, so
    # cross-season logic (promotion priors, Elo season regression) is only
    # approximately right for these two.
    Competition("Liga MX Apertura",     "Mexico",    "league", 1, 9111, 0.52, "", "Liga MX Apertura",
                calendar_season=True, teams_n=18,
                releg_spots=0, promo_spots=0, euro_spots=0),
    Competition("Liga MX Clausura",     "Mexico",    "league", 1, 9112, 0.52, "", "Liga MX Clausura",
                calendar_season=True, teams_n=18,
                releg_spots=0, promo_spots=0, euro_spots=0),
    # Continental competitions. Libertadores/Sudamericana link the South
    # American leagues to EACH OTHER, which makes their relative strengths
    # identifiable within that confederation — the same mechanism UEFA
    # competition provides in Europe. They do not link either continent to
    # the other.
    Competition("Copa Libertadores",    "South America", "europe", 0, 9113, 0.70, "", "Copa Libertadores"),
    Competition("Copa Sudamericana",    "South America", "europe", 0, 9114, 0.58, "", "Copa Sudamericana"),
    Competition("CAF Champions League", "Africa",    "europe", 0, 9115, 0.50, "", "CAF Champions League"),
]

BY_NAME = {c.name: c for c in COMPETITIONS}
BY_API_ID = {c.api_id: c for c in COMPETITIONS}

# BSD league names that are covered free (football only requires a BSD key, all leagues free)
# Maps BSD league name (lowercase) -> Competition name in our registry
# Populated lazily — BSD may use slightly different names; this handles common variants
BSD_LEAGUE_ALIASES: dict[str, str] = {
    "premier league": "Premier League",
    "efl championship": "Championship",
    "championship": "Championship",
    "efl league one": "League One",
    "league one": "League One",
    "efl league two": "League Two",
    "league two": "League Two",
    "fa cup": "FA Cup",
    "efl cup": "EFL Cup",
    "carabao cup": "EFL Cup",
    "league cup": "EFL Cup",
    "scottish premiership": "Scottish Premiership",
    "scottish championship": "Scottish Championship",
    "scottish league one": "Scottish League One",
    "scottish league two": "Scottish League Two",
    "scottish cup": "Scottish Cup",
    "scottish league cup": "Scottish League Cup",
    "bundesliga": "Bundesliga",
    "1. bundesliga": "Bundesliga",
    "dfb-pokal": "DFB-Pokal",
    "dfb pokal": "DFB-Pokal",
    "serie a": "Serie A",
    "coppa italia": "Coppa Italia",
    "ligue 1": "Ligue 1",
    "coupe de france": "Coupe de France",
    "la liga": "La Liga",
    "laliga": "La Liga",
    "copa del rey": "Copa del Rey",
    "uefa champions league": "Champions League",
    "champions league": "Champions League",
    "uefa europa league": "Europa League",
    "europa league": "Europa League",
    "uefa europa conference league": "Conference League",
    "conference league": "Conference League",
    "europa conference league": "Conference League",
    "uefa super cup": "UEFA Super Cup",
    # ── P2 expansion ──────────────────────────────────────────────────────
    # Verified against a live /api/v2/leagues/ pull (72 leagues), so these are
    # BSD's actual strings rather than assumed ones. Exact-match only remains
    # essential here: BSD's catalogue is global, and several of these names
    # are ambiguous in isolation. "Super League" is Switzerland's (BSD id 15)
    # — Greece's is "Stoiximan Super League" (id 24) and Portugal's top flight
    # is "Liga Portugal Betclic" (id 2). "Superliga" is Romania's (id 23);
    # Denmark's Superliga is not in BSD at all, it comes from fd.co.uk. A
    # substring rule would merge these three continents' worth of leagues.
    "eredivisie": "Eredivisie",
    "liga portugal betclic": "Liga Portugal",
    "liga 3": "Liga 3",
    "pro league": "Belgian Pro League",
    "trendyol super lig": "Super Lig",
    "stoiximan super league": "Greek Super League",
    "super league": "Swiss Super League",
    "superliga": "Romanian Superliga",
    "allsvenskan": "Allsvenskan",
    "eliteserien": "Eliteserien",
    "ekstraklasa": "Ekstraklasa",
    "veikkausliiga": "Veikkausliiga",
    "parva liga": "Parva Liga",
    "segunda división": "Segunda División",
    "segunda division": "Segunda División",
    # ── Non-UEFA club competitions ────────────────────────────────────────
    # Exact-match only remains essential. Two of these are the very names the
    # original substring rule collided on: "USL Championship" against England's
    # "Championship", and "Brasileirão Serie A" / "Chinese Super League"
    # against Serie A and the Swiss Super League. They are now registered
    # deliberately, which makes exact matching load-bearing rather than
    # merely cautious — a substring rule here would merge four continents.
    "mls": "MLS",
    "usl championship": "USL Championship",
    "j1 league": "J1 League",
    "k league 1": "K League 1",
    "brasileirão serie a": "Brasileirao Serie A",
    "brasileirao serie a": "Brasileirao Serie A",
    "brasileirão serie b": "Brasileirao Serie B",
    "brasileirao serie b": "Brasileirao Serie B",
    "categoría primera a": "Categoria Primera A",
    "categoria primera a": "Categoria Primera A",
    "saudi pro league": "Saudi Pro League",
    "chinese super league": "Chinese Super League",
    "botola pro": "Botola Pro",
    "liga mx apertura": "Liga MX Apertura",
    "liga mx clausura": "Liga MX Clausura",
    "copa libertadores": "Copa Libertadores",
    "copa sudamericana": "Copa Sudamericana",
    "caf champions league": "CAF Champions League",
}

# football-data.org: competitions covered on the free tier
FDORG_COMPETITIONS = {c.name: c for c in COMPETITIONS if c.fdorg_code}


def names() -> list[str]:
    return [c.name for c in COMPETITIONS]


def public_rows() -> list[dict]:
    rows = []
    for c in COMPETITIONS:
        row = asdict(c)
        row["api_football_id"] = c.api_id
        rows.append(row)
    return rows


def get(name: str | None) -> Competition | None:
    if not name:
        return None
    return BY_NAME.get(str(name).strip())


def _load_comp_strength() -> dict:
    """Fitted competition-strength overrides (P4.4) — cached for the process
    lifetime. Consulted by strength() ONLY when the file's "active" flag is
    true (report-only/gated-OFF by default, per the promotion discipline)."""
    global _comp_strength_cache
    if _comp_strength_cache is None:
        if _COMP_STRENGTH_FILE.exists():
            try:
                _comp_strength_cache = json.loads(_COMP_STRENGTH_FILE.read_text())
            except Exception:
                _comp_strength_cache = {}
        else:
            _comp_strength_cache = {}
    return _comp_strength_cache


def reload_comp_strength() -> None:
    """Force the next strength() call to re-read comp_strength.json from
    disk (tests / --fit-comp-strength change the file mid-process)."""
    global _comp_strength_cache
    _comp_strength_cache = None


def strength(name: str | None) -> float:
    c = get(name)
    base = c.strength if c else 0.75
    fitted = _load_comp_strength()
    if fitted.get("active") and name in fitted:
        return float(fitted[name])
    return base


def comp_from_bsd_league(bsd_name: str) -> Competition | None:
    """Resolve a BSD league name to a Competition.

    Exact match only (via BSD_LEAGUE_ALIASES or the bsd_league hint field).
    A prior substring-containment fallback caused real cross-continent
    collisions once BSD's fuller league catalog was queried, e.g. "USL
    Championship" (USA) matching our "Championship" (England) because
    "championship" in "usl championship", and "CAF Champions League" /
    "Brasileirão Serie A" likewise matching "Champions League" / "Serie A".
    Unrecognised names are printed by callers and simply skipped — safer
    than silently blending in a different continent's results.
    """
    low = str(bsd_name).strip().lower()
    if low in BSD_LEAGUE_ALIASES:
        return BY_NAME.get(BSD_LEAGUE_ALIASES[low])
    for c in COMPETITIONS:
        hint = (c.bsd_league or c.name).lower()
        if hint and hint == low:
            return c
    return None
