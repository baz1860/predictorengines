"""Engine registry bootstrap.

To add an engine: import its adapter here and registry.register() it.
That is the ONLY wiring step — the server and UI discover everything else.
"""
from contracts import registry
from .worldcup import WorldCupAdapter
from .cfb import CFBAdapter
from .golf import GolfAdapter
from .club_soccer import ClubSoccerAdapter
from .tennis import TennisAdapter
from .nhl import NHLAdapter

registry.register(WorldCupAdapter())
registry.register(CFBAdapter())
registry.register(GolfAdapter())
registry.register(ClubSoccerAdapter())
registry.register(TennisAdapter())
registry.register(NHLAdapter())

__all__ = ["registry"]
