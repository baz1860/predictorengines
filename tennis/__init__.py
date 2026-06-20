"""Tennis (ATP + WTA) match-outcome prediction engine.

Surface-split Bradley-Terry skill model fitted on Jeff Sackmann's public match
archives, with a Markov-chain match simulator for set/game sub-markets and a
bracket Monte-Carlo for outright (win/final/SF/QF) markets. Mirrors the golf
engine's `fetch -> fit -> predict -> simulate -> edge` backbone and wires into
the shared engine contract via app/engines/tennis.py.
"""
