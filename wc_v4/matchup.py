"""Compatibility shim for :mod:`research.matchup`."""
from research.matchup import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.matchup", run_name="__main__")
