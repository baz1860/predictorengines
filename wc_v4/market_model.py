"""Compatibility shim for :mod:`research.market_model`."""
from research.market_model import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.market_model", run_name="__main__")
