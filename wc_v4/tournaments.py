"""Compatibility shim for :mod:`research.tournaments`."""
from research.tournaments import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.tournaments", run_name="__main__")
