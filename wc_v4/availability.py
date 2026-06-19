"""Compatibility shim for :mod:`research.availability`."""
from research.availability import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.availability", run_name="__main__")
