"""Compatibility shim for :mod:`research.probability`."""
from research.probability import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.probability", run_name="__main__")
