"""Compatibility shim for :mod:`research.feature_store`."""
from research.feature_store import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.feature_store", run_name="__main__")
