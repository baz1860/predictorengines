"""Compatibility shim for :mod:`research.validate_v4`."""
from research.validate_v4 import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.validate_v4", run_name="__main__")
