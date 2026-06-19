"""Compatibility shim for :mod:`operations.operations`."""
from operations.operations import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("operations.operations", run_name="__main__")
