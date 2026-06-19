"""Compatibility shim for :mod:`governance.report`."""
from governance.report import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("governance.report", run_name="__main__")
