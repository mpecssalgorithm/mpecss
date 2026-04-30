"""Public package interface for MPECSS."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mpecss")
except PackageNotFoundError:
    __version__ = "0+unknown"

from mpecss.phase_2.homotopy import run_mpecss  # noqa: F401
