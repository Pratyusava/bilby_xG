# Licensed under an MIT style license -- see LICENSE

__author__ = ["Pratyusava Baral <pbaral@uwm.edu>", "Soichiro Morisaki"]

try:
    from importlib.metadata import version
    __version__ = version("bilby_xG")
except Exception:  # development mode
    __version__ = "unknown"
