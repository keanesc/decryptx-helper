"""
DecryptX Helper Library for Round 3 - Data Cleaning Contest

This library provides tools for participating in the DecryptX Round 3
data cleaning contest. Teams clean a FIFA dataset, train models,
and submit their scores to the leaderboard.
"""

from decryptx.auth import login
from decryptx.config import configure, get_config
from decryptx.data import load_data
from decryptx.submit import submit

__version__ = "0.1.0"
__all__ = [
    "login",
    "load_data",
    "submit",
    "configure",
    "get_config",
]
