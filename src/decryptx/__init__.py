"""
DecryptX Helper Library for Round 3 - Data Cleaning Contest

This library provides tools for participating in the DecryptX Round 3
data cleaning contest. Teams clean a FIFA21 dataset, train models,
and submit their scores to the leaderboard.
"""

from decryptx.auth import login
from decryptx.config import configure, get_config
from decryptx.data import get_train_test_split, load_data
from decryptx.evaluate import evaluate
from decryptx.submit import submit

__version__ = "0.1.0"
__all__ = [
    "login",
    "load_data",
    "get_train_test_split",
    "evaluate",
    "submit",
    "configure",
    "get_config",
]
