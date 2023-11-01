from enum import Enum


class RunMode(str, Enum):
    """Define a run mode for subproject 1. Either run in naive mode (a la Project 2) or SPIMI mode"""

    SPIMI = 'spimi'
    NAIVE = 'naive'
