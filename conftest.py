"""Puts the repository root on sys.path so `belot` and `scripts` import
under pytest without an editable install."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
