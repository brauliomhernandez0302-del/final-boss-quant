"""Shared fixtures for the test suite."""
import sys
import os

# Ensure project root is on the path so all imports resolve without installation.
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
