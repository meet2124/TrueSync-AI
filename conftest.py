# conftest.py — pytest root configuration
# Ensures the repo root is on sys.path so `from backend.X import Y` works in tests.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
