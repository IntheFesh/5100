import os, sys, pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _load import load_pqn


@pytest.fixture(scope="module")
def pqn():
    """pqn_v5.py definitions with the default (10M-step, E=2) configuration."""
    return load_pqn(["--steps", "1e7"])
