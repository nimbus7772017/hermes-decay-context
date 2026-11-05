"""Isolate tests under a temp HERMES_HOME; put package root on sys.path."""
from __future__ import annotations
import sys
from pathlib import Path
import os
import pytest

# Pins vocative/agent-name handling is configurable; tests use a generic name.
os.environ.setdefault("HERMES_PIN_AGENT_NAME", "hermes")

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PIN_OPEN_LOOPS_BRIDGE", "0")
    monkeypatch.delenv("HERMES_PIN_HELPER_URL", raising=False)
    monkeypatch.delenv("HERMES_PIN_DB_PASSWORD", raising=False)
    yield home
