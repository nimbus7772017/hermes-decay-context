"""DecayEngine -> operator pin digest hook (2026-10-04)."""
import json

import pytest

from pins import operator_pins as pins


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PIN_OPEN_LOOPS_BRIDGE", "0")
    monkeypatch.setenv("HERMES_PIN_DIGEST_HELPER", "0")   # keyword path only; no network
    yield tmp_path


def _engine():
    from engine import DecayEngine
    return object.__new__(DecayEngine)


def test_hook_marks_likely_done_before_rebuild():
    pins.pin_operator_message("yeah clean those mikrotik hosts up")
    d = ("STATE AT EVICTION: Unlinked template 18872 from all 6 Mikrotik hosts; duplicate keys removed.\n"
         "OPEN LOOPS: (as of eviction) None\nLAST SEAM: x")
    _engine()._pin_digest_hook(d)
    rec = json.loads(pins._pins_path().read_text().splitlines()[0])
    assert rec["state"] == "likely_done"
    assert "digest says done" in pins.format_operator_pins()


def test_hook_never_raises(monkeypatch):
    monkeypatch.setattr(pins, "apply_digest_hints", lambda d: (_ for _ in ()).throw(RuntimeError("x")))
    _engine()._pin_digest_hook("OPEN LOOPS: x")   # must not raise

@pytest.fixture(autouse=True)
def _fix7_filter_off(monkeypatch):
    # these tests use synthetic one-word "orders"; the fix7 directive filter has its own tests
    monkeypatch.setenv("HERMES_PIN_DIRECTIVE_FILTER", "0")
