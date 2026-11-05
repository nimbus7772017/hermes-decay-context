"""fix1 + fix7 (2026-10-30): pins actually retire; only directives get pinned."""
import json, time
import pytest
from pins import operator_pins as pins

@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setattr(pins, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PIN_DIRECTIVE_FILTER", "1")
    yield

def _recs():
    return [json.loads(l) for l in pins._pins_path().read_text().splitlines() if l.strip()]

CHITCHAT = [
    'i like ALTAIR and the band name will be "Models"',
    'yeah like theres a really good band called "Battles" right?',
    "i will play drums and the agent will do a lot of the rest",
    "Saw a deer by the creek this morning",
    "No, I mean he jumped to the other trail",
    "Oneal hill is here  37.42900, -107.13994",
]
DIRECTIVES = [
    "can you do the monthly subscriber cancellation audit please",
    "Yes, please complete the audit including shutting down service",
    "Then I want you to do another sweep and make sure that anyone that emailed us requesting to cancel their account got canceled for October",
    "Hey Hermes, send me a voice clip summary of the hunting trip",
    "Okay now can you find me the back hall link between the towers",
    "nice thats another cool thread. go look on the wiki and see if it is there",
    "why are the LTE radios at Menefee down this morning?",
    "don't restart the gateway until I say so",
]

@pytest.mark.parametrize("t", CHITCHAT)
def test_chitchat_not_pinned(t):
    assert pins.pin_operator_message(t) is False

@pytest.mark.parametrize("t", DIRECTIVES)
def test_directives_pinned_with_ids(t):
    assert pins.pin_operator_message(t) is True
    assert re.match(r"^D\d+$", _recs()[-1]["id"])

import re
def test_ids_increment_and_render():
    pins.pin_operator_message("please check zabbix for menefee")
    pins.pin_operator_message("can you send the report to jake")
    ids = [r["id"] for r in _recs()]
    assert ids == ["D1", "D2"]
    assert "- D2 " in pins.format_operator_pins()

def test_decision_reply_to_question_pinned():
    assert pins.pin_operator_message("yes", prev_assistant="Want me to restart it now?") is True

def test_filter_kill_switch(monkeypatch):
    monkeypatch.setenv("HERMES_PIN_DIRECTIVE_FILTER", "0")
    assert pins.pin_operator_message(CHITCHAT[0]) is True

DIGEST_1027 = """WORKING GOAL
As of the span: Finalize October email-sweep cancellations and verify 12 total cohort (12 cleared, 0 billing).

STATE AT EVICTION
Sweep of Oct 1-27 inbox complete; 7 customers found (4 pre-cleared by Jake, 3 acted upon today). All 7 now fully cleared.

OPEN LOOPS
None; all identified cancellation requests from Oct 1-27 are resolved.

LAST SEAM
Awaiting next operator instruction.
"""

def test_heading_own_line_digest_parsed():
    parts = pins._digest_parts(DIGEST_1027)
    assert parts["open"] == []
    assert any("cancellation requests" in d for d in parts["done"])

def _seed(text, age_h=1.0):
    now = time.time()
    pins._write_all([{"t": "x", "text": text, "state": "pinned", "id": "D1",
                      "ts_epoch": now - age_h * 3600, "expires_at": now - age_h * 3600 + pins.PIN_TTL_SECS}])

CANCEL = DIRECTIVES[2]

def test_cancel_sweep_pin_gets_likely_done_and_persists():
    _seed(CANCEL)
    res = pins.apply_digest_hints(DIGEST_1027, use_helper=True, helper=lambda p, s: True)
    assert res["likely_done"] == 1
    assert _recs()[0]["state"] == "likely_done"          # persisted to disk
    pins.pin_operator_message("please check zabbix for menefee")   # another writer must not clobber it
    assert _recs()[0]["state"] == "likely_done"
    assert "auto-retires" in pins.format_operator_pins()

def test_helper_no_means_no_hint():
    _seed(CANCEL)
    assert pins.apply_digest_hints(DIGEST_1027, use_helper=True, helper=lambda p, s: False)["likely_done"] == 0
    assert _recs()[0]["state"] == "pinned"

def test_likely_done_auto_retires_after_grace(monkeypatch):
    _seed(CANCEL)
    pins.apply_digest_hints(DIGEST_1027, use_helper=True, helper=lambda p, s: True)
    later = time.time() + pins.LIKELY_DONE_GRACE_SECS + 5
    assert pins.active_pins(now=later) == []               # stops riding the block
    monkeypatch.setattr(pins.time, "time", lambda: later)
    pins.pin_operator_message("please check zabbix for menefee")   # any write settles state
    r = _recs()[0]
    assert r["state"] == "satisfied" and r["retired_via"] == "digest-auto" and "evidence" in r

def test_expired_is_persisted(monkeypatch):
    _seed("please check zabbix", age_h=30)
    pins.pin_operator_message("can you send the report to jake")
    assert _recs()[0]["state"] == "expired"

def test_new_pins_ttl_24h():
    pins.pin_operator_message("please check zabbix")
    r = _recs()[0]
    assert r["expires_at"] - r["ts_epoch"] == pytest.approx(24 * 3600)
