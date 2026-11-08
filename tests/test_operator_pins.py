"""Tests for gateway/operator_pins.py — verbatim operator directive pins + lifecycle
(9/24 lifecycle restored 2026-10-04 with the operator's tweaks)."""
import json
import time

import pytest

from pins import operator_pins as pins


@pytest.fixture(autouse=True)
def _pin_home(tmp_path, monkeypatch):
    """Redirect the pin file into tmp_path (never touch real HERMES_HOME)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PIN_OPEN_LOOPS_BRIDGE", "0")   # never touch the real DB
    yield tmp_path


def _recs():
    return [json.loads(l) for l in pins._pins_path().read_text().splitlines() if l.strip()]


# --- capture (pre-existing contract) -------------------------------------------------

def test_pin_short_message_verbatim():
    assert pins.pin_operator_message("finish through Dynamic vegetation index (NDVI/NDMI)", "slack")
    data = _recs()[0]
    assert data["text"] == "finish through Dynamic vegetation index (NDVI/NDMI)"
    assert data["via"] == "slack"
    assert data["state"] == "pinned"
    assert data["expires_at"] - data["ts_epoch"] == pytest.approx(24 * 3600)  # fix1 2026-10-30


def test_thread_context_stripped():
    text = ('[Thread context — prior messages in this thread (not yet in conversation history):]\n'
            'someone: earlier chatter\n[End of thread context]\n\n'
            'leave the weather file alone')
    assert pins.pin_operator_message(text)
    assert _recs()[0]["text"] == "leave the weather file alone"


def test_long_pastes_and_noise_skipped():
    assert not pins.pin_operator_message("x" * 1600)
    assert not pins.pin_operator_message("...")
    assert not pins.pin_operator_message("")
    assert not pins._pins_path().exists()


def test_directive_essays_pinned():
    essay = ("climate is the whole game. the rut rides the climate curve. " * 12
             + "\n1. Debug the MODIS GIBS requests\n2. Continue with the LANDFIRE integration")
    assert len(essay) > 500 and len(essay) <= 1500
    assert pins.pin_operator_message(essay)
    assert "LANDFIRE integration" in _recs()[0]["text"]


def test_fifo_cap():
    for i in range(pins.PIN_KEEP * 2 + 5):
        pins.pin_operator_message("directive %d" % i)
    assert len(_recs()) == pins.PIN_KEEP


def test_format_block_newest_first():   # fix2 2026-10-30: was newest-last + OUTRANK
    pins.pin_operator_message("first order")
    pins.pin_operator_message("second order")
    block = pins.format_operator_pins()
    assert "OUTRANK" not in block and "newest first" in block
    assert block.index("second order") < block.index("first order")


def test_format_empty_when_no_pins():
    assert pins.format_operator_pins() == ""


def test_corrupt_file_never_breaks():
    pins._pins_path().write_text("not json\n{bad\n")
    assert pins.format_operator_pins() == ""


# --- skip acknowledgements / greetings ------------------------------------------------

@pytest.mark.parametrize("text", [
    "ok", "OK.", "okay", "Thanks!", "thank you so much", "thx", "good morning", "Good morning Hermes",
    "hello", "hey hermes", "are you there?", "you there", "still there?", "ok glad to have that done",
    "glad that's done 👍", "sounds good, thanks hermes!", "ok thanks", "good night", "great job",
    "perfect", ":thumbsup:", "lol",
])
def test_ack_not_pinned(text):
    assert pins.is_ack(text)
    assert not pins.pin_operator_message(text)


@pytest.mark.parametrize("text", [
    "yes fix that", "go ahead", "ok do it", "yeah clean those mikrotik hosts up",
    "hey hermes, delete those hosts", "did you identify the PtP between site A and site B?",
    "good, now check eNB 300", "thanks, also send me the list",
])
def test_directives_and_approvals_pinned(text):
    assert not pins.is_ack(text)
    assert pins.pin_operator_message(text)


# --- context for short replies --------------------------------------------------------

def test_bare_yes_no_pinned_only_with_context():
    assert not pins.pin_operator_message("yes")
    assert pins.pin_operator_message("yes", prev_assistant="Should I delete the 6 stale hosts?")
    assert pins.pin_operator_message("no", prev_assistant="Restart the gateway now?")
    assert [r["text"] for r in _recs()] == ["yes", "no"]


def test_short_reply_gets_context_from_previous_question():
    prev = ("I found 6 Mikrotik hosts inheriting system.name from two templates. "
            "Want me to unlink template 18872 from all six? I'll keep 10233.")
    assert pins.pin_operator_message("yes fix that", "slack", prev_assistant=prev)
    rec = _recs()[0]
    assert rec["re"] == "Want me to unlink template 18872 from all six?"
    assert '(re: "Want me to unlink template 18872' in pins.format_operator_pins()


def test_context_truncated_and_only_for_short_replies():
    prev = "x " * 400
    assert pins.pin_operator_message("do it", prev_assistant=prev)
    assert len(_recs()[-1]["re"]) <= pins.PIN_CONTEXT_CHARS
    long_msg = "please " + " ".join("word%d" % i for i in range(20))
    assert pins.pin_operator_message(long_msg, prev_assistant="Should I?  Anything else to check here?")
    assert "re" not in _recs()[-1]


# --- expiry ---------------------------------------------------------------------------

def test_expired_pins_leave_the_block_but_stay_on_disk():
    pins.pin_operator_message("old order")
    recs = _recs()
    recs[0]["ts_epoch"] -= 49 * 3600
    recs[0]["expires_at"] = recs[0]["ts_epoch"] + pins.PIN_TTL_SECS
    pins._write_all(recs)
    pins.pin_operator_message("new order")
    block = pins.format_operator_pins()
    assert "old order" not in block and "new order" in block
    assert len(_recs()) == 2


def test_legacy_records_migrate_and_expire_by_t():
    old = time.strftime("%Y-%m-%d %H:%M:%S MDT", time.localtime(time.time() - 3 * 86400))
    new = time.strftime("%Y-%m-%d %H:%M:%S MDT", time.localtime(time.time() - 3600))
    pins._pins_path().write_text(json.dumps({"t": old, "text": "legacy old", "via": "slack"}) + "\n"
                                 + json.dumps({"t": new, "text": "legacy new", "via": "slack"}) + "\n")
    res = pins.migrate_pin_file()
    assert res == {"records": 2, "migrated": 2}
    assert all(r["state"] == "pinned" and "expires_at" in r for r in _recs())
    assert pins.migrate_pin_file()["migrated"] == 0   # idempotent
    block = pins.format_operator_pins()
    assert "legacy new" in block and "legacy old" not in block


# --- retire ---------------------------------------------------------------------------

def test_retire_by_substring_with_evidence():
    pins.pin_operator_message("yeah clean those mikrotik hosts up")
    pins.pin_operator_message("check eNB 300 cells")
    hits = pins.retire_operator_directive("MikroTik hosts", evidence="unlinked 18872 from 6 hosts",
                                          actor="hermes-tool")
    assert len(hits) == 1
    rec = [r for r in _recs() if "mikrotik" in r["text"]][0]
    assert rec["state"] == "satisfied" and rec["evidence"] == "unlinked 18872 from 6 hosts"
    assert rec["retired_via"] == "hermes-tool"
    block = pins.format_operator_pins()
    pin_lines = "\n".join(l for l in block.splitlines() if l.startswith("- "))   # fix5: retired text may appear in the CLOSED ledger line
    assert "mikrotik" not in pin_lines and "eNB 300" in pin_lines
    assert "CLOSED (finished" in block and "mikrotik" in block.split("CLOSED (finished")[1]


def test_retire_last_and_no_match():
    pins.pin_operator_message("first order here")
    pins.pin_operator_message("second order here")
    assert pins.retire_operator_directive("nothing like this") == []
    hits = pins.retire_operator_directive("last")
    assert [h["text"] for h in hits] == ["second order here"]
    assert pins.retire_operator_directive("second order") == []   # already retired


def test_tool_requires_evidence_and_retires(monkeypatch):
    """Evidence gate and one-shot retire go through the handler register() publishes.

    The old tools.operator_directive_tool module is not part of this plugin.
    Calling that import would skip the registered tool; this uses the handler
    Hermes receives for operator_directive_done.
    """
    import importlib.util
    import sys
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "hdc_operator_tool", pkg / "__init__.py", submodule_search_locations=[str(pkg)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(pkg)]
    sys.modules["hdc_operator_tool"] = mod
    spec.loader.exec_module(mod)
    registered = {}

    class _Ctx:
        def register_tool(self, name, toolset, schema, handler, **kwargs):
            registered[name] = handler

        def register_hook(self, *args, **kwargs):
            pass

        def register_command(self, *args, **kwargs):
            pass

        def register_context_engine(self, engine):
            pass

        def register_auxiliary_task(self, *args, **kwargs):
            pass

    mod.register(_Ctx())
    done = registered["operator_directive_done"]
    assert pins.pin_operator_message("restart the plates window")
    empty = json.loads(done({"substring": "plates window", "evidence": ""}))
    assert "evidence required" in empty["error"]
    assert any("plates window" in p["text"] for p in pins.active_pins())
    out = json.loads(done({
        "substring": "plates window",
        "evidence": "hermes-plates active 10:41",
    }))
    assert out["retired"] == 1
    assert not any("plates window" in p["text"] for p in pins.active_pins())
    again = json.loads(done({"substring": "plates window", "evidence": "again"}))
    assert "error" in again


def test_bridge_called_on_retire(monkeypatch):
    seen = []
    monkeypatch.setattr(pins, "_bridge_open_loop_async", lambda rec: seen.append(rec))
    pins.pin_operator_message("fix the API access")
    pins.retire_operator_directive("api access", evidence="token refreshed")
    assert seen and seen[0]["evidence"] == "token refreshed"


def test_bridge_skips_quietly_when_db_unreachable(monkeypatch):
    monkeypatch.setenv("HERMES_PIN_OPEN_LOOPS_BRIDGE", "1")
    def boom(*a, **k):
        raise FileNotFoundError("pass")
    monkeypatch.setattr(pins.subprocess, "run", boom)
    assert pins._bridge_open_loop({"ts_epoch": time.time(), "text": "x", "evidence": "y"}) is False


# --- trim order -----------------------------------------------------------------------

def test_trim_drops_retired_first_then_expired_then_oldest():
    now = time.time()
    recs = []
    for i in range(pins.PIN_KEEP):
        recs.append({"t": "t%d" % i, "text": "order %d" % i, "state": "pinned",
                     "ts_epoch": now - 100 + i, "expires_at": now + 3600})
    recs[10]["state"] = "satisfied"          # retired, mid-list
    recs[3]["expires_at"] = now - 1          # expired
    pins._write_all(recs)
    pins.pin_operator_message("new order A")   # 26 -> drop the retired one
    texts = [r["text"] for r in _recs()]
    # fix1 2026-10-30: expired records are settled to state "expired" (a retired state),
    # so the oldest retired-or-expired record (order 3) drops first now.
    assert "order 3" not in texts and "order 0" in texts and "order 10" in texts
    pins.pin_operator_message("new order B")   # next: the expired one
    texts = [r["text"] for r in _recs()]
    assert "order 3" not in texts and "order 0" in texts
    pins.pin_operator_message("new order C")   # then the oldest active
    texts = [r["text"] for r in _recs()]
    assert "order 0" not in texts and texts[-1] == "new order C"
    assert len(texts) == pins.PIN_KEEP


# --- digest hook: likely_done / open ---------------------------------------------------

DIGEST = """WORKING GOAL: as of the span: the operator requested cleanup of Mikrotik hosts.
STATE AT EVICTION: Unlinked template 18872 from all 6 Mikrotik hosts; duplicate system.name keys removed and verified via API.
DECISIONS MADE:
- Kept template 10233 because it carries the SNMP items.
OPEN LOOPS:
- (as of eviction) eNB 300 tower identity still unresolved
KEY FACTS & IDS:
k: bad_template_id: 18872
LAST SEAM: reporting to the operator."""


def test_digest_marks_likely_done_and_open_never_closes():
    pins.pin_operator_message("yeah clean those mikrotik hosts up")
    pins.pin_operator_message("figure out which tower eNB 300 is")
    pins.pin_operator_message("call Tommy about the invoice")
    res = pins.apply_digest_hints(DIGEST, helper=lambda p, s: True)
    assert res == {"open": 1, "likely_done": 1}
    st = {r["text"]: r for r in _recs()}
    assert st["yeah clean those mikrotik hosts up"]["state"] == "likely_done"
    assert "18872" in st["yeah clean those mikrotik hosts up"]["hint"]
    assert st["figure out which tower eNB 300 is"]["state"] == "open"
    assert st["call Tommy about the invoice"]["state"] == "pinned"
    block = pins.format_operator_pins()
    assert '(digest says done: "' in block and "confirm and close" in block
    assert "[open — digest lists it under OPEN LOOPS]" in block
    assert "Close only with evidence; don't start work to satisfy a pin." in block
    # still active: likely_done is a hint, not a close
    assert len(pins.active_pins()) == 3


def test_digest_helper_no_or_error_is_noop():
    pins.pin_operator_message("yeah clean those mikrotik hosts up")
    assert pins.apply_digest_hints(DIGEST, helper=lambda p, s: False)["likely_done"] == 0
    assert pins.apply_digest_hints(DIGEST, helper=lambda p, s: None)["likely_done"] == 0
    assert _recs()[0]["state"] == "pinned"


def test_digest_keyword_only_when_helper_disabled():
    pins.pin_operator_message("yeah clean those mikrotik hosts up")
    assert pins.apply_digest_hints(DIGEST, use_helper=False)["likely_done"] == 1


def test_digest_open_overrides_older_likely_done():
    pins.pin_operator_message("yeah clean those mikrotik hosts up")
    pins.apply_digest_hints(DIGEST, helper=lambda p, s: True)
    d2 = "OPEN LOOPS:\n- (as of eviction) mikrotik hosts cleanup still pending on HQRack3\nLAST SEAM: x"
    pins.apply_digest_hints(d2, helper=lambda p, s: True)
    assert _recs()[0]["state"] == "open"


def test_digest_skips_cpu_card_and_retired_pins():
    pins.pin_operator_message("yeah clean those mikrotik hosts up")
    card = "CONTEXT CARD (deterministic, no model — venue degraded; 40 turns moved to memory)\n" + DIGEST
    assert pins.apply_digest_hints(card, helper=lambda p, s: True) == {"open": 0, "likely_done": 0}
    pins.retire_operator_directive("mikrotik", evidence="done")
    pins.apply_digest_hints(DIGEST, helper=lambda p, s: True)
    assert _recs()[0]["state"] == "satisfied"


def test_digest_none_open_loops_ignored():
    pins.pin_operator_message("clean up the eNB list")
    d = "OPEN LOOPS: (as of eviction) None—eNB list cleanup initiated\nLAST SEAM: x"
    assert pins.apply_digest_hints(d, helper=lambda p, s: True) == {"open": 0, "likely_done": 0}


# --- nudge ----------------------------------------------------------------------------

def test_nudge_plain_and_likely_done():
    pins.pin_operator_message("check eNB 300 cells")
    block = pins.format_operator_pins()
    assert block.splitlines()[-1].startswith("[Pin check:")
    assert "operator_directive_done(substring, evidence)" in block
    assert "Close only with evidence; don't start work to satisfy a pin." in block
    assert "[Pin check" not in pins.format_operator_pins(nudge=False)


def test_list_pins_text():
    assert pins.list_pins_text() == "No active pinned directives."
    pins.pin_operator_message("check eNB 300 cells")
    out = pins.list_pins_text()
    assert "check eNB 300 cells" in out and "[pinned]" in out and "!done" in out

@pytest.fixture(autouse=True)
def _fix7_filter_off(monkeypatch):
    # these tests use synthetic one-word "orders"; the fix7 directive filter has its own tests
    monkeypatch.setenv("HERMES_PIN_DIRECTIVE_FILTER", "0")
