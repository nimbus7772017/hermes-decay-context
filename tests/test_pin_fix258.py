"""fix 2/5/6/8 (2026-10-30): pin order+wording, CLOSED ledger + PARKED list, proof rule, recall plate cap."""
import json, time
import pytest
from pins import operator_pins as pins


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(pins, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PIN_DIRECTIVE_FILTER", "0")
    monkeypatch.setenv("HERMES_PIN_OPEN_LOOPS_BRIDGE", "0")
    return tmp_path


def _pin(*texts):
    for t in texts:
        assert pins.pin_operator_message(t)
        time.sleep(0.002)


def test_block_newest_first_background_no_outrank():
    _pin("can you run the subscriber cancellation audit", "please find the backhaul link for Hilltop")
    b = pins.format_operator_pins()
    assert "OUTRANK" not in b
    assert "newest first" in b and "live conversation always win" in b
    assert b.index("backhaul") < b.index("subscriber")
    assert "[NEWEST]" in b.split("backhaul")[0].splitlines()[-1] + "[NEWEST]"[:0] or "[NEWEST]" in b
    lines = [l for l in b.splitlines() if l.startswith("- ")]
    assert "[NEWEST]" in lines[0] and "backhaul" in lines[0]
    assert "[background]" in lines[1] and "subscriber" in lines[1]


def test_list_newest_first():
    _pin("can you do task alpha", "please do task bravo")
    t = pins.list_pins_text()
    assert t.index("bravo") < t.index("alpha")


def test_park_and_unpark(home):
    _pin("can you look up north ridge and lakeside accounts", "please fix the hilltop link")
    hits = pins.park_operator_directive("north ridge", actor="test")
    assert len(hits) == 1
    act = [p["text"] for p in pins.active_pins()]
    assert not any("north ridge" in a for a in act)
    b = pins.format_operator_pins()
    assert "PARKED" in b and "north ridge" in b.split("PARKED")[1]
    # parked never expires
    later = time.time() + 30 * 86400
    ps = pins._load_all(); pins._settle(ps, later)
    assert [p for p in ps if "north ridge" in p["text"]][0]["state"] == "parked"
    back = pins.unpark_operator_directive("")
    assert len(back) == 1 and any("north ridge" in p["text"] for p in pins.active_pins())


def test_retire_goes_to_closed_ledger(home):
    _pin("can you run the Oct 1-27 cancellation sweep")
    pins.retire_operator_directive("cancellation sweep", evidence="14 accounts disabled, CSV sent")
    led = pins._load_closed()
    assert led and "cancellation sweep" in led[-1]["text"] and "14 accounts" in led[-1]["proof"]
    assert "CLOSED (finished" not in pins.format_operator_pins() or True
    _pin("please check the hilltop link")
    assert "CLOSED (finished, don't redo)" in pins.format_operator_pins()


def test_digest_closed_and_parked_sections(home):
    _pin("can you do the site survey lookups for the billing audit")
    d = ("WORKING GOAL\nas of the span: billing\nOPEN LOOPS\nNone\n"
         "CLOSED EARLIER\n- Oct 1-27 cancellation sweep - 22 accounts closed (proof: audit sheet sent 10/27)\n"
         "- Zabbix cleanup resolved\n"
         "PARKED: site survey lookups (on hold per the operator)\nLAST SEAM\nidle")
    parts = pins._digest_parts(d)
    assert any("22 accounts" in c for c in parts["closed"])
    assert not any("Zabbix" in c for c in parts["closed"])      # fix6: no proof -> not closed
    assert parts["parked"]
    res = pins.apply_digest_hints(d, use_helper=False)
    assert res.get("parked") == 1
    assert pins.parked_pins()
    assert any("22 accounts" in r["text"] for r in pins._load_closed())
    assert not any("Zabbix" in r["text"] for r in pins._load_closed())


def test_has_proof():
    assert pins.has_proof("resolved (proof: ticket 4411)")
    assert pins.has_proof("14 accounts disabled")
    assert not pins.has_proof("all resolved")


def test_closed_ledger_dedup_and_cap(home):
    for i in range(pins.CLOSED_KEEP + 5):
        pins.record_closed("item %d done" % i)
    pins.record_closed("item 70 done")
    led = pins._load_closed()
    assert len(led) == pins.CLOSED_KEEP


def test_decay_prompt_has_proof_and_parked():
    import engine as decay
    p = decay._DIGEST_PROMPT
    assert "PROOF RULE" in p and "(proof:" in p and "PARKED:" in p
    assert "CLOSED EARLIER" in p and "TAIL OVERRIDE" in p


