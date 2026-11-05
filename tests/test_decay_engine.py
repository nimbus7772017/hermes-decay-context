"""Tests for the decay context engine: digest-on-eviction, tail-pruning
fallback, token estimator, and the configure() late-binding hook."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine import (
    DecayEngine,
    _est_tokens,
    _text_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**kwargs):
    kwargs.setdefault("config_context_length", 65536)
    return DecayEngine(**kwargs)


def _conversation(n_turns: int, body_size: int = 0, head: int = 3):
    """head protected messages + n_turns of user/assistant(+tool) exchange."""
    msgs = [
        {"role": "system", "content": "you are a test agent"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi, ready"},
    ][:head]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"do thing {i}"})
        if body_size:
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{\"cmd\": \"x\"}"},
                }],
            })
            # JSON-ish punctuation so the estimator treats it like tool output
            msgs.append({
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": ("{\"row\": [1, 2, 3], \"ok\": true} " * (body_size // 34)),
            })
        else:
            msgs.append({"role": "assistant", "content": f"did thing {i}"})
    return msgs


class _FakeProvider:
    def __init__(self, ok=True):
        self.ok = ok
        self.docs = []

    def retain_document(self, content, *, kind, session_id="", metadata=None):
        self.docs.append({"content": content, "kind": kind, "metadata": metadata})
        return self.ok


class _FakeMM:
    def __init__(self, providers):
        self.providers = providers


def _digest_response(text="GOAL: keep testing\nSTATE: midway\nDECISIONS: none\nOPEN LOOPS: finish\nKEY FACTS: x=1"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture
def mock_digest_llm(monkeypatch):
    """Patch the auxiliary compression call the engine uses for digests."""
    import agent.auxiliary_client as aux
    calls = []

    def fake_call_llm(task=None, *, messages, max_tokens=None, **kw):
        calls.append({"task": task, "messages": messages, "max_tokens": max_tokens})
        return _digest_response()

    monkeypatch.setattr(aux, "call_llm", fake_call_llm)
    return calls


# ---------------------------------------------------------------------------
# Digest on eviction
# ---------------------------------------------------------------------------

class TestDigestOnEviction:
    def test_digest_generated_and_retained(self, mock_digest_llm):
        engine = _make_engine()
        provider = _FakeProvider()
        engine.configure(memory_manager=_FakeMM([provider]))
        msgs = _conversation(25)  # 3 + 50 = 53 msgs; middle = 53-3-14 = 36

        out = engine.compress(msgs)

        assert len(out) == 3 + 1 + 14
        marker = out[3]
        assert marker["role"] == "user"
        assert "session_digest" in marker["content"]
        # aux call used the compression task and rolled no previous digest
        assert mock_digest_llm[0]["task"] == "compression"
        assert "(none — this is the first digest)" in mock_digest_llm[0]["messages"][0]["content"]
        # retained with kind + metadata
        assert len(provider.docs) == 1
        assert provider.docs[0]["kind"] == "session_digest"
        assert provider.docs[0]["metadata"]["evicted_turns"] == 36
        assert engine._last_digest.startswith("GOAL: keep testing")
        assert engine.compression_count == 1

    def test_second_eviction_rolls_previous_digest_forward(self, mock_digest_llm):
        engine = _make_engine()
        engine.configure(memory_manager=_FakeMM([_FakeProvider()]))
        engine.compress(_conversation(25))
        engine.compress(_conversation(30))

        assert "GOAL: keep testing" in mock_digest_llm[1]["messages"][0]["content"]
        assert "(none — this is the first digest)" not in mock_digest_llm[1]["messages"][0]["content"]

    def test_digest_failure_still_evicts(self, monkeypatch):
        import agent.auxiliary_client as aux

        def boom(**kw):
            raise RuntimeError("aux down")

        monkeypatch.setattr(aux, "call_llm", boom)
        engine = _make_engine()
        provider = _FakeProvider()
        engine.configure(memory_manager=_FakeMM([provider]))
        msgs = _conversation(25)

        out = engine.compress(msgs)

        assert len(out) == 3 + 1 + 14
        assert "moved to long-term memory" in out[3]["content"]
        # 2026-08-25 contract change: a failed/empty LLM digest falls through
        # to the deterministic CPU digest instead of leaving a state hole —
        # and the fallback digest retains like any other.
        assert "--- end session digest ---" in out[3]["content"]
        assert len(provider.docs) == 1
        assert provider.docs[0]["kind"] == "session_digest"
        assert engine.compression_count == 1

    def test_unretained_digest_is_embedded_inline(self, mock_digest_llm):
        engine = _make_engine()
        engine.configure(memory_manager=_FakeMM([_FakeProvider(ok=False)]))
        out = engine.compress(_conversation(25))

        assert "Summary of the moved turns" in out[3]["content"]
        assert "GOAL: keep testing" in out[3]["content"]

    def test_no_memory_manager_still_evicts(self, mock_digest_llm):
        engine = _make_engine()  # configure() never called with a manager
        out = engine.compress(_conversation(25))
        assert len(out) == 3 + 1 + 14
        # digest generated but not retainable -> embedded inline
        assert "Summary of the moved turns" in out[3]["content"]

    def test_digest_disabled_by_config(self, mock_digest_llm):
        engine = _make_engine()
        engine.configure(config={"context": {"decay": {"digest_enabled": False}}})
        out = engine.compress(_conversation(25))
        assert mock_digest_llm == []
        assert "moved to long-term memory" in out[3]["content"]


# ---------------------------------------------------------------------------
# Tail pruning
# ---------------------------------------------------------------------------

class TestTailPruning:
    def test_over_budget_tail_is_pruned_without_eviction(self):
        # n == h + t + 1 -> no eviction possible; giant tool bodies must shrink.
        engine = _make_engine(threshold_percent=0.06, config_context_length=65536)
        # budget = max(65536*0.06, 4000) = 4000 tokens; each body ~4.5k chars ≈ 1.4k tok,
        # so the first trim pass (recent 2 kept verbatim) lands back under budget.
        msgs = _conversation(5, body_size=4500)  # 3 + 15 = 18 = 3+14+1
        assert len(msgs) == 18

        out = engine.compress(msgs)

        assert out is not msgs
        assert engine.compression_count == 1
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        # the two most recent tool results stay verbatim
        assert len(tool_msgs[-1]["content"]) > 4000
        assert len(tool_msgs[-2]["content"]) > 4000
        # older ones got trimmed
        assert any("chars trimmed" in m["content"] for m in tool_msgs[:-2])
        # trim was surfaced to the model
        assert any("[Context note:" in str(m.get("content")) and "trimmed" in str(m["content"])
                   for m in out if m.get("role") == "user")
        # result fits the budget now
        assert _est_tokens(out) <= max(engine.threshold_tokens, 4000)

    def test_tool_pairs_stay_valid_after_pruning(self):
        engine = _make_engine(threshold_percent=0.06, config_context_length=65536)
        msgs = _conversation(7, body_size=12000)
        out = engine.compress(msgs)

        resp_ids = {m.get("tool_call_id") for m in out
                    if m.get("role") == "tool" and m.get("tool_call_id")}
        req_ids = {tc.get("id") for m in out for tc in (m.get("tool_calls") or []) if tc.get("id")}
        assert resp_ids <= req_ids  # no orphan tool responses

    def test_repeated_compress_keeps_making_progress(self):
        engine = _make_engine(threshold_percent=0.06, config_context_length=65536)
        msgs = _conversation(20, body_size=12000)  # eviction + pruning
        out1 = engine.compress(msgs)
        est1 = _est_tokens(out1)
        out2 = engine.compress(out1)
        # second pass must not grow and must not raise; either it fits now or shrank
        assert _est_tokens(out2) <= est1
        assert _est_tokens(out2) <= max(engine.threshold_tokens, 4000)

    def test_eviction_then_prune_marks_both(self, mock_digest_llm):
        engine = _make_engine(threshold_percent=0.06, config_context_length=65536)
        engine.configure(memory_manager=_FakeMM([_FakeProvider()]))
        msgs = _conversation(20, body_size=12000)
        out = engine.compress(msgs)
        marker = out[3]["content"]
        assert "moved to long-term memory" in marker
        assert "trimmed" in marker


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class TestEstimator:
    def test_json_estimates_higher_than_prose(self):
        jsonish = '{"key": [1, 2, 3], "nested": {"a": true}} ' * 100
        prose = ("The quick brown fox jumps over the lazy dog. " * 120)[:len(jsonish)]
        assert len(prose) == len(jsonish)
        assert _text_tokens(jsonish) > _text_tokens(prose) * 1.18

    def test_empty_and_short(self):
        assert _text_tokens("") == 0
        assert _text_tokens("hi") >= 1

    def test_message_list_counts_tool_calls(self):
        msgs = [{"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}" * 200}}]}]
        assert _est_tokens(msgs) > 50


# ---------------------------------------------------------------------------
# configure() hook
# ---------------------------------------------------------------------------

class TestConfigure:
    def test_config_overrides_reach_engine(self):
        engine = _make_engine()
        engine.configure(config={"context": {"decay": {
            "threshold_percent": 0.60,
            "protect_last_n": 10,
            "protect_first_n": 2,
            "digest_max_tokens": 500,
        }}})
        assert engine.threshold_percent == 0.60
        assert engine.threshold_tokens == max(int(65536 * 0.60), 4000)
        assert engine.protect_last_n == 10
        assert engine.protect_first_n == 2
        assert engine._digest_max_tokens == 500

    def test_memory_manager_stored(self):
        engine = _make_engine()
        mm = _FakeMM([_FakeProvider()])
        engine.configure(memory_manager=mm)
        assert engine._memory_manager is mm

    def test_empty_config_keeps_defaults(self):
        engine = _make_engine()
        engine.configure(config={})
        assert engine.threshold_percent == 0.75
        assert engine.protect_last_n == 14

    def test_session_reset_clears_digest(self, mock_digest_llm):
        engine = _make_engine()
        engine.configure(memory_manager=_FakeMM([_FakeProvider()]))
        engine.compress(_conversation(25))
        assert engine._last_digest is not None
        engine.on_session_reset()
        assert engine._last_digest is None


class TestDigestTailOverride:
    """2026-08-24 regression: a 3k outer-capped tail serialization dropped a
    credential resolution into the elided middle, and the digest reported the
    task as still-failing AFTER it succeeded.  The tail must keep its NEWEST
    turns, and the digest prompt must treat the tail as overriding the span.
    """

    def test_serialize_tail_keeps_newest_when_over_budget(self):
        msgs = [
            {"role": "user", "content": f"turn-{i:02d} " + "x" * 300}
            for i in range(20)
        ]
        out = DecayEngine._serialize_for_digest(msgs, max_chars=1500, keep="tail")
        assert "turn-19" in out and "turn-18" in out
        assert "turn-00" not in out
        assert "elided from the start" in out

    def test_serialize_default_keeps_outer(self):
        """Evicted-span serialization is unchanged: head + tail of the span."""
        msgs = [
            {"role": "user", "content": f"turn-{i:02d} " + "x" * 300}
            for i in range(20)
        ]
        out = DecayEngine._serialize_for_digest(msgs, max_chars=1500)
        assert "turn-00" in out and "turn-19" in out
        assert "elided from the middle" in out

    def test_digest_prompt_marks_tail_as_overriding(self, mock_digest_llm):
        engine = _make_engine()
        engine.configure(memory_manager=_FakeMM([_FakeProvider()]))
        engine.compress(_conversation(25))
        prompt = mock_digest_llm[0]["messages"][0]["content"]
        assert "TAIL WINS" in prompt
        assert "NEWER than the evicted span" in prompt
        tail_section = prompt.split("=== KEPT RECENT TAIL")[1]
        assert "did thing 24" in tail_section  # newest protected turn survives


class TestEstimatorCalibration:
    """2026-08-24: the static estimator runs 1.3-1.7x low on tool-JSON/code
    (measured against the live brain's /tokenize), which let real context
    exceed the physical window before estimated thresholds fired.  The engine
    now learns a rolling multiplier from server-reported prompt_tokens.
    """

    def test_learns_multiplier_from_real_counts(self, mock_digest_llm):
        engine = _make_engine()
        engine.configure(memory_manager=_FakeMM([_FakeProvider()]))
        msgs = _conversation(25, body_size=400)  # big enough for the >=2000-est guard
        raw_est = _est_tokens(msgs)
        assert raw_est > 2000
        assert engine._est_cal == 1.0
        engine.compress(msgs, current_tokens=int(raw_est * 1.7))
        assert engine._est_cal > 1.15            # EMA moved toward 1.7
        first = engine._est_cal
        engine.compress(_conversation(25, body_size=400), current_tokens=int(raw_est * 1.7))
        assert engine._est_cal > first           # keeps converging

    def test_caltok_applies_multiplier(self):
        engine = _make_engine()
        engine._est_cal = 1.5
        msgs = _conversation(4)
        assert engine._caltok(msgs) == int(_est_tokens(msgs) * 1.5)

    def test_guards_reject_absurd_and_tiny_samples(self):
        engine = _make_engine()
        before = engine._est_cal
        engine._learn_est_cal(10**9, _conversation(25))          # absurd ratio
        engine._learn_est_cal(100, [{"role": "user", "content": "hi"}])  # tiny est
        engine._learn_est_cal(0, _conversation(25))              # no actual
        assert engine._est_cal == before

    def test_cal_persists_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        engine = _make_engine()
        engine._est_cal = 1.61
        engine._est_cal_n = 7
        engine._save_est_cal()
        fresh = _make_engine()
        assert fresh._est_cal == 1.61
        assert fresh._est_cal_n == 7

    def test_asymmetric_learning_rate(self):
        """Rises fast (danger), falls slowly (safe) — the operator's 'beta'."""
        engine = _make_engine()
        msgs = _conversation(25, body_size=400)
        raw_est = _est_tokens(msgs)

        # ratio above current cal -> fast alpha (0.25)
        engine._learn_est_cal(int(raw_est * 2.0), msgs)
        rose = engine._est_cal - 1.0
        assert 0.2 < rose < 0.3                     # ~0.25 * (2.0 - 1.0)

        # ratio below current cal -> slow alpha (0.08)
        engine2 = _make_engine()
        engine2._est_cal = 1.6
        engine2._learn_est_cal(int(raw_est * 1.0), msgs)
        fell = 1.6 - engine2._est_cal
        assert 0.03 < fell < 0.07                   # ~0.08 * (1.6 - 1.0)


class TestDigestFallback:
    """2026-08-25: LLM digest seat died (remote card) 2 min after a good
    digest; the timeout path evicted 116 turns with no continuity at all.
    An empty/failed LLM digest must now fall through to the deterministic
    CPU digest."""

    def test_llm_failure_falls_through_to_cpu_digest(self, mock_digest_llm, monkeypatch):
        import agent.auxiliary_client as aux

        def boom(**kw):
            raise TimeoutError("Request timed out.")

        monkeypatch.setattr(aux, "call_llm", boom)  # properly restored by monkeypatch
        engine = _make_engine()
        engine.configure(memory_manager=_FakeMM([_FakeProvider()]))
        out = engine.compress(_conversation(25))
        marker = out[3]
        assert "CONTEXT CARD" in marker["content"]
        assert "OBSERVATION" in marker["content"] and "DECISION POINT" in marker["content"]

    def test_llm_success_still_wins(self, mock_digest_llm):
        engine = _make_engine()
        engine.configure(memory_manager=_FakeMM([_FakeProvider()]))
        engine.compress(_conversation(25))
        assert mock_digest_llm, "LLM digest should be attempted first"
        assert engine._last_digest.startswith("GOAL: keep testing")
