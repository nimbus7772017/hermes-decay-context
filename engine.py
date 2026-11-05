"""Decay context engine — clean eviction + recall instead of lossy summarization.

Fixes the "announce-then-stop" stall. The built-in compressor SUMMARIZES the
conversation when it grows, and the lossy summary severs the agentic tool-loop:
the model reads a digest of its own history, reverts to a chat reply, and stops
(finish_reason=stop, no tool call). This engine instead DROPS the oldest turns —
which are already retained in Hindsight via the memory plugin's sync_turn — and
keeps a coherent recent window VERBATIM, so the tool-loop is never broken. The
agent's existing `recall` tool brings back specifics on demand. No summary, no
flow-break, no thrash (one eviction gets well under budget).

Rolling session digest: before the middle span is evicted it is folded into a
compact state digest (goal / state / decisions / open loops / key facts) by the
auxiliary compression model, and the digest is retained into Hindsight as
kind=session_digest. Raw turns answer "what was that ID?"; the digest answers
"what were we in the middle of and why?". The digest is purely additive — if the
summarizer or the retain path fails, eviction proceeds exactly as before (a
generated-but-unretained digest is embedded in the marker instead, so no state
is silently dropped).

Tail-pruning fallback: eviction alone cannot shrink the protected tail. When
head+marker+tail is still over budget (the "tail full of giant tool results"
failure that used to burn all compression attempts and force a session reset),
the engine trims old tool-result bodies, then drops oldest tail messages, and as
a last resort trims even recent tool bodies — so every compress() call is
guaranteed to make progress and the retry loop can always recover.

Config: set `context.engine: decay` in config.yaml. Optional overrides under
`context.decay:` (protect_last_n, protect_first_n, threshold_percent,
digest_enabled, digest_max_tokens). Overrides reach the engine via the
configure() hook run_agent.py calls after engine selection.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

# 2026-10-13 ctxdyn: last X-Club-Ctx-Limit seen from club-router (process-wide, so a fresh agent per turn
# starts from it instead of the 131072 config floor). (limit, monotonic time)
_ROUTER_CTX = [0, 0.0]
# ctx262k 2026-10-30 (the operator: "work with 262 not 131"): the 15-min TTL made the agent forget the router's
# 262144 after a restart or a quiet spell and fall back to the 131072 floor (early compact 20:37).
# Now: 24 h TTL, last value persisted to <HERMES_HOME>/router_ctx.json and reloaded at import, and the
# config floor itself is 262144. An overflow step-down (update_model with a lower value) still wins.
_ROUTER_CTX_TTL_S = 24 * 3600
_ROUTER_CTX_FILE = "router_ctx.json"


def _router_ctx_path():
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / _ROUTER_CTX_FILE
    except Exception:
        return None


def _router_ctx_save(limit: int) -> None:
    try:
        import json as _j, time as _t
        p = _router_ctx_path()
        if p is None:
            return
        tmp = p.with_suffix(".tmp")
        tmp.write_text(_j.dumps({"limit": int(limit), "wall": _t.time()}), encoding="utf-8")
        import os as _o
        _o.replace(tmp, p)
    except Exception:
        pass


_ROUTER_CTX_LOADED_FROM = [None]


def _router_ctx_load() -> None:
    """Restore the last router-reported card limit across a gateway restart (wall-clock aged,
    same TTL). Lazy and keyed by HERMES_HOME so tests / other profiles never read the live file."""
    try:
        import json as _j, time as _t
        p = _router_ctx_path()
        if p is None or _ROUTER_CTX_LOADED_FROM[0] == str(p):
            return
        _ROUTER_CTX_LOADED_FROM[0] = str(p)
        if not p.exists():
            return
        d = _j.loads(p.read_text(encoding="utf-8"))
        lim, wall = int(d.get("limit") or 0), float(d.get("wall") or 0)
        age = _t.time() - wall
        if lim >= 8192 and 0 <= age < _ROUTER_CTX_TTL_S and not _ROUTER_CTX[0]:
            _ROUTER_CTX[0], _ROUTER_CTX[1] = lim, _t.monotonic() - age
    except Exception:
        pass

# cardfollow2 2026-11-02 (the operator: "compaction follows the card she's on", root fix, not caps).
# The X-Club-Ctx-Limit header is the LARGEST placeable card (262K), so on smaller cards (131K)
# compaction at 0.8 x 262K could never fire before the card filled. Now, before each compaction
# decision, ask the router (/router/status, live per-card ctx_tokens from each llama-server /props+/slots)
# which card this session will land on:
#   1. warm: the card holding this session's most recent slot (kv_policy prefers warm) -> that card's n_ctx
#   2. cold: smallest healthy card that can hold the prompt (+8K reserve) -> its n_ctx ("smallest it could
#      be served on"; tiny cards it cannot fit are excluded, e.g. a 41K card at 100K+)
#   3. router unreachable: the card that served the last reply (X-Club-N-Ctx), else X-Club-Ctx-Limit, else config.
# Hysteresis: no repeat compaction for a session until the prompt grows >= _CF_REGROW tokens past its
# post-compaction size, and a card change is logged once (no flip-flop logging / thrash).
_CF_STATUS = [None, 0.0]            # (status json, monotonic fetched)
_CF_STATUS_TTL_S = 20.0
_CF_SERVED = [None, 0, 0.0]         # last serving backend, its n_ctx, monotonic
_CF_POST = {}                       # session_id -> post-compaction token size
_CF_REGROW = 8192
_CF_RESERVE = 8192
_CF_LAST_LOG = {}                   # session_id -> (card, limit) last logged


def _os_env(k, d=None):
    import os as _o
    return _o.environ.get(k, d)


def _cf_router_conf():
    try:
        import yaml as _y
        from hermes_constants import get_hermes_home
        d = _y.safe_load((get_hermes_home() / "config.yaml").read_text(encoding="utf-8")) or {}
        m = d.get("model") or {}
        base = str(m.get("base_url") or "").rstrip("/")
        # Optional router-aware context sizing (off by default): only when HERMES_DECAY_ROUTER_STATUS=1 and the
        # model endpoint exposes GET /router/status with X-Club-* headers (club-router compatible). Fail-open.
        if not base or str(_os_env("HERMES_DECAY_ROUTER_STATUS", "0")).strip().lower() not in ("1", "true", "yes", "on"):
            return None, None
        if base.endswith("/v1"):
            base = base[:-3]
        return base, str(m.get("api_key") or "")
    except Exception:
        return None, None


def _cf_status():
    import time as _t
    if _CF_STATUS[0] is not None and _t.monotonic() - _CF_STATUS[1] < _CF_STATUS_TTL_S:
        return _CF_STATUS[0]
    base, key = _cf_router_conf()
    if not base:
        return None
    try:
        import json as _j, urllib.request as _u
        req = _u.Request(base + "/router/status", headers={"Authorization": f"Bearer {key}"} if key else {})
        with _u.urlopen(req, timeout=1.5) as r:
            st = _j.loads(r.read().decode("utf-8"))
        _CF_STATUS[0], _CF_STATUS[1] = st, _t.monotonic()
        return st
    except Exception as e:
        logger.debug("DecayEngine cardfollow: /router/status failed: %s", e)
        _CF_STATUS[1] = _t.monotonic() - _CF_STATUS_TTL_S + 5   # retry in ~5 s, not every call
        return _CF_STATUS[0] if _CF_STATUS[0] is not None and _t.monotonic() - _CF_STATUS[1] < 300 else None


def cf_predict(status, session_id, prompt_tokens):
    """-> (card, n_ctx, why) for this session's next request, or (None, 0, why). Pure (testable)."""
    if not isinstance(status, dict):
        return None, 0, "no status"
    cards = {b.get("name"): b for b in status.get("backends") or []
             if b.get("enabled") and b.get("healthy") and int(b.get("ctx_tokens") or 0) >= 8192}
    if not cards:
        return None, 0, "no healthy card"
    pt = int(prompt_tokens or 0)
    if session_id:
        best = None
        for name, c in (((status.get("kv_policy") or {}).get("cards")) or {}).items():
            if name not in cards:
                continue
            for s in c.get("slots") or []:
                if s and s.get("session_id") == session_id and int(s.get("tokens") or 0) > 0:
                    age = int(s.get("age_s") or 0)
                    if best is None or age < best[0]:
                        best = (age, name)
        if best and int(cards[best[1]]["ctx_tokens"]) >= max(pt + _CF_RESERVE // 2, 65536):
            return best[1], int(cards[best[1]]["ctx_tokens"]), f"warm slot (age {best[0]}s)"
    # cards under 64K (a 41K card) are never the session's limit: Hermes refuses windows < 64K and a
    # 0.8 x 41K threshold would compact a 33K session (thrash). Same as the warm case's sanity floor.
    fit = sorted((int(c["ctx_tokens"]), n) for n, c in cards.items()
                 if int(c["ctx_tokens"]) >= max(pt + _CF_RESERVE, 65536))
    if fit:
        return fit[0][1], fit[0][0], "cold: smallest card that fits"
    n, c = max(cards.items(), key=lambda kv: int(kv[1]["ctx_tokens"]))
    return n, int(c["ctx_tokens"]), "none fits: largest card"


_NEW_MSG_LABEL = __import__("os").environ.get("HERMES_DECAY_NEW_MSG_LABEL", "[New user message]")
_DEFAULT_CTX = 65536  # the agent (typical local model) runs ~64K; safe fallback if probe fails
_MARKER = ("[Context note: {n} earlier turn(s) were moved to long-term memory to keep this "
           "conversation focused. Nothing was lost — they are retained in memory. Ask me to "
           "recall anything specific from earlier if you need it.]")
_MARKER_DIGEST = ("[Context note: {n} earlier turn(s) were moved to long-term memory to keep this "
                  "conversation focused. A rolling session digest was written for review at:\n"
                  "  {path}\n"
                  "and retained as kind=session_digest. Recall session digest for earlier work, "
                  "or open that file. Nothing was lost.]\n\n"
                  "--- session digest (compact review) ---\n"
                  "{digest}\n"
                  "--- end session digest ---")
_TRIM_NOTE = ("\n[Additional note: some older tool outputs in the recent window were trimmed to "
              "fit the context budget. Re-run the tool or re-read the file if you need the "
              "full output.]")
_TRIM_MARKER = ("[Context note: older tool outputs in this conversation were trimmed to fit the "
                "context budget. Re-run the tool or re-read the file if you need the full output.]")
_DIGEST_PROMPT = """You are writing a continuity LEDGER for an AI agent whose oldest conversation turns were just evicted to long-term memory. She will read this ledger plus the recent tail that survived — the tail is the present and always overrides this ledger. The ledger's job is to record, not to direct: facts and seams age well, commands written by the past do not. NEVER prescribe what she should do next — choosing the next action is her job, from the live tail plus the seam you record.

Rules:
- Plain text, these exact sections: WORKING GOAL / STATE AT EVICTION / DECISIONS MADE / OPEN LOOPS / KEY FACTS & IDS / LAST SEAM
- The evicted span may cover MULTIPLE tasks. Weight the CURRENT one — the task the KEPT RECENT TAIL is in — heavily.
- Older, completed tasks from the span: at most one line each under a "CLOSED EARLIER:" line (one line, outcome only, e.g. "CLOSED EARLIER: Zabbix audit — 558 unreachable hosts found, report delivered"). Never give them GOAL/OPEN LOOPS billing.
- WORKING GOAL: the ONE task actually in flight at the END of the evicted span (the active task at eviction), one line, framed "as of eviction: ...". State only that task — no history, no "first X, later shifted to Y" story, no list of earlier topics (those belong in SPAN COVERAGE). Never assert it as the present goal — the live tail may have moved on.
- STATE AT EVICTION: a factual SNAPSHOT of where things stood when the span ended — current status of each live item (sites/hosts, readings like voltages/counts, tools built and their paths, open findings, what just failed). Short "item: status" clauses, NOT a narrative; never retell or reorder the day (no "earlier...", "then...", "later..."). Chronology lives only in SPAN COVERAGE.
  RECENCY RULE: the END of the evicted span is the newest state. If something was
  broken early in the span but FIXED/diagnosed by the end, STATE AT EVICTION must say
  the RESOLVED state (and note the fix), never the stale broken one. Latest wins.
- TAIL OVERRIDE: the KEPT RECENT TAIL (when present) happened AFTER the evicted
  span. If it contradicts or resolves anything in the span — an auth that later
  succeeded, a task that finished, a decision that changed — the tail wins:
  STATE AT EVICTION / KEY FACTS & IDS / OPEN LOOPS must state the resolved state and
  must never list as failed/broken something the tail shows working.
- OPERATOR DOMINANCE: any message from the operator inside the evicted span is the
  highest-value content in it — his one-line course corrections decide what the
  work IS. Scan the span for his messages FIRST; WORKING GOAL / STATE AT EVICTION /
  OPEN LOOPS / LAST SEAM must absorb his latest instruction, and if he rescinded or
  redirected something, the card must show the NEW position, never the rescinded one.
- ANTI-RE-ISSUE: never ship a card that is unchanged from the previous card while
  the span contained unabsorbed operator messages. If the operator spoke in the span,
  something on this card MUST move. A re-issued card means you dropped him.
- DECISIONS MADE: choices already locked in (do not relitigate). Each decision MUST carry its one-clause "because <why>" — the span contains her reasoning; the rationale is the most valuable trace in the span and the first thing a lazy summary drops. A decision without its why is fragile; a decision with it survives the seam.
- OPEN LOOPS: anything unfinished or promised, each marked "(as of eviction)". Finished sweeps/audits go under CLOSED EARLIER, never here.
- PARKED: (optional, one line) items the operator explicitly put on hold, deferred or said "later"/"not now" about — e.g. "PARKED: Elk Stream Ranch Su/Boone VISP lookups (on hold per the operator)". Parked items are NOT open loops; do not repeat them under OPEN LOOPS.
- PROOF RULE: never write "resolved", "done", "closed" or "complete" (in any section, CLOSED EARLIER included) unless the span shows the proof, and put it in the line as "(proof: <what shows it — tool result, count, id, file, the operator's confirmation>)", e.g. "CLOSED EARLIER: Oct 2-3 cancellations — 14 accounts disabled (proof: audit CSV sent 10/03)". If the span only claims it without evidence, write "reported done, unverified" and keep it in OPEN LOOPS. The tail-override and recency rules still apply, but they need proof too.
- KEY FACTS & IDS: paths, IPs, names, numbers, versions worth keeping — as compact "k: v" lines, NO raw JSON. Current-task facts first.
  VERBATIM ONLY: every value in KEY FACTS & IDS (emails, IDs, paths, IPs, numbers) must be copied character-for-character from the text below. NEVER infer, guess, or construct a value (no "inferred from context"). If a value is not in the text, write "unknown" or omit the line.
- LAST SEAM: what was literally in flight when the span was evicted — the pending tool call, the file being written, the question awaiting an answer, the approval waiting on the operator. One or two lines, purely descriptive. This is the "continue from here" anchor; it must NEVER contain a recommendation.
- SPAN COVERAGE (2026-10-31, REQUIRED): the evicted span is usually HOURS of work across several topics. Add a "SPAN COVERAGE:" section right after WORKING GOAL listing EVERY major topic/task in the evicted span in STRICT CHRONOLOGICAL order, OLDEST FIRST, newest last (never reverse it), one line each: "~HH:MM–HH:MM <topic> — <outcome / key facts / what the operator said>". Older topics are NOT less important just because they are older — they are about to exist nowhere else in her context. Never drop a topic the operator spent real time on (interviews, drafts, applications, personal facts he shared).
- NO TAIL REHASH: the KEPT RECENT TAIL is still verbatim in her context. Use it ONLY to resolve/override span facts. Do NOT restate tail content (its results, numbers, patches, replies) — spend the budget on the EVICTED span. If the span's last task continues into the tail, one line saying so is enough.
- When the evicted span is given as CHUNK NOTES (chunked pre-summaries in time order), treat them as the span: every chunk must be represented in SPAN COVERAGE.
- Max 2400 chars total. No filler, no apologies, no meta-commentary."""

_DIGEST_CHUNK_PROMPT = """You are pre-summarizing ONE chunk (part {i} of {n}, in time order) of an AI agent's conversation with the operator that is being evicted from her context. Write compact plain-text notes, max 900 chars:
- TOPICS: one line per distinct topic/task in this chunk: "~HH:MM <topic> — <what the operator asked/said> — <outcome>". Use times from "[Current time: ...]" stamps when present.
- FACTS: key facts worth keeping as "k: v" (names, numbers, IPs, paths, dates, decisions + because). Copy values character-for-character; never invent. Never copy passwords or secrets — write "<secret given>" instead.
- OPEN: anything left unfinished or promised at the end of this chunk (or "none").
No filler, no advice, no meta-commentary."""

# evicted-span serialization caps for the LLM digest path
_DIGEST_LLM_MAX_CHARS = 16000
_DIGEST_LLM_TIMEOUT_S = 90
# 2026-10-31: chunked digest — spans larger than _DIGEST_LLM_MAX_CHARS are no
# longer truncated to the outer 8k+8k (that dropped hours of mid-span work and
# left a card that re-described the kept tail). Instead the whole span is cut
# into time-ordered chunks, each pre-summarized, then merged by the card pass.
_DIGEST_CHUNK_CHARS = 24000      # serialized chars per chunk (~7k tokens; 9B seat has 128k ctx)
_DIGEST_CHUNK_PER_MSG = 900      # per-message cap inside a chunk
_DIGEST_MAX_CHUNKS = 8           # chunk size grows past this rather than adding calls
_DIGEST_CHUNK_WORKERS = 2        # seat runs --parallel 2


def _digest_line_cap(text: str, max_line: int = 400) -> str:
    """Per-line cap: one runaway line must never starve the rest of the card
    (2026-08-28: a degenerate repetition line ate c10's entire budget — no
    OPEN LOOPS, no KEY FACTS, no LAST SEAM survived)."""
    out = []
    for ln in (text or "").splitlines():
        out.append(ln if len(ln) <= max_line else ln[:max_line] + " …")
    return "\n".join(out)


def _digest_degenerate(text: str) -> Optional[str]:
    """Detect degeneration in an LLM-written digest. Returns a reason string or None.

    2026-08-28 c10: the aux seat emitted 'proprietary ' x several hundred — a
    repetition loop that also hallucinated a fact contradicting the agent's own
    earlier (correct) analysis. A degenerate card is worse than the CPU fallback:
    it reads authoritative while being garbage.
    """
    if not text:
        return "empty"
    words = text.split()
    if len(words) < 12:
        return None  # too short to judge; short cards are legal
    # consecutive-identical-word loop
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run > 8:
            return "consecutive-word loop (%r x%d)" % (words[i], run)
    # single-word dominance (catches non-consecutive loops too)
    from collections import Counter
    top_word, top_n = Counter(words).most_common(1)[0]
    if top_n / len(words) > 0.25:
        return "single-word dominance (%r = %.0f%% of card)" % (top_word, 100 * top_n / len(words))
    return None


_VF_TOKEN_RES = [
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),                 # email
    re.compile(r"(?:~|\.{1,2})?/[\w.@~+-]+(?:/[\w.@~+-]+)+/?"),   # path
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"),            # IP[:port]
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),  # UUID
    re.compile(r"\b\d{8}_\d{6}_[0-9a-f]{4,}\b"),                     # session id
]


def _verify_key_facts(digest: str, middle, recent, prior: str = "") -> str:
    """Deterministic post-check (2026-10-11: 9B seat wrote
    'recipient_email: user@example.com (inferred from context)' — the
    real address had been elided from its input).  In the KEY FACTS & IDS
    section, drop any fact whose email/path/IP/UUID/session-id does not appear
    verbatim in the RAW evicted span, kept tail, or prior digest, and any fact
    self-labelled as inferred/guessed.  Never raises."""
    try:
        src_parts = [prior or ""]
        for m in list(middle or []) + list(recent or []):
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
            src_parts.append(str(c or ""))
            if m.get("tool_calls"):
                src_parts.append(str(m.get("tool_calls")))
        src = "\n".join(src_parts)
        src_l = src.lower()
        sec = re.search(r"(KEY FACTS(?: & IDS)?:?)(.*?)(?=\n\s*LAST SEAM\b|\Z)", digest, re.S)
        if not sec:
            return digest
        body = sec.group(2)
        # facts are "k: v" separated by ';' or newlines
        pieces = re.split(r"(;|\n)", body)
        kept, dropped = [], []
        for pc in pieces:
            if pc in (";", "\n") or not pc.strip():
                kept.append(pc); continue
            bad = None
            if re.search(r"\b(inferr?ed|guess(ed)?|assumed|presumably|likely)\b", pc, re.I):
                bad = "self-labelled inference"
            else:
                for rx in _VF_TOKEN_RES:
                    for tok in rx.findall(pc):
                        t = tok.rstrip(".,)`'\"")
                        if t and t.lower() not in src_l:
                            bad = "not verbatim in input: %r" % t; break
                    if bad:
                        break
            if bad:
                dropped.append((pc.strip(), bad))
            else:
                kept.append(pc)
        if not dropped:
            return digest
        for f, why in dropped:
            logger.warning("DecayEngine: KEY FACTS post-check dropped %r (%s)", f[:200], why)
        new_body = re.sub(r"(;\s*){2,}", "; ", "".join(kept))
        new_body = re.sub(r"^[;\s]*", " ", new_body).rstrip(" ;") + "\n"
        return digest[:sec.start(2)] + new_body + digest[sec.end(2):].lstrip(" ")
    except Exception as e:  # never break eviction
        logger.info("DecayEngine: KEY FACTS post-check failed (%s) — digest unchanged", e)
        return digest


# Digest LLM is wired from register(ctx) via ctx.llm (public PluginContext
# surface). Directory-engine installs that skip register() keep _DIGEST_LLM
# unset and fall through to the CPU digest — never import agent.auxiliary_client
# (catalog rule 9).
_DIGEST_LLM = None  # callable matching call_llm(task=, messages=, max_tokens=, timeout=)
_DIGEST_LLM_TASK = "decay_digest"  # plugin-owned aux slot; inherit_from=compression


def wire_digest_llm(ctx, *, task_key: str | None = _DIGEST_LLM_TASK) -> None:
    """Bind the rolling-digest seat to ``ctx.llm`` (public). Idempotent."""
    global _DIGEST_LLM
    llm = getattr(ctx, "llm", None)
    if llm is None or not hasattr(llm, "complete"):
        return

    def _call(*, task=None, messages, max_tokens=None, timeout=None, **kw):
        # Call sites still pass task="compression" (legacy aux name). Route
        # through the plugin-owned slot when registered; otherwise the user's
        # active main model (omit task=).
        call_kw: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "purpose": "decay.session_digest",
        }
        if task_key:
            call_kw["task"] = task_key
        return llm.complete(**call_kw)

    _DIGEST_LLM = _call


def set_digest_llm_callable(fn) -> None:
    """Test/host seam: inject a digest callable without touching Hermes internals."""
    global _DIGEST_LLM
    _DIGEST_LLM = fn


def _llm_digest_callable():
    """Return the wired digest callable, or raise so callers fall back to CPU."""
    if _DIGEST_LLM is None:
        raise RuntimeError(
            "digest LLM not wired — call wire_digest_llm(ctx) from register(), "
            "or set_digest_llm_callable() in tests"
        )
    return _DIGEST_LLM


def _llm_extract(resp: Any) -> str:
    """Normalize ctx.llm results and OpenAI-shaped responses to plain text."""
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""

# Tail-pruning knobs
_TOOL_KEEP_RECENT = 2     # tool results kept verbatim in the first trim pass
_TOOL_BODY_CAP = 400      # chars kept from the start of a trimmed tool result
_TOOL_BODY_TAIL = 200     # chars kept from the end of a trimmed tool result
_MIN_TAIL = 2             # never drop below this many recent messages


def _probe_ctxlen(model: str, **kw) -> int | None:
    """Best-effort model context-length lookup; None if unavailable."""
    if not model:
        return None
    for path in ("agent.context_compressor", "agent.model_utils", "agent.tokens", "agent.token_utils"):
        try:
            mod = __import__(path, fromlist=["get_model_context_length"])
            fn = getattr(mod, "get_model_context_length", None)
            if fn:
                allowed = {k: v for k, v in kw.items()
                           if k in ("base_url", "api_key", "provider", "config_context_length")}
                return int(fn(model, **allowed))
        except Exception:
            continue
    return None


def _text_tokens(s: str) -> int:
    """Token estimate for one string, aware of punctuation density.

    A flat 4-chars-per-token rule is ~20% optimistic on JSON/code/tool output
    (braces, quotes and separators tokenize poorly). Sample the text and use a
    denser ratio when the punctuation share is high.
    """
    if not s:
        return 0
    n = len(s)
    sample = s[:4000]
    punct = sum(1 for ch in sample if not ch.isalnum() and not ch.isspace())
    cpt = 3.3 if (punct / max(len(sample), 1)) > 0.12 else 4.0
    return int(n / cpt) + 1


def _est_tokens(msgs: List[Dict[str, Any]]) -> int:
    """Cheap token estimate over serialized message content."""
    total = 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
        total += _text_tokens(str(c or ""))
        for tc in (m.get("tool_calls") or []):
            total += _text_tokens(str(tc))
    return total


def _is_marker(m: Dict[str, Any]) -> bool:
    c = m.get("content")
    return isinstance(c, str) and c.startswith("[Context note:")


class DecayEngine(ContextEngine):
    """Drops old turns (kept in Hindsight) instead of summarizing them."""

    def __init__(self, *args, **kwargs):
        cfg = kwargs.get("config") or {}
        dcfg = {}
        try:
            dcfg = (cfg.get("context", {}) or {}).get("decay", {}) or {}
        except Exception:
            dcfg = {}
        self.threshold_tokens_cap = int(dcfg.get("threshold_tokens_cap", 0) or 0)   # lightchat pilot; 0 = off
        self.protect_first_n = int(kwargs.get("protect_first_n", dcfg.get("protect_first_n", 3)))
        self.protect_last_n = int(kwargs.get("protect_last_n", dcfg.get("protect_last_n", 14)))
        self.threshold_percent = float(kwargs.get("threshold_percent", dcfg.get("threshold_percent", 0.75)))
        self.model = kwargs.get("model", "") or ""
        self.context_length = int(kwargs.get("config_context_length") or _DEFAULT_CTX)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        # Estimator calibration: _text_tokens runs 1.3-1.7x LOW on
        # tool-JSON/code-heavy turns (measured against the live brain's
        # /tokenize 2026-08-24), which let real context exceed the physical
        # window before estimated thresholds fired.  Learn a rolling multiplier
        # from server-reported prompt_tokens each time compress() runs with
        # real counts, and apply it to internal estimate-based checks.
        self._est_cal = 1.0
        self._est_cal_n = 0
        self._load_est_cal()
        # Rolling digest state
        self._memory_manager = None
        self._last_digest: Optional[str] = None
        self._last_digest_path: str = ""
        self._digest_enabled = bool(dcfg.get("digest_enabled", True))
        self._digest_max_tokens = int(dcfg.get("digest_max_tokens", 700))
        self._digest_llm = bool(dcfg.get("digest_llm", True))
        # 2026-08-18: resolve context length with FULL reliability. The plugin
        # loader passes no config to __init__, configure() receives only the
        # agent: section, and run_agent's update_model() then stomps whatever
        # we have with the auto-probe (which reads 32768/65536 for some local
        # alias). So: read context.decay.context_length from the config FILE
        # ourselves (profile-aware), and let _ctx_from_config shield it from
        # update_model(). Last-ditch: probe the llama server /props n_ctx.
        _ctx = self._ctx_from_config_file()
        if _ctx > 0:
            self.context_length = _ctx
            self._ctx_from_config = True
        else:
            self._ctx_from_config = False
        # threshold_percent has the same plumbing gap — file wins when the
        # kwargs/configure path never delivered it.
        if "threshold_percent" not in kwargs:
            try:
                import os
                import yaml
                from hermes_constants import get_hermes_home
                _data = yaml.safe_load(
                    (get_hermes_home() / "config.yaml").read_text(encoding="utf-8")) or {}
                _tp = float((((_data.get("context") or {}).get("decay") or {})).get("threshold_percent", 0) or 0)
                if _tp > 0:
                    self.threshold_percent = _tp
            except Exception:
                pass
        self._recompute()

    # -- identity ----------------------------------------------------------
    @property
    def name(self) -> str:
        return "decay"

    @property
    def session_digest(self) -> str:
        """The current rolling session digest ("" if none has been built yet).

        Public accessor for the post-compact briefing hook in run_agent.py
        (``build_post_compact_briefing``) so it can re-inject the state of
        the just-evicted work without reaching into private state.
        """
        return self._last_digest or ""

    def is_available(self) -> bool:
        return True

    # -- wiring ------------------------------------------------------------
    @staticmethod
    def _ctx_from_config_file() -> int:
        """Read context.decay.context_length from the hermes config file.

        Self-contained on purpose: the plugin loader gives __init__ no config
        and run_agent's configure() hook only forwards the agent: section, so
        context.decay.* never reaches us through those channels (2026-08-18:
        this gap is why the laptop thrash-compacted at ~46k despite the knob).
        """
        try:
            import os
            import yaml
            from hermes_constants import get_hermes_home
            cfg_path = get_hermes_home() / "config.yaml"
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            decay = (data.get("context", {}) or {}).get("decay", {}) or {}
            v = int(decay.get("context_length", 0) or 0)
            return v if v > 0 else 0
        except Exception:
            return 0

    def configure(self, config: dict = None, memory_manager: Any = None, **kwargs) -> None:
        """Late-binding hook called by run_agent.py after engine selection.

        The plugin loader instantiates engines with no arguments, so config and
        the live memory provider only reach the engine through this hook.
        """
        if memory_manager is not None:
            self._memory_manager = memory_manager
        dcfg = {}
        try:
            dcfg = ((config or {}).get("context", {}) or {}).get("decay", {}) or {}
        except Exception:
            dcfg = {}
        if dcfg:
            self.protect_first_n = int(dcfg.get("protect_first_n", self.protect_first_n))
            self.protect_last_n = int(dcfg.get("protect_last_n", self.protect_last_n))
            self.threshold_percent = float(dcfg.get("threshold_percent", self.threshold_percent))
            self._digest_enabled = bool(dcfg.get("digest_enabled", self._digest_enabled))
            self._digest_max_tokens = int(dcfg.get("digest_max_tokens", self._digest_max_tokens))
            self.threshold_tokens_cap = int(dcfg.get("threshold_tokens_cap", getattr(self, "threshold_tokens_cap", 0)) or 0)
            # 2026-08-16: aux-LLM digests are safe again — the compression seat
            # (7B) lives off-box on a remote host, so a digest
            # call never touches the agent's GPU. CPU heuristic stays as fallback.
            self._digest_llm = bool(dcfg.get("digest_llm", self._digest_llm))
            # 2026-08-18: explicit context length for threshold math. Nothing in
            # run_agent calls set_context_length, so without this the engine ran
            # at _DEFAULT_CTX (65536) forever — on the 131072-window laptop that
            # compacted every few minutes once a dig passed ~46k tokens (thrash).
            _ctx = int(dcfg.get("context_length", 0) or 0)
            if _ctx > 0:
                self.context_length = _ctx
                self._ctx_from_config = True
            self._recompute()
        # cardfollow 2026-10-30 (the operator: "follow the card she's running on"): config is only the
        # fallback; a live/persisted router card limit wins regardless of hook order.
        self._apply_router_ctx()
        logger.info(
            "DecayEngine configured: threshold=%.2f (%d tok) protect_first=%d protect_last=%d digest=%s",
            self.threshold_percent, self.threshold_tokens, self.protect_first_n,
            self.protect_last_n, self._digest_enabled,
        )

    # -- setup -------------------------------------------------------------
    def _recompute(self) -> None:
        t = int(self.context_length * self.threshold_percent)
        # lightchat pilot (2026-10-16): optional absolute ceiling on the threshold (context.decay.
        # threshold_tokens_cap). Default 0 = off, so every existing profile is unchanged. A cap is
        # used instead of a small context_length because Hermes refuses windows under 64k and the
        # router's X-Club-Ctx-Limit keeps rewriting context_length.
        cap = int(getattr(self, "threshold_tokens_cap", 0) or 0)
        if cap > 0:
            t = min(t, cap)
        self.threshold_tokens = max(t, 4000)

    def set_context_length(self, context_length: int) -> None:
        self.context_length = int(context_length) or _DEFAULT_CTX
        self._recompute()

    def update_model(self, model: str, context_length: int, **kwargs) -> None:
        self.model = model or self.model
        # An explicit context.decay.context_length in config wins over the
        # auto-detected/probed value — run_agent calls update_model AFTER
        # configure() and would otherwise stomp the override back to the probe.
        # ctxdyn 2026-10-13: a LOWER value (context-overflow step-down) is always taken.
        if not getattr(self, "_ctx_from_config", False) or (
                int(context_length or 0) and int(context_length) < self.context_length):
            self.context_length = int(context_length) or _DEFAULT_CTX
            if getattr(self, "_ctx_from_config", False):
                _ROUTER_CTX[0] = 0       # overflow: forget the router's bigger limit until it says so again
                try:
                    p = _router_ctx_path()
                    if p is not None and p.exists():
                        p.unlink()
                except Exception:
                    pass
        self._recompute()

    def set_router_ctx_limit(self, limit: int) -> None:
        """ctxdyn 2026-10-13: club-router's X-Club-Ctx-Limit (largest card n_ctx it can place on now; it never puts
        a request on a card it does not fit). Threshold = threshold_percent x that. The config value stays the
        floor used before any router answer / after the TTL."""
        import time as _t
        try:
            limit = int(limit)
        except Exception:
            return
        if limit < 8192:
            return
        if not getattr(self, "_router_seen_this_turn", False):
            # cardfollow-everyturn: the first router report of each turn (fresh agent per turn)
            # is adopted outright and logged, so a stale saved value never outlives it.
            self._router_seen_this_turn = True
            logger.info("DecayEngine router card this turn: ctx %d (threshold %d; saved was %s)",
                        limit, max(int(limit * self.threshold_percent), 4000), _ROUTER_CTX[0] or "none")
        _new = limit != _ROUTER_CTX[0] or _t.monotonic() - _ROUTER_CTX[1] > 600
        _ROUTER_CTX[0], _ROUTER_CTX[1] = limit, _t.monotonic()
        if _new:
            _router_ctx_save(limit)   # ctx262k: survive restarts (throttled to ~10 min)
        if limit != self.context_length:
            logger.info("DecayEngine router ctx limit: %d -> %d (threshold %d)", self.context_length, limit,
                        max(int(limit * self.threshold_percent), 4000))
            self.context_length = limit
            self._recompute()

    def set_router_card(self, backend, n_ctx) -> None:
        """cardfollow2 2026-11-02: X-Club-Backend / X-Club-N-Ctx of the card that served the last reply.
        Fallback when /router/status is unreachable; a SMALLER serving card is adopted at once (safe side)."""
        import time as _t
        try:
            n_ctx = int(n_ctx)
        except Exception:
            return
        if n_ctx < 8192:
            return
        _CF_SERVED[0], _CF_SERVED[1], _CF_SERVED[2] = backend, n_ctx, _t.monotonic()
        if n_ctx < self.context_length:
            logger.info("DecayEngine cardfollow: served on %s n_ctx=%d < %d -> threshold %d",
                        backend, n_ctx, self.context_length, max(int(n_ctx * self.threshold_percent), 4000))
            self.context_length = n_ctx
            self._recompute()

    def _apply_router_ctx(self) -> None:
        import time as _t
        _router_ctx_load()
        if (_ROUTER_CTX[0] and _t.monotonic() - _ROUTER_CTX[1] < _ROUTER_CTX_TTL_S
                and self.context_length != _ROUTER_CTX[0]):
            self.context_length = _ROUTER_CTX[0]
            self._recompute()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        # digestkey 2026-10-31: set thread key/session id FIRST so a failure in the
        # ctx probe/router lookup below cannot leave digests keyed "session".
        try:
            self.session_id = session_id or getattr(self, "session_id", None)
            _gsk0 = str(kwargs.get("gateway_session_key") or "").strip()
            if _gsk0:
                self._digest_thread_key = _gsk0
        except Exception:
            pass
        self.model = kwargs.get("model") or self.model
        probed = _probe_ctxlen(self.model, **kwargs)
        # ctx262k: an explicit config context_length is the floor; a model-metadata probe
        # (e.g. generic "qwen" -> 131072) must not silently shrink it.
        if probed and not (getattr(self, "_ctx_from_config", False) and int(probed) < self.context_length):
            self.context_length = probed
        self._apply_router_ctx()
        self._last_digest = None
        self._last_digest_path = ""
        # fix3 2026-10-30: digests are keyed per thread (gateway session key), not "session"
        self.session_id = session_id or getattr(self, "session_id", None)
        _gsk = str(kwargs.get("gateway_session_key") or "").strip()
        if _gsk:
            self._digest_thread_key = _gsk
        self._recompute()
        logger.info("DecayEngine on_session_start: model=%s ctx=%d threshold=%d protect_last_n=%d",
                    self.model or "?", self.context_length, self.threshold_tokens, self.protect_last_n)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._last_digest = None
        self._last_digest_path = ""

    # -- token tracking ----------------------------------------------------
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens",
                                           self.last_prompt_tokens + self.last_completion_tokens)

    def _card_follow(self, pt) -> None:
        """cardfollow2 2026-11-02: set context_length/threshold from the card this session will be served on."""
        import time as _t
        sid = getattr(self, "session_id", None)
        card, lim, why = cf_predict(_cf_status(), sid, pt)
        if not lim and _CF_SERVED[1] and _t.monotonic() - _CF_SERVED[2] < 900:
            card, lim, why = _CF_SERVED[0], _CF_SERVED[1], "last serving card (router status unavailable)"
        if not lim:
            self._apply_router_ctx()        # old path: X-Club-Ctx-Limit / config floor
            return
        if lim != self.context_length:
            self.context_length = lim
            self._recompute()
        _ROUTER_CTX[0], _ROUTER_CTX[1] = lim, _t.monotonic()   # gateway hygiene follows the same card
        key = sid or "?"
        if _CF_LAST_LOG.get(key) != (card, lim):
            _CF_LAST_LOG[key] = (card, lim)
            logger.info("DecayEngine cardfollow: session=%s card=%s n_ctx=%d threshold=%d prompt=%s (%s)",
                        key, card, lim, self.threshold_tokens, pt, why)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        pt = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        try:
            self._card_follow(pt)
        except Exception as e:
            logger.debug("DecayEngine cardfollow failed: %s", e)
            self._apply_router_ctx()
        if pt is None or pt < self.threshold_tokens:
            return False
        post = _CF_POST.get(getattr(self, "session_id", None) or "?")
        if post and pt < post + _CF_REGROW and pt < self.context_length - _CF_RESERVE:
            logger.info("DecayEngine cardfollow: skip repeat compaction (prompt %d < post-compact %d + %d)",
                        pt, post, _CF_REGROW)
            return False
        return True

    # -- estimator calibration ------------------------------------------------
    _EST_CAL_ALPHA = 0.25      # EMA rate when the ratio says "estimator too low" (danger side: learn fast)
    _EST_CAL_ALPHA_DOWN = 0.08 # and when it says "too high" (safe side: sag slowly — the 'beta')
    _EST_CAL_MIN = 0.8         # clamps: never trust an insane sample
    _EST_CAL_MAX = 2.5

    def _est_cal_path(self):
        from pathlib import Path as _P
        import os as _os
        return _P(_os.environ.get("HERMES_HOME", str(_P.home() / ".hermes"))) / "decay_est_cal.json"

    def _load_est_cal(self) -> None:
        try:
            import json as _json
            d = _json.loads(self._est_cal_path().read_text())
            cal = float(d.get("cal", 1.0))
            if self._EST_CAL_MIN <= cal <= self._EST_CAL_MAX:
                self._est_cal = cal
                self._est_cal_n = int(d.get("n", 0))
        except Exception:
            pass  # cold start at 1.0 is fine

    def _save_est_cal(self) -> None:
        try:
            import json as _json
            self._est_cal_path().write_text(_json.dumps({
                "cal": self._est_cal, "n": self._est_cal_n,
                "note": "EMA multiplier for _est_tokens; learned from server prompt_tokens",
            }))
        except Exception:
            pass

    def _learn_est_cal(self, actual_tokens: int, msgs: List[Dict[str, Any]]) -> None:
        """EMA-calibrate the estimator against a server-reported prompt size.

        Asymmetric on purpose (the operator's "beta to hold it up"): the costs of
        the two error directions differ — underestimate and the real context
        blows the physical window; overestimate and she compacts slightly
        early.  So the multiplier RISES fast (danger side) and FALLS slowly.
        """
        if not actual_tokens or not msgs:
            return
        est = _est_tokens(msgs)
        if est < 2000:          # tiny samples make meaningless ratios
            return
        ratio = actual_tokens / est
        if not (0.5 <= ratio <= 4.0):   # absurd guard: usage mismatch != estimator error
            return
        a = self._EST_CAL_ALPHA if ratio > self._est_cal else self._EST_CAL_ALPHA_DOWN
        self._est_cal = round(min(max(
            self._est_cal * (1 - a) + ratio * a, self._EST_CAL_MIN), self._EST_CAL_MAX), 4)
        self._est_cal_n += 1
        logger.debug("DecayEngine: est_cal -> %.3f (sample ratio %.2f, alpha=%.2f, n=%d)",
                     self._est_cal, ratio, a, self._est_cal_n)
        self._save_est_cal()

    def _caltok(self, msgs: List[Dict[str, Any]]) -> int:
        """Calibrated token estimate — use for threshold/budget decisions."""
        return int(_est_tokens(msgs) * self._est_cal)

    # -- core: evict (with digest), prune when the tail alone is too big ---
    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None, **kwargs) -> List[Dict[str, Any]]:
        msgs = list(messages)
        n = len(msgs)
        h, t = self.protect_first_n, self.protect_last_n
        out = msgs

        # Calibrate BEFORE deciding anything: compress() is called exactly when
        # real counts exist (run_agent passes last_prompt_tokens as
        # current_tokens), which is also exactly when estimate accuracy matters.
        if current_tokens:
            self._learn_est_cal(current_tokens, msgs)

        if n > h + t + 1:
            middle = msgs[h:n - t]
            digest = self._summarize_middle(middle, recent=msgs[n - t:])
            digest_path = self._write_digest_review(digest) if digest else ""
            retained = self._retain_digest(digest, len(middle)) if digest else False
            if digest:
                self._last_digest = digest
                self._last_digest_path = digest_path
                self._pin_digest_hook(digest)
            marker = {"role": "user", "content": self._marker_text(len(middle), digest, retained, digest_path)}
            out = self._sanitize_tool_pairs(msgs[:h] + [marker] + msgs[n - t:])
            logger.info("DecayEngine: evicted %d middle turn(s) to memory (digest=%s, retained=%s)",
                        len(middle), bool(digest), retained)

        budget = max(self.threshold_tokens, 4000)
        if self._caltok(out) > budget:
            before = self._caltok(out)
            out, changed = self._prune_to_budget(out, budget)
            if changed:
                out = self._note_trim(out)
                logger.info("DecayEngine: tail over budget (~%d > %d tok) — pruned to ~%d tok",
                            before, budget, self._caltok(out))

        if out is not msgs:
            self.compression_count += 1
            try:   # cardfollow2 hysteresis: remember the post-compaction size for this session
                _CF_POST[getattr(self, "session_id", None) or "?"] = self._caltok(out)
            except Exception:
                pass
        return out

    # -- rolling session digest --------------------------------------------
    def _summarize_middle(self, middle: List[Dict[str, Any]], recent: List[Dict[str, Any]] = None,
                          timeout_s: int = _DIGEST_LLM_TIMEOUT_S) -> Optional[str]:
        """Build the post-compact continuity card.

        2026-08-16: aux-LLM digest (7B compression seat, off-box on a remote host — no GPU contention with the agent's brain). Per the
        decay-engine test contract: if the aux call fails, evict WITHOUT a
        digest (plain marker) rather than injecting the lossy CPU heuristic —
        digest call times out (2026-08-25: remote card died 2 min after the
        previous digest; the timeout path here left a 116-turn eviction with
        NO continuity at all).  The CPU digest is deterministic extraction —
        it cannot hallucinate — so a degraded card beats a state hole.
        An empty LLM result also falls through.
        timeout_s: LLM fold budget; the close path passes a shorter one
        (2026-09-09).
        """
        if not self._digest_enabled or not middle:
            return None
        if self._digest_llm:
            try:
                text = self._build_llm_digest(middle, recent or [], timeout_s=timeout_s)
                if text:
                    _deg = _digest_degenerate(text)
                    if _deg:
                        logger.warning(
                            "DecayEngine: LLM digest degenerate (%s) — falling back to CPU digest",
                            _deg)
                    else:
                        return _digest_line_cap(text)
                else:
                    logger.info("DecayEngine: LLM digest empty — falling back to CPU digest")
            except Exception as e:
                logger.info("DecayEngine: LLM digest failed (%s) — falling back to CPU digest", e)
        try:
            return self._build_cpu_digest(middle, recent or [])
        except Exception as e:
            logger.info("DecayEngine: CPU digest failed (%s) — evicting without digest", e)
            return None

    def _build_llm_digest(self, middle: List[Dict[str, Any]], recent: List[Dict[str, Any]],
                          timeout_s: int = _DIGEST_LLM_TIMEOUT_S) -> Optional[str]:
        """Continuity card written by the aux compression seat (7B)."""
        call_llm = _llm_digest_callable()
        full = self._serialize_for_digest(middle, max_chars=10**9)
        if len(full) <= _DIGEST_LLM_MAX_CHARS:
            span_hdr = "=== EVICTED TURNS (being removed from context) ==="
            span = full
        else:
            span_hdr = ("=== EVICTED SPAN as CHUNK NOTES (whole span, time order — every chunk "
                        "must appear in SPAN COVERAGE) ===")
            span = self._chunk_notes(middle, call_llm, timeout_s)
        tail = self._serialize_for_digest(recent, max_chars=5000, keep="tail") if recent else ""
        prior = (self._last_digest or "")[:2000]
        prompt_parts = [_DIGEST_PROMPT, "", span_hdr, span]
        if tail:
            prompt_parts += ["", "=== KEPT RECENT TAIL (NEWER than the evicted span — still in context. "
                             "If it contradicts or resolves anything in the evicted span, the TAIL WINS "
                             "and the digest must state the resolution) ===", tail]
        prompt_parts += ["", "=== PREVIOUS DIGEST (roll forward anything still true) ===",
                         prior if prior else "(none — this is the first digest)"]
        resp = call_llm(
            task="compression",
            messages=[{"role": "user", "content": "\n".join(prompt_parts)}],
            max_tokens=max(self._digest_max_tokens, 1000),
            timeout=timeout_s,
        )
        text = _llm_extract(resp)
        if text:
            text = _verify_key_facts(text, middle, recent, self._last_digest or "")
        return text or None

    def _chunk_notes(self, middle: List[Dict[str, Any]], call_llm, timeout_s: int) -> str:
        """Cover the WHOLE evicted span: split into time-ordered chunks and
        pre-summarize each (2026-10-31). A chunk whose call fails/times out
        degrades to a deterministic line list of the operator's messages in it, so
        no part of the span is ever silently dropped."""
        import time as _t
        from concurrent.futures import ThreadPoolExecutor
        def _strip_ctx(m):
            # Slack user turns carry a long "[Thread context ...]" preamble; a
            # per-message cap would keep only the preamble and drop the request.
            if m.get("role") != "user":
                return m
            t = self._msg_text(m)
            if "[End of thread context]" not in t:
                return m
            stamp = re.search(r"\[Current time:[^\]]*\]", t)
            body = t.split("[End of thread context]", 1)[1].strip()
            if _NEW_MSG_LABEL in body:   # ctxorder 2026-11-04: keep the request, not the wrappers
                body = body.rsplit(_NEW_MSG_LABEL, 1)[1].strip()
            return {**m, "content": ((stamp.group(0) + " ") if stamp else "") + body}
        middle = [_strip_ctx(m) for m in middle]
        lines = []
        for m in middle:
            lines.append(self._serialize_for_digest([m], max_chars=10**9,
                                                    per_msg=_DIGEST_CHUNK_PER_MSG))
        total = sum(len(x) + 1 for x in lines)
        size = max(_DIGEST_CHUNK_CHARS, total // _DIGEST_MAX_CHUNKS + 1)
        groups, cur, cur_len, cur_msgs, gmsgs = [], [], 0, [], []
        for ln, m in zip(lines, middle):
            if cur and cur_len + len(ln) > size:
                groups.append("\n".join(cur)); gmsgs.append(cur_msgs)
                cur, cur_len, cur_msgs = [], 0, []
            cur.append(ln[:size]); cur_len += len(ln) + 1; cur_msgs.append(m)
        if cur:
            groups.append("\n".join(cur)); gmsgs.append(cur_msgs)
        n = len(groups)

        def _fallback(msgs):
            users = [self._one_line(self._msg_text(m), 200) for m in msgs
                     if m.get("role") == "user" and not self._is_systemish_user(self._msg_text(m))]
            return "(notes call failed — the operator's messages in this chunk:)\n" + \
                "\n".join("- " + u for u in users[:20])

        def _one(i):
            prompt = _DIGEST_CHUNK_PROMPT.format(i=i + 1, n=n) + "\n\n=== CHUNK ===\n" + groups[i]
            resp = call_llm(task="compression",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=500, timeout=timeout_s)
            return _llm_extract(resp)

        deadline = _t.monotonic() + max(2 * timeout_s, 60)
        notes = [None] * n
        ex = ThreadPoolExecutor(max_workers=_DIGEST_CHUNK_WORKERS)
        futs = {ex.submit(_one, i): i for i in range(n)}
        try:
            for f, i in futs.items():
                try:
                    notes[i] = f.result(timeout=max(1.0, deadline - _t.monotonic()))
                except Exception as e:
                    logger.info("DecayEngine: chunk %d/%d notes failed (%s) — deterministic fallback",
                                i + 1, n, e)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        out = []
        for i in range(n):
            body = (notes[i] or "").strip() or _fallback(gmsgs[i])
            out.append(f"--- CHUNK {i + 1}/{n} ---\n{body}")
        logger.info("DecayEngine: chunked digest — %d msgs, %d chars → %d chunks (%d LLM ok)",
                    len(middle), total, n, sum(1 for x in notes if x))
        return "\n".join(out)

    def _build_cpu_digest(self, middle: List[Dict[str, Any]], recent: List[Dict[str, Any]] = None) -> str:
        """Deterministic continuity card — content-agnostic."""
        users: List[str] = []
        tools_done: List[str] = []
        pending_tool: Optional[str] = None
        assistant_bits: List[str] = []
        user_blob: List[str] = []
        tool_blob: List[str] = []

        for m in middle:
            role = m.get("role") or "?"
            text = self._msg_text(m)
            if role == "user" and text.strip():
                if self._is_systemish_user(text):
                    continue
                users.append(self._one_line(text, 280))
                user_blob.append(text)
            if role == "assistant":
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        name = str((tc.get("function") or {}).get("name") or "?")
                        args = str((tc.get("function") or {}).get("arguments") or "")
                        pending_tool = name
                        hint = self._one_line(args, 100)
                        tools_done.append(f"{name}({hint})" if hint else name)
                        tool_blob.append(name + " " + args)
                elif text.strip():
                    assistant_bits.append(self._one_line(text, 200))
            if role == "tool":
                name = pending_tool or m.get("name") or "tool"
                tools_done.append(f"{name} -> {self._tool_outcome(text)}")
                tool_blob.append(text)
                pending_tool = None

        # Working set = salient fragments from the span (not typed fields)
        ws = self._salient_fragments(
            user_texts=user_blob,
            tool_texts=tool_blob,
            assistant_texts=[self._msg_text(m) for m in middle if m.get("role") == "assistant"],
            prior_digest=self._last_digest or "",
        )

        # "Right before compact" anchor — from the KEPT tail, not the evicted
        # span. Without it the card only describes what was thrown away, and
        # post-compact she reverts to the oldest task framing (the head) and
        # loops on it. This section tells her what she was doing when compact hit.
        recent = recent or []
        recent_users: List[str] = []
        recent_assistant: List[str] = []
        recent_tools: List[str] = []
        pending_recent_tool: Optional[str] = None
        for m in recent:
            role = m.get("role") or "?"
            text = self._msg_text(m)
            if role == "user" and text.strip():
                if not self._is_systemish_user(text):
                    recent_users.append(self._one_line(text, 280))
            if role == "assistant":
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        pending_recent_tool = str((tc.get("function") or {}).get("name") or "?")
                elif text.strip():
                    recent_assistant.append(self._one_line(text, 200))
            if role == "tool":
                name = pending_recent_tool or m.get("name") or "tool"
                recent_tools.append(f"{name} -> {self._tool_outcome(text)}")
                pending_recent_tool = None

        if users:
            now_lines = users[-4:]
        elif recent_users:
            # Span was pure tool work — the operator's latest words are still in
            # context (kept tail), say so instead of implying data loss.
            now_lines = [f"{u}  (still in context)" for u in recent_users[-4:]]
        else:
            now_lines = ["(nothing recent from the operator)"]

        # OPEN: recent user lines — all of them are open context, not only "?" shape
        open_loops = users[-5:] if users else (recent_users[-5:] or ["(nothing open)"])

        seen_t = set()
        dnr: List[str] = []
        for t in tools_done:
            base = t.split(" -> ")[0].split("(")[0]
            if base in seen_t:
                continue
            seen_t.add(base)
            dnr.append(t)
        dnr = dnr[-12:]
        done = self._collapse_repeats([t for t in tools_done if " -> " in t])[-10:]
        recent_tools = self._collapse_repeats(recent_tools)

        # 2026-08-25: the fallback card speaks in venue phases
        # (SPEC_geometry-context-ordering: OBSERVATION / BRIDGE / CURRENT
        # EVIDENCE / DECISION POINT).  Room semantics kept under the phase
        # names; "RIGHT BEFORE COMPACT" stays near the top because the
        # in-context inject caps the card at ~1200 chars (see _marker_text).
        parts = [
            f"CONTEXT CARD (deterministic, no model — venue degraded; {len(middle)} turns moved to memory)",
            "",
            "OBSERVATION — recent operator drives:",
            *[f"  - {x}" for x in now_lines],
        ]
        if recent_tools or recent_assistant:
            parts += [
                "",
                "CURRENT EVIDENCE — latest moves (CONTINUE from here; do NOT restart the oldest task):",
                *[f"  - did: {x}" for x in recent_tools[-4:]],
                *[f"  - said: {x}" for x in recent_assistant[-2:]],
            ]
        parts += [
            "",
            "CURRENT EVIDENCE — held from the span (distinctive bits, not a form):",
        ]
        if ws:
            parts.extend(f"  - {x}" for x in ws)
        else:
            parts.append("  - (span had little distinctive content beyond the lines above)")
        parts += [
            "",
            "  tool ledger (span):",
            *([f"  - {x}" for x in done] if done else ["  - (none)"]),
        ]
        if assistant_bits:
            parts += [
                "  her ledger (span):",
                *[f"  - {x}" for x in assistant_bits[-4:]],
            ]
        parts += [
            "",
            "DECISION POINT — next:",
            "  - continue the latest moves above",
            "",
            "  still in play:",
            *[f"  - {x}" for x in open_loops[:6]],
            "",
            "  already tried (do not re-run):",
            *([f"  - {x}" for x in dnr] if dnr else ["  - (none)"]),
        ]

        text = "\n".join(parts)
        if len(text) > 3500:
            text = text[:3500].rstrip() + "\n… (venue card capped)"
        return text

    @staticmethod
    def _is_systemish_user(text: str) -> bool:
        t = text.strip()
        if t.startswith("["):
            return True
        head = t[:120]
        if "Context note:" in head or "post-compact" in head:
            return True
        if head.startswith("<post-compact") or head.startswith("<compact"):
            return True
        return False

    @staticmethod
    def _msg_text(m: Dict[str, Any]) -> str:
        c = m.get("content")
        if isinstance(c, list):
            return " ".join(
                str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c
            )
        return str(c or "")

    @staticmethod
    def _one_line(s: str, n: int) -> str:
        s = " ".join(str(s).split())
        if len(s) > n:
            return s[: n - 1].rstrip() + "…"
        return s

    @staticmethod
    def _collapse_repeats(lines: List[str]) -> List[str]:
        """Collapse consecutive identical lines into 'line (xN)'. A stuck tool
        loop (same failing command 9x) would otherwise fill every anchor slot
        with copies of one error."""
        out: List[List[Any]] = []
        for line in lines:
            if out and out[-1][0] == line:
                out[-1][1] += 1
            else:
                out.append([line, 1])
        return [f"{line} (x{n})" if n > 1 else line for line, n in out]

    @staticmethod
    def _tool_outcome(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return "(empty)"
        low = t.lower()
        if "error" in low[:200] or "traceback" in low[:200] or "failed" in low[:200]:
            return "ERR " + DecayEngine._one_line(t, 120)
        if t.startswith("{") or t.startswith("["):
            return "json " + DecayEngine._one_line(t, 120)
        return DecayEngine._one_line(t, 140)

    # English function words only — not domain slots. Used to *downweight* noise,
    # not to decide what topics are allowed.
    _STOP = frozenset(
        "a an the and or but if then else when while for of to in on at by from "
        "with without into onto over under as is are was were be been being am "
        "it its this that these those i you he she we they me him her us them "
        "my your his our their mine yours what which who whom whose how why "
        "do does did done doing have has had having will would can could should "
        "may might must shall about than so not no yes ok okay just like very "
        "also too more most some any all each every few many much such only "
        "own same other another both either neither because until unless "
        "there here where out up down off again further once".split()
    )

    def _salient_fragments(
        self,
        *,
        user_texts: List[str],
        tool_texts: List[str],
        assistant_texts: List[str],
        prior_digest: str,
    ) -> List[str]:
        """Content-agnostic salience — no domain wordlists, no typed slots.

        Prefer:
          1) short phrases that show up in BOTH what you said and tool output
             (the conversation's own anchors)
          2) high-information tokens (digits, mixed, paths, long IDs) from the span
          3) quoted spans and label: value pairs as they appear in text
          4) a few distinctive multi-word phrases from your lines

        Never requires "sound like a radio ticket" or fill an IP field.
        """
        user_join = "\n".join(user_texts)
        tool_join = "\n".join(tool_texts)
        all_join = "\n".join(user_texts + tool_texts + assistant_texts)

        out: List[str] = []
        seen = set()

        def add(s: str, cap: int = 120) -> None:
            s = self._one_line(s, cap)
            if len(s) < 2:
                return
            k = s.lower()
            if k in seen:
                return
            seen.add(k)
            out.append(s)

        # 1) Quoted material (people naturally quote names, phrases)
        # NOTE: ' is NOT a quote delimiter — treating it as one shredded every
        # contraction into garbage fragments ("ll say", "t need") in the
        # 2026-08-13 newsletter digest.
        for q in re.findall(r'[\"“]([^\"“”]{2,80})[\"”]', all_join):
            add(q.strip())

        # 2) Loose "label: value" / "label = value" as written — any label
        for lab, val in re.findall(
            r"(?m)^\s*([A-Za-z][A-Za-z0-9 _/-]{1,40})\s*[:=]\s*(.+)$",
            all_join,
        ):
            lab, val = lab.strip(), val.strip()
            if lab and val and len(val) < 100:
                add(f"{lab}: {val}")

        # 3) Tokens that co-occur in user speech AND tools (anchors of the thread)
        user_toks = self._info_tokens(user_join)
        tool_toks = self._info_tokens(tool_join)
        for t in user_toks:
            if t in tool_toks:
                add(t, 80)

        # 4) High-info tokens from the whole span (digits / mixed / long)
        for t in self._info_tokens(all_join)[:24]:
            add(t, 80)

        # 5) Distinctive bigrams/trigrams from user lines only (how you actually talk)
        for phrase in self._user_phrases(user_join)[:6]:
            add(phrase, 100)

        # 6) Prior digest fragments that reappear in this span (continuity without slots)
        for line in self._prior_held_lines(prior_digest):
            # keep if any contentful token still appears in the new span
            toks = self._info_tokens(line)
            if toks and any(t in all_join.lower() for t in toks[:5]):
                add(line, 120)

        return out[:28]

    def _info_tokens(self, text: str) -> List[str]:
        """Tokens that carry information density — shape-free, not domain-typed."""
        if not text:
            return []
        # split on non-alphanumeric but keep internal . _ - / @ +
        raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/@+-]{1,64}", text)
        scored: List[tuple] = []
        for tok in raw:
            low = tok.lower()
            if low in self._STOP:
                continue
            if len(tok) < 3 and not any(c.isdigit() for c in tok):
                continue
            score = 0
            if any(c.isdigit() for c in tok):
                score += 4
            if re.search(r"[._/@+$-]", tok):
                score += 3
            if any(c.isupper() for c in tok[1:]):  # camel / acronyms
                score += 1
            if len(tok) >= 7:
                score += 2
            elif len(tok) >= 5:
                score += 1
            # bare short dictionary-ish words without digits/punct: skip
            if score < 3 and not any(c.isdigit() for c in tok) and not re.search(r"[._/@+$-]", tok):
                if len(tok) < 8:
                    continue
            scored.append((score, tok))
        scored.sort(key=lambda x: (-x[0], -len(x[1])))
        # unique preserve order of score
        seen = set()
        out: List[str] = []
        for _, t in scored:
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
        return out

    def _user_phrases(self, text: str) -> List[str]:
        """Short multi-word phrases from user speech; stopwords only for glue."""
        if not text:
            return []
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9']{0,40}", text)
        phrases: List[str] = []
        n = len(words)
        for i in range(n):
            # bigram / trigram if not all stop
            for width in (3, 2):
                if i + width > n:
                    continue
                chunk = words[i : i + width]
                content = [w for w in chunk if w.lower() not in self._STOP and len(w) >= 4]
                if not content:
                    continue
                # need substance: a longer content word or a digit somewhere
                if not any(len(w) >= 5 or any(c.isdigit() for c in w) for w in content):
                    if not any(any(c.isdigit() for c in w) for w in chunk):
                        continue
                phrases.append(" ".join(chunk))
        # rank by presence of digits / length uniqueness
        def pscore(p: str) -> int:
            s = 0
            if any(c.isdigit() for c in p):
                s += 3
            if any(w.lower() not in self._STOP for w in p.split()):
                s += 1
            s += min(len(p), 40) // 10
            return s

        phrases.sort(key=lambda p: (-pscore(p), -len(p)))
        seen = set()
        out: List[str] = []
        for p in phrases:
            k = p.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        return out

    def _prior_held_lines(self, digest: str) -> List[str]:
        if not digest:
            return []
        lines = []
        capture = False
        for line in digest.splitlines():
            s = line.strip()
            if s.startswith("HELD ONTO") or s.startswith("WORKING SET"):
                capture = True
                continue
            if capture:
                if s.endswith(":") and not s.startswith("-"):
                    break
                if s.startswith("- "):
                    lines.append(s[2:].strip())
        return lines[:20]


    def _retain_digest(self, digest: str, turn_count: int, trigger: str = "compact") -> bool:
        """Persist the digest into the memory bank as kind=session_digest."""
        providers = getattr(self._memory_manager, "providers", None) or []
        for p in providers:
            fn = getattr(p, "retain_document", None)
            if fn is None:
                continue
            try:
                ok = bool(fn(digest, kind="session_digest", metadata={
                    "evicted_turns": turn_count,
                    "digest_index": self.compression_count + 1,
                    "trigger": trigger,
                }))
                if ok:
                    # DISTILL INVARIANT audit (SPEC_distill-invariant.md, path B): provenance on
                    # every retained digest — fact without provenance is a rumor.
                    try:
                        import hashlib, json as _json, os as _os, time as _time
                        _fid = hashlib.sha256(digest.encode()).hexdigest()[:12]
                        logger.info("[distill] fact_id=%s path=hindsight kind=session_digest turns=%d idx=%d",
                                    _fid, turn_count, self.compression_count + 1)
                        with open(_os.path.join(_os.environ.get("HERMES_HOME") or _os.path.expanduser("~/.hermes"), "decay_distill_audit.jsonl"), "a") as _af:
                            _af.write(_json.dumps({"t": int(_time.time()), "fact_id": _fid,
                                                   "path": "hindsight", "kind": "session_digest",
                                                   "evicted_turns": turn_count,
                                                   "digest_index": self.compression_count + 1}) + "\n")
                    except Exception:
                        pass   # audit must never break retention
                return ok
            except Exception as e:
                logger.info("DecayEngine: digest retain failed (%s)", e)
                return False
        logger.debug("DecayEngine: no memory provider accepts documents — digest not retained")
        return False

    @staticmethod
    def _serialize_for_digest(middle: List[Dict[str, Any]], max_chars: int = 14000,
                              per_msg: int = 1200, keep: str = "outer") -> str:
        lines = []
        for m in middle:
            role = m.get("role", "?")
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
            text = str(c or "")
            if m.get("tool_calls"):
                names = [str((tc.get("function") or {}).get("name") or "?")
                         for tc in m["tool_calls"]]
                text += " [called tools: " + ", ".join(names) + "]"
            if len(text) > per_msg:
                text = text[:per_msg] + f" …[{len(text) - per_msg} chars elided]"
            lines.append(f"[{role}] {text}")
        blob = "\n".join(lines)
        if len(blob) > max_chars:
            if keep == "tail":
                # For the protected recent tail, NEWEST turns matter most —
                # elide the oldest, never the end (2026-08-24: a 3k outer-capped
                # tail dropped a credential resolution into the elided middle
                # and the digest reported it as still-failed).
                cut = blob[-max_chars:]
                nl = cut.find("\n")
                if 0 < nl < 200:  # don't cut mid-message if a break is near
                    cut = cut[nl + 1:]
                blob = f"…[{len(blob) - len(cut)} chars elided from the start]…\n" + cut
            else:
                half = max_chars // 2
                blob = (blob[:half] + f"\n…[{len(blob) - max_chars} chars elided from the middle]…\n"
                        + blob[-half:])
        return blob

    def _pin_digest_hook(self, digest: str) -> None:
        """Operator-pin digest hook (2026-10-04): match active pins against
        this digest — OPEN LOOPS -> state "open", completed content -> "likely_done"
        (a hint rendered as "digest says done: …, confirm and close"; never closes a
        pin). Runs before the post-compact rebuild injects the pin block. Fail-open."""
        try:
            try:
                from .pins.operator_pins import apply_digest_hints  # type: ignore  # plugin install
            except ImportError:
                from pins.operator_pins import apply_digest_hints  # flat sys.path (tests)
            res = apply_digest_hints(digest)
            if res.get("open") or res.get("likely_done"):
                logger.info("DecayEngine: pin digest hints open=%d likely_done=%d",
                            res.get("open", 0), res.get("likely_done", 0))
        except Exception as e:
            logger.debug("pin digest hook skipped: %s", e)

    # -- fix3/fix4 2026-10-30 (ordered) -------------------
    # fix3: every thread gets its own chain at session_digests/threads/<slug>/LATEST.md,
    #   keyed by the gateway session key (one per Slack thread/DM), so one thread can no
    #   longer overwrite another thread's LATEST.  Readers (memory-provider recall) use the
    #   thread file.  Global session_digests/LATEST.md stays as an operator convenience.
    # fix4: an idle/close digest never moves LATEST backwards:
    #   - thread LATEST is not rewritten when no new user turn arrived since the last
    #     digest of that thread (same last-user fingerprint);
    #   - global LATEST is only rewritten by a compaction (live activity) or when this
    #     thread's last user turn is at least as new as the one global LATEST came from.
    @staticmethod
    def digest_thread_slug(key: str) -> str:
        import hashlib
        key = str(key or "").strip() or "session"
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", key)[-60:].strip("_") or "session"
        return f"{base}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"

    def _thread_key(self) -> str:
        return str(getattr(self, "_digest_thread_key", "") or getattr(self, "session_id", "") or "session")

    @staticmethod
    def _user_activity(messages) -> Tuple[Optional[float], str]:
        """(epoch of last real user turn from the gateway [Current time:] stamp, fingerprint)."""
        import hashlib
        from datetime import datetime
        last_txt, last_ts, n = "", None, 0
        for m in messages or []:
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
            c = str(c or "")
            if c.startswith("[Context note:") or not c.strip():
                continue
            n += 1
            last_txt = c
            mt = re.match(r"\[Current time: ([^\]]+?)(?: [A-Z]{2,5})?\]", c)
            if mt:
                try:
                    last_ts = datetime.strptime(mt.group(1).strip(), "%A, %B %d, %Y %I:%M %p").timestamp()
                except Exception:
                    pass
        fp = hashlib.sha1(f"{n}|{last_txt[-2000:]}".encode("utf-8", "replace")).hexdigest()[:16] if n else ""
        return last_ts, fp

    def _write_digest_review(self, digest: str, trigger: str = "compact", messages=None) -> str:
        """Write digest to a reviewable path (human-readable exec/cat target)."""
        try:
            import json as _json, time as _time
            _home0 = __import__("pathlib").Path(__import__("os").environ.get(
                "HERMES_HOME", str(__import__("pathlib").Path.home() / ".hermes")))
            tkey = self._thread_key()
            slug = self.digest_thread_slug(tkey)
            tdir = _home0 / "session_digests" / "threads" / slug
            tdir.mkdir(parents=True, exist_ok=True)
            if trigger == "compact":
                act_ts, fp = _time.time(), ""     # compaction happens inside a live turn
            else:
                act_ts, fp = self._user_activity(messages)
            tmeta_p = tdir / "meta.json"
            try:
                tmeta = _json.loads(tmeta_p.read_text())
            except Exception:
                tmeta = {}
            write_thread = True
            if trigger != "compact" and fp and tmeta.get("last_user_fp") == fp:
                write_thread = False   # fix4: no new user turn since this thread's last digest
            gmeta_p = _home0 / "session_digests" / "LATEST.meta.json"
            try:
                gmeta = _json.loads(gmeta_p.read_text())
            except Exception:
                gmeta = {}
            if trigger == "compact" or gmeta.get("thread_key") == tkey:
                write_global = write_thread
            else:
                g_ts = gmeta.get("last_user_ts")
                write_global = write_thread and act_ts is not None and (g_ts is None or act_ts >= float(g_ts))

            import os
            from datetime import datetime
            from pathlib import Path
            home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
            out_dir = home / "session_digests"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            idx = getattr(self, "compression_count", 0) + 1
            sid = slug
            # 2026-09-09: on-close digests carry their trigger so the review file
            # and filename can't be mistaken for a threshold compaction.
            suffix = f"c{idx}" if trigger == "compact" else f"{trigger}-{idx}"
            path = out_dir / f"{ts}_{sid}_{suffix}.md"
            body = (
                f"# Session digest ({'post-compact' if trigger == 'compact' else trigger} review)\n\n"
                f"- written: {datetime.now().isoformat(timespec='seconds')}\n"
                f"- session: {sid}\n"
                f"- thread_key: {tkey}\n"
                f"- compact_index: {idx}\n"
                + (f"- trigger: {trigger}\n" if trigger != "compact" else "")
                + f"\n## Digest\n\n"
                f"{digest.strip()}\n"
            )
            path.write_text(body, encoding="utf-8")
            meta = {"thread_key": tkey, "session_id": str(getattr(self, "session_id", "") or ""),
                    "trigger": trigger, "written": datetime.now().isoformat(timespec="seconds"),
                    "path": str(path), "last_user_ts": act_ts,
                    "last_user_fp": fp or tmeta.get("last_user_fp", "")}
            if write_thread:
                (tdir / "LATEST.md").write_text(body, encoding="utf-8")
                tmeta_p.write_text(_json.dumps(meta, indent=1))
            if write_global:
                (out_dir / "LATEST.md").write_text(body, encoding="utf-8")
                gmeta_p.write_text(_json.dumps(meta, indent=1))
                # also an "exec output" twin for quick tail/cat review
                exec_path = out_dir / "LATEST.exec.txt"
                exec_path.write_text(
                    f"$ cat {path}\n\n{digest.strip()}\n",
                    encoding="utf-8",
                )
            logger.info("DecayEngine: digest review file written %s (%d chars) thread=%s "
                        "thread_latest=%s global_latest=%s trigger=%s",
                        path, len(digest), slug, write_thread, write_global, trigger)
            return str(path)
        except Exception as e:
            logger.debug("digest review write failed: %s", e)
            return ""

    # -- digest-before-sleep (2026-09-09, Shen — ordered by the operator) --------
    # Digests used to exist only when a decay-threshold compaction fired
    # mid-session; on close/reset/idle the un-digested tail evaporated
    # (observed 2026-09-08: 07:14 reset, last digest from 18:41 the day
    # before — the whole evening's work never folded).  This folds the tail
    # through the SAME machinery (`_summarize_middle` + the writer/retain
    # above), labeled so post-compact readers know it isn't a threshold digest.
    _CLOSE_DIGEST_TIMEOUT_S = 45   # close path is best-effort; never stall a reset
    _CLOSE_MIN_SUBSTANTIVE = 2     # closing small talk isn't worth a digest

    def write_close_digest(self, messages: List[Dict[str, Any]], trigger: str = "session-close") -> str:
        """Fold the un-digested tail of a closing session; return the review path.

        Only messages after the last eviction marker are newer than
        `_last_digest` — everything before was folded already.  Fail-open by
        contract: returns "" on any error or when the tail is trivial, so
        callers proceed with the reset unconditionally.  Runs on a snapshot;
        never mutates `messages`.
        """
        try:
            if not self._digest_enabled or not messages:
                return ""
            seam = 0
            for i, m in enumerate(messages):
                c = m.get("content")
                if isinstance(c, str) and c.startswith("[Context note:"):
                    seam = i + 1
            span = messages[seam:]
            substantive = []
            for m in span:
                if m.get("role") not in ("user", "assistant"):
                    continue
                c = self._msg_text(m)
                if m.get("role") == "user":
                    # The gateway stamps every inbound turn "[Current time: ...]"
                    # — strip it before the systemish check, which would treat
                    # the stamp as synthetic.  (Without this, ALL production
                    # gateway turns count as systemish.)
                    if c.startswith("[Current time:"):
                        nl = c.find("\n")
                        c = c[nl + 1:] if nl > 0 else ""
                    if not c.strip() or self._is_systemish_user(c):
                        continue
                if len(c.strip()) >= 40:
                    substantive.append(c)
            if len(substantive) < self._CLOSE_MIN_SUBSTANTIVE:
                return ""
            digest = self._summarize_middle(span, recent=[], timeout_s=self._CLOSE_DIGEST_TIMEOUT_S)
            if not digest:
                return ""
            path = self._write_digest_review(digest, trigger=trigger, messages=span)
            retained = self._retain_digest(digest, len(span), trigger=trigger)
            self._last_digest = digest
            self._last_digest_path = path
            self._pin_digest_hook(digest)
            logger.info(
                "DecayEngine: close digest (%s) folded %d turn(s) — written=%s retained=%s",
                trigger, len(span), bool(path), retained)
            return path
        except Exception as e:
            logger.info("DecayEngine: close digest failed (%s) — reset proceeds without it", e)
            return ""

    def _marker_text(self, n: int, digest: Optional[str], retained: bool,
                     digest_path: str = "") -> str:
        if digest and retained:
            body = digest.strip()
            if len(body) > 1200:
                body = body[:1200].rstrip() + "\n… (full digest in file)"
            return _MARKER_DIGEST.format(
                n=n,
                path=digest_path or "~/.hermes/session_digests/LATEST.md",
                digest=body,
            )
        if digest:
            # Generated but could not be retained — keep the state inline so it
            # survives (and gets folded into the next digest on the next eviction).
            path_note = f"\nReview file: {digest_path}" if digest_path else ""
            return _MARKER.format(n=n) + path_note + "\n\nSummary of the moved turns:\n" + digest
        return _MARKER.format(n=n)

    # -- tail pruning (guaranteed progress when eviction alone isn't enough) --
    def _prune_to_budget(self, msgs: List[Dict[str, Any]], budget: int) -> Tuple[List[Dict[str, Any]], bool]:
        out, changed = self._truncate_old_tool_bodies(msgs, keep_recent=_TOOL_KEEP_RECENT)
        if self._caltok(out) <= budget:
            return out, changed
        # Drop oldest tail messages, always keeping the most recent exchange.
        while self._caltok(out) > budget:
            idx = self._first_tail_index(out)
            if len(out) - idx <= _MIN_TAIL:
                break
            del out[idx]
            out = self._sanitize_tool_pairs(out)
            changed = True
        if self._caltok(out) > budget:
            # Last resort: trim even the recent tool bodies.
            out, c2 = self._truncate_old_tool_bodies(out, keep_recent=0)
            changed = changed or c2
        return out, changed

    def _first_tail_index(self, msgs: List[Dict[str, Any]]) -> int:
        h = self.protect_first_n
        if len(msgs) > h and _is_marker(msgs[h]):
            return h + 1
        return h

    @staticmethod
    def _truncate_old_tool_bodies(msgs: List[Dict[str, Any]], keep_recent: int) -> Tuple[List[Dict[str, Any]], bool]:
        tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
        protected = set(tool_idx[-keep_recent:]) if keep_recent else set()
        out = list(msgs)
        changed = False
        for i in tool_idx:
            if i in protected:
                continue
            c = out[i].get("content")
            if not isinstance(c, str) or len(c) <= _TOOL_BODY_CAP + _TOOL_BODY_TAIL + 100:
                continue
            trimmed = len(c) - _TOOL_BODY_CAP - _TOOL_BODY_TAIL
            m2 = dict(out[i])
            m2["content"] = (
                c[:_TOOL_BODY_CAP]
                + f"\n…[{trimmed:,} chars trimmed to fit the context budget — "
                  "re-run the tool if you need the full output]…\n"
                + c[-_TOOL_BODY_TAIL:]
            )
            out[i] = m2
            changed = True
        return out, changed

    def _note_trim(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        h = self.protect_first_n
        out = list(msgs)
        if len(out) > h and _is_marker(out[h]):
            m = dict(out[h])
            m["content"] = m["content"] + _TRIM_NOTE
            out[h] = m
        else:
            out.insert(h, {"role": "user", "content": _TRIM_MARKER})
        return out

    @staticmethod
    def _sanitize_tool_pairs(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Guarantee a valid OpenAI sequence after cutting: no orphan tool messages,
        no assistant tool_calls whose responses were dropped."""
        resp_ids = {m.get("tool_call_id") for m in msgs
                    if m.get("role") == "tool" and m.get("tool_call_id")}
        req_ids = {tc.get("id") for m in msgs for tc in (m.get("tool_calls") or []) if tc.get("id")}
        out: List[Dict[str, Any]] = []
        for m in msgs:
            if m.get("role") == "tool":
                if m.get("tool_call_id") in req_ids:
                    out.append(m)               # keep only tool responses whose request survives
                continue
            tcs = m.get("tool_calls")
            if tcs:
                kept = [tc for tc in tcs if tc.get("id") in resp_ids]
                m2 = dict(m)
                if kept:
                    m2["tool_calls"] = kept
                else:
                    m2.pop("tool_calls", None)  # strip dangling calls; keep as plain assistant text
                    if not m2.get("content"):
                        m2["content"] = "(earlier tool actions moved to memory)"
                out.append(m2)
            else:
                out.append(m)
        return out
