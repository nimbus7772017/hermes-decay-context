"""Operator directive pins — short operator instructions, copied VERBATIM.

Why this exists (2026-08-28): a load-bearing one-line instruction ("finish through
NDVI/NDMI") fell out of the protected tail and the rolling digest never captured it —
summarization is lossy exactly where it matters most. Pins never pass through a model:
they are copied, not described, and re-injected after every compaction.

Capture / inject: the hermes-decay-context plugin registers /pins /done /park /unpark
slash commands and a pre_llm_call hook that injects format_operator_pins(). Hosts that
already call pin_operator_message() from a gateway may keep doing so; this package does
not patch Hermes core.

The file is plain JSONL at get_hermes_home()/operator_directives.jsonl, capped at
PIN_KEEP records.

Lifecycle (restored 2026-10-04 from the 2026-09-24 build, with the operator's tweaks):
  state "pinned"       — captured, active (rides the block)
  state "open"         — active; the compaction digest listed it under OPEN LOOPS
  state "likely_done"  — active; the digest's completed content matched it. This is a
                         HINT ONLY: it never closes a pin. Rendered with
                         "(digest says done: …, confirm and close)".
  state "satisfied"    — retired with evidence: the agent's operator_directive_done tool
                         (door A) or the operator's "!done <words>" / "done with <words>"
                         DM (door B). Stops riding the block.
  expired              — computed at read time: ts_epoch + PIN_TTL_SECS (48 h,
                         door C). Never written to disk.
Pure acknowledgements/greetings ("ok", "thanks", "good morning", "are you there",
"glad that's done") are not pinned. Short replies ("yes fix that") carry a one-line
"re" context from the agent's previous message so "that" stays meaningful.
When trimming to PIN_KEEP, retired pins go first, then expired, then the oldest.
Optional open-loops bridge is OFF by default (HERMES_PIN_OPEN_LOOPS_BRIDGE=1 to enable)
and reads credentials only from env (HERMES_PIN_DB_USER / HERMES_PIN_DB_PASSWORD) — never
from pass(1) or credential files.
"""

import json
import logging
import os


import re

# Agent / operator display names (generic by default; set for your install).
_AGENT_NAME = os.environ.get("HERMES_PIN_AGENT_NAME", "").strip().lower()
_AGENT_ALT = ("|" + re.escape(_AGENT_NAME)) if _AGENT_NAME else ""
_OPERATOR_NAME = os.environ.get("HERMES_PIN_OPERATOR_NAME", "").strip() or "the operator"

import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

PIN_FILE = "operator_directives.jsonl"
PIN_KEEP = 25          # cap on disk (retired first, then expired, then oldest)
PIN_INJECT_LAST = 6    # how many active pins ride into the block
PIN_MAX_CHARS = 1500   # directives often arrive as essays ending in a numbered list
                       # (was 500 — that cap skipped the operator's 2026-08-28 climate-first
                       # steering message; documents/pastes are still excluded above this)
PIN_TTL_SECS = 24 * 3600  # fix1 2026-10-30: 48h -> 24h (was: 48 * 3600)
LIKELY_DONE_GRACE_SECS = 6 * 3600  # fix1: a "digest says done" pin auto-retires after 6 h unless re-opened
_LEGACY_TTL_SECS = 48 * 3600  # 2026-10-04 (the operator): pins expire out of the block after 48 h
                          # (was 7 days in the 2026-09-24 build). Computed at read time.
PIN_CONTEXT_MAX_WORDS = 12   # replies this short get a "re" context line
PIN_CONTEXT_CHARS = 200      # length of that context line

ACTIVE_STATES = ("pinned", "open", "likely_done")
RETIRED_STATES = ("satisfied", "expired")   # fix1: expiry is now persisted on write
PARKED_STATE = "parked"   # fix5: on hold by the operator/digest — not open, not retired, never expires
CLOSED_FILE = "closed_work.jsonl"   # fix5: CLOSED ledger of finished work (with proof)
CLOSED_KEEP = 60
CLOSED_SHOW = 3   # ledger lines shown under the pin block
PARKED_SHOW = 4

_THREAD_CTX_START = "[Thread context —"
_THREAD_CTX_END = "[End of thread context]"


def _pins_path():
    return get_hermes_home() / PIN_FILE


@contextmanager
def _locked():
    """Serialize read-modify-write across gateway threads/processes (gateway capture,
    the agent's tool, the digest hook and the !done verb can race)."""
    lock_path = get_hermes_home() / (PIN_FILE + ".lock")
    fh = None
    try:
        import fcntl
        fh = open(lock_path, "a+")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except Exception:
        fh = None  # locking is best-effort; never block a turn on it
    try:
        yield
    finally:
        if fh is not None:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            fh.close()


def _write_all(pins: List[Dict[str, Any]]) -> None:
    """Atomic rewrite (tmp + replace). Raises on failure."""
    path = _pins_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pins),
                   encoding="utf-8")
    os.replace(tmp, path)


def strip_thread_context(text: str) -> str:
    """Remove a platform-prepended thread-context block, leaving the operator's words."""
    t = (text or "").lstrip()
    if t.startswith(_THREAD_CTX_START):
        end = t.find(_THREAD_CTX_END)
        if end != -1:
            t = t[end + len(_THREAD_CTX_END):]
    return t.strip()


# ---------------------------------------------------------------------------
# Acknowledgement / greeting filter (2026-10-04). Conservative: the WHOLE message
# must be made of these phrases (plus the agent name, punctuation, emoji). Approvals such
# as "yes", "do it", "go ahead" are NOT acks — they are pinned with context.
# ---------------------------------------------------------------------------
_ACK_PHRASES = [
    # greetings / sign-offs
    r"hi", r"hello", r"hey", r"hiya", r"yo", r"howdy",
    r"good (?:morning|afternoon|evening|night)", r"morning", r"evening", r"gm", r"gn",
    r"good ?night", r"night", r"bye", r"see (?:you|ya)(?: later| tomorrow)?", r"later",
    r"how are you(?: doing)?", r"how(?:'?s| is) it going", r"what'?s up", r"sup",
    # presence checks
    r"(?:are )?you (?:there|around|up|awake|here)", r"still (?:there|around|here)",
    r"are you (?:still )?(?:there|around|up|awake|here)", r"ping",
    # thanks / acks
    r"ok(?:ay)?", r"k+", r"kk", r"alright", r"all right", r"roger(?: that)?", r"copy(?: that)?",
    r"thanks?(?: you)?(?: (?:so|very) much)?", r"thank you(?: (?:so|very) much)?", r"thx", r"ty",
    r"much appreciated", r"appreciate it", r"cheers",
    r"got it", r"gotcha", r"understood", r"noted", r"makes sense", r"sounds good",
    r"cool", r"great", r"nice", r"perfect", r"awesome", r"excellent", r"wonderful", r"sweet",
    r"good", r"very good", r"good (?:job|work|stuff)", r"nice (?:job|work|one)",
    r"well done", r"great (?:job|work)", r"lol", r"haha+", r"ha", r"wow",
    r"no problem", r"np", r"no worries",
    # closers
    r"glad (?:that'?s|that is|it'?s|it is|to have (?:that|it)|we got (?:that|it)) (?:done|finished|fixed|sorted)",
    r"(?:that'?s|that is) (?:done|great|good|perfect|fine)",
]
_ACK_RE = re.compile(r"^(?:(?:%s)\b[\s,.!?]*)+$" % "|".join("(?:%s)" % p for p in _ACK_PHRASES))


def is_ack(text: str) -> bool:
    """True when *text* is only greetings / thanks / acknowledgements."""
    t = (text or "").lower()
    t = t.replace("\u2019", "'")
    t = re.sub(r":[a-z0-9_+\-]+:", " ", t)              # slack :emoji:
    t = re.sub(r"[^\w\s',.!?]", " ", t)                   # unicode emoji / symbols
    if _AGENT_NAME:
        t = re.sub(r"\b" + re.escape(_AGENT_NAME) + r"\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.!?")
    if not t:
        return True
    return bool(_ACK_RE.match(t + " "))


# ---------------------------------------------------------------------------
# Plain-language close requests (2026-10-04 10:56, the operator approved). "you can
# close that investigation", "please close this job", "it's intentionally
# orphaned" are NOT new directives: they are never pinned, and the gateway
# treats them like "done with" (retire the best-matching active pin with the
# message as evidence). Conservative regexes; no model call.
# ---------------------------------------------------------------------------
CLOSE_NOTE_MARK = "[close-only request:"
CLOSE_NOTE = ("[close-only request: the operator is asking to close something, not to start "
              "anything. Close the matching pinned directive(s) with "
              "operator_directive_done(substring, evidence), using his words as the evidence, "
              "and reply briefly. Do not start new work, sweeps, or documentation for this "
              "message.%s]")

# Things you "close" on a network that are real work, not a pin/job closure.
_CLOSE_OBJ_EXCLUDE = (r"port|ports|firewall|connection|connections|session|sessions|socket|"
                      r"sockets|tunnel|tunnels|vpn|window|windows|tab|tabs|door|account|"
                      r"accounts|ticket|tickets|case|cases|circuit|circuits|gap|gaps|hole|holes|"
                      r"access|interface|interfaces|link|links|loop\s*hole")
_CLOSE_DEICTIC = r"(?:that|this|it|these|those|them)"
_CLOSE_NOUN = (r"(?:job|jobs|investigation|investigations|loop|loops|open\s+loop|pin|pins|"
               r"directive|directives|task|tasks|item|items|topic|question|inquiry|thing|"
               r"one|issue|request)")
_CLOSE_NOT_WORK = (r"(?!\s+(?:(?:the|that|this|those|these|a|an|my|our|your|all|out|off|up|down)\s+)*"
                   r"(?:[\w\-/.:]+\s+)?(?:%s)\b)" % _CLOSE_OBJ_EXCLUDE)
_CLOSE_RES = [
    # "close that/this/it" — not "close that port"
    re.compile(r"\bclose\b%s\s+(?:(?:out|off)\s+)?%s\b" % (_CLOSE_NOT_WORK, _CLOSE_DEICTIC), re.I),
    # "close (that|this|the) job / investigation / loop / pin ..."
    re.compile(r"\bclose\b%s\s+(?:out\s+)?(?:(?:the|that|this|my|your|our)\s+)?(?:[\w\-/.]+\s+)?%s\b"
               % (_CLOSE_NOT_WORK, _CLOSE_NOUN), re.I),
    re.compile(r"\b(?:you\s+can|go\s+ahead\s+and|ok(?:ay)?\s+to|feel\s+free\s+to|please|pls|plz|plase|pleas)"
               r"\s+close\b%s" % _CLOSE_NOT_WORK, re.I),
    re.compile(r"\bclose\s+(?:it|that|this)\s+(?:out|off)\b", re.I),
    re.compile(r"\bintentionally\s+(?:orphaned|unused|excluded|abandoned|dropped|decommissioned)\b", re.I),
    re.compile(r"\b(?:intentionally|deliberately|on\s+purpose)\s+left\b.{0,60}?\b(?:out|off|behind|alone)\b", re.I),
    re.compile(r"\bleft\b.{0,60}?\bout\b.{0,40}?\b(?:intentionally|deliberately|on\s+purpose)\b", re.I),
    re.compile(r"\b(?:mark|consider)\s+(?:it|that|this|the\s+\w+)\s+(?:as\s+)?(?:done|closed|resolved|complete|finished)\b", re.I),
    re.compile(r"\bno\s+longer\s+(?:in\s+use|used|needed|relevant)\b.{0,40}?\bclose\b%s" % _CLOSE_NOT_WORK, re.I),
    re.compile(r"\b(?:forget\s+about|stop\s+(?:investigating|chasing|working\s+on))\s+(?:that|this|it)\b", re.I),
]
# Imperatives that mean "also go do something" — such a message is not close-only.
_WORK_RE = re.compile(
    r"\b(?:fix|add|move|migrate|check|investigate|look\s+(?:at|into)|find|document|record|"
    r"write\s+(?:up|down|it)|update|create|configure|set\s+up|setup|install|run|send|email|"
    r"call|deploy|restart|reboot|delete|remove|disable|enable|change|rename|build|"
    r"test|verify|scan|sweep|audit|research|figure\s+out|dig\s+into|ping|telnet|"
    r"ssh|log\s+in|login|start|schedule|replace|upgrade|patch|clean\s+up|tell|ask|"
    r"draft|reply|make|put|also|and\s+then|next|after\s+that|instead)\b", re.I)


def strip_close_note(text: str) -> str:
    t = text or ""
    i = t.find(CLOSE_NOTE_MARK)
    return t[:i].rstrip() if i != -1 else t


def _close_body(text: str) -> str:
    return strip_close_note(strip_thread_context(text or ""))


def has_close_intent(text: str) -> bool:
    """True when the operator's own words (thread context stripped) ask to close
    something: "you can close that investigation", "please close this job",
    "it's intentionally orphaned". "close port 22" / "close that ticket" are work."""
    body = _close_body(text)
    if not body or len(body) > 600:
        return False
    flat = re.sub(r"\s+", " ", body.replace("\u2019", "'"))
    return any(r.search(flat) for r in _CLOSE_RES)


def is_close_only(text: str) -> bool:
    """Close intent and no other imperative ("...and then fix X" is not close-only)."""
    if not has_close_intent(text):
        return False
    flat = re.sub(r"\s+", " ", _close_body(text))
    # remove the close phrases themselves before looking for work verbs
    for r in _CLOSE_RES:
        flat = r.sub(" ", flat)
    return not _WORK_RE.search(flat)


def is_close_request(text: str) -> bool:
    """Never pin these (close-only requests)."""
    return is_close_only(text)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def _record_ts_epoch(rec: Dict[str, Any], now: float) -> float:
    """Epoch for a record. Legacy records carry only a ``t`` string like
    ``2026-09-24 16:45:33 MDT``; %Z parsing is platform-flaky, so parse the
    date+time portion and read it as local time (the pin file is written with
    local %Z stamps). Unparseable timestamps fall back to ``now`` — a pin we
    cannot date stays active rather than being silently retired."""
    try:
        return float(rec["ts_epoch"])
    except (KeyError, TypeError, ValueError):
        pass
    t = str(rec.get("t") or "").strip()
    if t:
        core = " ".join(t.split()[:2])  # "YYYY-MM-DD HH:MM:SS", TZ word dropped
        try:
            return float(time.mktime(time.strptime(core, "%Y-%m-%d %H:%M:%S")))
        except Exception:
            pass
    return now


def _normalize_record(rec: Dict[str, Any], now: float) -> Dict[str, Any]:
    """Backfill lifecycle defaults in memory: state ("pinned" when absent),
    ts_epoch, expires_at (= ts_epoch + TTL). Retirement adds evidence /
    retired_at / retired_via; the digest hook adds hint / hint_at."""
    rec.setdefault("state", "pinned")
    rec["ts_epoch"] = _record_ts_epoch(rec, now)
    if not rec.get("expires_at"):
        rec["expires_at"] = rec["ts_epoch"] + PIN_TTL_SECS
    return rec


def _load_all() -> List[Dict[str, Any]]:
    try:
        out = []
        for line in _pins_path().read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        now = time.time()
        return [_normalize_record(r, now) for r in out]
    except FileNotFoundError:
        return []
    except Exception:
        return []  # a corrupt pin file must never break a turn


_PROOF_RE = re.compile(
    r"(?:\b(?:proof|evidence|verified|confirmed|ticket|id|count|report|sent to|logged)\b|\d|[/@]|\.(?:md|csv|json|txt|log)\b)",
    re.I)


def has_proof(text: str) -> bool:
    """fix6: a "done/resolved" claim counts only when it carries some evidence —
    an explicit proof:/evidence: marker, a number/count/id, a path or email, or a
    verified/confirmed word. Bare "resolved" with nothing behind it does not."""
    return bool(_PROOF_RE.search(str(text or "")))


def _closed_path():
    return get_hermes_home() / CLOSED_FILE


def _load_closed() -> List[Dict[str, Any]]:
    out = []
    try:
        for ln in _closed_path().read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("closed ledger unreadable: %s", e)
    return out


def record_closed(text: str, proof: str = "", source: str = "", now: float = None) -> bool:
    """fix5: append one finished item to the CLOSED ledger (dedup on text, capped)."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()[:240]
    if not text:
        return False
    now = time.time() if now is None else now
    try:
        recs = _load_closed()
        key = text.lower()
        if any(str(r.get("text", "")).lower() == key for r in recs):
            return False
        recs.append({"t": time.strftime("%Y-%m-%d %H:%M", time.localtime(now)), "ts_epoch": now,
                     "text": text, "proof": str(proof or "")[:240], "source": source})
        recs = recs[-CLOSED_KEEP:]
        path = _closed_path()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        os.replace(tmp, path)
        logger.info("closed ledger += %r (%s)", text[:80], source)
        return True
    except Exception as e:
        logger.warning("closed ledger write failed: %s", e)
        return False


def _ledger_retired(p: Dict[str, Any]) -> None:
    record_closed("%s%s" % ((p.get("id") + " ") if p.get("id") else "", str(p.get("text") or "")[:180]),
                  proof=str(p.get("evidence") or ""), source=str(p.get("retired_via") or "pin"))


def _likely_done_lapsed(p: Dict[str, Any], now: float) -> bool:
    return (p.get("state") == "likely_done"
            and now >= float(p.get("hint_at") or now) + LIKELY_DONE_GRACE_SECS)


def _is_active(p: Dict[str, Any], now: float) -> bool:
    return (p.get("state") in ACTIVE_STATES and now < float(p.get("expires_at") or 0)
            and not _likely_done_lapsed(p, now))


def _settle(pins: List[Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
    """fix1 2026-10-30: write lifecycle transitions to disk instead of leaving every
    record "pinned" forever. likely_done past its grace -> satisfied (retired_via
    digest-auto, evidence = the digest line); active past expires_at -> expired."""
    for p in pins:
        st = p.get("state")
        if _likely_done_lapsed(p, now):
            p["state"] = "satisfied"
            p["retired_via"] = "digest-auto"
            p["retired_at"] = now
            p.setdefault("evidence", "digest says done: %s" % str(p.get("hint") or "")[:200])
            _ledger_retired(p)
        elif st in ACTIVE_STATES and now >= float(p.get("expires_at") or 0):
            p["state"] = "expired"
            p["retired_at"] = now
    return pins


def active_pins(now: float = None) -> List[Dict[str, Any]]:
    """Records eligible for injection: an active state AND not expired (door C).
    Expiry is computed here, never written."""
    now = time.time() if now is None else float(now)
    return [p for p in _load_all() if _is_active(p, now)]


def _trim(pins: List[Dict[str, Any]], keep: int = None, now: float = None) -> List[Dict[str, Any]]:
    """Cap at *keep* records, dropping retired pins first (oldest first), then
    expired ones, then the oldest active ones. Order is preserved."""
    keep = PIN_KEEP if keep is None else keep
    now = time.time() if now is None else now
    _settle(pins, now)
    if len(pins) <= keep:
        return pins
    excess = len(pins) - keep
    drop = set()
    tiers = (
        lambda p: p.get("state") in RETIRED_STATES,
        lambda p: p.get("state") not in RETIRED_STATES and p.get("state") != PARKED_STATE and not _is_active(p, now),
        lambda p: p.get("state") == PARKED_STATE,   # fix5: parked outlives expired, yields to active
        lambda p: True,
    )
    for tier in tiers:
        for i, p in enumerate(pins):
            if excess <= 0:
                break
            if i not in drop and tier(p):
                drop.add(i)
                excess -= 1
        if excess <= 0:
            break
    return [p for i, p in enumerate(pins) if i not in drop]


def migrate_pin_file() -> Dict[str, int]:
    """One-shot, idempotent on-disk migration: backfill state/ts_epoch/
    expires_at into legacy records. Expiry stays computed at read."""
    now = time.time()
    with _locked():
        try:
            lines = _pins_path().read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return {"records": 0, "migrated": 0}
        recs = []
        migrated = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "state" not in rec or "ts_epoch" not in rec or "expires_at" not in rec:
                migrated += 1
            recs.append(_normalize_record(rec, now))
        _write_all(recs)
    return {"records": len(recs), "migrated": migrated}


def _context_line(prev_assistant: str) -> str:
    """One line (~PIN_CONTEXT_CHARS) saying what a short reply refers to: the last
    question in the agent's previous message if there is one, else its tail."""
    t = re.sub(r"\s+", " ", str(prev_assistant or "")).strip()
    if not t:
        return ""
    qs = re.findall(r"[^.!?\n]*\?", t)
    if qs:
        q = qs[-1].strip()
        if len(q) >= 12:
            t = q
    if len(t) > PIN_CONTEXT_CHARS:
        t = "…" + t[-(PIN_CONTEXT_CHARS - 1):]
    return t


# fix7 2026-10-30: only real instructions become pins. Banter ("the band name will be
# Models", "a really good band called Battles right?") and status chatter ("Dad shot an
# elk") used to be pinned verbatim and then steered the post-compact rebuild.
_VOICE_PREFIX = re.compile(r"^\[The user sent a voice message[^\]:]*:\s*", re.I)
_LEAD = r"(?:(?:ok(?:ay)?|yeah|yes|yep|no|now|then|and|so|also|hey|hi|please" + _AGENT_ALT + r"|alright|right|cool|great|nice|sorry|oh|well|but)[\s,.!:-]+)*"
_DIRECTIVE_RES = [
    re.compile(r"\b(?:can|could|would|will|can't|cant)\s+(?:you|u|ya)\b", re.I),
    re.compile(r"\bplease\b|\bpls\b|\bplz\b", re.I),
    re.compile(r"\bi\s+(?:want|need|would\s+like|'d\s+like|d\s+like)\s+(?:you|u" + _AGENT_ALT + r"|to\s+have\s+you|it|that|this|the|a|an)\b", re.I),
    re.compile(r"\b(?:make\s+sure|go\s+ahead|from\s+now\s+on|going\s+forward|remember\s+(?:to|that)|keep\s+(?:an\s+eye|watching|track))\b", re.I),
    re.compile(r"(?:^|[.!?;]\s+|\n)" + _LEAD + r"(?:don'?t|do\s+not|never|always|stop|quit)\s+(?!worry\b|know\b|think\b|care\b|mean\b)\w+", re.I),
    re.compile(r"\blet'?s\s+(?!see\b|go\b\s*$)\w+", re.I),
    re.compile(r"(?:^|[.!?;]\s+|\n)" + _LEAD +
               r"(?:go|try|run|send|find|check|look|fix|build|make|write|draft|update|add|remove|delete|"
               r"create|schedule|call|text|email|ping|tell|ask|show|give|pull|grab|restart|reboot|deploy|install|"
               r"cancel|close|finish|complete|test|verify|research|search|compare|review|summari[sz]e|list|"
               r"set|turn|put|move|book|order|do|get|start|open|read|reply|respond|post|share|upload|download|"
               r"export|import|merge|push|apply|enable|disable|configure|investigate|audit|sweep|scan|ping|"
               r"track|monitor|watch|figure|work|keep|use|switch|change|rename|copy|save|note|log|plan|map|"
               r"estimate|quote|price|calculate|count|sort|clean|drop|pin|unpin|kill|shut|bring|hold|wait|"
               r"skip|ignore|retry|redo|undo|rerun|continue|resume|proceed|finalize|confirm|label|tag|mark)\b",
               re.I),
]
_WH_Q = re.compile(r"(?:^|[.!?]\s+)" + _LEAD +
                   r"(?:what|why|how|where|when|which|who|whats|what's|is|are|did|does|do|has|have|was|were)\b[^?]{12,}\?", re.I)
_DECISION_REPLY = re.compile(r"^\s*(?:yes|yep|yeah|no|nope|do\s+it|go|go\s+for\s+it|ship\s+it|approved?|"
                             r"sounds\s+good|option\s+\w+|the\s+(?:first|second|third|last)\s+one|[a-d1-9])"
                             r"\b[\s,.!]*", re.I)


def is_directive(text: str, prev_assistant: str = "") -> bool:
    """True when *text* asks the agent to do / not do something (or answers her question
    with a decision). Disable with HERMES_PIN_DIRECTIVE_FILTER=0."""
    if os.environ.get("HERMES_PIN_DIRECTIVE_FILTER", "1").strip().lower() in ("0", "false", "no", "off"):
        return True
    t = _VOICE_PREFIX.sub("", str(text or "")).strip().rstrip("]").strip().strip('"\u201c\u201d\'').strip()
    if not t:
        return False
    if any(r.search(t) for r in _DIRECTIVE_RES):
        return True
    if _WH_Q.search(t) and len(t.split()) >= 5 and not re.search(r"\bright\s*\?\s*$", t, re.I):
        return True
    if prev_assistant and "?" in str(prev_assistant) and len(t.split()) <= PIN_CONTEXT_MAX_WORDS \
            and _DECISION_REPLY.match(t):
        return True
    return False


def _next_pin_id(pins: List[Dict[str, Any]]) -> str:
    n = 0
    for p in pins:
        m = re.match(r"^D(\d+)$", str(p.get("id") or ""))
        if m:
            n = max(n, int(m.group(1)))
    return "D%d" % (n + 1)


def pin_operator_message(text: str, source_desc: str = "", prev_assistant: str = "") -> bool:
    """Pin a short operator message verbatim. Returns True if pinned.

    Skips: empty, long pastes (>PIN_MAX_CHARS), pure noise (<4 alnum chars), and
    pure acknowledgements/greetings (is_ack). A short reply (<= PIN_CONTEXT_MAX_WORDS
    words) gets ``re``: one line from the agent's previous message (*prev_assistant*).
    """
    body = strip_close_note(strip_thread_context(text))
    if not body or len(body) > PIN_MAX_CHARS:
        return False
    # Noise floor: <4 alnum chars — except a bare "yes"/"no" answering the agent's
    # question (prev_assistant given), which is a decision worth keeping.
    if sum(ch.isalnum() for ch in body) < (2 if prev_assistant else 4):
        return False
    if is_ack(body):
        return False
    if is_close_request(body):
        return False  # 2026-10-04: a close request is not a new directive
    if not is_directive(body, prev_assistant):
        logger.info("operator pin skipped (not a directive): %r", body[:80])
        return False  # fix7 2026-10-30
    now = time.time()
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S %Z").strip(),
           "text": body, "via": source_desc,
           "state": "pinned", "ts_epoch": now,
           "expires_at": now + PIN_TTL_SECS}
    if prev_assistant and len(body.split()) <= PIN_CONTEXT_MAX_WORDS:
        ctx = _context_line(prev_assistant)
        if ctx:
            rec["re"] = ctx
    try:
        with _locked():
            pins = _load_all()
            rec["id"] = _next_pin_id(pins)   # fix7: open-loop id (D<n>)
            pins.append(rec)
            _write_all(_trim(pins, now=now))
    except Exception:
        return False  # pin failure must never break message handling
    return True


# ---------------------------------------------------------------------------
# Retirement (doors A and B)
# ---------------------------------------------------------------------------

def retire_operator_directive(substring: str, evidence: str = "", actor: str = "") -> List[Dict[str, Any]]:
    """Mark active directives whose text contains ``substring`` (case-insensitive)
    as state="satisfied" with evidence. ``substring`` "last" retires the newest
    active pin. Records stay in the file (until trimmed, retired-first).

    Returns the matched records. Raises on a file-write failure so callers can
    report that nothing actually changed."""
    sub = (substring or "").strip().strip('"').strip("'").strip().lower()
    if not sub:
        return []
    now = time.time()
    hits = []
    with _locked():
        pins = _load_all()
        if sub == "last":
            act = [p for p in pins if _is_active(p, now)]
            targets = act[-1:] if act else []
        else:
            targets = [p for p in pins
                       if p.get("state") in ACTIVE_STATES and sub in str(p.get("text") or "").lower()]
        for p in targets:
            p["state"] = "satisfied"
            p["retired_at"] = now
            if evidence:
                p["evidence"] = evidence
            if actor:
                p["retired_via"] = actor
            hits.append(p)
        if hits:
            _write_all(pins)
    if hits:
        for h in hits:
            _ledger_retired(h)
            _bridge_open_loop_async(h)
    return hits


def park_operator_directive(substring: str, reason: str = "", actor: str = "") -> List[Dict[str, Any]]:
    """fix5: move active pins containing *substring* ("last" = newest) to the parked
    list: they stop riding the block as open work but stay listed as PARKED and never
    expire. unpark_operator_directive() brings one back."""
    sub_ = (substring or "").strip().strip('"').strip("'").strip().lower()
    if not sub_:
        return []
    now = time.time()
    hits = []
    with _locked():
        pins = _load_all()
        if sub_ == "last":
            act = [p for p in pins if _is_active(p, now)]
            targets = act[-1:]
        else:
            targets = [p for p in pins if p.get("state") in ACTIVE_STATES
                       and sub_ in str(p.get("text") or "").lower()]
        for p in targets:
            p["state"] = PARKED_STATE
            p["parked_at"] = now
            p["parked_why"] = (reason or "")[:200]
            if actor:
                p["parked_via"] = actor
            hits.append(p)
        if hits:
            _write_all(pins)
            logger.info("parked %d pin(s) (%s)", len(hits), actor or "?")
    return hits


def unpark_operator_directive(substring: str) -> List[Dict[str, Any]]:
    """fix5: bring parked pins back to active ("" / "last" = the most recently parked)."""
    sub_ = (substring or "").strip().strip('"').strip("'").strip().lower()
    now = time.time()
    with _locked():
        pins = _load_all()
        pk = [p for p in pins if p.get("state") == PARKED_STATE]
        if sub_ in ("", "last"):
            targets = sorted(pk, key=lambda p: float(p.get("parked_at") or 0))[-1:]
        else:
            targets = [p for p in pk if sub_ in str(p.get("text") or "").lower()]
        for p in targets:
            p["state"] = "pinned"
            p["expires_at"] = now + PIN_TTL_SECS
        if targets:
            _write_all(pins)
    return targets


def parked_pins() -> List[Dict[str, Any]]:
    return [p for p in _load_all() if p.get("state") == PARKED_STATE]


# ---------------------------------------------------------------------------
# Close-request matching (2026-10-04 10:56). Same idea as "done with <words>",
# but the words are a whole sentence, so score pins on shared IDs / IPs / subnets
# and content words instead of a literal substring.
# ---------------------------------------------------------------------------
_CLOSE_STOPWORDS = {
    # the close vocabulary itself
    "close", "closi", "pleas", "plase", "inten", "orpha", "left", "longe", "inves", "loop",
    "direc", "thing", "issue", "reque", "migra", "think", "actua", "becau", "wonde", "found",
    "corre", "right", "done", "finis", "resol", "consi", "mark", "needs", "neede", "used",
    # function words that _STOP may not cover (a shared "between" is not evidence)
    "betwe", "about", "there", "their", "where", "which", "would", "could", "shoul", "these",
    "those", "still", "just", "what", "when", "have", "been", "were", "into", "onto", "some",
    "also", "only", "more", "very", "here", "over", "under", "after", "befor", "again",
    "other", "anoth", "every", "direc", "sure", "pretty", "prett", "yeah", "okay"}


def _norm_for_match(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"(?i)\b(?:p2p|ptp|point[- ]to[- ]point)\b", " ptplink ", t)
    return t


def _close_terms(text: str):
    t = _norm_for_match(text)
    ids, words = _terms(t)
    nets = set()
    for ip in re.findall(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.\d{1,3}\b", t):
        nets.add("net:%s.%s.%s" % ip)
    words = {w for w in words if w not in _CLOSE_STOPWORDS}
    return ids, nets, words


def match_close_request(text: str, now: float = None) -> Dict[str, Any]:
    """Score active pins against a close request. Returns
    {"best": pin or None, "candidates": [pins with any score], "ambiguous": bool}.
    A pin qualifies on a shared ID/IP, a shared /24 (10.9.8.0/30 ~ 10.9.8.2), or
    >= 2 shared content words covering >= 1/3 of the pin's words. The best must be
    unique (strictly higher score than the runner-up)."""
    now = time.time() if now is None else now
    body = _close_body(text)
    mid, mnet, mw = _close_terms(body)
    scored = []
    for p in active_pins(now):
        pid, pnet, pw = _close_terms("%s %s" % (p.get("text", ""), p.get("re", "")))
        id_hit = len(pid & mid)
        net_hit = len(pnet & mnet)
        w_hit = len(pw & mw)
        ok = bool(id_hit or net_hit or (w_hit >= 2 and pw and w_hit / float(len(pw)) >= 0.34))
        if not ok:
            continue
        scored.append((3 * id_hit + 2 * net_hit + w_hit, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = {"best": None, "candidates": [p for _, p in scored], "ambiguous": False}
    if not scored:
        return out
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        out["ambiguous"] = True
        return out
    out["best"] = scored[0][1]
    return out


def retire_pin_record(rec: Dict[str, Any], evidence: str, actor: str = "") -> List[Dict[str, Any]]:
    """Retire exactly one record (matched by ts_epoch + text) as satisfied."""
    now = time.time()
    hits = []
    with _locked():
        pins = _load_all()
        for p in pins:
            if (p.get("state") in ACTIVE_STATES and p.get("text") == rec.get("text")
                    and abs(float(p.get("ts_epoch") or 0) - float(rec.get("ts_epoch") or 0)) < 1e-3):
                p["state"] = "satisfied"
                p["retired_at"] = now
                p["evidence"] = evidence
                if actor:
                    p["retired_via"] = actor
                hits.append(p)
                break
        if hits:
            _write_all(pins)
    for h in hits:
        _ledger_retired(h)
        _bridge_open_loop_async(h)
    return hits


def close_note_for(text: str, candidates: List[Dict[str, Any]] = None) -> str:
    """The note appended to a close request that goes on to the agent."""
    extra = ""
    if candidates:
        extra = " Possible matches: " + "; ".join(
            '"%s"' % (str(c.get("text") or "")[:70].replace("\n", " ")) for c in candidates[:3]) + "."
    return CLOSE_NOTE % extra


def list_pins_text(now: float = None) -> str:
    """Human-readable list of active pins (for the operator's !pins)."""
    now = time.time() if now is None else now
    act = [p for p in _load_all() if _is_active(p, now)]
    if not act:
        return "No active pinned directives."
    lines = ["Active pinned directives (%d, newest first; the newest %d ride every turn):"
             % (len(act), PIN_INJECT_LAST)]
    for p in reversed(act):
        lines.append("- %s [%s]: \"%s\"" % (p.get("t", "?"), p.get("state"),
                                             str(p.get("text") or "")[:160]))
    pk = parked_pins()
    if pk:
        lines.append("Parked (on hold, not open): " + "; ".join(
            str(p.get("text") or "")[:70].replace("\n", " ") for p in pk[-PARKED_SHOW:]))
    lines.append('Close one with "!done <a few words from it>" or "!done last"; '
                 'put one on hold with "!park <words>" ("!unpark <words>" brings it back).')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional open-loops DB bridge (OFF by default). Enable with
# HERMES_PIN_OPEN_LOOPS_BRIDGE=1 and set HERMES_PIN_DB_USER / HERMES_PIN_DB_PASSWORD
# (optional HERMES_PIN_DB_HOST/PORT/NAME). Password via MYSQL_PWD env to the client —
# never on argv, never logged. Failures skip quietly.
# ---------------------------------------------------------------------------
_BRIDGE_ENABLED_ENV = "HERMES_PIN_OPEN_LOOPS_BRIDGE"   # must be 1/true/yes/on to enable


def _sql_str(s: str) -> str:
    return "'" + str(s).replace("\\", "\\\\").replace("'", "''") + "'"


def _bridge_open_loop(rec: Dict[str, Any]) -> bool:
    # Catalog-safe: OFF unless explicitly enabled. Credentials from env only.
    if os.environ.get(_BRIDGE_ENABLED_ENV, "0").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    user = (os.environ.get("HERMES_PIN_DB_USER") or "").strip()
    pw = (os.environ.get("HERMES_PIN_DB_PASSWORD") or "").strip()
    host = (os.environ.get("HERMES_PIN_DB_HOST") or "127.0.0.1").strip()
    port = (os.environ.get("HERMES_PIN_DB_PORT") or "3306").strip()
    db = (os.environ.get("HERMES_PIN_DB_NAME") or "hermes").strip()
    if not user or not pw:
        logger.debug("pin open_loops bridge skipped (missing HERMES_PIN_DB_USER/PASSWORD)")
        return False
    try:
        ep = int(float(rec.get("ts_epoch") or time.time()))
        evidence = str(rec.get("evidence") or "retired via %s" % (rec.get("retired_via") or "?"))
        sql = (
            "INSERT INTO open_loops (loop_key, description, blocking_on, opened_at, closed_at, resolution) "
            "VALUES (%s, %s, '', FROM_UNIXTIME(%d), NOW(), %s) "
            "ON DUPLICATE KEY UPDATE closed_at=NOW(), resolution=VALUES(resolution)"
            % (_sql_str("pin-%d" % ep),
               _sql_str("Operator directive (pin %s): %s" % (rec.get("t", "?"), str(rec.get("text") or "")[:900])),
               ep, _sql_str(evidence[:1000]))
        )
        env = dict(os.environ, MYSQL_PWD=pw)
        r = subprocess.run(["mariadb", "--skip-ssl", "-h", host, "-P", port,
                            "-u", user, "-D", db, "-e", sql],
                           capture_output=True, text=True, timeout=10, env=env)
        if r.returncode != 0:
            logger.debug("pin open_loops bridge skipped (rc=%s)", r.returncode)
            return False
        return True
    except Exception as e:
        logger.debug("pin open_loops bridge skipped: %s", type(e).__name__)
        return False


def _bridge_open_loop_async(rec: Dict[str, Any]) -> None:
    try:
        threading.Thread(target=_bridge_open_loop, args=(dict(rec),), daemon=True,
                         name="pin-open-loops-bridge").start()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Digest hook (2026-10-04, the operator's idea): after a compaction digest is written,
# match each active pin against the digest's completed content and its OPEN LOOPS.
#   OPEN LOOPS match -> state "open" (confirmed open; clears an older hint)
#   completed match  -> state "likely_done" + hint (NEVER closes the pin)
# Matching: keywords + IDs, and for done-candidates a strict yes/no JSON check by
# the 7B helper (short timeout). Any helper error -> no-op for that pin.
# ---------------------------------------------------------------------------
_DIGEST_SECTIONS = ("WORKING GOAL", "STATE AT EVICTION", "DECISIONS MADE", "OPEN LOOPS",
                    "KEY FACTS & IDS", "KEY FACTS", "LAST SEAM", "CLOSED EARLIER", "PARKED")
_DONE_WORDS = re.compile(
    r"\b(done|completed?|finished|fixed|resolved|delivered|sent|cleaned(?: up)?|removed|deleted|"
    r"applied|verified|confirmed|closed|merged|created|identified|found|unlinked|updated|"
    r"migrated|installed|restarted|configured|complete|cleared|canceled|cancelled|handled|shipped)\b", re.I)
_NEG_DONE = re.compile(r"\b(not|never|no longer|awaiting|pending|blocked|failed|unable|still)\b", re.I)
_STOP = set("""the and for with that this those these them they then than from into onto your you
your our out about have has had was were are is be been being will would should could can
just also only please make sure what when where which while who whom whose how why its it's
there their here some any all each every more most other such very want need lets let's yeah
okay yes fix do did does done get got going gonna up down over under again still need
thing things stuff check look see find figure""".split())
_HELPER_URL_DEFAULT = ""  # set HERMES_PIN_HELPER_URL to enable digest helper LLM
_HELPER_TIMEOUT_S = 8
_HELPER_MAX_CALLS = 3


def _digest_parts(digest: str) -> Dict[str, List[str]]:
    """Split a digest into {"open": [...lines], "done": [...snippets]}."""
    sec = None
    secs: Dict[str, List[str]] = {}
    closed: List[str] = []
    for raw in (digest or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # fix1 2026-10-30: the 9B writes headings both as "OPEN LOOPS: x" and as a bare
        # "OPEN LOOPS" line with content below; the old colon-only regex missed the
        # second form, so those digests produced no hints at all.
        m = re.match(r"^(?:#+\s*)?\**(%s)\**\s*(?::\s*(.*)|$)" % "|".join(re.escape(s) for s in _DIGEST_SECTIONS), line, re.I)
        if m:
            name = m.group(1).upper()
            rest = (m.group(2) or "").strip()
            if name == "CLOSED EARLIER":
                if rest:
                    closed.append(rest)
                sec = "CLOSED EARLIER"   # fix5: bullets under a bare heading count too
                continue
            sec = name
            secs.setdefault(sec, [])
            if rest:
                secs[sec].append(rest)
            continue
        if sec:
            secs.setdefault(sec, []).append(line.lstrip("-• ").strip())
    open_lines = []
    for l in secs.get("OPEN LOOPS", []):
        core = re.sub(r"^\(as of eviction\)\s*", "", l, flags=re.I).strip()
        if core and not re.match(r"^\(?(none|nothing open|n/a)\b", core, re.I):
            open_lines.append(l)
        elif core:
            # "None; all identified cancellation requests from Oct 1-27 are resolved."
            tail = re.sub(r"^\(?(none|nothing open|n/a)\)?[\s;:,.\-—]*", "", core, flags=re.I).strip()
            if tail and _DONE_WORDS.search(tail) and not _NEG_DONE.search(tail):
                closed.append(tail)
    closed += [l for l in secs.get("CLOSED EARLIER", []) if l]
    parked = [l for l in secs.get("PARKED", [])
              if l and not re.match(r"^\(?(none|nothing|n/a)\b", l, re.I)]
    # fix6: a closed/resolved claim without proof is not evidence of anything
    closed = [c for c in closed if has_proof(c)]
    done: List[str] = list(closed)
    for name in ("STATE AT EVICTION", "DECISIONS MADE"):
        for l in secs.get(name, []):
            for s in re.split(r"(?<=[.;])\s+", l):
                if _DONE_WORDS.search(s) and not _NEG_DONE.search(s):
                    done.append(s.strip())
    return {"open": open_lines, "done": done, "closed": closed, "parked": parked}


def _terms(text: str):
    t = str(text or "")
    ids = set(m.lower() for m in re.findall(
        r"\b(?:\d{1,3}(?:\.\d{1,3}){3}|[A-Za-z]*\d[\w\-./]*|[A-Z]{2,}[\w\-]*)\b", t))
    ids = {i.rstrip(".-/") for i in ids if len(i.rstrip(".-/")) >= 2}
    words = set()
    for w in re.findall(r"[A-Za-z][A-Za-z\-']{3,}", t):
        w = w.lower().strip("'-")
        if w in _STOP or len(w) < 4:
            continue
        words.add(w[:5])   # crude stem: cleanup~clean, hosts~host
    return ids, words


def _pin_match(pin: Dict[str, Any], snippet: str) -> bool:
    pid, pw = _terms("%s %s" % (pin.get("text", ""), pin.get("re", "")))
    sid, sw = _terms(snippet)
    id_hit = len(pid & sid)
    w_hit = len(pw & sw)
    if id_hit and w_hit:
        return True
    if not pw:
        return False
    # fix1: long directives rarely share half their words with a terse digest line;
    # accept 2+ shared stems at >=25% (or 3+ outright) — the 7B helper confirms.
    return w_hit >= 3 or (w_hit >= 2 and w_hit / float(len(pw)) >= 0.25)


def _helper_says_done(pin: Dict[str, Any], snippet: str, url: str = None,
                      timeout: float = _HELPER_TIMEOUT_S) -> Optional[bool]:
    """Strict yes/no from the 7B helper. Returns True/False, or None on any error."""
    try:
        import urllib.request
        url = (url or os.environ.get("HERMES_PIN_HELPER_URL") or _HELPER_URL_DEFAULT or "").strip()
        if not url:
            return None
        prompt = (
            "You check whether a work log line says an instruction was COMPLETED.\n"
            "Instruction: %s\n%s"
            "Work log line: %s\n"
            'Answer ONLY with JSON: {"done": true} if the log line clearly says this exact '
            'instruction was completed, otherwise {"done": false}.'
            % (str(pin.get("text", ""))[:400],
               ("Context it replied to: %s\n" % pin["re"][:200]) if pin.get("re") else "",
               snippet[:500]))
        body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 12, "temperature": 0}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        m = re.search(r"\{[^{}]*\}", text or "")
        if not m:
            return None
        val = json.loads(m.group(0)).get("done")
        return val if isinstance(val, bool) else None
    except Exception as e:
        logger.debug("pin digest helper skipped: %s", type(e).__name__)
        return None


def apply_digest_hints(digest: str, use_helper: Optional[bool] = None, helper=None) -> Dict[str, int]:
    """Mark active pins "open" / "likely_done" from a fresh digest. Never closes a
    pin. Returns counts. Deterministic CPU cards (no model) are skipped: their
    "still in play" list is just the last user lines and would match everything."""
    res = {"open": 0, "likely_done": 0}
    d = str(digest or "")
    if not d.strip() or d.lstrip().startswith("CONTEXT CARD (deterministic"):
        return res
    parts = _digest_parts(d)
    for c in parts.get("closed", []):   # fix5: finished sweeps -> CLOSED ledger (proof-checked, fix6)
        record_closed(c, proof=c, source="digest")
    if not parts["open"] and not parts["done"] and not parts.get("parked"):
        return res
    if use_helper is None:
        use_helper = os.environ.get("HERMES_PIN_DIGEST_HELPER", "1").strip().lower() not in (
            "0", "false", "no", "off")
    helper = helper or _helper_says_done
    now = time.time()
    # Decide outside the lock (the helper call can take seconds), then re-apply.
    decisions: Dict[float, Dict[str, Any]] = {}
    calls = 0
    helper_down = False
    for p in active_pins(now):
        key = float(p.get("ts_epoch") or 0)
        park_hit = next((l for l in parts.get("parked", []) if _pin_match(p, l)), None)
        if park_hit:
            decisions[key] = {"state": PARKED_STATE, "hint": park_hit[:200]}
            continue
        open_hit = next((l for l in parts["open"] if _pin_match(p, l)), None)
        if open_hit:
            decisions[key] = {"state": "open", "hint": open_hit[:200]}
            continue
        done_hit = next((s for s in parts["done"] if _pin_match(p, s)), None)
        if not done_hit:
            continue
        if use_helper:
            if calls >= _HELPER_MAX_CALLS or helper_down:
                continue
            calls += 1
            verdict = helper(p, done_hit)
            if verdict is None:
                helper_down = True   # one error -> no more helper calls this digest
            if verdict is not True:
                continue   # False or error -> no-op
        decisions[key] = {"state": "likely_done", "hint": done_hit[:200]}
    if not decisions:
        return res
    try:
        with _locked():
            pins = _load_all()
            changed = False
            for p in pins:
                dec = decisions.get(float(p.get("ts_epoch") or 0))
                if not dec or p.get("state") not in ACTIVE_STATES:
                    continue
                if p.get("state") != dec["state"] or p.get("hint") != dec["hint"]:
                    p["state"] = dec["state"]
                    p["hint"] = dec["hint"]
                    p["hint_at"] = now
                    if dec["state"] == PARKED_STATE:
                        p["parked_at"] = now
                        p["parked_via"] = "digest"
                    changed = True
                res[dec["state"]] = res.get(dec["state"], 0) + 1
            before = [p.get("state") for p in pins]
            _settle(pins, now)
            if changed or before != [p.get("state") for p in pins]:
                _write_all(pins)
                logger.info("pin digest hints persisted: %s", res)
    except Exception as e:
        logger.warning("pin digest hints not written: %s", e)
        return {"open": 0, "likely_done": 0}
    return res


# ---------------------------------------------------------------------------
# Injection block
# ---------------------------------------------------------------------------

_NUDGE = ("[Pin check: if a pinned directive above is already satisfied, close it now with "
          "operator_directive_done(substring, evidence). Close only with evidence; don't start "
          "work to satisfy a pin.]")
_NUDGE_LIKELY = ("[Pin check: %d pin(s) above are tagged \"digest says done\". If the evidence is "
                 "already in front of you, close each with operator_directive_done(substring, "
                 "evidence); if not, leave it. Close only with evidence; don't start work to "
                 "satisfy a pin.]")


def format_operator_pins(nudge: bool = True) -> str:
    """The injection block ("" when there are no active pins).

    Only active (pinned / open / likely_done), unexpired records ride; satisfied
    (doors A/B) and expired (door C) directives stay in the file but stop steering.
    With *nudge*, a one-line close-with-evidence reminder follows the list."""
    pins = active_pins()[-PIN_INJECT_LAST:]
    if not pins:
        return ""
    # fix2 2026-10-30: newest first; older pins are BACKGROUND. The live conversation
    # (the operator's latest messages) is the current topic and wins over any pin; pins only
    # outrank the compaction digest's guesses, never what he is saying right now.
    lines = [
        f"[OPERATOR DIRECTIVES — {_OPERATOR_NAME}'s exact words, newest first. Background memory, NOT the",
        "current topic: their latest messages in the live conversation always win. A pin beats only",
        "the session digest's summary of the past. Do not switch to an older pin unless they bring",
        "it back up. If one looks stale or contradicted, say so and ask. To retire a done one, ask",
        f"{_OPERATOR_NAME} to send \"/done <words>\" (\"/park",
        "<words>\" puts one on hold); pins also expire out of this list after 24 h.]",
    ]
    n_likely = 0
    pins = list(reversed(pins))
    for i, p in enumerate(pins):
        tag = "NEWEST" if i == 0 else "background"
        line = '- %s%s [%s]: "%s"' % ((p["id"] + " ") if p.get("id") else "", p.get("t", "?"), tag, p.get("text", ""))
        if p.get("re"):
            line += ' (re: "%s")' % p["re"]
        st = p.get("state")
        if st == "open":
            line += " [open — digest lists it under OPEN LOOPS]"
        elif st == "likely_done":
            n_likely += 1
            line += (' (digest says done: "%s", confirm and close; auto-retires ~%dh after tagging '
                     'unless re-opened)' % (str(p.get("hint") or "")[:200], LIKELY_DONE_GRACE_SECS // 3600))
        lines.append(line)
    pk = parked_pins()
    if pk:
        lines.append("PARKED (on hold — not open work, don't chase): " + "; ".join(
            str(p.get("text") or "")[:90].replace("\n", " ") for p in pk[-PARKED_SHOW:]))
    cl = _load_closed()[-CLOSED_SHOW:]
    if cl:
        lines.append("CLOSED (finished, don't redo): " + "; ".join(
            str(r.get("text") or "")[:90] for r in reversed(cl)))
    if nudge:
        lines.append(_NUDGE_LIKELY % n_likely if n_likely else _NUDGE)
    return "\n".join(lines)
