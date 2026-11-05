"""hermes-decay-context — Decay context engine + operator Pins.

Install as a general Hermes plugin (drop into ``~/.hermes/plugins/hermes-decay-context/``
or ``hermes plugins install``) **or** place ``engine.py`` (+ ``pins/``) under
``plugins/context_engine/decay/`` for the directory-engine path.

As a general plugin, ``register(ctx)`` wires:

* ``ctx.register_context_engine(DecayEngine())`` — select with ``context.engine: decay``
* slash commands ``/pins``, ``/done``, ``/park``, ``/unpark``
* ``pre_llm_call`` hook that injects ``format_operator_pins()`` as context

No Hermes core patches (catalog rule 9). Digest LLM uses ``ctx.llm`` +
``register_auxiliary_task(..., inherit_from="compression")``. Optional
open-loops DB bridge is OFF unless explicitly enabled via env (see README).
"""
from __future__ import annotations

from typing import Any

try:
    from .engine import DecayEngine  # noqa: F401
except ImportError:  # flat sys.path / pytest collecting parent dir
    from engine import DecayEngine  # noqa: F401


def _op():
    """pins.operator_pins as a submodule of this plugin (Hermes loads plugins as
    hermes_plugins.<name>, so the package dir is NOT on sys.path); flat fallback for tests."""
    try:
        from .pins import operator_pins as m  # type: ignore
    except ImportError:
        from pins import operator_pins as m  # type: ignore
    return m


def _cmd_pins(raw_args: str = "") -> str:
    _m = _op(); list_pins_text = _m.list_pins_text
    return list_pins_text()


def _cmd_done(raw_args: str = "") -> str:
    _m = _op(); retire_operator_directive = _m.retire_operator_directive; list_pins_text = _m.list_pins_text
    arg = (raw_args or "").strip()
    if not arg:
        return (
            'Usage: "/done <a few words from the pin>" or "/done last". '
            '"/pins" lists the active pins.'
        )
    ok = retire_operator_directive(arg)
    if not ok:
        return f'No active pinned directive matches "{arg[:60]}". Send "/pins" to list them.'
    return f'Retired pin matching "{arg[:60]}".\n' + list_pins_text()


def _cmd_park(raw_args: str = "") -> str:
    _m = _op(); park_operator_directive = _m.park_operator_directive
    arg = (raw_args or "").strip()
    if not arg:
        return 'Usage: "/park <a few words from the pin>" or "/park last".'
    ok = park_operator_directive(arg)
    if not ok:
        return f'No active pin matches "{arg}". "/pins" lists them.'
    return f'Parked pin matching "{arg[:60]}".'


def _cmd_unpark(raw_args: str = "") -> str:
    _m = _op(); unpark_operator_directive = _m.unpark_operator_directive
    arg = (raw_args or "").strip()
    if not arg:
        return 'Usage: "/unpark <a few words from the parked pin>" or "/unpark last".'
    ok = unpark_operator_directive(arg)
    if not ok:
        return f'No parked pin matches "{arg}". "/pins" lists them.'
    return f'Unparked pin matching "{arg[:60]}".'


def _pre_llm_inject(**kwargs: Any):
    """Inject active operator pins into the turn context (fail-open).

    Set HERMES_DECAY_PINS_INJECT=0 when the host gateway already appends
    format_operator_pins() (host dual-install) to avoid double injection.
    """
    import os
    if os.environ.get("HERMES_DECAY_PINS_INJECT", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return None
    try:
        _m = _op(); format_operator_pins = _m.format_operator_pins
        block = format_operator_pins()
        if block and str(block).strip():
            return {"context": str(block)}
    except Exception:
        return None
    return None


_DONE_SCHEMA = {
    "name": "operator_directive_done",
    "description": (
        "Retire one of the operator's pinned OPERATOR DIRECTIVES when it is already satisfied. "
        "substring: a few verbatim words from the directive (case-insensitive), or \"last\". "
        "evidence: one short line proving completion. Close only with evidence; never start work "
        "just to satisfy a pin. Retired directives stop being injected; nothing is deleted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "substring": {"type": "string", "description": "Verbatim words from the pin (or \"last\")."},
            "evidence": {"type": "string", "description": "Short completion evidence."},
        },
        "required": ["substring", "evidence"],
    },
}


def _tool_done(args: dict, **_kw) -> str:
    import json
    _m = _op(); retire_operator_directive = _m.retire_operator_directive
    sub = str((args or {}).get("substring") or "")
    ev = str((args or {}).get("evidence") or "").strip()
    if not ev:
        return json.dumps({"error": "evidence required: one short line proving the directive is done"})
    try:
        hits = retire_operator_directive(sub, evidence=ev, actor="agent-tool")
    except Exception as exc:  # fail-open: report, never raise into the agent loop
        return json.dumps({"error": f"retire failed: {exc}"})
    if not hits:
        return json.dumps({"error": f'no active pinned directive matched "{sub[:60]}"'})
    return json.dumps({"retired": len(hits) if isinstance(hits, list) else 1}, ensure_ascii=False)


def _reg_cmd(ctx, name, fn, description, args_hint=None):
    """register_command across Hermes versions (args_hint is newer than 0.10)."""
    if args_hint:
        try:
            return ctx.register_command(name, fn, description=description, args_hint=args_hint)
        except TypeError:
            pass
    return ctx.register_command(name, fn, description=description)


def register(ctx) -> None:
    """Hermes general-plugin entrypoint."""
    # Public digest seat: plugin-owned aux task that follows auxiliary.compression
    # until the operator pins decay_digest directly. Falls back to main-model
    # ctx.llm.complete() when register_auxiliary_task is unavailable.
    task_key = None
    try:
        ctx.register_auxiliary_task(
            "decay_digest",
            display_name="Decay digest",
            description="Rolling session digest for the decay context engine",
            inherit_from="compression",
        )
        task_key = "decay_digest"
    except Exception:
        pass
    try:
        from .engine import wire_digest_llm
    except ImportError:
        from engine import wire_digest_llm  # type: ignore
    try:
        wire_digest_llm(ctx, task_key=task_key)
    except Exception:
        pass

    try:
        ctx.register_context_engine(DecayEngine())
    except Exception:
        pass

    _reg_cmd(ctx, "pins", _cmd_pins, "List active operator-directive pins")
    _reg_cmd(ctx, "done", _cmd_done, "Retire a pin by substring or 'last'", "<words|last>")
    _reg_cmd(ctx, "park", _cmd_park, "Park (hold) a pin by substring or 'last'", "<words|last>")
    _reg_cmd(ctx, "unpark", _cmd_unpark, "Unpark a held pin by substring or 'last'", "<words|last>")
    ctx.register_hook("pre_llm_call", _pre_llm_inject)
    # Agent-side retirement tool. HERMES_DECAY_PINS_TOOL=0 skips it when the host already ships one.
    import os
    if os.environ.get("HERMES_DECAY_PINS_TOOL", "1").strip().lower() not in ("0", "false", "no", "off"):
        try:
            ctx.register_tool(
                name="operator_directive_done", toolset="decay_pins", schema=_DONE_SCHEMA,
                handler=_tool_done, description=_DONE_SCHEMA["description"], emoji="📌",
            )
        except Exception:
            pass
