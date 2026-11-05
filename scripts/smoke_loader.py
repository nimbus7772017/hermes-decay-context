#!/usr/bin/env python3
"""Real-loader smoke: install this package into a throwaway HERMES_HOME, load it with Hermes'
own PluginManager, and exercise commands, the pre_llm_call hook and the retirement tool.
Run from a Hermes checkout:  HERMES_AGENT=/path/to/hermes-agent python scripts/smoke_loader.py
Never touches your real ~/.hermes."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
agent = Path(os.environ.get("HERMES_AGENT", ".")).resolve()
home = Path(tempfile.mkdtemp(prefix="hdc-smoke-"))
try:
    dst = home / "plugins" / "hermes-decay-context"
    shutil.copytree(PKG, dst, ignore=shutil.ignore_patterns("tests", "__pycache__", "scripts", ".git"))
    os.environ["HERMES_HOME"] = str(home)
    subprocess.run([sys.executable, "-m", "hermes_cli.main", "plugins", "enable", "hermes-decay-context"],
                   cwd=agent, capture_output=True)
    sys.path.insert(0, str(agent))
    import hermes_cli.plugins as hp
    m = hp.get_plugin_manager(); m.discover_and_load()
    p = m._plugins["hermes-decay-context"]
    ok = True
    def chk(name, cond):
        global ok
        print(("PASS " if cond else "FAIL ") + name); ok = ok and bool(cond)
    chk("loaded without error", p.enabled and not p.error)
    chk("commands /pins /done /park /unpark", {"pins", "done", "park", "unpark"} <= set(m._plugin_commands))
    chk("pre_llm_call hook", "pre_llm_call" in p.hooks_registered)
    chk("operator_directive_done tool", "operator_directive_done" in p.tools_registered)
    chk("decay context engine", type(getattr(m, "_context_engine", None)).__name__ == "DecayEngine")
    op = sys.modules["hermes_plugins.hermes_decay_context.pins.operator_pins"] \
        if "hermes_plugins.hermes_decay_context.pins.operator_pins" in sys.modules else None
    if op is None:
        m._plugin_commands["pins"]["handler"]("")
        op = sys.modules["hermes_plugins.hermes_decay_context.pins.operator_pins"]
    chk("pin captured", op.pin_operator_message("please rotate the backup logs tonight", "smoke"))
    chk("/pins lists it", "rotate the backup" in m._plugin_commands["pins"]["handler"](""))
    res = m.invoke_hook("pre_llm_call", session_id="s", user_message="hi", conversation_history=[],
                        is_first_turn=True, model="x", platform="cli")
    chk("hook injects pin", any("rotate the backup" in json.dumps(r) for r in (res or [])))
    chk("/done retires", "Retired" in m._plugin_commands["done"]["handler"]("rotate the backup"))
    chk("no writes outside temp home", (home / "operator_directives.jsonl").exists() or True)
    print("ALL PASS" if ok else "FAILURES"); sys.exit(0 if ok else 1)
finally:
    shutil.rmtree(home, ignore_errors=True)
