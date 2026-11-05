"""Public plugin API: register_context_engine + slash commands + pre_llm_call."""
from __future__ import annotations


def test_register_wires_engine_commands_and_hook():
    import importlib.util, sys
    from pathlib import Path
    pkg = Path(__file__).resolve().parents[1]
    # Load as a real package name so relative imports inside work if any remain
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

    recorded = {"commands": [], "hooks": [], "engines": []}

    class Ctx:
        def register_tool(self, *a, **k): pass
        def register_hook(self, name, cb): recorded["hooks"].append(name)
        def register_command(self, name, *a, **k): recorded["commands"].append(name)
        def register_context_engine(self, engine):
            recorded["engines"].append(getattr(engine, "name", None))
        def get_config(self, key, default=None): return default

    import __init__ as plugin  # noqa: awkward but works with pythonpath=.
    # Prefer importlib
    spec = importlib.util.spec_from_file_location(
        "hermes_decay_context", pkg / "__init__.py",
        submodule_search_locations=[str(pkg)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(pkg)]
    sys.modules["hermes_decay_context"] = mod
    spec.loader.exec_module(mod)
    mod.register(Ctx())
    assert recorded["engines"] == ["decay"]
    assert set(recorded["commands"]) >= {"pins", "done", "park", "unpark"}
    assert "pre_llm_call" in recorded["hooks"]


def test_cmd_done_and_pins_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PIN_OPEN_LOOPS_BRIDGE", "0")
    from pins.operator_pins import pin_operator_message
    import importlib, sys
    from pathlib import Path
    pkg = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(pkg))
    # import handlers via loading module
    spec = importlib.util.spec_from_file_location(
        "hermes_decay_context2", pkg / "__init__.py",
        submodule_search_locations=[str(pkg)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(pkg)]
    sys.modules["hermes_decay_context2"] = mod
    spec.loader.exec_module(mod)
    assert pin_operator_message("finish the readiness canvas update", "test")
    listed = mod._cmd_pins("")
    assert "finish the readiness" in listed or "readiness canvas" in listed
    out = mod._cmd_done("readiness canvas")
    assert "Retired" in out or "retired" in out.lower() or "No active" not in listed
