"""Catalog hygiene (2026-11-05): tool registration, fleet-strip, router sizing opt-in."""
from __future__ import annotations
import importlib.util, json, re, sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
FLEET = re.compile(r"(?i)\b(nina|bythos|michael|pleroma|cyno|amber|phronesis|nimbus|tailscale)\b|100\.\d+\.\d+\.\d+|:8122")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PKG / "__init__.py", submodule_search_locations=[str(PKG)])
    mod = importlib.util.module_from_spec(spec); mod.__path__ = [str(PKG)]
    sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def test_no_fleet_names_in_shipped_code():
    for f in ["__init__.py", "engine.py", "pins/operator_pins.py", "plugin.yaml"]:
        txt = (PKG / f).read_text(encoding="utf-8")
        txt = txt.replace("author: Nimbus Solutions", "")
        hits = [m.group(0) for m in FLEET.finditer(txt)]
        assert not hits, f"{f}: {sorted(set(hits))}"


def test_register_tool_and_tool_roundtrip(monkeypatch):
    mod = _load("hdc_tool")
    tools = {}
    class Ctx:
        def register_tool(self, name, toolset, schema, handler, **k): tools[name] = (schema, handler)
        def register_hook(self, *a, **k): pass
        def register_command(self, *a, **k): pass
        def register_context_engine(self, e): pass
    mod.register(Ctx())
    assert "operator_directive_done" in tools
    schema, h = tools["operator_directive_done"]
    assert schema["parameters"]["required"] == ["substring", "evidence"]
    assert "error" in json.loads(h({"substring": "x", "evidence": ""}))
    from pins.operator_pins import pin_operator_message
    assert pin_operator_message("please rotate the backup logs tonight", "test")
    out = json.loads(h({"substring": "rotate the backup", "evidence": "rotated"}))
    assert out.get("retired") == 1


def test_tool_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_DECAY_PINS_TOOL", "0")
    mod = _load("hdc_tool_off"); tools = []
    class Ctx:
        def register_tool(self, name, *a, **k): tools.append(name)
        def register_hook(self, *a, **k): pass
        def register_command(self, *a, **k): pass
        def register_context_engine(self, e): pass
    mod.register(Ctx())
    assert tools == []


def test_router_status_does_not_read_or_forward_model_key(tmp_path, monkeypatch):
    import engine
    import urllib.request
    sentinel = "sentinel-model-key-not-real"
    (tmp_path / "config.yaml").write_text(
        "model:\n  base_url: http://127.0.0.1:9/v1\n  api_key: %s\n" % sentinel
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DECAY_ROUTER_STATUS", "1")
    calls = []

    def _refuse(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("router probe must not open a connection")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    assert engine._cf_status() is None
    assert calls == []
    assert not hasattr(engine, "_cf_router_conf")
    src = (PKG / "engine.py").read_text(encoding="utf-8")
    assert "import yaml" not in src
    assert "router/status" not in src
    assert sentinel not in src


def test_generic_new_message_label():
    import engine
    assert engine._NEW_MSG_LABEL == "[New user message]"


def test_register_command_without_args_hint():
    """Hermes 0.10 PluginContext.register_command has no args_hint kwarg."""
    mod = _load("hdc_old_api"); cmds = []
    class Ctx:
        def register_tool(self, *a, **k): pass
        def register_hook(self, *a, **k): pass
        def register_command(self, name, handler, description=""): cmds.append(name)
        def register_context_engine(self, e): pass
    mod.register(Ctx())
    assert set(cmds) == {"pin", "pins", "done", "park", "unpark"}
