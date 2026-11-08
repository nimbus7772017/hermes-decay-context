"""Review blockers from the catalog review of 4ce34d74.

Pin capture is an explicit /pin command registered on the plugin context.
The router probe does not read or forward a model key. The mariadb child
does not inherit the process environment. context.decay settings load through
hermes_yaml. Profile writes follow hermes_constants.get_hermes_home().
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, PKG / "__init__.py", submodule_search_locations=[str(PKG)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(PKG)]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Ctx:
    def __init__(self):
        self.commands = {}
        self.hooks = {}

    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, name, callback):
        self.hooks[name] = callback

    def register_command(self, name, handler, description="", args_hint=None):
        self.commands[name] = handler

    def register_context_engine(self, engine):
        pass

    def register_auxiliary_task(self, *args, **kwargs):
        pass


def test_pin_command_registered_and_hook_does_not_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PIN_OPEN_LOOPS_BRIDGE", "0")
    mod = _load("hdc_pin_cmd")
    ctx = _Ctx()
    mod.register(ctx)
    assert "pin" in ctx.commands
    hook = ctx.hooks["pre_llm_call"]

    from pins.operator_pins import active_pins

    directive = "please check the backup logs tonight"
    assert hook(user_message=directive) is None
    assert active_pins() == []

    usage = ctx.commands["pin"]("  ")
    assert usage.startswith("Usage:")
    assert active_pins() == []

    listed = ctx.commands["pin"](directive)
    assert listed.startswith("Pinned.")
    assert "backup logs" in listed
    assert len(active_pins()) == 1

    injected = hook(user_message="please check the backup logs tonight")
    assert injected and "backup logs" in injected["context"]
    assert len(active_pins()) == 1


def test_mariadb_child_env_drops_provider_sentinel(monkeypatch):
    from pins import operator_pins as pins

    sentinel = "sentinel-provider-key-do-not-inherit"
    monkeypatch.setenv("HERMES_PIN_OPEN_LOOPS_BRIDGE", "1")
    monkeypatch.setenv("HERMES_PIN_DB_USER", "pinuser")
    monkeypatch.setenv("HERMES_PIN_DB_PASSWORD", "pin-db-sentinel")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", sentinel)
    captured = {}

    class _Result:
        returncode = 0

    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env")
        return _Result()

    monkeypatch.setattr(pins.subprocess, "run", _fake_run)
    assert pins._bridge_open_loop({"ts_epoch": 1, "text": "check logs", "evidence": "done"}) is True
    env = captured["env"]
    assert set(env) == {"PATH", "MYSQL_PWD"}
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["MYSQL_PWD"] == "pin-db-sentinel"
    assert sentinel not in env.values()
    joined = " ".join(captured["argv"])
    assert "pin-db-sentinel" not in joined
    assert sentinel not in joined


def test_nondefault_decay_settings_follow_profile_not_process_env(tmp_path, monkeypatch):
    import builtins

    process_home = tmp_path / "process"
    profile_home = tmp_path / "profile-b"
    process_home.mkdir()
    profile_home.mkdir()
    (process_home / "config.yaml").write_text(
        "context:\n  decay:\n    protect_last_n: 4\n    threshold_percent: 0.55\n"
        "    context_length: 70000\n    digest_enabled: true\n    digest_max_tokens: 111\n",
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        "context:\n  engine: decay\n  decay:\n"
        "    protect_last_n: 9\n    protect_first_n: 1\n"
        "    threshold_percent: 0.42\n    digest_enabled: false\n"
        "    digest_max_tokens: 321\n    context_length: 131072\n"
        "    threshold_tokens_cap: 8000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(profile_home)
    real_import = builtins.__import__

    def _guard(name, *args, **kwargs):
        if name == "yaml" or name.startswith("yaml."):
            raise AssertionError("PyYAML must not be imported; use hermes_yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard)
    try:
        from engine import DecayEngine
        engine = DecayEngine()
        explicit = DecayEngine(protect_last_n=11)
    finally:
        reset_hermes_home_override(token)

    assert engine.protect_last_n == 9
    assert engine.protect_first_n == 1
    assert engine.threshold_percent == pytest.approx(0.42)
    assert engine.context_length == 131072
    assert engine._digest_enabled is False
    assert engine._digest_max_tokens == 321
    assert engine.threshold_tokens_cap == 8000
    assert engine.threshold_tokens == 8000
    assert explicit.protect_last_n == 11
    assert not (process_home / "decay_est_cal.json").exists()


class _Provider:
    def retain_document(self, content, *, kind, metadata=None, **kwargs):
        return True


class _MM:
    providers = [_Provider()]


def test_profile_override_isolates_digest_cal_and_audit(tmp_path, monkeypatch):
    process_home = tmp_path / "process"
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    for path in (process_home, profile_a, profile_b):
        path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from engine import DecayEngine

    def _write(home, body, limit):
        token = set_hermes_home_override(home)
        try:
            engine = DecayEngine(config_context_length=65536)
            engine._est_cal = 1.25
            engine._est_cal_n = 3
            engine._save_est_cal()
            engine.session_id = "sess"
            engine._digest_thread_key = "thread-" + home.name
            engine._memory_manager = _MM()
            path = engine._write_digest_review(body, trigger="compact")
            assert engine._retain_digest(body, 2) is True
            engine.set_router_ctx_limit(limit)
            return path
        finally:
            reset_hermes_home_override(token)

    path_a = _write(profile_a, "GOAL: profile a only", 90000)
    path_b = _write(profile_b, "GOAL: profile b only", 100000)

    assert path_a.startswith(str(profile_a))
    assert path_b.startswith(str(profile_b))
    assert "profile a only" in (profile_a / "session_digests" / "LATEST.md").read_text(encoding="utf-8")
    assert "profile b only" in (profile_b / "session_digests" / "LATEST.md").read_text(encoding="utf-8")
    assert "profile b only" not in (profile_a / "session_digests" / "LATEST.md").read_text(encoding="utf-8")
    assert (profile_a / "decay_est_cal.json").is_file()
    assert (profile_b / "decay_est_cal.json").is_file()
    audit_a = (profile_a / "decay_distill_audit.jsonl").read_text(encoding="utf-8")
    audit_b = (profile_b / "decay_distill_audit.jsonl").read_text(encoding="utf-8")
    assert audit_a.strip() and audit_b.strip()
    assert "session_digest" in audit_a
    for name in ("session_digests", "decay_est_cal.json", "decay_distill_audit.jsonl", "router_ctx.json"):
        assert not (process_home / name).exists()
        assert (profile_a / name).exists()
        assert (profile_b / name).exists()
