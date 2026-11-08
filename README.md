# hermes-decay-context

**Decay + Pins** — primary Hermes Plugin Hackathon submit.

## What it does

1. **Decay context engine** (`name: decay`) — drops oldest turns (already retained
   in memory) instead of lossy summarization, so the agentic tool-loop is not
   severed. Folds evicted turns into a rolling `session_digest` and trims
   oversized tool results when the protected tail alone exceeds budget.
2. **Operator Pins** — short operator directives stored **verbatim**. Create one
   with `/pin <text>` (that command calls `pin_operator_message`). They ride
   every turn (via a `pre_llm_call` inject) and are managed with `/pins`,
   `/done <words|last>`, `/park`, `/unpark`. The inject hook does **not** read
   `user_message` and does not pin ordinary chat.

No Hermes core patches (catalog rule 9). This plugin declares no
`secret_scope` credential and does not read `model.api_key` (or any other
key) out of `config.yaml`. Optional bridges are **off by default**.

## Install

```bash
# General plugin (commands + engine registration)
cp -R hermes-decay-context ~/.hermes/plugins/
# or: hermes plugins install /path/to/hermes-decay-context
```

Select the engine in `~/.hermes/config.yaml`:

```yaml
context:
  engine: decay
  decay:
    protect_last_n: 20
    threshold_percent: 0.8
    digest_enabled: true
```

Restart the gateway / CLI so the plugin loads.

### Directory-engine alternative

You can also drop `engine.py` + `pins/` into `plugins/context_engine/decay/`
(rename `engine.py` → `__init__.py` or re-export `DecayEngine`). Directory
loaders do **not** call `register()`, so slash commands / the inject hook only
come from the general-plugin install.

## Optional env (all off by default)

| Env | Purpose |
|-----|---------|
| `HERMES_PIN_AGENT_NAME` | Agent's name, so "hey <name>, …" is parsed as a vocative (default none) |
| `HERMES_PIN_OPERATOR_NAME` | Name shown in the injected pin block (default "the operator") |
| `HERMES_DECAY_NEW_MSG_LABEL` | Label your gateway puts before the newest user message (default `[New user message]`) |
| `HERMES_DECAY_PINS_INJECT=0` | Skip the pre_llm_call pin inject (host already injects pins) |
| `HERMES_DECAY_PINS_TOOL=0` | Skip registering the `operator_directive_done` tool |
| `HERMES_PIN_OPEN_LOOPS_BRIDGE=1` | Write retired pins to an open_loops SQL table |
| `HERMES_PIN_DB_USER` / `HERMES_PIN_DB_PASSWORD` | DB user and password, from the environment only when the bridge is on (never from `config.yaml` or `pass`) |
| `HERMES_PIN_DB_HOST` / `PORT` / `NAME` | Defaults `127.0.0.1` / `3306` / `hermes` |
| `HERMES_PIN_HELPER_URL` | Optional OpenAI-compatible URL for digest "likely done" helper |

## Validate / test

```bash
export HERMES_HOME=$(mktemp -d)
hermes plugins validate /path/to/hermes-decay-context --install-deps
pytest /path/to/hermes-decay-context/tests -q
```

Digest LLM calls go through ``ctx.llm`` and a plugin-owned ``decay_digest``
auxiliary task (``inherit_from: compression``). Directory-engine installs that
skip ``register()`` fall back to the deterministic CPU digest — they never
import ``agent.auxiliary_client`` (catalog rule 9).

## Layout

```
hermes-decay-context/
├── plugin.yaml
├── __init__.py          # register(ctx)
├── engine.py            # DecayEngine
├── pins/
│   ├── __init__.py
│   └── operator_pins.py
├── README.md
└── tests/
```

## Agent tool

`operator_directive_done(substring, evidence)` lets the agent retire a pin once it can show the proof.
It needs evidence to run, and it never deletes anything.

`context.decay.*` (including `protect_last_n`, `threshold_percent`,
`digest_enabled`, `context_length`) is loaded with Hermes' `hermes_yaml` from
the active profile `config.yaml`. Hermes does not ship PyYAML, and this plugin
does not declare it. The home directory is `hermes_constants.get_hermes_home()`
so a context-local profile override in one process is not the process-wide
`HERMES_HOME`.

## Disclosures

- **Files** (under the active profile from `get_hermes_home()`, not a hardcoded `~/.hermes`):
  - `operator_directives.jsonl` (and its lock) and `closed_work.jsonl` — pins
  - `session_digests/` — rolling digest review (`LATEST.md`, per-thread copies, timestamped snapshots)
  - `decay_distill_audit.jsonl` — one audit line when a digest is retained
  - `decay_est_cal.json` — token-estimator calibration
  - `router_ctx.json` — last host-reported context limit, if the host calls `set_router_ctx_limit`
- **Secrets:** none declared. No `agent.secret_scope` entry. Model credentials in
  `config.yaml` are not read and are not forwarded.
- **Network:** none by default. The optional digest helper (`HERMES_PIN_HELPER_URL`) is opt-in.
  There is no router-status probe.
- **Shell:** only with the optional open-loops bridge. The `mariadb` child is started with
  `PATH` and `MYSQL_PWD` only. Provider keys and the rest of the Hermes process environment
  are not inherited. The password is not placed on argv.
- **Background work:** the optional bridge may start one short-lived daemon thread. Nothing else.

## Tests

```bash
HERMES_HOME=$(mktemp -d) PYTHONPATH=.:/path/to/hermes-agent python -m pytest -q --import-mode=importlib tests
```

Loader smoke that imported ``hermes_cli.plugins`` was removed (private core
import under rule 9). Use ``hermes plugins validate`` for admission checks.
