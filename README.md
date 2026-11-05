# hermes-decay-context

**Decay + Pins** — primary Hermes Plugin Hackathon submit.

## What it does

1. **Decay context engine** (`name: decay`) — drops oldest turns (already retained
   in memory) instead of lossy summarization, so the agentic tool-loop is not
   severed. Folds evicted turns into a rolling `session_digest` and trims
   oversized tool results when the protected tail alone exceeds budget.
2. **Operator Pins** — short operator directives stored **verbatim**. They ride
   every turn (via a `pre_llm_call` inject) and are managed with slash commands:
   `/pins`, `/done <words|last>`, `/park`, `/unpark`.

No Hermes core patches (catalog rule 9). Secrets only via env / `get_secret`
patterns (rule 11) — optional bridges are **off by default**.

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
| `HERMES_DECAY_ROUTER_STATUS=1` | Router-aware sizing: read `/router/status` + `X-Club-*` headers from a compatible router (off) |
| `HERMES_DECAY_PINS_INJECT=0` | Skip the pre_llm_call pin inject (host already injects pins) |
| `HERMES_DECAY_PINS_TOOL=0` | Skip registering the `operator_directive_done` tool |
| `HERMES_PIN_OPEN_LOOPS_BRIDGE=1` | Write retired pins to an open_loops SQL table |
| `HERMES_PIN_DB_USER` / `HERMES_PIN_DB_PASSWORD` | DB creds (never from `pass`) |
| `HERMES_PIN_DB_HOST` / `PORT` / `NAME` | Defaults `127.0.0.1` / `3306` / `hermes` |
| `HERMES_PIN_HELPER_URL` | Optional OpenAI-compatible URL for digest "likely done" helper |
| `HERMES_DECAY_PINS_INJECT=0` | Disable pre_llm_call pin inject (use when gateway already injects) |

## Validate / test

```bash
export HERMES_HOME=$(mktemp -d)
hermes plugins validate /path/to/hermes-decay-context
pytest /path/to/hermes-decay-context/tests -q
```

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

## Disclosures

- **Files:** pins live in `$HERMES_HOME/operator_directives.jsonl`. A digest audit line goes to
  `$HERMES_HOME/decay_distill_audit.jsonl`.
- **Network:** none by default. The optional digest helper LLM (`HERMES_PIN_HELPER_URL`) and router-aware sizing
  (`HERMES_DECAY_ROUTER_STATUS=1`, which calls your configured model `base_url`) are both opt-in.
- **Shell:** only with the optional open-loops bridge (`mysql` client, password passed via the `MYSQL_PWD` env var).
- **Background work:** none.

## Tests

```bash
HERMES_HOME=$(mktemp -d) PYTHONPATH=.:/path/to/hermes-agent python -m pytest -q --import-mode=importlib tests
HERMES_AGENT=/path/to/hermes-agent python scripts/smoke_loader.py   # real PluginManager, temp HERMES_HOME
```
