# cursor-auto-model

A client-side replacement for Cursor Agent CLI's server-side `auto` mode.
Given a prompt, picks the best Cursor model for the task (optionally
decomposes the prompt into a DAG of sub-tasks and runs each with its own
model), using a local [Ollama](https://ollama.com) model as a classifier /
planner.

- **Zero dependencies** beyond `python3` (stdlib only), `bash`, `cursor-agent`, and `ollama`.
- **13 routing buckets, 93 Cursor model ids** — every listed model is reachable.
- **Wave-parallel task execution** with file-write conflict detection.
- **SHA-256 prompt cache** — cold classify 5–7 s, warm hit ~55 ms.
- **Regex intent fast-path** — trivial prompts (greetings, typo fixes, `rename X to Y`, …) skip Ollama entirely (~70 ms).
- **Agent role registry** — `agents/*.yaml` defines personas (`coder`, `reviewer`, `tester`, `architect`, `security-architect`, `doc-writer`). Planner picks one per task; orchestrator prepends the role's system prompt to `cursor-agent --print`.
- **Optional LLM-as-judge** on a sample of runs for offline quality signal.
- **Safe by default**: on any failure, falls back to a known-good model and exits 0.

> ⚠ Cursor's `--model auto` on the CLI is *not* client-side. The CLI asks
> `api2.cursor.sh` what to use and every call we observed returned
> `composer-2-fast`. This project exists because that wasn't enough. See
> [`DESIGN.md`](./DESIGN.md) for the full investigation and
> [`PLAN.md`](./PLAN.md) for the roadmap.

---

## Quickstart

```bash
# 1. Prereqs
ollama pull qwen2.5:latest               # the classifier / planner model
cursor-agent --list-models > cursor-models.tsv   # refresh the catalogue

# 2. Validate the ladders against your catalogue
python3 router.py --validate
# → router: 13 buckets, 145 ladder entries, catalogue=93 ids
#   ok: every ladder entry is a real Cursor model id

# 3. Single-model routing (drop-in for `cursor-agent`)
./cursor-auto-router -p "write a python fib function"
./cursor-auto-router -p "refactor this module into async" --yolo

# 4. Multi-task planning + orchestrated execution
./cursor-auto-plan          -p "add dark mode toggle"      # print the plan
./cursor-auto-plan --json   -p "add dark mode toggle"      # plan as JSON
./cursor-auto-plan --exec   -p "add dark mode toggle"      # run it
./cursor-auto-plan --exec --dry-run -p "add dark mode toggle"   # print commands
```

---

## How it works

1. **Classify** the prompt with a local Ollama model (`qwen2.5:latest` by default) into a `(bucket, latency, effort)` triple.
2. **Resolve** to a Cursor model id via the ladders in [`models.yaml`](./models.yaml). Cheapest-first; `effort=max` walks the top of the ladder.
3. **In planner mode**, decompose the prompt into a list of tasks with `depends_on` links, validate the DAG, and compute wave-ordered execution via Kahn's topological sort.
4. **Execute** each wave in parallel (default `--max-parallel 3`), tail merged logs to stdout, persist per-task logs + `plan.json` to `.cursor-auto-plan/<run-id>/`.

On any parse / classify / Ollama failure at any stage, we fall back to a default model or a single-task plan and never block the caller.

---

## Files

| File | What it is |
|---|---|
| [`router.py`](./router.py) | Classifier + planner + resolver + cache + judge + fast-path. Single file, stdlib only. |
| [`models.yaml`](./models.yaml) | Bucket → ladder configuration. The only place model policy lives. |
| [`fastpath.yaml`](./fastpath.yaml) | Regex intent shortcuts. Matches bypass the Ollama classifier. Edit freely; `router.py --validate` lints it. |
| [`agents/`](./agents) | One YAML per persona (system-prompt preamble, optional bucket/effort hints). `agents/README.md` documents the schema. |
| [`cursor-models.tsv`](./cursor-models.tsv) | Authoritative list of Cursor model ids (dumped from `cursor-agent --list-models`). |
| [`cursor-auto-router`](./cursor-auto-router) | Bash wrapper: classify → `cursor-agent --model <picked>`. |
| [`cursor-auto-plan`](./cursor-auto-plan) | Bash orchestrator: plan → wave-parallel execution. |
| [`logger.js`](./logger.js) | Optional Node.js `--require` preload that logs + redacts Cursor's HTTP traffic. Used in phase-1 investigation; not on the hot path. |
| [`cursor-agent-logged`](./cursor-agent-logged) | Thin wrapper that runs `cursor-agent` with `logger.js` preloaded. |
| [`analyze.js`](./analyze.js) | Offline parser for `cursor-auto.log`. |
| [`DESIGN.md`](./DESIGN.md) | Phase-1 engineering log (investigation + rationale). |
| [`PLAN.md`](./PLAN.md) | Roadmap (past / present / future). |

---

## Configuration

All via environment variables. Nothing is read from `~` — the project is
self-contained in its own directory.

### Router / planner

| Variable | Default | What it controls |
|---|---|---|
| `ROUTER_OLLAMA_URL` | `http://localhost:11434` | Classifier Ollama endpoint |
| `ROUTER_OLLAMA_MODEL` | `qwen2.5:latest` | Classifier model |
| `ROUTER_TIMEOUT_SEC` | `12` | Classifier timeout |
| `ROUTER_PLANNER_URL` | = `ROUTER_OLLAMA_URL` | Planner Ollama endpoint (can be remote) |
| `ROUTER_PLANNER_MODEL` | `qwen2.5:latest` | Planner model |
| `ROUTER_PLANNER_TIMEOUT_SEC` | `60` | Planner timeout |
| `ROUTER_MODELS_FILE` | `./models.yaml` | Override bucket config |
| `ROUTER_CATALOGUE_FILE` | `./cursor-models.tsv` | Override model catalogue |
| `ROUTER_LOG_FILE` | `./router.log` | Classifier decision log (newline-JSON) |
| `ROUTER_PLAN_LOG_FILE` | `./plan.log` | Planner decision log (newline-JSON) |

### Fast-path (regex shortcuts)

| Variable | Default | What it controls |
|---|---|---|
| `ROUTER_FASTPATH` | `1` | `0` disables the regex fast-path entirely |
| `ROUTER_FASTPATH_FILE` | `./fastpath.yaml` | Override the fast-path rule file |

### Agent role personas

| Variable | Default | What it controls |
|---|---|---|
| `ROUTER_ROLES` | `1` | `0` disables role personas entirely (planner stops emitting `role`, orchestrator stops prepending preambles) |
| `ROUTER_ROLES_DIR` | `./agents` | Override the directory scanned for `*.yaml` personas |

### Cache

| Variable | Default | What it controls |
|---|---|---|
| `ROUTER_CACHE_DIR` | `./.cache` | Where classifier + planner cache live |
| `ROUTER_CACHE_TTL_SEC` | `604800` (1 week) | `0` disables cache entirely |

### LLM-as-judge (opt-in)

| Variable | Default | What it controls |
|---|---|---|
| `ROUTER_JUDGE_SAMPLE` | `0` | Fraction `0..1` of runs to judge. Set `0.1` to judge 10%. |
| `ROUTER_JUDGE_URL` | = `ROUTER_OLLAMA_URL` | Judge Ollama endpoint |
| `ROUTER_JUDGE_MODEL` | `qwen2.5:latest` | Judge model (ideally different from classifier!) |
| `ROUTER_JUDGE_TIMEOUT_SEC` | `30` | Judge timeout |
| `ROUTER_JUDGE_LOG_FILE` | `./judge.log` | Judge verdict log (newline-JSON) |

### Orchestrator

| Variable | Default | What it controls |
|---|---|---|
| `CAP_YOLO` | `1` in `--exec`, unset otherwise | Pass `--force --sandbox disabled --approve-mcps --trust` to `cursor-agent`. Set `0` to require manual approval (headless runs will hang). |
| `CURSOR_AGENT_BIN` | `cursor-agent` | Override the binary. |

### CLI flags — `cursor-auto-plan`

```
-p, --print PROMPT       the prompt to plan/execute
--json                   print plan as JSON (still goes to stdout)
--exec                   execute the plan (default is dry-plan)
--dry-run                with --exec, print commands but don't run
--keep-going             don't stop on first failed wave
--max-parallel N         max concurrent tasks per wave (default 3)
--explain                verbose planner reasoning to stderr
```

---

## Tuning the router

Everything routing-related is in [`models.yaml`](./models.yaml). Two moves
cover 80% of tuning needs:

**Add / retire a Cursor model.** Re-dump the catalogue:

```bash
cursor-agent --list-models > cursor-models.tsv
python3 router.py --validate   # will complain about any stale ids
```

Then edit `models.yaml` to add the new id to the right bucket's `ladder` /
`fast_ladder`. No code change needed.

**The router is consistently picking too cheap / too expensive.** Either:

- Edit the `good_for:` description of the bucket — the classifier reads
  those descriptions verbatim as part of its prompt.
- Edit the ladder order (cheaper at the front, stronger at the back).
- Turn on `ROUTER_JUDGE_SAMPLE=0.1` for a day, then grep `judge.log` for
  `"verdict":"over"` / `"verdict":"under"` and adjust accordingly.

---

## Development

```bash
# Smoke test
python3 router.py --validate
echo "fix a typo" | python3 router.py --explain
python3 router.py --plan --prompt "add dark mode" --explain
./cursor-auto-plan --exec --dry-run -p "add dark mode"

# See every classifier / planner / judge decision:
tail -f router.log plan.log judge.log
```

Runtime artifacts (`router.log`, `plan.log`, `judge.log`, `cursor-auto.log`,
`.cache/`, `.cursor-auto-plan/`) are `.gitignore`d; they're per-user and
regenerate on demand.

---

## License

MIT. See [`LICENSE`](./LICENSE).
