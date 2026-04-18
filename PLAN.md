# PLAN.md — full structural plan

Where we started, where we are, and where we're going.

For the detailed technical journey of **phase 1** (investigating Cursor's
server-side `auto`, building the TLS logger, discovering the Rust native
addon), see [`DESIGN.md`](./DESIGN.md). This file is the forward-looking
roadmap; DESIGN.md is the backward-looking engineering log.

---

## 0. Where we started

**Original question:** *How does Cursor Agent CLI's `auto` mode actually pick
a model, and can we predict or replicate it client-side?*

This turned into a multi-stage investigation that changed the goal twice
along the way:

| Phase | Goal | Outcome |
|---|---|---|
| 0 | Understand `auto` by reading the CLI | `auto` is server-side; CLI just asks `api2.cursor.sh` via `GetDefaultModelForCli`. |
| 1 | Log the wire format to observe `auto` | Built `logger.js` (Node `--require` preload, TLS hook, HTTP framing, decompression, redaction). Captured catalogue + telemetry, but **chat RPC was invisible** — it goes through a Rust native addon with its own network stack, bypassing Node.js. |
| 2 | Since we can't observe it, **replace it** | Built `router.py`: a two-stage classifier (local Ollama → bucket → deterministic ladder → Cursor model id). One model per prompt. |
| 3 | One model per prompt is a toy; upgrade to a planner | Added planner mode: decompose prompt → DAG of subtasks → wave-ordered execution with per-task model selection. `cursor-auto-plan` wraps this. |
| 4 | Steal good ideas from existing OSS routers | Researched 5 projects (`agent-orchestrator`, `virtusoul-router`, `optimus-code`, `Routerly`, `llm-router`). Shipped Tier-1 ports: yolo-flag bundle, SHA-256 prompt cache, 4-tier JSON parser, long-prompt stdin delivery, LLM-as-judge. |

---

## 1. Where we are today

### 1.1 System shape

```
             ┌──────────────────────┐
  prompt ──▶ │  cursor-auto-plan    │  ← orchestrator (bash)
             │   (--exec --dry-run) │
             └──────────┬───────────┘
                        │
             ┌──────────▼───────────┐
             │  router.py --plan    │  ← planner (Python, stdlib only)
             │   --json             │
             └──────┬────────┬──────┘
                    │        │
             ┌──────▼──┐  ┌──▼───────────┐
             │ Ollama  │  │ models.yaml  │  ← bucket ladders
             │ planner │  │ cursor-      │  ← every real Cursor id
             │ LLM     │  │  models.tsv  │
             └─────────┘  └──────────────┘
                    │
              plan.json (DAG)
                    │
      ┌─────────────▼─────────────┐
      │  Wave executor (bash)     │
      │    wave 1 ║ wave 2 ║ …   │
      │  each task:               │
      │    cursor-agent --model X │
      │      --force --sandbox    │
      │      disabled             │
      │      --approve-mcps       │
      │      --trust              │
      │      --print "<subtask>"  │
      └───────────────────────────┘
```

### 1.2 Components

| File | Role | Lines |
|---|---|--:|
| `router.py` | Classifier + planner + resolver + cache + judge (single file, stdlib-only). | ~1100 |
| `models.yaml` | Bucket definitions. 13 buckets, 145 ladder entries. The **only** place model policy lives — retiring/adding Cursor models is a YAML edit. |
| `cursor-models.tsv` | Authoritative catalogue (93 ids) dumped from `cursor-agent --list-models`. Used for ladder validation. |
| `cursor-auto-router` | Thin bash wrapper: classify prompt → `cursor-agent --model <picked>`. Single-prompt mode. |
| `cursor-auto-plan` | Full orchestrator: plan → DAG → wave-parallel execution with per-task logs. |
| `logger.js` | Node.js `--require` preload. Captures + redacts Cursor's HTTP/1.1 traffic. Kept for forensic work but **not on the hot path** — chat is invisible to it. |
| `cursor-agent-logged` | Thin wrapper that runs `cursor-agent` with `logger.js` preloaded. |
| `analyze.js` | Offline parser for `cursor-auto.log`. |
| `DESIGN.md` | The phase-1 engineering log. |

### 1.3 What the router actually does on a prompt

1. **Normalise** the prompt and look up `sha256(prompt ‖ model ‖ catalogue_sig ‖ buckets_sig)` in `.cache/{cls,plan}/`.
2. On miss: call local Ollama with `format=json` + `temperature=0`.
3. **4-tier parse** the response:
   (a) direct `json.loads`, (b) strip ```\`\`\`json ... \`\`\` ``` fences,
   (c) first balanced `{...}` via brace-counting (string-aware),
   (d) truncation repair (close unterminated strings, drop dangling keys, balance brackets).
4. Classifier mode: `(bucket, latency, effort)` → walk `models.yaml` ladder → Cursor model id.
5. Planner mode: validate task DAG, assign a model per task via the same resolver, run Kahn's topological sort into **waves**, auto-serialise any tasks within a wave whose `writes` overlap.
6. **Cache-put** the decision so the next identical prompt is a 40 ms hit.
7. With probability `ROUTER_JUDGE_SAMPLE`, fork a detached judge process that asks a cheap Ollama model whether the routing was `ok`/`over`/`under`/`wrong_bucket` and appends to `judge.log`. **Not** fed back into routing automatically.

### 1.4 Measured latency (local qwen2.5:latest)

| Path | Cold (miss) | Warm (hit) |
|---|--:|--:|
| Classifier (`router.py`) | ~5–7 s | ~55 ms |
| Planner (`router.py --plan`) | ~20–25 s | ~40 ms |
| Validator (`router.py --validate`) | ~250 ms | — |

### 1.5 Safety defaults

- Headless execution (`cursor-auto-plan --exec`) always sets
  `--force --sandbox disabled --approve-mcps --trust`. Without these,
  `cursor-agent` blocks on the first approval prompt and the whole wave
  stalls. Opt out with `CAP_YOLO=0`.
- Router **never blocks the user**: every failure mode falls back to a
  default model or a single-task plan and exits 0.
- No secrets in the repo: redaction in `logger.js`, `.gitignore` excludes
  all runtime logs and `.cache/`.

---

## 2. Where we're going — roadmap

Ordered by value × (1 / cost). Tier 1 already shipped.

### Tier 1 — shipped ✅

| # | Item | Status | Notes |
|---|---|---|---|
| T1.1 | Headless yolo-flag bundle (`--force --sandbox disabled --approve-mcps --trust`) | ✅ | `CAP_YOLO=1` default in `--exec`. |
| T1.2 | SHA-256 prompt cache (classifier + planner) | ✅ | 1-week TTL; `ROUTER_CACHE_TTL_SEC=0` disables. |
| T1.3 | 4-tier JSON parser | ✅ | Verified on 10 pathological inputs incl. truncation mid-string. |
| T1.4 | Long-prompt stdin delivery (`printf %s \| --print`) | ✅ | Kicks in at 64 KB. |
| T1.5 | Fire-and-forget LLM-as-judge | ✅ | `ROUTER_JUDGE_SAMPLE=0.1` to enable; writes `judge.log`. |

### Tier 2 — next up 🎯

Bigger lifts, higher payoff. All three interact with each other, so we'll
do them as one themed batch.

| # | Item | Cost | Why |
|---|---|---|---|
| T2.1 | **Learned routing memory** (port from llm-router's `memory/profiles.py`) | ~3 h | After N corrections (user manually picked model X for bucket Y three times in a row), auto-override for that bucket. Persisted in `memory.json`. |
| T2.2 | **Multi-policy scoring** (port from Routerly) | ~4 h | Instead of `ladder[tier]` = one-shot pick, score each candidate against a pipeline of policies (`cheapest`, `fastest`, `last_known_healthy`, `capability`, `memory_override`) and return the top-k. Enables graceful degradation if a model is rate-limited. |
| T2.3 | **Per-task git worktrees** (port from Optimus Code / agent-orchestrator) | ~4 h | When two waves-1 tasks touch the same file, auto-branch into worktrees and execute each task in isolation. Requires a merge-back step (probably: just land them serially on failure, parallel on success, and leave the worktree around for the user to inspect on conflict). |

**Design constraints for Tier 2:**

- No new runtime dependencies. Stdlib + `cursor-agent` + `ollama` only.
  If a design needs `sentence-transformers`, we defer it to Tier 3.
- Every new feature must have an **off-by-default** env var. Power users
  opt in; new users get the current behaviour unchanged.
- Every change must keep `router.py --validate` green.

### Tier 3 — probably 🤔

Bigger surface-area changes. Each is a weekend project.

| # | Item | Cost | Why |
|---|---|---|---|
| T3.1 | **Local ML classifier** (port VirtuSoul's `MiniLM + LogisticRegression`) | ~1 day | Replace the Ollama classifier with a 15 ms embedder. Huge latency win but adds `sentence-transformers` dependency and a small pre-trained model artefact. Gate behind `ROUTER_CLASSIFIER=ml` env var; keep Ollama as default + fallback. |
| T3.2 | **Judge → memory loop** | ~4 h | Today the judge writes `judge.log` and humans read it. Next step: when the judge says `over`/`under` with confidence > 0.8 for the same bucket three times, automatically promote/demote the default effort in `memory.json`. Requires T2.1. |
| T3.3 | **MCP-based multi-agent orchestration** (idea from Optimus Code) | ~2 days | Give each sub-task agent an MCP channel back to the planner so it can report partial results / request re-planning. Currently each sub-task is a black-box `cursor-agent --print`. |
| T3.4 | **Telemetry dashboard** | ~1 day | `router.log`, `plan.log`, `judge.log` are all newline-JSON. A tiny Flask / static HTML tool that renders hit-rate, avg latency, bucket-distribution, and judge-disagreement-rate over time. |

### Tier 4 — maybe ⏸

Speculative. Listed for completeness, not scheduled.

- **Streaming plans.** Start executing wave 1 while the planner is still
  emitting wave 2. Needs a streaming JSON parser and rework of the
  orchestrator wave loop.
- **Cross-machine planner.** The planner already honours `ROUTER_PLANNER_URL`,
  so running a larger Ollama on a beefier box via SSH tunnel works today.
  Formalise it with a `scripts/remote-planner.sh` and a health check.
- **Policy learning from git diffs.** Post-execution, diff what the agent
  wrote vs. what the user kept after review. Feed that signal into the
  memory layer. Needs reliable git-diff-of-agent-commits detection.
- **Give up on Cursor's HTTPS-in-Rust-addon.** If we ever need to observe
  chat on the wire, the approach would be to `LD_PRELOAD` a shim around
  `SSL_write`/`SSL_read`. Deferred indefinitely — we pivoted away from
  passive observation for a reason.

---

## 3. Non-goals

Explicitly out of scope so we don't accidentally scope-creep:

1. **Replacing `cursor-agent`.** This project always shells out to the
   real CLI. We're a router + orchestrator, not an agent.
2. **Supporting non-Cursor model providers.** The ladders in
   `models.yaml` are Cursor model ids. A port to OpenAI / Anthropic
   directly would need a different resolver and a different value
   proposition — out of scope here.
3. **Any GUI.** CLI + JSON logs. If someone wants a dashboard they can
   render the newline-JSON themselves (T3.4 is a tiny static renderer,
   not a product).
4. **Formal correctness of the DAG.** Kahn's topo-sort gives a valid
   execution order for any DAG we build. If the planner LLM produces
   non-DAG output (cycle / dangling ref), we fall back to a 1-task plan.
   We do **not** try to heal the DAG.

---

## 4. Open questions

Things we've deliberately left undecided:

- **Cache invalidation when `models.yaml` changes.** Currently the cache key
  includes a hash of bucket names, so renaming a bucket invalidates all
  cached decisions for that bucket. But changing a bucket's `ladder`
  without renaming the bucket does **not** invalidate. Acceptable today
  because ladders change rarely; revisit if we start editing ladders more
  frequently than weekly.
- **Fairness across waves.** Today wave N+1 waits for *all* of wave N. A
  stuck task in wave N blocks independent downstream work. Could be
  fixed by running the topo-sort at task granularity instead of wave
  granularity, but then logs become much harder to read. Not urgent.
- **Judge model choice.** Right now the same `qwen2.5:latest` judges its
  own classifier output. That's suspicious. Using a different small
  model (e.g. `llama3.2:3b`) as the judge would be more honest.
  Parameterised via `ROUTER_JUDGE_MODEL`, so a flip is one env var away.

---

## 5. Contribution flow

For future-us or anyone reading this:

1. **Edit `models.yaml`**, then `python3 router.py --validate`. That's the
   lowest-risk change and covers ~80% of the surface area.
2. **Router/planner logic** lives in `router.py`. Keep it stdlib-only.
3. **Orchestrator** lives in `cursor-auto-plan`. Keep bash POSIX-ish.
4. **Never commit `cursor-auto.log`, `router.log`, `plan.log`,
   `judge.log`, `.cache/`, `.cursor-auto-plan/`.** `.gitignore` already
   excludes them; don't `-f` them in.
5. **Smoke test before pushing:**
   ```bash
   python3 router.py --validate
   echo "fix a typo" | python3 router.py --explain
   python3 router.py --plan --prompt "add dark mode"
   ./cursor-auto-plan --exec --dry-run -p "add dark mode"
   ```

That's the plan.
