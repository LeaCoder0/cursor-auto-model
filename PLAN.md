# PLAN.md — full structural plan

Where we started, where we are, and where we're going.

For the detailed technical journey of **phase 1** (investigating Cursor's
server-side `auto`, building the TLS logger, discovering the Rust native
addon), see [`DESIGN.md`](./DESIGN.md). This file is the forward-looking
roadmap; DESIGN.md is the backward-looking engineering log.

> **Scope guard.** This project is a **client-side router + orchestrator
> for `cursor-agent`**. It is intentionally a zero-dependency Python + bash
> tool. Several ideas we steal from bigger frameworks (`ruflo`,
> `llm-router`, `Routerly`, etc.) have a scaled-down adaptation planned
> here. Every planned item below is re-scored against: *can it be done in
> stdlib Python + bash, without new runtime deps, without GPU?*

---

## 0. Where we started

**Original question:** *How does Cursor Agent CLI's `auto` mode actually pick
a model, and can we predict or replicate it client-side?*

This turned into a multi-stage investigation that changed the goal three
times along the way:

| Phase | Goal | Outcome |
|---|---|---|
| 0 | Understand `auto` by reading the CLI | `auto` is server-side; the CLI just asks `api2.cursor.sh` via `GetDefaultModelForCli`. |
| 1 | Log the wire format to observe `auto` | Built `logger.js` (Node `--require` preload, TLS hook, HTTP framing, decompression, redaction). Captured catalogue + telemetry, but **chat RPC was invisible** — it goes through a Rust native addon with its own network stack, bypassing Node.js. |
| 2 | Since we can't observe it, **replace it** | Built `router.py`: a two-stage classifier (local Ollama → bucket → deterministic ladder → Cursor model id). One model per prompt. |
| 3 | One model per prompt is a toy; upgrade to a planner | Added planner mode: decompose prompt → DAG of subtasks → wave-ordered execution with per-task model selection. `cursor-auto-plan` wraps this. |
| 4 | Steal good ideas from existing OSS routers | Researched 5 small projects (`agent-orchestrator`, `virtusoul-router`, `optimus-code`, `Routerly`, `llm-router`). Shipped Tier-1: yolo-flag bundle, SHA-256 prompt cache, 4-tier JSON parser, long-prompt stdin delivery, LLM-as-judge. |
| 5 | Look at the biggest OSS agent-orchestration platform (`ruflo` / ex-claude-flow, 32k ★) | Identified 8 concepts that map onto our design without pulling in their Rust/WASM stack. Slotted into Tier-2 / Tier-3 / Tier-4 below. |

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

| File | Role |
|---|---|
| `router.py` | Classifier + planner + resolver + cache + judge (single file, stdlib-only). |
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
6. **Cache-put** the decision so the next identical prompt is a ~40 ms hit.
7. With probability `ROUTER_JUDGE_SAMPLE`, fork a detached judge process that asks a cheap Ollama model whether the routing was `ok`/`over`/`under`/`wrong_bucket` and appends to `judge.log`. **Not** fed back into routing automatically (Tier-3 closes that loop).

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

## 2. Ideas harvested from `ruflo` (ruvnet/ruflo, 32k★)

ruflo is a much larger project than ours (100+ agent definitions, WASM
kernels, Rust vector DB, Raft/BFT consensus, a 313-tool MCP server). Most of
that is infeasible to port into a stdlib-only Python tool. But a handful of
**concepts** map very cleanly onto what we already have. For each, we list
what ruflo does and what we'd *actually* build.

| # | ruflo concept | Our adaptation | Where it lands |
|---|---|---|---|
| A | **Agent Booster (WASM)** — skip LLM entirely for deterministic edits like `var→const`, `add-types`, `remove-console`. | An **intent-classifier fast-path** in `router.py` that matches ~10 high-confidence regex patterns and routes to `composer-2-fast` with `effort=normal` *without* a classifier call. Saves 5–7 s on the cheap case. | Tier 2 (T2.A) |
| B | **Agent role registry** — one YAML per role (`coder.yaml`, `reviewer.yaml`, `architect.yaml`, `tester.yaml`, `security-architect.yaml`). | A sibling `agents/` directory with one YAML per "agent persona". Each task in a plan can reference a persona which contributes a system-prompt preamble passed to `cursor-agent --print`. Buckets keep deciding **model**; personas decide **voice**. | Tier 2 (T2.B) |
| C | **`RETRIEVE → JUDGE → DISTILL → CONSOLIDATE → ROUTE` learning loop** (SONA / ReasoningBank). | Our local version: (1) RETRIEVE = cache + memory lookup, (2) JUDGE = we already ship this, (3) DISTILL = take N judge verdicts per bucket and summarise to a `notes.md` per bucket, (4) CONSOLIDATE = on ≥3 corrections, promote to override in `memory.json`, (5) ROUTE = classifier reads `memory.json` as a pre-filter. | Tier 2 (T2.C, merges old T2.1) |
| D | **Hooks** — `pre-task`, `post-task`, `pre-wave`, `post-wave`, `on-failure` scripts that Claude Code runs automatically. | `cursor-auto-plan` scans `.cursor-auto-plan/hooks/<stage>/*.sh` and runs each executable for that stage. Default: none. Typical use: lint check after a wave, auto-revert on failure, post to Slack. | Tier 3 (T3.A) |
| E | **Anti-drift checkpoints** — every few tasks, ask "are we still on-goal?" and reconcile. | Between waves, if `ROUTER_DRIFT_CHECK=1`, run one extra judge pass that reads the original prompt + the list of completed task descriptions, and returns `on_track` / `drift` / `stop`. Orchestrator halts on `stop`. | Tier 3 (T3.B) |
| F | **Multi-provider failover** (Anthropic → OpenAI → Ollama). | We already do *intra-Cursor* ladder fallback. ruflo's insight: also fall back on *transient* errors (rate-limit, 5xx from Cursor). Wrap each `cursor-agent --model X` call in a 1-retry-with-next-ladder-entry shim in `cursor-auto-plan`. | Tier 3 (T3.C) |
| G | **Token optimiser** — compress context via pattern cache. | Over time, `.cache/cls/*.json` becomes a corpus of `(prompt, bucket)` pairs. Export as a tiny `training/examples.jsonl` so any future local-ML classifier (existing Tier 3) has ground-truth to train on. No runtime change needed, just a collection habit. | Tier 3 (T3.D) |
| H | **Queen / worker hive-mind with Raft/BFT consensus** (the headline ruflo feature). | **Explicitly declined.** We have one orchestrator and N single-shot subprocesses — nothing to reach consensus about, no Byzantine faults to tolerate. Documented as a non-goal. | Non-goal §3 |

### 2.1 Why most of ruflo is out of scope

| ruflo subsystem | Why we skip |
|---|---|
| SONA / EWC++ / Flash Attention / LoRA | Requires training a neural model. Our routing has ~13 classes over ~93 model ids; a decision tree or the concept **C** loop above solves 95% of what they solve. |
| HNSW + ONNX MiniLM vector memory | A file-based key→value cache with SHA-256 keys already gives us ~40ms retrieval for exact matches; nearest-neighbour retrieval for *semantic* matches is a Tier-3 ML-classifier concern (already tracked as T3.1 before), not a separate HNSW store. |
| 100+ specialized agents + SPARC methodology | Marketing. The concrete stealable part is the **YAML-per-role** format — that's T2.B. |
| MCP server, plugin SDK, IPFS marketplace | Distribution concerns for a platform. We're a dot-sh + dot-py in a single repo. |
| Rust / WASM kernels | Our hot path is a local Ollama call (seconds) or a cache hit (ms). Optimising Python or Rust-ifying anything gives ≤1% end-to-end. |

---

## 3. Roadmap

Ordered by value × (1 / cost). Tier 1 has shipped. Each tier is a
self-contained shipping window.

### Tier 1 — shipped ✅

| # | Item | Notes |
|---|---|---|
| T1.1 | Headless yolo-flag bundle (`--force --sandbox disabled --approve-mcps --trust`) | `CAP_YOLO=1` default in `--exec`. |
| T1.2 | SHA-256 prompt cache (classifier + planner) | 1-week TTL; `ROUTER_CACHE_TTL_SEC=0` disables. |
| T1.3 | 4-tier JSON parser | Verified on 10 pathological inputs incl. truncation mid-string. |
| T1.4 | Long-prompt stdin delivery (`printf %s \| --print`) | Kicks in at 64 KB. |
| T1.5 | Fire-and-forget LLM-as-judge | `ROUTER_JUDGE_SAMPLE=0.1` enables; writes `judge.log`. |

### Tier 2 — next up 🎯 (3 themed batches)

All three batches interact. Ship them as one sprint.

| # | Item | Est. cost | Why | Source |
|---|---|---|---|---|
| T2.A | **Intent-classifier fast-path**: before calling Ollama, match prompt against ~10 high-confidence regex patterns (`/^fix (the )?typo/i`, `/^rename .* to .*/i`, etc.) and short-circuit to `composer-2-fast`. Miss falls through to the classifier. | ~2 h | Saves 5–7 s on the common trivial case. Inspired by ruflo Agent Booster. | ruflo A |
| T2.B | **Agent role registry** (`agents/*.yaml`): each YAML declares `name`, `system_prompt`, `preferred_bucket`, `effort`. Planner emits `role` per task; orchestrator prepends `[Role: <system_prompt>]` to the task description before `--print`. | ~3 h | Better prompts without touching the classifier. Ported verbatim-pattern from ruflo's `agents/`. | ruflo B |
| T2.C | **Learned routing memory** (supersedes old T2.1). Implements the full 5-stage `RETRIEVE→JUDGE→DISTILL→CONSOLIDATE→ROUTE` loop: cache and judge already exist (R+J), add `router.py --distill` (daily cron / manual) that summarises `judge.log` into per-bucket `memory/*.md`, and `memory.json` overrides read at route-time. After ≥3 same-direction corrections for a bucket, auto-promote to a hard override. | ~5 h | Turns the judge from "interesting log I never read" into a closed loop. | ruflo C + llm-router/profiles.py |
| T2.D | **Multi-policy scoring** (was T2.2). Replace `ladder[tier]` one-shot pick with a scored candidate list: `cheapest`, `fastest`, `last_known_healthy`, `capability`, `memory_override`. Top-1 is returned; top-3 is logged for future debugging. | ~4 h | Graceful degradation when a model is rate-limited or deprecated. | Routerly |
| T2.E | **Per-task git worktrees** (was T2.3). When two tasks in the same wave have overlapping `writes`, spawn each in its own `git worktree` branch; leave them for the user to diff/merge on conflict. Non-conflicting tasks still run in the main tree. | ~4 h | Safe parallelism for file-writing agents. Defer the auto-merge; humans merge. | Optimus Code + agent-orchestrator |

**Shared design constraints for all of Tier 2:**

- No new runtime dependencies. Stdlib Python + `cursor-agent` + `ollama` only.
- Every new feature has an **off-by-default** env var. Opt-in.
- `python3 router.py --validate` stays green.

### Tier 3 — probably 🤔

Bigger surface, each is a weekend project.

| # | Item | Est. cost | Source |
|---|---|---|---|
| T3.A | **Stage hooks** (`pre-wave`/`post-wave`/`pre-task`/`post-task`/`on-failure`) executed from `.cursor-auto-plan/hooks/<stage>/*.sh`. Stdout passes environment like `CF_TASK_ID`, `CF_MODEL`, `CF_WAVE`. | ~4 h | ruflo D |
| T3.B | **Anti-drift checkpoint** between waves (`ROUTER_DRIFT_CHECK=1`). Extra judge call that compares completed-subtask summaries against the original prompt and returns `on_track`/`drift`/`stop`. | ~3 h | ruflo E |
| T3.C | **Transient-error retry** around `cursor-agent` in the orchestrator: on non-zero exit from a known retryable class (rate-limit, 5xx), retry once with the next ladder entry. | ~2 h | ruflo F |
| T3.D | **Training-corpus export**: `router.py --export-training path.jsonl` dumps every cached `(prompt, parsed_bucket, picked_model)` tuple as a JSONL stream, ready to train a local ML classifier. | ~1 h | ruflo G |
| T3.E | **Local ML classifier** (was T3.1). Replace Ollama in the classifier stage with `MiniLM + LogisticRegression` or similar. Opt-in via `ROUTER_CLASSIFIER=ml`; Ollama stays as default + fallback. Requires `sentence-transformers` dependency — added only under an `extras` install flag. Training data from T3.D. | ~1 day | VirtuSoul |
| T3.F | **Judge → memory auto-loop** (was T3.2, now upgraded). When judge says `over`/`under` with confidence >0.8 for the same bucket ≥3 times, update `memory.json` defaults automatically. Requires T2.C. | ~4 h | Our extension |
| T3.G | **Telemetry dashboard** (was T3.4). Tiny static HTML that renders newline-JSON logs: cache hit-rate, avg latency, bucket distribution, judge disagreement over time, drift-check outcomes. | ~1 day | — |
| T3.H | **Cross-machine planner**. Formalise the already-supported remote-Ollama pattern with `scripts/remote-planner.sh`, an SSH-tunnel helper, and a health-check. | ~2 h | — |

### Tier 4 — maybe ⏸ (speculative, unscheduled)

- **Streaming plans.** Execute wave 1 while planner still emits wave 2.
- **Policy learning from git diffs.** Post-run, diff what the agent wrote vs. what the user kept after review. Feed into memory.
- **Observer shim for Cursor chat RPC.** If we ever need on-wire chat observability, the move is `LD_PRELOAD` around `SSL_write`/`SSL_read` in `cursor-agent`'s Rust addon. Deferred indefinitely — we pivoted away from passive observation for a reason.
- **MCP server wrapper.** Expose `cursor-auto-plan` as an MCP tool so it can be called from within `cursor-agent` sessions itself. Recursive but useful.
- **Dual-mode (Cursor + Codex/Claude Code).** ruflo supports both. We pin to Cursor deliberately; revisit only if a user asks.

---

## 4. Non-goals

Explicitly out of scope so we don't accidentally scope-creep:

1. **Replacing `cursor-agent`.** This project always shells out to the real CLI. We're a router + orchestrator, not an agent.
2. **Supporting non-Cursor model providers.** The ladders in `models.yaml` are Cursor model ids. A port to OpenAI / Anthropic direct would need a different resolver and a different value proposition — out of scope.
3. **Any GUI.** CLI + JSON logs. T3.G is a static renderer over those logs, not a product.
4. **Formal DAG correctness.** Kahn's topo-sort gives a valid execution order for any DAG we build. If the planner LLM produces non-DAG output (cycle / dangling ref), we fall back to a 1-task plan. We do not try to heal the DAG.
5. **Multi-agent consensus (Raft / Byzantine / Gossip).** ruflo's headline feature. We have one orchestrator and N single-shot subprocesses. Nothing to reach consensus about.
6. **Self-training neural models (SONA / EWC++ / Flash Attention / LoRA).** Infrastructure mismatch: requires GPU, WASM runtime, or a training loop. The learning loop in T2.C achieves the same end goal (better routing over time) in ~200 lines of Python.
7. **On-the-wire observability of Cursor's chat RPC.** Deferred indefinitely — the Rust addon bypasses Node.js and `LD_PRELOAD` is the only route.

---

## 5. Open questions

Things we've deliberately left undecided:

- **Cache invalidation when `models.yaml` changes.** Cache key includes a hash of bucket *names*, so renaming invalidates. Changing a `ladder` entry under the same bucket name does **not** invalidate. Acceptable today; revisit if ladder churn picks up.
- **Fairness across waves.** Wave N+1 waits for *all* of wave N. A stuck task in wave N blocks independent downstream work. Could run topo-sort at task granularity instead of wave granularity, but logs become much harder to read. Not urgent.
- **Judge model choice.** Same `qwen2.5:latest` judging its own classifier output is suspicious. A different small model (e.g. `llama3.2:3b`) would be more honest. One env var flip — `ROUTER_JUDGE_MODEL` — once T2.C lands and we actually start reading the judge log.
- **Should the intent-classifier fast-path (T2.A) pass through user intent to the task description?** E.g. match `^fix the typo in (.+)$`, bind `$1` as a parameter, and prepend "Fix the typo in `$1` and nothing else." to the cursor-agent `--print` arg. Probably yes, but first design the regex→template mapping carefully.
- **Role registry (T2.B) — should `role` also override `bucket`?** Currently designed so `role` only adds a system-prompt preamble; `bucket` is still chosen by the classifier. An alternative: `role=security-architect` forces `bucket=coding_hard` regardless. Leaning toward "preamble-only" to keep the mental model simple.

---

## 6. Contribution flow

For future-us or anyone reading this:

1. **Edit `models.yaml`**, then `python3 router.py --validate`. Lowest-risk change, ~80% of the surface area.
2. **Router/planner logic** lives in `router.py`. Keep it stdlib-only.
3. **Orchestrator** lives in `cursor-auto-plan`. Keep bash POSIX-ish.
4. **Never commit** `cursor-auto.log`, `router.log`, `plan.log`, `judge.log`, `.cache/`, `.cursor-auto-plan/`. `.gitignore` excludes them; don't `-f` them in.
5. **Smoke test before pushing:**
   ```bash
   python3 router.py --validate
   echo "fix a typo" | python3 router.py --explain
   python3 router.py --plan --prompt "add dark mode"
   ./cursor-auto-plan --exec --dry-run -p "add dark mode"
   ```

---

## 7. TL;DR for the next sprint

**Tier 2, in order:**

1. **T2.A** Intent-classifier fast-path — 10 regex shortcuts → skip Ollama. *2 h, 5–7 s saved per hit.*
2. **T2.B** Agent role registry — `agents/*.yaml` + system-prompt prepending. *3 h.*
3. **T2.C** Learned routing memory — close the `judge.log → memory.json → router` loop. *5 h.*
4. **T2.D** Multi-policy scoring — replace one-shot ladder pick with a scored candidate list. *4 h.*
5. **T2.E** Per-task git worktrees — safe parallel writes. *4 h.*

Total ~18 hours of work. Everything is opt-in, everything preserves the
existing fallback guarantees, nothing adds a runtime dependency.

That's the plan.
