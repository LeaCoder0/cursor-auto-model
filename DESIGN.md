# cursor-auto-model — design notes

A client-side replacement for Cursor's server-side "auto" model router. Given
a user prompt, it picks the best Cursor model for the task using a local
Ollama model as a classifier, then invokes `cursor-agent` with that model.

This document captures the full journey: what was investigated, what was
learned about Cursor internals, why the approach changed, and how the
current system works.

---

## 1. Original question

> How does auto mode work in `cursor-agent`? What models does it pick?

### What we found

- `cursor-agent --model auto` does **not** locally decide anything.
- On startup the CLI issues a Connect-RPC call to `api2.cursor.sh`:
  - `GetDefaultModelForCli` — server returns the *default* model for this
    account/CLI. In every run we observed, the answer was `composer-2-fast`.
  - `GetUsableModels` — server returns the full model catalogue for the
    account (22 families, 93 ids in our case).
- The actual chat turn is sent to the resolved model. The CLI also reports
  the chosen model back to Cursor via a telemetry RPC (`TrackEvents` /
  `SubmitLogs`), which is how the "Auto selected X" UI text is populated.

So `auto` on the CLI just means:

1. Ask the server what model to use.
2. Use that model.

The routing logic lives on Cursor's servers and we have no direct visibility
into it. From the outside, it always picked `composer-2-fast` for us.

---

## 2. Investigation into the CLI wire format

To confirm the above, we built a preload-based network logger.

### `logger.js` (Node.js `--require` preload)

Invoked via:

```bash
NODE_OPTIONS="--require /path/to/logger.js" cursor-agent ...
```

It hooks `tls.connect` and captures all plaintext TLS traffic for hosts
matching the Cursor API. Evolutions along the way:

- **Synchronous logging** (`fs.appendFileSync`) — `process.exit` was
  truncating async writes.
- **HTTP/1.1 framing** — parse request line, headers, and bodies out of the
  raw TLS stream.
- **gzip/brotli decompression** — `zlib.gunzipSync`, `zlib.brotliDecompressSync`.
- **Redaction** — strip `Authorization` / `Cookie` headers and inline
  JWT-like tokens before writing the log.
- **Hostname matching fix** — initially filtered only on `servername`;
  extended to check `servername`, `rawHost`, and `host` because the CLI
  sometimes passes an already-resolved IP as `host`.

### Supporting scripts

- `cursor-agent-logged` — bash wrapper that runs `cursor-agent` with
  `NODE_OPTIONS=--require logger.js --yolo`. Produces `cursor-auto.log`.
- `analyze.js` — parses `cursor-auto.log` and prints the key findings:
  `GetDefaultModelForCli` default, `GetUsableModels` catalogue, telemetry
  with actual model used.

### Key discovery: chat traffic bypasses Node.js

The chat RPC (`StreamUnifiedChatWithTools`) **never** appeared in the
`tls.connect` logs. Investigation with `strace` showed many TCP connections
to Cursor's API that Node's `tls.connect` didn't see. Reason: the chat
functionality is implemented in a **Rust native addon**
(`file_service.linux-x64-gnu.node`) which uses its own network stack
(`reqwest` / `hyper` / `rustls`), completely bypassing Node's TLS module.

Consequence: a pure Node-space logger can observe config RPCs and telemetry
but cannot capture the chat turn itself. To decrypt the chat traffic we'd
need either an `SSLKEYLOGFILE`-aware build of the Rust addon or an
`LD_PRELOAD` hook into `rustls`, neither of which were pursued.

### What this means

We never saw proof of a richer client-side "auto" algorithm. All the
evidence is consistent with the CLI asking the server for a model and
sending the chat to that model.

---

## 3. Pivot: build our own router

User intent clarified:

> I want some model to pick the best Cursor model out of the available ones
> based on the task given to Cursor. Basically "auto picking of the model"
> functionality, but without using `auto` as the Cursor model.

So instead of *observing* Cursor's auto, we build our own client-side one
that:

- Uses Cursor's real catalogue (not a subset).
- Runs fully offline via Ollama (no extra cost, no extra API calls).
- Emits a single deterministic Cursor model id the CLI can consume via
  `--model <id>`.
- Keeps all of Cursor's native tools and codebase context (because we're
  still calling `cursor-agent`, just with an explicit model).

---

## 4. Architecture

```
  ┌────────────────────────┐
  │    cursor-auto-router  │  (bash wrapper)
  │                        │
  │  parses args, extracts │
  │  prompt, short-circuits│
  └───────┬────────────────┘
          │ prompt
          ▼
  ┌────────────────────────┐      ┌───────────────────────┐
  │       router.py        │─────▶│  Ollama (classifier)  │
  │                        │◀─────│  qwen2.5 / gemma3 ... │
  │  build prompt, parse   │      └───────────────────────┘
  │  JSON, resolve ladder, │
  │  log decision, print id│
  └───────┬────────────────┘
          │ picked model id
          ▼
  ┌────────────────────────┐
  │  cursor-agent          │
  │  --model <picked>      │
  │  ...original args...   │
  └────────────────────────┘
```

Two stages, by design:

1. **Classify** (LLM-based, fuzzy) — Ollama picks a `bucket`, `latency`
   (`fast`/`normal`), and `effort` (`normal`/`high`/`max`) from a fixed
   taxonomy.
2. **Resolve** (deterministic, table-driven) — `router.py` maps
   `(bucket, latency, effort)` to a single Cursor model id via the ladders
   in `models.yaml`.

This keeps the LLM's job small (pick 3 labels from a finite vocab) and the
model-id mapping fully auditable, without the LLM needing to know the
Cursor model namespace at all.

---

## 5. Files and what they do

```
cursor-auto-model/
├── cursor-auto-router      # bash wrapper, entry point
├── router.py               # classifier + resolver (core logic)
├── models.yaml             # 13-bucket taxonomy + ladders per bucket
├── cursor-models.tsv       # authoritative list from `cursor-agent --list-models`
├── router.log              # one JSON line per routing decision (audit trail)
│
├── logger.js               # Node preload TLS logger (from investigation phase)
├── cursor-agent-logged     # wrapper that runs cursor-agent with logger.js
├── analyze.js              # summariser for cursor-auto.log
├── cursor-auto.log         # output from logger.js (not used by the router)
└── DESIGN.md               # this file
```

Only the first four files are needed at runtime. The rest are artefacts
from the investigation and can be ignored if you only want routing.

---

## 6. The taxonomy (`models.yaml`)

13 buckets, chosen to cover the observed 93-id catalogue without leaving
any model unreachable. Each bucket has:

- `good_for` — plain-English description the Ollama classifier reads.
- `ladder` — ordered list of Cursor ids from `effort=normal` to `effort=max`.
- `fast_ladder` (optional) — fast variants preferred when `latency=fast`.

Buckets:

| bucket                 | purpose                                            |
|------------------------|----------------------------------------------------|
| `trivial_chat`         | 1-sentence answers, greetings, jokes               |
| `quick_qa`             | short factual Q&A, no code                         |
| `light_write`          | commit msgs, release notes, rephrasing             |
| `simple_code_edit`     | typo fixes, rename, one-liner edits                |
| `coding_medium`        | single-file feature work, one function from a spec |
| `coding_hard`          | multi-file refactor, non-obvious bugs, test suites |
| `coding_extreme`       | concurrency, security, perf, subtle bugs           |
| `codex_spark_preview`  | explicit preview-model benchmarking                |
| `long_context_read`    | summarising large inputs (logs, repos, docs)       |
| `deep_reason`          | design docs, proofs, trade-off analysis            |
| `agentic_tool_loop`    | many-step tool loops, iterate-until-green          |
| `vision_or_multimodal` | screenshots, diagrams, UI mockups                  |
| `creative_long_form`   | blog posts, stories, marketing copy                |

Fallbacks:

- `default_id: composer-2-fast` — returned when Ollama fails or picks an
  invalid id.
- `default_bucket: simple_code_edit` — bucket used if Ollama returns an
  unknown bucket name.

All 92 reachable Cursor ids appear in at least one ladder
(verified by `router.py --validate`).

---

## 7. `router.py` in detail

### CLI surface

```
router.py --prompt "..."            # print picked id to stdout
router.py --prompt "..." --explain  # also print reason to stderr
router.py --prompt "..." --dry-run  # don't print id; only reason to stderr
router.py --validate                # check ladders against cursor-models.tsv
```

Reads stdin if `--prompt` is absent.

### Environment

| var                     | default                          |
|-------------------------|----------------------------------|
| `ROUTER_OLLAMA_URL`     | `http://localhost:11434`         |
| `ROUTER_OLLAMA_MODEL`   | `qwen2.5:latest`                 |
| `ROUTER_TIMEOUT_SEC`    | `12`                             |
| `ROUTER_MODELS_FILE`    | `<dir>/models.yaml`              |
| `ROUTER_CATALOGUE_FILE` | `<dir>/cursor-models.tsv`        |
| `ROUTER_LOG_FILE`       | `<dir>/router.log`               |

### Flow

1. Load `models.yaml` and `cursor-models.tsv`.
2. Read user prompt (arg or stdin).
3. Build an LLM prompt that embeds the bucket names + `good_for` strings
   and asks for a strict JSON response:
   ```json
   {"bucket": "...", "latency": "fast|normal", "effort": "normal|high|max"}
   ```
4. Parse JSON, validate against the known bucket set. On parse or validation
   failure, fall back to `default_bucket` / `default_id`.
5. Resolve `(bucket, latency, effort)` to a concrete id by walking the
   ladder. `effort=normal` takes the first entry, `high` takes mid,
   `max` takes the last; `latency=fast` prefers `fast_ladder` if it exists.
6. Re-validate the id exists in `cursor-models.tsv` (otherwise fall back).
7. Append a structured JSON line to `router.log` (timestamp, ollama model,
   prompt head, classification, picked id, duration, notes).
8. Print id to stdout (unless `--dry-run`).

### Robustness

- Never raises on bad LLM output; always returns *something*.
- Catches Ollama timeouts / connection errors and falls back.
- `--validate` gates `models.yaml` against the catalogue so a broken file
  is caught before it ships a bad id.

---

## 8. `cursor-auto-router` in detail

A thin bash wrapper. Its job:

1. Parse `-p` / `--print` to extract the prompt. Also read stdin if no `-p`.
2. Detect `--model <x>` — if set, the user is explicit; skip routing.
3. If `ROUTER_DISABLE=1`, skip routing.
4. If no prompt (e.g. interactive `chat` mode), skip routing.
5. Otherwise, pipe prompt into `router.py`, capture the id, and
   `exec cursor-agent --model <id> <original args>`.

All short-circuit paths use `exec` so there's zero process overhead vs
calling `cursor-agent` directly.

---

## 9. Why Ollama, why this taxonomy

Constraints that shaped the design:

- **No extra cost.** Any client-side router that calls a hosted API
  (OpenAI, Anthropic, Groq) defeats the point — you're paying twice per
  turn. Ollama is free and local.
- **Low latency.** A 3s routing overhead on a 30s chat is fine. A 30s
  routing overhead on a 3s chat is awful. Hard cap: ~5s median.
- **Deterministic output.** `cursor-agent --model` needs a single id. We
  can't pass the LLM's raw guess directly; it'll hallucinate ids (saw
  `gpt-5.4`, `gpt-5.4-mega`, etc.). So LLM picks *labels*, code picks *id*.
- **Auditability.** `router.log` must let you ask "why did you pick X for
  this prompt" after the fact.
- **Coverage.** Every real Cursor id should be reachable for some
  `(bucket, latency, effort)` combination — otherwise part of the user's
  paid-for catalogue is dead weight. Enforced by `--validate`.

---

## 10. Benchmarks (current, as of last run)

17-prompt labelled test set covering all 13 buckets, at three difficulty
tiers. Each prompt has `ok_buckets`, `ok_effort`, `ok_latency` sets; a run
counts as "full" correct only if all three match.

10 Ollama models evaluated:

```
MODEL                      BKT    FULL   MEDIAN     MAX   FAILS
-----------------------------------------------------------------
qwen2.5:latest         100.0%  100.0%    3.12s   7.55s   0
gemma3:4b               94.0%   94.0%    3.42s   6.47s   1
phi3.5:3.8b             88.0%   88.0%    2.69s   7.83s   2
mistral:latest          82.0%   82.0%    3.41s   9.11s   3
qwen2.5:1.5b            82.0%   76.0%    1.90s   3.74s   4
qwen2.5-coder:latest    76.0%   76.0%    3.74s  12.17s   4
llama3.2:3b             76.0%   71.0%    1.98s   5.10s   5
qwen2.5:3b              59.0%   59.0%    2.11s   4.46s   7
llama3.2:1b             35.0%   35.0%    1.62s   4.25s  11
gemma3:1b               18.0%   18.0%    1.66s   3.03s  14
tinyllama:latest        12.0%   12.0%    9.23s  12.08s  15
```

### Takeaways

- **Best accuracy: `qwen2.5:latest` (7B)** — 100% @ 3.1s median. Current
  default.
- **Best speed/quality tradeoff: `gemma3:4b`** — 94% @ 3.4s, at 3.3GB disk
  footprint (vs 4.7GB for qwen2.5:7b).
- **Fastest usable: `qwen2.5:1.5b`** — 82% bucket accuracy @ 1.9s median.
  Failures are all over-classifying writing prompts as code (acceptable if
  most of your use is code).
- **Do not use:**
  - `tinyllama` (collapses to the default bucket, 12% acc).
  - `gemma3:1b` / `llama3.2:1b` (too small, <40% acc).
  - `qwen2.5:3b` (surprisingly worse than `qwen2.5:1.5b` — the quant seems
    bad at structured classification).
  - `qwen2.5-coder:latest` — coder tuning *hurts* routing by classifying
    everything as coding.

### Recommendation

Stay on `qwen2.5:latest` unless latency becomes a problem. If it does,
switch to `gemma3:4b` (nearly the same accuracy at lower memory) or
`qwen2.5:1.5b` (40% faster, coding-prompt heavy users only).

---

## 11. Known limitations

- **No chat-turn visibility.** Because the chat RPC is in a Rust native
  addon with its own TLS, we can't independently verify what model the
  server ultimately used. We trust `cursor-agent --model <x>` and confirm
  via Cursor's own telemetry/logs.
- **First-turn only classification.** The router sees the current prompt.
  In an interactive `chat` session, we short-circuit and let Cursor pick
  the default model for the session. Per-turn routing would need a
  different integration point.
- **Classifier drift.** If Cursor adds/removes models, `cursor-models.tsv`
  and `models.yaml` must be refreshed. `router.py --validate` will flag
  drift in the ladders; there's no automatic refresh yet.
- **No multimodal detection from file attachments.** The `vision_or_multimodal`
  bucket only triggers if the prompt *mentions* images. If you drop
  a screenshot into Cursor's context without mentioning it, the router
  won't know.
- **Fixed ladders.** Ladder order is hand-tuned. There's no feedback loop
  from Cursor's usage telemetry back into ladder weights.

---

## 12. Decisions log

Chronological record of decisions made and why.

1. **Investigate auto via docs first** → docs don't specify the algorithm,
   move to code.
2. **Inspect bundled JS** → auto is resolved over the network, not locally.
3. **Add Node preload logger** → confirmed via `GetDefaultModelForCli`.
4. **Try to capture the actual chat RPC** → `tls.connect` hook missed it;
   later confirmed chat is in a Rust native addon.
5. **Abandon observing server-side auto** → build our own instead.
6. **Use Ollama, not a hosted API** → keep it free, local, private.
7. **Two-stage (LLM label → code-resolved id), not one-stage** → LLMs
   hallucinate ids; bucket vocabularies are finite and stable.
8. **YAML-configurable taxonomy, not hardcoded** → policy changes shouldn't
   require code changes.
9. **Strict JSON output from the classifier** → easy to parse, easy to fail
   safely.
10. **Enforce catalogue coverage via `--validate`** → prevents silent
    model-deprecation bugs.
11. **Keep `composer-2-fast` as the final fallback** → it's what Cursor's
    own auto picks, so worst case we match their default.
12. **Benchmark-driven model choice** → ran 17 prompts × 11 Ollama models;
    `qwen2.5:latest` wins on accuracy, `gemma3:4b` is the best Pareto
    point, `qwen2.5:1.5b` is the right choice for latency-sensitive users.
13. **Keep the Node logger even though it doesn't see chat** → still useful
    for auditing catalogue changes and telemetry; zero cost to leave in.

---

## 13. Usage

```bash
# One-shot prompt, router picks the model
./cursor-auto-router -p "refactor foo.py into async and add tests"

# Show which model was picked and why
ROUTER_EXPLAIN=1 ./cursor-auto-router -p "fix the typo in README"

# Force a specific model (router skipped)
./cursor-auto-router --model claude-4.6-opus-high -p "..."

# Temporarily disable routing
ROUTER_DISABLE=1 ./cursor-auto-router -p "..."

# Use a faster (but slightly less accurate) router model
ROUTER_OLLAMA_MODEL=qwen2.5:1.5b ./cursor-auto-router -p "..."

# Validate policy against the live catalogue
./router.py --validate

# Refresh the catalogue after Cursor updates
cursor-agent --list-models \
  | awk 'NF>=2 {print $1 "\t" $2}' \
  > cursor-models.tsv
./router.py --validate   # check nothing broke
```

---

## 14. Possible next steps

- Add `ROUTER_PROFILE=fast|balanced|accuracy` to flip between
  `qwen2.5:1.5b` / `gemma3:4b` / `qwen2.5:latest` in one place.
- Auto-refresh `cursor-models.tsv` on a timer and run `--validate` as a
  pre-flight check inside `cursor-auto-router`.
- Feedback loop: tag `router.log` lines with "user changed model"
  (observable via Cursor telemetry) and retrain ladder ordering.
- Per-turn routing inside an interactive session (requires deeper
  integration than the current CLI wrapper).
- Explore `LD_PRELOAD`+`SSLKEYLOGFILE` against the Rust addon to finally
  observe chat ground truth — only worth it if we want to verify server-
  side overrides.
