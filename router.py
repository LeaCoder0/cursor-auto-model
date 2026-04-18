#!/usr/bin/env python3
"""
router.py — pick a Cursor model id for a given prompt using a local Ollama model.

Two-stage routing:
  1. A local LLM classifies the prompt into a `bucket` (task category) and
     optionally sets `latency` (fast|normal) and `effort` (normal|max).
  2. A deterministic resolver maps (bucket, latency, effort) to a single
     concrete Cursor model id using the ladders in models.yaml.

The taxonomy is maintained in models.yaml so that:
  - Adding or retiring Cursor models only needs a YAML edit, no code change.
  - Every real Cursor id is reachable via some bucket + effort combination.

On any failure (Ollama down, bad JSON, unknown bucket, unknown id, timeout)
we fall back to `default_id` from models.yaml and exit 0 — so the outer
wrapper can always trust stdout to be a valid Cursor model id.

Usage:
  echo "write a python fib function" | router.py
  router.py --prompt "what is 1+1"
  router.py --prompt "..." --ollama-model qwen2.5:latest --timeout 15
  router.py --prompt "..." --explain   # prints picked id + reason to stderr
  router.py --prompt "..." --dry-run   # don't print id to stdout
  router.py --validate                 # check every ladder id exists in cursor-models.tsv

  # Planner mode: decompose a prompt into tasks + a wave-ordered DAG.
  router.py --plan --prompt "..."           # prints human-readable plan
  router.py --plan --json --prompt "..."    # prints JSON plan

Env:
  ROUTER_OLLAMA_URL     (default http://localhost:11434)
  ROUTER_OLLAMA_MODEL   (default qwen2.5:latest) — classifier (single-model mode)
  ROUTER_PLANNER_URL    (default: same as ROUTER_OLLAMA_URL)
  ROUTER_PLANNER_MODEL  (default qwen2.5:latest) — planner (decomposer)
  ROUTER_TIMEOUT_SEC    (default 12)
  ROUTER_PLANNER_TIMEOUT_SEC (default 60)
  ROUTER_MODELS_FILE    (default <script_dir>/models.yaml)
  ROUTER_CATALOGUE_FILE (default <script_dir>/cursor-models.tsv)
  ROUTER_LOG_FILE       (default <script_dir>/router.log)
  ROUTER_PLAN_LOG_FILE  (default <script_dir>/plan.log)
  ROUTER_CACHE_DIR      (default <script_dir>/.cache) — prompt → decision cache
  ROUTER_CACHE_TTL_SEC  (default 604800, one week) — 0 disables cache entirely
  ROUTER_JUDGE_SAMPLE   (default 0.0) — fraction [0,1] of completed runs to
                        asynchronously evaluate with an LLM judge. 0 disables.
  ROUTER_JUDGE_URL      (default: ROUTER_OLLAMA_URL)
  ROUTER_JUDGE_MODEL    (default qwen2.5:latest)
  ROUTER_JUDGE_LOG_FILE (default <script_dir>/judge.log)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Prompt cache.
#
# SHA-256(prompt + key-salts) → JSON decision file on disk. Two independent
# namespaces: "cls" (classifier) and "plan" (planner). Cache is bypassed if
# ROUTER_CACHE_TTL_SEC == "0".
#
# Keeping this file-based (no sqlite, no LRU thread) keeps the router a
# zero-dependency script and makes cache inspection trivial:
#
#   ls -lah .cache/cls/  .cache/plan/
#
# Llm-router does the same thing; it's cheap insurance and a big latency win
# for the same prompt fired twice in a row (common during iterative use).
# ---------------------------------------------------------------------------

def _cache_root() -> Path:
    root = os.environ.get("ROUTER_CACHE_DIR") or str(SCRIPT_DIR / ".cache")
    return Path(root)


def _cache_ttl() -> int:
    try:
        return int(os.environ.get("ROUTER_CACHE_TTL_SEC", "604800"))
    except ValueError:
        return 604800


def _cache_key(namespace: str, *parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def cache_get(namespace: str, key: str) -> dict | None:
    ttl = _cache_ttl()
    if ttl <= 0:
        return None
    path = _cache_root() / namespace / f"{key}.json"
    try:
        st = path.stat()
    except OSError:
        return None
    if (time.time() - st.st_mtime) > ttl:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def cache_put(namespace: str, key: str, value: dict) -> None:
    if _cache_ttl() <= 0:
        return
    d = _cache_root() / namespace
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{key}.json.tmp"
        tmp.write_text(json.dumps(value))
        tmp.replace(d / f"{key}.json")
    except OSError:
        pass

# ---------------------------------------------------------------------------
# Minimal YAML loader (stdlib-only).
#
# Handles the subset used by our models.yaml:
#   - top-level scalars:             `key: value`
#   - top-level lists of dicts:      `key:\n  - a: 1\n    b: [x, y]`
#   - folded strings:                `key: >` followed by indented lines
#   - flow-style lists:              `key: [a, b, c]`
# ---------------------------------------------------------------------------

def _strip_comment(s: str) -> str:
    # Keep `#` inside quoted strings; our YAML has none of those, so a simple
    # split is enough.
    idx = s.find(" #")
    return s if idx < 0 else s[:idx]


def _parse_flow_list(s: str) -> list:
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"not a flow list: {s!r}")
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [x.strip().strip('"').strip("'") for x in inner.split(",")]


def _parse_scalar(s: str):
    s = s.strip()
    if s == "":
        return None
    if s.startswith("[") and s.endswith("]"):
        return _parse_flow_list(s)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def load_yaml_simple(path: Path) -> dict:
    raw_lines = path.read_text().splitlines()
    # Pre-strip comments but keep indentation and blank lines.
    lines: list[str] = []
    for ln in raw_lines:
        ls = ln.lstrip()
        if ls.startswith("#"):
            lines.append("")
        else:
            lines.append(_strip_comment(ln).rstrip())

    top: dict = {}
    i = 0
    n = len(lines)

    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if indent_of(line) != 0:
            i += 1
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        # Case A: top-level scalar on same line.
        if rest != "":
            top[key] = _parse_scalar(rest)
            i += 1
            continue
        # Case B: top-level list of dicts.
        items: list = []
        i += 1
        while i < n:
            ln = lines[i]
            if not ln.strip():
                i += 1
                continue
            ind = indent_of(ln)
            if ind == 0:
                break
            # Expect "- foo: bar" at indent 2.
            if ln.lstrip().startswith("- "):
                item: dict = {}
                first = ln.lstrip()[2:]
                fk, _, fv = first.partition(":")
                fk = fk.strip()
                fv_stripped = fv.strip()
                if fv_stripped in (">", "|"):
                    # Folded scalar starting on the next line; collect until
                    # we see a line at or below the list-item indent.
                    j = i + 1
                    acc: list[str] = []
                    item_ind = ind
                    while j < n:
                        lj = lines[j]
                        if not lj.strip():
                            j += 1
                            continue
                        if indent_of(lj) <= item_ind:
                            break
                        acc.append(lj.strip())
                        j += 1
                    item[fk] = " ".join(acc)
                    i = j
                else:
                    item[fk] = _parse_scalar(fv_stripped)
                    i += 1
                # Collect sibling keys of this item (same indent as "-" but
                # without the "-").
                while i < n:
                    ln2 = lines[i]
                    if not ln2.strip():
                        i += 1
                        continue
                    ind2 = indent_of(ln2)
                    if ind2 <= ind or ln2.lstrip().startswith("- "):
                        break
                    kk, _, vv = ln2.lstrip().partition(":")
                    kk = kk.strip()
                    vv_stripped = vv.strip()
                    if vv_stripped in (">", "|"):
                        j = i + 1
                        acc = []
                        while j < n:
                            lj = lines[j]
                            if not lj.strip():
                                j += 1
                                continue
                            if indent_of(lj) <= ind2:
                                break
                            acc.append(lj.strip())
                            j += 1
                        item[kk] = " ".join(acc)
                        i = j
                    else:
                        item[kk] = _parse_scalar(vv_stripped)
                        i += 1
                items.append(item)
            else:
                i += 1
        top[key] = items
    return top


# ---------------------------------------------------------------------------
# Catalogue (all real Cursor model ids) and validation.
# ---------------------------------------------------------------------------

def load_catalogue(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        mid = line.split("\t", 1)[0].strip()
        if mid:
            ids.add(mid)
    return ids


def validate_buckets(cfg: dict, catalogue: set[str]) -> list[str]:
    """Return a list of problems; empty list means everything is fine."""
    problems: list[str] = []
    default_id = cfg.get("default_id")
    if default_id and catalogue and default_id not in catalogue:
        problems.append(f"default_id {default_id!r} not in catalogue")
    for b in cfg.get("buckets", []):
        name = b.get("name", "?")
        for ladder_key in ("ladder", "fast_ladder"):
            for mid in b.get(ladder_key) or []:
                if catalogue and mid not in catalogue:
                    problems.append(f"bucket {name!r} {ladder_key} references unknown id {mid!r}")
    return problems


# ---------------------------------------------------------------------------
# Resolver: (bucket, latency, effort) -> concrete id.
# ---------------------------------------------------------------------------

def resolve_id(cfg: dict, bucket_name: str, latency: str, effort: str) -> tuple[str, str]:
    """
    Pick one id from the bucket's ladder.

      - `latency="fast"` prefers `fast_ladder` when non-empty.
      - `effort="max"` escalates toward the end of the ladder.
      - `effort="normal"` stays near the front.

    Returns (id, note).
    """
    buckets = cfg.get("buckets") or []
    bucket = next((b for b in buckets if b.get("name") == bucket_name), None)
    if bucket is None:
        return cfg.get("default_id"), f"unknown bucket {bucket_name!r}; used default_id"

    use_fast = (latency == "fast") and bool(bucket.get("fast_ladder"))
    ladder = bucket.get("fast_ladder") if use_fast else bucket.get("ladder")
    ladder = [x for x in (ladder or []) if isinstance(x, str)]
    if not ladder:
        return cfg.get("default_id"), f"bucket {bucket_name!r} has empty ladder; used default_id"

    if effort == "max":
        pick = ladder[-1]
        tier = "max"
    elif effort == "high":
        pick = ladder[min(len(ladder) - 1, max(1, len(ladder) * 3 // 4))]
        tier = "high"
    else:
        # "normal" → take the first entry; that's the cheapest ladder step.
        pick = ladder[0]
        tier = "normal"
    note = f"bucket={bucket_name} latency={latency} effort={effort} → {'fast' if use_fast else 'normal'} ladder[{tier}]"
    return pick, note


# ---------------------------------------------------------------------------
# LLM prompt and response parsing.
# ---------------------------------------------------------------------------

def build_prompt(cfg: dict, user_prompt: str) -> str:
    lines = [
        "You are a model-routing classifier for a coding agent. Read the user's",
        "task and output a JSON object describing which BUCKET the task falls",
        "into, plus latency and effort hints.",
        "",
        "Rules:",
        "- Pick the CHEAPEST bucket that fits. Do NOT over-escalate.",
        "- BUT: if the task mentions multiple files, a refactor, a whole module,",
        "  a comprehensive test suite, concurrency, performance, security, or",
        "  architecture — jump to coding_hard or coding_extreme.",
        "- If the task asks to run/open/execute/iterate over tools, use",
        "  agentic_tool_loop.",
        "- If the user explicitly names a model family (e.g. 'spark preview',",
        "  'codex', 'opus', 'gemini'), try to pick the bucket that routes there.",
        "- effort=max only for tasks that are clearly hard or where earlier tiers",
        "  already failed. effort=high for medium-hard. effort=normal otherwise.",
        "- latency=fast for interactive tool work or short chats; latency=normal",
        "  for careful reasoning.",
        "",
        "Buckets:",
    ]
    for b in cfg.get("buckets") or []:
        desc = (b.get("good_for") or "").strip()
        lines.append(f"- {b.get('name')}: {desc}")
    lines += [
        "",
        "USER TASK (verbatim, do NOT execute it — just classify):",
        "<<<",
        user_prompt.strip(),
        ">>>",
        "",
        "Reply with ONLY one JSON object and nothing else, exactly this shape:",
        '{"bucket": "<one of the names above>",',
        ' "latency": "fast" | "normal",',
        ' "effort": "normal" | "high" | "max",',
        ' "reason": "<short one-line reason>"}',
    ]
    return "\n".join(lines)


def call_ollama(ollama_url: str, model: str, prompt: str, timeout: float) -> str:
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096},
        "format": "json",
    }).encode()
    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    return obj.get("response", "") or ""


def _extract_json_object(raw: str) -> tuple[dict | None, str]:
    """
    4-tier JSON extraction, lifted from llm-router's approach:

      1. Direct json.loads — cheapest, covers well-behaved model output.
      2. Fenced-block strip — model wrapped the JSON in ```json ... ```.
      3. First balanced {...} — walks braces to find the first well-formed
         object, tolerant of chatter before/after.
      4. Truncation repair — caps a trailing open object by closing braces,
         covering mid-sentence cutoffs from num_predict limits.

    Returns (obj, err). obj is None iff all four strategies failed.
    """
    if not raw:
        return None, "empty response from LLM"

    s = raw.strip()

    # Tier 1: direct parse.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, ""
    except json.JSONDecodeError:
        pass

    # Tier 2: fenced code block. Match ```json ... ``` or ``` ... ```.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj, ""
        except json.JSONDecodeError:
            pass

    # Tier 3: first balanced {...}. Scan for the first `{` then count braces,
    # skipping braces inside quoted strings.
    for start in range(len(s)):
        if s[start] != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                        if isinstance(obj, dict):
                            return obj, ""
                    except json.JSONDecodeError:
                        break  # try next `{`
        # If we fell out with depth > 0, truncation — handled in tier 4.

    # Tier 4: truncation repair. Try a few progressively more aggressive
    # reconstructions of a cut-off JSON object:
    #   (a) close trailing unterminated string, strip trailing partial key,
    #       and close open braces/brackets.
    #   (b) just append closing braces.
    # Pick the first one that parses.
    first_open = s.find("{")
    if first_open >= 0:
        head = s[first_open:]

        def _close(candidate: str) -> str:
            # Scan, tracking string state and bracket depth, then append
            # whatever is needed to balance.
            in_str = False
            esc = False
            stack: list[str] = []
            for c in candidate:
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == "{":
                    stack.append("}")
                elif c == "[":
                    stack.append("]")
                elif c in "}]" and stack and stack[-1] == c:
                    stack.pop()
            tail = ""
            if in_str:
                tail += '"'
            # If we terminated a string but are mid key (e.g. `,"latency":`),
            # strip the dangling key/colon so we can close the object cleanly.
            closed = candidate + tail
            closed = re.sub(r",\s*\"[^\"]*\"\s*:\s*$", "", closed)
            closed = re.sub(r",\s*\"[^\"]*\"\s*$", "", closed)
            closed = re.sub(r",\s*$", "", closed)
            return closed + "".join(reversed(stack))

        candidates = [_close(head), head + "}", head + "}}", head + "}]}"]
        for cand in candidates:
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj, ""
            except json.JSONDecodeError:
                continue

    return None, f"non-JSON from LLM; head={raw[:200]!r}"


def parse_router_output(raw: str, valid_buckets: set[str]) -> tuple[dict | None, str]:
    if not raw:
        return None, "empty response from router LLM"
    obj, err = _extract_json_object(raw)
    if obj is None:
        return None, f"router returned non-JSON: {err}"
    bucket = obj.get("bucket")
    if not isinstance(bucket, str) or bucket not in valid_buckets:
        return None, f"router picked unknown bucket {bucket!r}"
    latency = obj.get("latency", "normal")
    effort = obj.get("effort", "normal")
    if latency not in ("fast", "normal"):
        latency = "normal"
    if effort not in ("normal", "high", "max"):
        effort = "normal"
    reason = str(obj.get("reason", "")).strip() or "(no reason)"
    return {"bucket": bucket, "latency": latency, "effort": effort, "reason": reason}, ""


def log_event(log_file: Path, event: dict) -> None:
    try:
        with log_file.open("a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Planner mode: decompose a prompt into a DAG of tasks.
#
# The planner LLM returns:
#
#   {
#     "summary": "...",
#     "tasks": [
#       {
#         "id": "t1",
#         "description": "...",
#         "bucket": "coding_hard",
#         "effort": "high",                # normal | high | max
#         "latency": "normal",             # fast | normal
#         "depends_on": [],                # list of task ids
#         "reads": ["path/or/glob", ...],  # optional file hints
#         "writes": ["path/or/glob", ...]  # optional file hints
#       },
#       ...
#     ]
#   }
#
# router.py fills in the Cursor `model` per task (via resolve_id), validates
# the DAG (no cycles, no dangling refs, no unknown buckets), and computes
# `waves` — a list of task-id lists where tasks in the same wave can run in
# parallel. On ANY validation failure we fall back to a 1-task plan using
# the existing single-model classifier so callers always get a usable plan.
# ---------------------------------------------------------------------------

def build_planner_prompt(cfg: dict, user_prompt: str) -> str:
    bucket_lines = []
    for b in cfg.get("buckets") or []:
        desc = (b.get("good_for") or "").strip()
        bucket_lines.append(f"- {b.get('name')}: {desc}")

    return "\n".join([
        "You are a TASK-DECOMPOSITION PLANNER for a coding agent.",
        "Read the user's request and break it into the smallest number of",
        "concrete subtasks that can each be handed to a separate agent run.",
        "",
        "Rules:",
        "- If the request is ALREADY a single concrete task (e.g. 'fix this typo',",
        "  'explain recursion'), return EXACTLY ONE task. Do not invent subtasks.",
        "- Prefer fewer, larger tasks over many tiny ones. Aim for 1-6 tasks.",
        "- Each task.description MUST be a self-contained instruction an agent",
        "  can execute without re-reading the original prompt. Include file",
        "  names, function names, scope.",
        "- Populate depends_on with ids of tasks that MUST finish first. If two",
        "  tasks touch the same file with writes, one MUST depend on the other.",
        "- Parallelism matters: if tasks are truly independent (different files,",
        "  no shared state), leave depends_on empty so they can run in parallel.",
        "- Pick the CHEAPEST bucket that fits each task. Use coding_hard /",
        "  coding_extreme / deep_reason only for genuinely large or tricky work.",
        "- reads/writes: list files, directories, or globs that the task is",
        "  expected to READ or WRITE (best effort; omit if unsure).",
        "",
        "Buckets:",
        *bucket_lines,
        "",
        "USER REQUEST (verbatim, do NOT execute it — only plan):",
        "<<<",
        user_prompt.strip(),
        ">>>",
        "",
        "Reply with ONLY one JSON object and nothing else, exactly this shape:",
        '{',
        '  "summary": "<one-line summary of the whole request>",',
        '  "tasks": [',
        '    {',
        '      "id": "t1",',
        '      "description": "<what this subtask does, self-contained>",',
        '      "bucket": "<one of the bucket names above>",',
        '      "latency": "fast" | "normal",',
        '      "effort": "normal" | "high" | "max",',
        '      "depends_on": ["t_other", ...],',
        '      "reads": ["path/or/glob", ...],',
        '      "writes": ["path/or/glob", ...]',
        '    }',
        '  ]',
        '}',
    ])


def _parse_plan_json(raw: str) -> tuple[dict | None, str]:
    obj, err = _extract_json_object(raw or "")
    if obj is None:
        return None, f"planner returned non-JSON: {err}"
    if "tasks" not in obj or not isinstance(obj["tasks"], list) or not obj["tasks"]:
        return None, "planner JSON has no non-empty 'tasks' list"
    return obj, ""


def _validate_and_enrich_plan(plan: dict, cfg: dict, catalogue: set[str]) -> tuple[dict | None, list[str]]:
    """
    Validate the decoded plan, fill in resolved Cursor model ids per task,
    and compute `waves` via Kahn's topological sort.

    Returns (enriched_plan, problems). If problems is non-empty, the plan is
    considered unusable and the caller should fall back.
    """
    problems: list[str] = []
    valid_buckets = {b["name"] for b in (cfg.get("buckets") or []) if "name" in b}
    default_id = cfg.get("default_id")

    raw_tasks = plan.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return None, ["plan has no tasks"]

    # Normalise tasks and assign ids.
    cleaned: list[dict] = []
    seen_ids: set[str] = set()
    for idx, t in enumerate(raw_tasks):
        if not isinstance(t, dict):
            problems.append(f"task[{idx}] is not an object")
            continue
        tid = str(t.get("id") or f"t{idx+1}").strip() or f"t{idx+1}"
        if tid in seen_ids:
            problems.append(f"duplicate task id {tid!r}")
            continue
        seen_ids.add(tid)
        desc = str(t.get("description") or "").strip()
        if not desc:
            problems.append(f"task {tid!r} has empty description")
            continue
        bucket = t.get("bucket")
        if bucket not in valid_buckets:
            problems.append(f"task {tid!r} has unknown bucket {bucket!r}")
            continue
        latency = t.get("latency", "normal")
        if latency not in ("fast", "normal"):
            latency = "normal"
        effort = t.get("effort", "normal")
        if effort not in ("normal", "high", "max"):
            effort = "normal"
        deps = t.get("depends_on") or []
        if not isinstance(deps, list):
            problems.append(f"task {tid!r} depends_on is not a list")
            continue
        deps = [str(x) for x in deps]
        reads = [str(x) for x in (t.get("reads") or []) if isinstance(x, (str, int))]
        writes = [str(x) for x in (t.get("writes") or []) if isinstance(x, (str, int))]

        mid, note = resolve_id(cfg, bucket, latency, effort)
        if not mid or (catalogue and mid not in catalogue):
            mid = default_id
            note = (note or "") + " | resolver fallback → default_id"

        cleaned.append({
            "id": tid,
            "description": desc,
            "bucket": bucket,
            "latency": latency,
            "effort": effort,
            "depends_on": deps,
            "reads": reads,
            "writes": writes,
            "model": mid,
            "resolver_note": note.strip(" |"),
        })

    if problems:
        return None, problems

    # Check deps reference known ids.
    ids = {t["id"] for t in cleaned}
    for t in cleaned:
        for d in t["depends_on"]:
            if d not in ids:
                problems.append(f"task {t['id']!r} depends on unknown task {d!r}")
    if problems:
        return None, problems

    # Kahn's topological sort → waves.
    indeg: dict[str, int] = {t["id"]: 0 for t in cleaned}
    children: dict[str, list[str]] = {t["id"]: [] for t in cleaned}
    for t in cleaned:
        for d in t["depends_on"]:
            indeg[t["id"]] += 1
            children[d].append(t["id"])

    remaining = dict(indeg)
    waves: list[list[str]] = []
    taken: set[str] = set()
    while len(taken) < len(cleaned):
        wave = sorted([tid for tid, n in remaining.items() if n == 0 and tid not in taken])
        if not wave:
            # Cycle.
            problems.append(f"cycle detected among tasks: {sorted(set(remaining) - taken)}")
            return None, problems
        waves.append(wave)
        for tid in wave:
            taken.add(tid)
            for ch in children[tid]:
                remaining[ch] -= 1

    # Safety: if two tasks in the same wave write to overlapping paths,
    # serialise them by pushing the later one into the next wave.
    # (Only a shallow string-equality / prefix check — good enough as a hint.)
    def _overlaps(a: list[str], b: list[str]) -> bool:
        sa, sb = set(a), set(b)
        if sa & sb:
            return True
        for x in sa:
            for y in sb:
                if x and y and (x.startswith(y) or y.startswith(x)):
                    return True
        return False

    by_id = {t["id"]: t for t in cleaned}
    fixed_waves: list[list[str]] = []
    warnings: list[str] = []
    for wave in waves:
        # Greedy: keep each task in this wave unless a prior task in the same
        # wave also writes an overlapping path — in that case push to the end.
        kept: list[str] = []
        pushed: list[str] = []
        for tid in wave:
            clash = False
            for k in kept:
                if _overlaps(by_id[tid]["writes"], by_id[k]["writes"]):
                    clash = True
                    warnings.append(
                        f"serialised {tid!r} after {k!r} — both write overlapping paths "
                        f"({by_id[tid]['writes']} vs {by_id[k]['writes']})"
                    )
                    break
            if clash:
                pushed.append(tid)
            else:
                kept.append(tid)
        fixed_waves.append(kept)
        if pushed:
            # Add dependencies so `depends_on` is still truthful.
            for tid in pushed:
                by_id[tid]["depends_on"] = sorted(set(by_id[tid]["depends_on"]) | set(kept))
            fixed_waves.append(pushed)

    enriched = {
        "summary": str(plan.get("summary") or "").strip(),
        "tasks": cleaned,
        "waves": fixed_waves,
        "warnings": warnings,
    }
    return enriched, []


def plan_prompt(
    cfg: dict,
    catalogue: set[str],
    user_prompt: str,
    ollama_url: str,
    ollama_model: str,
    timeout: float,
) -> tuple[dict, dict]:
    """
    Returns (plan, meta). `meta` contains debugging info: duration, raw LLM
    response head, whether a fallback was used, any validation problems, and
    whether the cache was hit.
    """
    started = time.time()
    meta: dict = {
        "planner_model": ollama_model,
        "fallback": False,
        "problems": [],
        "cache": "miss",
    }
    user_prompt = user_prompt.strip()

    if not user_prompt:
        meta["problems"].append("empty prompt")
        meta["fallback"] = True
        meta["dur_ms"] = int((time.time() - started) * 1000)
        return _single_task_fallback(cfg, catalogue, ""), meta

    # Cache lookup. Key over prompt + planner model + model catalogue hash +
    # bucket-list version, so upgrading models.yaml invalidates stale plans.
    cat_sig = hashlib.sha256(("\n".join(sorted(catalogue))).encode()).hexdigest()[:12]
    buckets_sig = hashlib.sha256(
        ("\n".join(sorted(b.get("name", "") for b in cfg.get("buckets") or []))).encode()
    ).hexdigest()[:12]
    ckey = _cache_key("plan", user_prompt, ollama_model, cat_sig, buckets_sig)
    cached = cache_get("plan", ckey)
    if cached is not None:
        meta["cache"] = "hit"
        meta["dur_ms"] = int((time.time() - started) * 1000)
        return cached, meta

    try:
        llm_prompt = build_planner_prompt(cfg, user_prompt)
        raw = call_ollama(ollama_url, ollama_model, llm_prompt, timeout)
        meta["raw_head"] = raw[:400]
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        meta["problems"].append(f"planner call failed: {type(e).__name__}: {e}")
        meta["fallback"] = True
        meta["dur_ms"] = int((time.time() - started) * 1000)
        return _single_task_fallback(cfg, catalogue, user_prompt), meta

    parsed, err = _parse_plan_json(raw)
    if parsed is None:
        meta["problems"].append(err)
        meta["fallback"] = True
        meta["dur_ms"] = int((time.time() - started) * 1000)
        return _single_task_fallback(cfg, catalogue, user_prompt), meta

    enriched, problems = _validate_and_enrich_plan(parsed, cfg, catalogue)
    if enriched is None:
        meta["problems"].extend(problems)
        meta["fallback"] = True
        meta["dur_ms"] = int((time.time() - started) * 1000)
        return _single_task_fallback(cfg, catalogue, user_prompt), meta

    cache_put("plan", ckey, enriched)
    meta["dur_ms"] = int((time.time() - started) * 1000)
    return enriched, meta


def _single_task_fallback(cfg: dict, catalogue: set[str], user_prompt: str) -> dict:
    """Build a 1-task plan using the existing classifier logic (best effort)."""
    valid_buckets = {b["name"] for b in (cfg.get("buckets") or []) if "name" in b}
    default_id = cfg.get("default_id")
    default_bucket = cfg.get("default_bucket") or (
        next(iter(valid_buckets)) if valid_buckets else None
    )
    bucket = default_bucket
    effort = "normal"
    latency = "normal"

    # Best-effort classification (may fail; we don't care, we still emit a plan).
    try:
        llm_prompt = build_prompt(cfg, user_prompt or "do nothing")
        raw = call_ollama(
            os.environ.get("ROUTER_OLLAMA_URL", "http://localhost:11434"),
            os.environ.get("ROUTER_OLLAMA_MODEL", "qwen2.5:latest"),
            llm_prompt,
            float(os.environ.get("ROUTER_TIMEOUT_SEC", "12")),
        )
        parsed, _ = parse_router_output(raw, valid_buckets)
        if parsed:
            bucket = parsed["bucket"]
            latency = parsed["latency"]
            effort = parsed["effort"]
    except Exception:
        pass

    mid, note = resolve_id(cfg, bucket, latency, effort)
    if not mid or (catalogue and mid not in catalogue):
        mid = default_id
    return {
        "summary": "single-task fallback plan",
        "tasks": [{
            "id": "t1",
            "description": user_prompt or "",
            "bucket": bucket,
            "latency": latency,
            "effort": effort,
            "depends_on": [],
            "reads": [],
            "writes": [],
            "model": mid,
            "resolver_note": note,
        }],
        "waves": [["t1"]],
        "warnings": [],
    }


def render_plan_markdown(plan: dict) -> str:
    lines: list[str] = []
    summary = plan.get("summary") or ""
    lines.append(f"Plan: {summary}" if summary else "Plan")
    lines.append("")
    by_id = {t["id"]: t for t in plan.get("tasks") or []}
    for wi, wave in enumerate(plan.get("waves") or [], start=1):
        parallel = "parallel" if len(wave) > 1 else "single"
        dep_note = ""
        if wi > 1:
            prev = plan["waves"][wi - 2]
            dep_note = f"  (waits for: {', '.join(prev)})"
        lines.append(f"  Wave {wi} — {parallel}, {len(wave)} task(s){dep_note}")
        for tid in wave:
            t = by_id[tid]
            lines.append(f"    {tid}  [{t['bucket']} / effort={t['effort']} / latency={t['latency']}]  →  {t['model']}")
            lines.append(f"        {t['description']}")
            extras = []
            if t["depends_on"]: extras.append(f"depends_on={t['depends_on']}")
            if t["reads"]:      extras.append(f"reads={t['reads']}")
            if t["writes"]:     extras.append(f"writes={t['writes']}")
            if extras:
                lines.append(f"        " + "  ".join(extras))
        lines.append("")
    for w in plan.get("warnings") or []:
        lines.append(f"  ! warning: {w}")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# LLM-as-judge.
#
# Inspired by llm-router's judge.py. The idea: on a user-configurable sample
# rate, after we commit to a (bucket, effort, latency, model) decision, ask a
# small local LLM whether that decision looks right for the prompt. The
# judge's verdict is appended to judge.log — we DON'T fold it back into
# routing automatically. It's a signal for us to look at later and tune
# models.yaml / prompt rules.
#
# Absolutely non-blocking: we fork a detached subprocess and return
# immediately. If Ollama is down, worst case: some child processes fail
# silently. The main path is unaffected.
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are a model-routing QA judge. Given a user task and
the routing decision that was made, judge whether the routed Cursor model is
reasonable. DO NOT execute the task. Be terse.

User task:
<<<
{prompt}
>>>

Routing decision:
- bucket: {bucket}
- effort: {effort}
- latency: {latency}
- chosen cursor model: {picked}
- router reason: {reason}

Reply with ONLY one JSON object, no prose:
{{"verdict": "ok" | "over" | "under" | "wrong_bucket",
 "confidence": 0.0-1.0,
 "suggested_bucket": "<same as above or a different one>",
 "notes": "<one short sentence>"}}
"""


def _maybe_judge(
    *,
    prompt: str,
    picked_id: str,
    parsed: dict | None,
    reason: str,
    cache_status: str,
    err: str | None,
) -> None:
    try:
        rate = float(os.environ.get("ROUTER_JUDGE_SAMPLE", "0") or "0")
    except ValueError:
        rate = 0.0
    if rate <= 0.0:
        return
    if cache_status == "hit":
        # No point judging a cached decision every time.
        return
    if err is not None or parsed is None:
        return
    if random.random() > rate:
        return

    payload = json.dumps({
        "prompt": prompt,
        "picked": picked_id,
        "parsed": parsed,
        "reason": reason,
    })
    try:
        # Detached child: new session, std* closed, never blocks us.
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--judge-one"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        ).stdin.write(payload.encode())
    except OSError:
        pass


def _run_judge_one() -> int:
    """Internal: consume a JSON payload on stdin, produce one judge.log line."""
    try:
        raw_in = sys.stdin.read()
        payload = json.loads(raw_in)
    except (OSError, json.JSONDecodeError):
        return 0

    url = os.environ.get("ROUTER_JUDGE_URL", os.environ.get("ROUTER_OLLAMA_URL", "http://localhost:11434"))
    model = os.environ.get("ROUTER_JUDGE_MODEL", "qwen2.5:latest")
    timeout = float(os.environ.get("ROUTER_JUDGE_TIMEOUT_SEC", "30"))
    log_file = Path(os.environ.get("ROUTER_JUDGE_LOG_FILE", str(SCRIPT_DIR / "judge.log")))

    parsed = payload.get("parsed") or {}
    llm_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=(payload.get("prompt") or "")[:2000],
        bucket=parsed.get("bucket", "?"),
        effort=parsed.get("effort", "?"),
        latency=parsed.get("latency", "?"),
        picked=payload.get("picked", "?"),
        reason=(payload.get("reason") or "")[:300],
    )
    started = time.time()
    verdict: dict | None = None
    err: str | None = None
    try:
        raw = call_ollama(url, model, llm_prompt, timeout)
        verdict, _ = _extract_json_object(raw)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        err = f"{type(e).__name__}: {e}"

    log_event(log_file, {
        "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge_model": model,
        "picked": payload.get("picked"),
        "bucket": parsed.get("bucket"),
        "effort": parsed.get("effort"),
        "latency": parsed.get("latency"),
        "prompt_head": (payload.get("prompt") or "")[:160],
        "verdict": verdict,
        "err": err,
        "dur_ms": int((time.time() - started) * 1000),
    })
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    # Internal sub-entry: dispatched by _maybe_judge's detached child.
    if len(sys.argv) >= 2 and sys.argv[1] == "--judge-one":
        return _run_judge_one()
    ap = argparse.ArgumentParser(description="Pick a Cursor model id via local Ollama")
    ap.add_argument("--prompt", help="prompt text (otherwise read stdin)")
    ap.add_argument("--ollama-url", default=os.environ.get("ROUTER_OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--ollama-model", default=os.environ.get("ROUTER_OLLAMA_MODEL", "qwen2.5:latest"))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("ROUTER_TIMEOUT_SEC", "12")))
    ap.add_argument("--models-file", default=os.environ.get("ROUTER_MODELS_FILE", str(SCRIPT_DIR / "models.yaml")))
    ap.add_argument("--catalogue-file", default=os.environ.get("ROUTER_CATALOGUE_FILE", str(SCRIPT_DIR / "cursor-models.tsv")))
    ap.add_argument("--log-file", default=os.environ.get("ROUTER_LOG_FILE", str(SCRIPT_DIR / "router.log")))
    ap.add_argument("--explain", action="store_true", help="print reason to stderr")
    ap.add_argument("--dry-run", action="store_true", help="don't print id to stdout, only reason to stderr")
    ap.add_argument("--validate", action="store_true", help="check ladders against cursor-models.tsv and exit")
    ap.add_argument("--plan", action="store_true", help="decompose the prompt into a DAG of tasks instead of picking a single model")
    ap.add_argument("--json", action="store_true", help="with --plan, emit JSON instead of human-readable plan")
    ap.add_argument("--planner-url", default=os.environ.get("ROUTER_PLANNER_URL", os.environ.get("ROUTER_OLLAMA_URL", "http://localhost:11434")))
    ap.add_argument("--planner-model", default=os.environ.get("ROUTER_PLANNER_MODEL", "qwen2.5:latest"))
    ap.add_argument("--planner-timeout", type=float, default=float(os.environ.get("ROUTER_PLANNER_TIMEOUT_SEC", "60")))
    ap.add_argument("--plan-log-file", default=os.environ.get("ROUTER_PLAN_LOG_FILE", str(SCRIPT_DIR / "plan.log")))
    args = ap.parse_args()

    models_path = Path(args.models_file)
    if not models_path.exists():
        print(f"router: models file not found: {models_path}", file=sys.stderr)
        return 2
    cfg = load_yaml_simple(models_path)
    buckets = cfg.get("buckets") or []
    if not buckets:
        print("router: models.yaml has no buckets", file=sys.stderr)
        return 2
    default_id = cfg.get("default_id") or buckets[0]["ladder"][0]
    default_bucket = cfg.get("default_bucket") or buckets[0]["name"]
    valid_buckets = {b["name"] for b in buckets if "name" in b}

    catalogue = load_catalogue(Path(args.catalogue_file))
    if args.validate:
        problems = validate_buckets(cfg, catalogue)
        if not catalogue:
            print(f"router: catalogue file empty or missing ({args.catalogue_file}); skipping id validation", file=sys.stderr)
        print(f"router: {len(buckets)} buckets, "
              f"{sum(len(b.get('ladder') or []) + len(b.get('fast_ladder') or []) for b in buckets)} ladder entries, "
              f"catalogue={len(catalogue)} ids")
        if problems:
            for p in problems:
                print(f"  ! {p}")
            return 1
        print("  ok: every ladder entry is a real Cursor model id")
        return 0

    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read()
    prompt = (prompt or "").strip()
    if not prompt:
        if args.plan:
            # Emit a minimal no-op plan so downstream tooling never breaks.
            empty_plan = _single_task_fallback(cfg, catalogue, "")
            if args.json:
                sys.stdout.write(json.dumps(empty_plan, indent=2))
            else:
                sys.stdout.write(render_plan_markdown(empty_plan))
            return 0
        if not args.dry_run:
            sys.stdout.write(default_id)
        if args.explain or args.dry_run:
            print(f"router: empty prompt → default {default_id}", file=sys.stderr)
        return 0

    if args.plan:
        plan, meta = plan_prompt(
            cfg,
            catalogue,
            prompt,
            args.planner_url,
            args.planner_model,
            args.planner_timeout,
        )
        log_event(Path(args.plan_log_file), {
            "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "planner_model": args.planner_model,
            "prompt_head": prompt[:240],
            "num_tasks": len(plan.get("tasks") or []),
            "num_waves": len(plan.get("waves") or []),
            "fallback": meta.get("fallback"),
            "problems": meta.get("problems"),
            "dur_ms": meta.get("dur_ms"),
            "cache": meta.get("cache"),
        })
        if args.explain:
            fb = " (FALLBACK)" if meta.get("fallback") else ""
            ch = " (cache hit)" if meta.get("cache") == "hit" else ""
            print(f"router: planned {len(plan.get('tasks') or [])} task(s) in "
                  f"{len(plan.get('waves') or [])} wave(s){fb}{ch} "
                  f"({meta.get('dur_ms')}ms, planner={args.planner_model})",
                  file=sys.stderr)
            for p in meta.get("problems") or []:
                print(f"  problem: {p}", file=sys.stderr)
        if args.json:
            sys.stdout.write(json.dumps(plan, indent=2))
        else:
            sys.stdout.write(render_plan_markdown(plan))
        return 0

    started = time.time()
    picked_id: str | None = None
    reason = ""
    err = None
    raw = ""
    parsed: dict | None = None
    cache_status = "miss"

    # Cache lookup for classifier, keyed on prompt + classifier model +
    # bucket-list signature.
    buckets_sig = hashlib.sha256(
        ("\n".join(sorted(valid_buckets))).encode()
    ).hexdigest()[:12]
    cls_key = _cache_key("cls", prompt, args.ollama_model, buckets_sig)
    cached_cls = cache_get("cls", cls_key)
    if cached_cls and isinstance(cached_cls.get("picked"), str):
        picked_id = cached_cls["picked"]
        parsed = cached_cls.get("parsed")
        reason = cached_cls.get("reason", "cache hit")
        cache_status = "hit"

    if picked_id is None:
        try:
            llm_prompt = build_prompt(cfg, prompt)
            raw = call_ollama(args.ollama_url, args.ollama_model, llm_prompt, args.timeout)
            parsed, parse_err = parse_router_output(raw, valid_buckets)
            if parsed is None:
                reason = parse_err
            else:
                picked_id, note = resolve_id(cfg, parsed["bucket"], parsed["latency"], parsed["effort"])
                if picked_id and (not catalogue or picked_id in catalogue):
                    reason = f"{parsed['reason']} | {note}"
                else:
                    picked_id = None
                    reason = f"resolver produced invalid id (bucket={parsed['bucket']}); {note}"
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            err = f"{type(e).__name__}: {e}"

    if not picked_id:
        # Fall back: use default_bucket with normal latency/effort, then finally default_id.
        fb_id, fb_note = resolve_id(cfg, default_bucket, "normal", "normal")
        if fb_id and (not catalogue or fb_id in catalogue):
            picked_id = fb_id
            reason = (reason or err or "fallback") + f" | {fb_note}"
        else:
            picked_id = default_id
            reason = (reason or err or "fallback") + " | used default_id"

    # Persist successful, non-fallback decisions so re-running the same prompt
    # is effectively free. We cache even "fallback" picks because the cache is
    # keyed by model+prompt — if Ollama was down once, we still want to try
    # fresh next time, which we do because fallbacks don't cache:
    if cache_status == "miss" and parsed is not None and err is None:
        cache_put("cls", cls_key, {
            "picked": picked_id,
            "parsed": parsed,
            "reason": reason,
        })

    dur_ms = int((time.time() - started) * 1000)
    log_event(Path(args.log_file), {
        "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ollama_model": args.ollama_model,
        "parsed": parsed,
        "picked": picked_id,
        "reason": reason,
        "err": err,
        "raw": raw[:500],
        "prompt_head": prompt[:160],
        "dur_ms": dur_ms,
        "cache": cache_status,
    })

    # Fire-and-forget LLM-as-judge. Disabled by default; user opts in with
    # ROUTER_JUDGE_SAMPLE > 0. We detach a background process so we never
    # delay the caller. Only samples non-cached, non-errored decisions.
    _maybe_judge(
        prompt=prompt,
        picked_id=picked_id,
        parsed=parsed,
        reason=reason,
        cache_status=cache_status,
        err=err,
    )

    if args.explain or args.dry_run:
        ch = " (cache hit)" if cache_status == "hit" else ""
        print(f"router: picked {picked_id} — {reason}{ch} ({dur_ms}ms)", file=sys.stderr)
    if not args.dry_run:
        sys.stdout.write(picked_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
