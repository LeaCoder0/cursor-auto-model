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
  router.py --prompt "..." --no-fastpath   # skip regex fast-path, force Ollama
  router.py --learn                    # distill judge.log -> memory.json
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
  ROUTER_FASTPATH       (default 1) — 0 disables regex intent fast-path
  ROUTER_FASTPATH_FILE  (default <script_dir>/fastpath.yaml) — regex rules
  ROUTER_ROLES          (default 1) — 0 disables role system-prompt prepending
  ROUTER_ROLES_DIR      (default <script_dir>/agents) — role YAML directory
  ROUTER_MEMORY         (default 0) — 1 enables learned-memory bucket overrides
  ROUTER_MEMORY_FILE    (default <script_dir>/memory.json) — distilled overrides
  ROUTER_MEMORY_MIN_AGREE (default 3) — min judge votes to learn an override
  ROUTER_SCORING        (default 1) — 0 falls back to legacy ladder[tier] picker
  ROUTER_HEALTH_FILE    (default <script_dir>/health.json) — per-model block list
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
import datetime
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
# Intent-classifier fast-path.
#
# Before calling the Ollama classifier, try to match the prompt against a
# small list of high-confidence regex shortcuts (fastpath.yaml). On a match
# we skip the LLM entirely: (bucket, latency, effort) are taken from the
# matching rule, and resolve_id() turns that into a Cursor model id as usual.
#
# Design rules (enforced by convention, not by code):
#
#   1. Rules must be UNAMBIGUOUS. If the same prompt could reasonably want
#      a different bucket, the rule does NOT belong here — let the
#      classifier decide.
#   2. Rules are CONSERVATIVE. We always pick the cheapest defensible
#      bucket. Undershooting on the fast-path is recoverable (the agent
#      produces a fast-but-fine answer); overshooting wastes money
#      silently and never gets corrected.
#   3. Keep the list SMALL (< 25 rules). At any larger size you've written
#      a bad hand-rolled classifier — use the real one instead.
#
# Matches are cached by the same prompt-cache machinery as Ollama decisions,
# so a repeated fast-path prompt skips even the regex walk on the second
# call.
# ---------------------------------------------------------------------------

def _fastpath_enabled() -> bool:
    return os.environ.get("ROUTER_FASTPATH", "1") != "0"


_FASTPATH_CACHE: tuple[list[tuple[str, re.Pattern, dict]], float] | None = None


def load_fastpath(path: Path) -> list[tuple[str, re.Pattern, dict]]:
    """
    Load and compile fastpath rules. Returns a list of
    (rule_id, compiled_regex, decision_dict). Returns [] if the file is
    missing, malformed, or contains no valid rules — fast-path simply
    becomes a no-op in that case.

    Re-reads + recompiles if the file's mtime changes so edits take effect
    without a restart.
    """
    global _FASTPATH_CACHE
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _FASTPATH_CACHE = None
        return []
    if _FASTPATH_CACHE is not None and _FASTPATH_CACHE[1] == mtime:
        return _FASTPATH_CACHE[0]

    try:
        cfg = load_yaml_simple(path)
    except (OSError, ValueError):
        _FASTPATH_CACHE = ([], mtime)
        return []

    compiled: list[tuple[str, re.Pattern, dict]] = []
    for r in cfg.get("rules") or []:
        rid = str(r.get("id") or "").strip()
        pat = r.get("pattern")
        bucket = r.get("bucket")
        if not rid or not isinstance(pat, str) or not bucket:
            continue
        try:
            rx = re.compile(pat, re.IGNORECASE)
        except re.error:
            # A broken rule shouldn't silently disable the whole file.
            # Skip just this one.
            continue
        latency = r.get("latency", "normal")
        if latency not in ("fast", "normal"):
            latency = "normal"
        effort = r.get("effort", "normal")
        if effort not in ("normal", "high", "max"):
            effort = "normal"
        compiled.append((rid, rx, {
            "bucket": str(bucket),
            "latency": latency,
            "effort": effort,
            "reason": f"fastpath:{rid}",
        }))

    _FASTPATH_CACHE = (compiled, mtime)
    return compiled


def try_fastpath(prompt: str, rules: list[tuple[str, re.Pattern, dict]]) -> tuple[dict | None, str | None]:
    """
    Walk rules top-to-bottom, return (decision, rule_id) on first match,
    (None, None) on no match. We strip the prompt because many real prompts
    have leading/trailing whitespace from stdin or editor paste, and all
    our rules anchor with ^/$.
    """
    if not rules:
        return None, None
    stripped = prompt.strip()
    for rid, rx, decision in rules:
        if rx.search(stripped):
            return decision, rid
    return None, None


# ---------------------------------------------------------------------------
# Agent role registry.
#
# Each YAML in agents/ defines a persona that can be attached to a task.
# Roles add a system-prompt preamble to `cursor-agent --print`; they do
# NOT change which model is picked (that's models.yaml's job).
#
# Each agents/<name>.yaml has this shape:
#   name: <str>                # required, must be unique
#   description: <str>         # optional, one-line summary
#   preferred_bucket: <str>    # optional, hint to the planner
#   preferred_effort: <str>    # optional, normal | high | max
#   system_prompt: |           # required, prepended to the task prompt
#     You are ...
#
# Loader returns {name: role_dict}. Invalid files are skipped with a
# warning to stderr; they never crash the router.
# ---------------------------------------------------------------------------

def _roles_enabled() -> bool:
    return os.environ.get("ROUTER_ROLES", "1") != "0"


def _parse_role_yaml(text: str) -> dict:
    """
    Tiny single-file YAML parser for our role shape. Handles top-level
    scalars plus block-literal strings (`key: |`). Does not handle nested
    dicts or lists — we don't need them here.
    """
    out: dict = {}
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        # Top-level key only.
        if line[:1] in (" ", "\t"):
            i += 1
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest in ("|", ">"):
            # Block scalar: collect indented lines until de-indent.
            i += 1
            acc: list[str] = []
            base_indent = None
            while i < n:
                ln = lines[i]
                if not ln.strip():
                    acc.append("")
                    i += 1
                    continue
                indent = len(ln) - len(ln.lstrip(" "))
                if base_indent is None:
                    if indent == 0:
                        break
                    base_indent = indent
                if indent < base_indent:
                    break
                acc.append(ln[base_indent:])
                i += 1
            joined = ("\n".join(acc)).rstrip()
            # `>` folds; `|` preserves. We do simple split+join for `>`.
            if rest == ">":
                joined = " ".join(s.strip() for s in joined.split("\n") if s.strip())
            out[key] = joined
        else:
            # Plain scalar; strip quotes if wrapped.
            if (rest.startswith('"') and rest.endswith('"')) or \
               (rest.startswith("'") and rest.endswith("'")):
                rest = rest[1:-1]
            out[key] = rest
            i += 1
    return out


_ROLES_CACHE: tuple[dict, float] | None = None


def load_roles(roles_dir: Path) -> dict:
    """
    Scan `roles_dir` for *.yaml files; return {name: role_dict}. Invalidates
    on directory mtime change. Missing directory → empty dict (no roles).
    """
    global _ROLES_CACHE
    try:
        mtime = roles_dir.stat().st_mtime
    except OSError:
        _ROLES_CACHE = None
        return {}
    if _ROLES_CACHE is not None and _ROLES_CACHE[1] == mtime:
        # Directory mtime is a weak signal (doesn't change when a file
        # inside is edited in place), so also check newest file mtime.
        try:
            newest = max((p.stat().st_mtime for p in roles_dir.glob("*.yaml")), default=mtime)
        except OSError:
            newest = mtime
        if newest == _ROLES_CACHE[1]:
            return _ROLES_CACHE[0]

    roles: dict = {}
    try:
        yaml_files = sorted(roles_dir.glob("*.yaml"))
    except OSError:
        yaml_files = []
    newest_mtime = mtime
    for p in yaml_files:
        try:
            text = p.read_text()
            st = p.stat()
            newest_mtime = max(newest_mtime, st.st_mtime)
        except OSError:
            continue
        try:
            role = _parse_role_yaml(text)
        except Exception:
            continue
        name = (role.get("name") or p.stem).strip()
        if not name:
            continue
        if not (role.get("system_prompt") or "").strip():
            # A role without a system prompt is useless.
            continue
        roles[name] = {
            "name": name,
            "description": str(role.get("description") or "").strip(),
            "preferred_bucket": str(role.get("preferred_bucket") or "").strip() or None,
            "preferred_effort": str(role.get("preferred_effort") or "").strip() or None,
            "system_prompt": role["system_prompt"].rstrip() + "\n",
        }

    _ROLES_CACHE = (roles, newest_mtime)
    return roles


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

# ---------------------------------------------------------------------------
# Multi-policy scorer (T2.D).
#
# The legacy resolve_id() picks ladder[tier] by index. That's brittle when a
# model is unhealthy or expensive — there's no graceful skip.
#
# We replace it with a small scoring pass over every candidate in the
# (relevant) ladder. Three policies vote, weights sum to 1.0:
#
#   capability  weight 0.6   how well the candidate matches the requested
#                            effort tier (1.0 perfect, 0.7 one off, 0.4 far)
#   cost        weight 0.2   1.0 fast / 0.5 normal / 0.2 thinking — inferred
#                            from the model id substring (no extra config)
#   health      weight 0.2   1.0 by default; 0.0 if the id appears in
#                            health.json's blocked map and the until-deadline
#                            hasn't passed
#
# Capability is dominant on purpose: the user said "max effort" so a max-tier
# model SHOULD win over a cheaper but less capable one. Cost and health are
# tiebreakers / safety nets, not the primary signal.
#
# health.json (manually editable for now, no auto-collection):
#   {
#     "version": 1,
#     "blocked": {
#       "<model-id>": {"reason": "rate-limited", "until": "2026-04-19T00:00Z"}
#     }
#   }
# An entry with `until` in the past is ignored. An entry with no `until` is
# treated as permanent (until you delete it).
#
# Set ROUTER_SCORING=0 to fall back to the legacy ladder[tier] behaviour
# entirely (kept available as a safety net during the rollout).
# ---------------------------------------------------------------------------

POLICY_WEIGHTS = {"capability": 0.6, "cost": 0.2, "health": 0.2}


def _scoring_enabled() -> bool:
    return os.environ.get("ROUTER_SCORING", "1") != "0"


_HEALTH_CACHE: tuple[dict, float] | None = None


def load_health(path: Path) -> dict:
    """
    Returns {"version": 1, "blocked": {id: {reason, until?}}}. Missing or
    corrupt file -> empty health (everyone is healthy).
    """
    global _HEALTH_CACHE
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"version": 1, "blocked": {}}
    if _HEALTH_CACHE is not None and _HEALTH_CACHE[1] == mtime:
        return _HEALTH_CACHE[0]
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    blocked = data.get("blocked") or {}
    if not isinstance(blocked, dict):
        blocked = {}
    out = {"version": 1, "blocked": blocked}
    _HEALTH_CACHE = (out, mtime)
    return out


def _is_blocked_now(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    until = entry.get("until")
    if not isinstance(until, str) or not until:
        return True
    try:
        # Accept "Z", "+00:00", and naive ISO. Compare as UTC.
        s = until.rstrip("Z")
        try:
            until_dt = datetime.datetime.fromisoformat(s)
        except ValueError:
            return True  # malformed → assume blocked (safer)
        if until_dt.tzinfo is None:
            until_dt = until_dt.replace(tzinfo=datetime.timezone.utc)
        return until_dt > datetime.datetime.now(datetime.timezone.utc)
    except Exception:
        return True


def _capability_score(idx: int, ladder_len: int, effort: str) -> float:
    """
    Map (effort, position-in-ladder) to a capability score in [0,1].
    Position 0 is cheapest; position -1 is strongest.

    For effort=normal the BOTTOM of the ladder is "perfect"; for effort=max
    the TOP is perfect.
    """
    if ladder_len <= 1:
        return 1.0
    norm_pos = idx / (ladder_len - 1)  # 0.0 .. 1.0
    if effort == "max":
        target = 1.0
    elif effort == "high":
        target = 0.75
    else:
        target = 0.0
    dist = abs(norm_pos - target)
    if dist <= 0.05:
        return 1.0
    if dist <= 0.25:
        return 0.7
    return 0.4


def _cost_score(model_id: str) -> float:
    """
    Cheap heuristic: substring sniff on the id. We don't have authoritative
    pricing here so we use the well-known naming conventions:
      - 'fast' / 'flash' / '-mini' / 'haiku' / 'tinyllama' => cheap (1.0)
      - 'thinking' / 'opus' / 'extreme' / 'high' / 'max' / '1m' => pricey (0.2)
      - everything else => normal (0.5)
    """
    s = model_id.lower()
    cheap_marks = ("fast", "flash", "-mini", "haiku", "small")
    pricey_marks = ("thinking", "opus", "extreme", "-1m", "1m-")
    if any(m in s for m in cheap_marks):
        return 1.0
    if any(m in s for m in pricey_marks):
        return 0.2
    if "high" in s or "max" in s:
        return 0.4
    return 0.5


def _health_score(model_id: str, health: dict) -> float:
    blocked = (health.get("blocked") or {}).get(model_id)
    return 0.0 if blocked and _is_blocked_now(blocked) else 1.0


def score_candidates(
    cfg: dict,
    bucket_name: str,
    latency: str,
    effort: str,
    health: dict,
    catalogue: set[str] | None = None,
) -> list[dict]:
    """
    Return a list of {id, capability, cost, health, total, blocked} dicts,
    sorted by total descending (then by cost descending so cheaper wins ties).
    Empty list if the bucket / ladder is empty. Models not in the catalogue
    are excluded entirely (they can't be picked anyway).
    """
    buckets = cfg.get("buckets") or []
    bucket = next((b for b in buckets if b.get("name") == bucket_name), None)
    if bucket is None:
        return []
    use_fast = (latency == "fast") and bool(bucket.get("fast_ladder"))
    ladder_raw = bucket.get("fast_ladder") if use_fast else bucket.get("ladder")
    ladder = [x for x in (ladder_raw or []) if isinstance(x, str)]
    if not ladder:
        return []

    out: list[dict] = []
    for idx, mid in enumerate(ladder):
        if catalogue and mid not in catalogue:
            continue
        cap = _capability_score(idx, len(ladder), effort)
        cost = _cost_score(mid)
        hp = _health_score(mid, health)
        total = (
            POLICY_WEIGHTS["capability"] * cap
            + POLICY_WEIGHTS["cost"] * cost
            + POLICY_WEIGHTS["health"] * hp
        )
        out.append({
            "id": mid,
            "capability": round(cap, 3),
            "cost": round(cost, 3),
            "health": round(hp, 3),
            "total": round(total, 3),
            "blocked": hp == 0.0,
            "ladder_pos": idx,
            "ladder_len": len(ladder),
            "fast_ladder": use_fast,
        })
    # Sort by total desc, then cost desc (cheaper wins tie), then ladder_pos asc.
    out.sort(key=lambda d: (-d["total"], -d["cost"], d["ladder_pos"]))
    return out


def resolve_id(
    cfg: dict,
    bucket_name: str,
    latency: str,
    effort: str,
    *,
    catalogue: set[str] | None = None,
    health: dict | None = None,
) -> tuple[str, str]:
    """
    Pick one Cursor model id for (bucket, latency, effort).

    By default (ROUTER_SCORING=1, the new path) we use the multi-policy
    scorer over the bucket's ladder, with health-blocked models excluded.
    With ROUTER_SCORING=0 we fall back to the legacy ladder[tier] index
    behaviour (kept as a safety net).

    Returns (id, note). Note string includes scoring breakdown when
    scoring is on, or the legacy "ladder[tier]" note otherwise — so
    log readers can tell which path produced a decision.
    """
    buckets = cfg.get("buckets") or []
    bucket = next((b for b in buckets if b.get("name") == bucket_name), None)
    if bucket is None:
        return cfg.get("default_id"), f"unknown bucket {bucket_name!r}; used default_id"

    use_fast = (latency == "fast") and bool(bucket.get("fast_ladder"))
    ladder = [x for x in (bucket.get("fast_ladder") if use_fast else bucket.get("ladder")) or [] if isinstance(x, str)]
    if not ladder:
        return cfg.get("default_id"), f"bucket {bucket_name!r} has empty ladder; used default_id"

    # ---- LEGACY PATH (ROUTER_SCORING=0) ------------------------------------
    if not _scoring_enabled():
        if effort == "max":
            pick = ladder[-1]
            tier = "max"
        elif effort == "high":
            pick = ladder[min(len(ladder) - 1, max(1, len(ladder) * 3 // 4))]
            tier = "high"
        else:
            pick = ladder[0]
            tier = "normal"
        note = f"bucket={bucket_name} latency={latency} effort={effort} → {'fast' if use_fast else 'normal'} ladder[{tier}] (legacy)"
        return pick, note

    # ---- SCORING PATH (default) --------------------------------------------
    health = health if health is not None else load_health(_health_path())
    cands = score_candidates(cfg, bucket_name, latency, effort,
                             health=health, catalogue=catalogue)
    if cands:
        winner = cands[0]
        note = (
            f"bucket={bucket_name} latency={latency} effort={effort} → "
            f"scored[{winner['id']} cap={winner['capability']} cost={winner['cost']} "
            f"health={winner['health']} total={winner['total']}] of {len(cands)} cand(s)"
        )
        return winner["id"], note

    # If scoring excluded everything (e.g. all blocked + catalogue mismatch),
    # fall back to the bottom of the ladder so we always return a usable id.
    return ladder[0], (
        f"bucket={bucket_name} latency={latency} effort={effort} → all candidates "
        f"excluded by health/catalogue; fell back to ladder[0]={ladder[0]}"
    )


def _health_path() -> Path:
    return Path(os.environ.get("ROUTER_HEALTH_FILE", str(SCRIPT_DIR / "health.json")))


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

def build_planner_prompt(cfg: dict, user_prompt: str, roles: dict | None = None) -> str:
    bucket_lines = []
    for b in cfg.get("buckets") or []:
        desc = (b.get("good_for") or "").strip()
        bucket_lines.append(f"- {b.get('name')}: {desc}")

    role_lines: list[str] = []
    if roles:
        for name, role in sorted(roles.items()):
            d = role.get("description") or "(no description)"
            role_lines.append(f"- {name}: {d}")

    lines = [
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
    ]
    if role_lines:
        lines += [
            "",
            "ROLES vs BUCKETS — these are TWO INDEPENDENT FIELDS. Do not confuse",
            "them. NEVER put a role name in the bucket field, and never put a",
            "bucket name in the role field.",
            "  - bucket: WHICH MODEL runs the task (capability/cost). Required.",
            "    Value MUST come from the Buckets list above.",
            "  - role:   HOW the agent should BEHAVE while running (persona).",
            "    Value MUST come from the Roles list below. OPTIONAL — omit if",
            "    no role fits cleanly. Empty role is fine and common.",
            "Example: a task to review code for security would have",
            "    bucket=coding_hard   (because audit needs strong reasoning)",
            "    role=security-architect   (because we want a security persona)",
        ]
    lines += [
        "",
        "Buckets:",
        *bucket_lines,
    ]
    if role_lines:
        lines += [
            "",
            "Roles:",
            *role_lines,
        ]
    lines += [
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
        '      "role": "<one of the role names above, OR omit>",',
        '      "depends_on": ["t_other", ...],',
        '      "reads": ["path/or/glob", ...],',
        '      "writes": ["path/or/glob", ...]',
        '    }',
        '  ]',
        '}',
    ]
    return "\n".join(lines)


def _parse_plan_json(raw: str) -> tuple[dict | None, str]:
    obj, err = _extract_json_object(raw or "")
    if obj is None:
        return None, f"planner returned non-JSON: {err}"
    if "tasks" not in obj or not isinstance(obj["tasks"], list) or not obj["tasks"]:
        return None, "planner JSON has no non-empty 'tasks' list"
    return obj, ""


def _validate_and_enrich_plan(
    plan: dict,
    cfg: dict,
    catalogue: set[str],
    roles: dict | None = None,
) -> tuple[dict | None, list[str]]:
    """
    Validate the decoded plan, fill in resolved Cursor model ids per task,
    and compute `waves` via Kahn's topological sort.

    Returns (enriched_plan, problems). If problems is non-empty, the plan is
    considered unusable and the caller should fall back.
    """
    problems: list[str] = []
    valid_buckets = {b["name"] for b in (cfg.get("buckets") or []) if "name" in b}
    default_id = cfg.get("default_id")
    roles = roles or {}

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
        # Role is optional. Unknown roles are silently dropped (not fatal)
        # so an older plan.json can still execute against a newer roles dir
        # and vice versa. We just note it in resolver_note for debugging.
        role_raw = t.get("role")
        role_note = ""
        if isinstance(role_raw, str) and role_raw.strip():
            role_name = role_raw.strip()
            if role_name in roles:
                role = role_name
            else:
                role = None
                role_note = f" | dropped unknown role {role_name!r}"
        else:
            role = None
        # Recovery for a common planner mistake: putting a role NAME into
        # the bucket field. If bucket is unknown but happens to be a known
        # role, treat it as if the planner had said role=<that> and pick a
        # bucket from the role's preferred_bucket (or coding_medium).
        if bucket not in valid_buckets and isinstance(bucket, str) and bucket in roles:
            recovered_role = bucket
            role = role or recovered_role
            preferred = roles[recovered_role].get("preferred_bucket")
            if preferred and preferred in valid_buckets:
                bucket = preferred
            elif "coding_medium" in valid_buckets:
                bucket = "coding_medium"
            else:
                bucket = next(iter(valid_buckets), None)
            role_note += f" | recovered: bucket was role-name {recovered_role!r}, remapped to bucket={bucket}"
        if bucket not in valid_buckets:
            problems.append(f"task {tid!r} has unknown bucket {bucket!r}")
            continue
        latency = t.get("latency", "normal")
        if latency not in ("fast", "normal"):
            latency = "normal"
        effort = t.get("effort", "normal")
        if effort not in ("normal", "high", "max"):
            effort = "normal"
        # Apply role-suggested effort if planner left it default and role
        # has a preferred_effort hint. Bucket choice still wins overall.
        if role and effort == "normal":
            pe = roles[role].get("preferred_effort")
            if pe in ("normal", "high", "max"):
                effort = pe
        deps = t.get("depends_on") or []
        if not isinstance(deps, list):
            problems.append(f"task {tid!r} depends_on is not a list")
            continue
        deps = [str(x) for x in deps]
        reads = [str(x) for x in (t.get("reads") or []) if isinstance(x, (str, int))]
        writes = [str(x) for x in (t.get("writes") or []) if isinstance(x, (str, int))]

        mid, note = resolve_id(cfg, bucket, latency, effort, catalogue=catalogue)
        if not mid or (catalogue and mid not in catalogue):
            mid = default_id
            note = (note or "") + " | resolver fallback → default_id"

        cleaned.append({
            "id": tid,
            "description": desc,
            "bucket": bucket,
            "latency": latency,
            "effort": effort,
            "role": role,
            "depends_on": deps,
            "reads": reads,
            "writes": writes,
            "model": mid,
            "resolver_note": (note + role_note).strip(" |"),
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
    roles: dict | None = None,
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
    roles = roles or {}

    if not user_prompt:
        meta["problems"].append("empty prompt")
        meta["fallback"] = True
        meta["dur_ms"] = int((time.time() - started) * 1000)
        return _single_task_fallback(cfg, catalogue, ""), meta

    # Cache lookup. Key over prompt + planner model + model catalogue hash +
    # bucket-list version + role-list signature, so adding/removing/renaming
    # a role invalidates stale plans.
    cat_sig = hashlib.sha256(("\n".join(sorted(catalogue))).encode()).hexdigest()[:12]
    buckets_sig = hashlib.sha256(
        ("\n".join(sorted(b.get("name", "") for b in cfg.get("buckets") or []))).encode()
    ).hexdigest()[:12]
    roles_sig = hashlib.sha256(
        ("\n".join(sorted(roles.keys()))).encode()
    ).hexdigest()[:12]
    ckey = _cache_key("plan", user_prompt, ollama_model, cat_sig, buckets_sig, roles_sig)
    cached = cache_get("plan", ckey)
    if cached is not None:
        meta["cache"] = "hit"
        meta["dur_ms"] = int((time.time() - started) * 1000)
        return cached, meta

    try:
        llm_prompt = build_planner_prompt(cfg, user_prompt, roles)
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

    enriched, problems = _validate_and_enrich_plan(parsed, cfg, catalogue, roles)
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

    mid, note = resolve_id(cfg, bucket, latency, effort, catalogue=catalogue)
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
            "role": None,
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
            role_tag = f" / role={t['role']}" if t.get("role") else ""
            lines.append(f"    {tid}  [{t['bucket']} / effort={t['effort']} / latency={t['latency']}{role_tag}]  →  {t['model']}")
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
# Learned routing memory.
#
# This is the DISTILL + CONSOLIDATE half of ruflo's
#   RETRIEVE -> JUDGE -> DISTILL -> CONSOLIDATE -> ROUTE
# loop, adapted to our stack:
#
#   JUDGE        -> already done in T1 (judge.log).
#   DISTILL      -> `router.py --learn` reads judge.log, groups verdicts by
#                   (prompt_signature, picked_bucket), and emits memory.json.
#   CONSOLIDATE  -> only "wrong_bucket" verdicts with sufficient agreement
#                   (>= MEMORY_MIN_AGREE, default 3) become overrides.
#   RETRIEVE     -> on each classify, we hash the current prompt the same way
#                   and look it up in memory.json BEFORE calling Ollama.
#   ROUTE        -> a memory hit short-circuits to the suggested bucket.
#                   resolve_id() then turns that bucket into a Cursor model.
#
# Safety properties:
#   - Memory is OFF by default (ROUTER_MEMORY=1 to enable). We only opt
#     users in once they've explicitly run --learn at least once.
#   - Memory is read-only at runtime. The ONLY writer is `--learn`.
#   - We require N independent agreements (default 3) before overriding.
#     One angry judge call cannot rewrite the world.
#   - We require a SINGLE majority bucket among judges. A tie is no-op.
#   - Memory hits are logged with {"memory": "<sig>"} in router.log so we
#     can audit drift.
#
# Prompt signature:
#   Lowercase, drop non-alphanumerics, dedupe words, sort the first 12
#   alphabetically, join with single spaces. Bag-of-words. Crude but
#   robust to paraphrase: "fix typo in readme" and "fix the typo in
#   README.md" produce the same signature.
# ---------------------------------------------------------------------------

MEMORY_MIN_AGREE_DEFAULT = 3
MEMORY_MAX_WORDS = 12


def _memory_enabled() -> bool:
    return os.environ.get("ROUTER_MEMORY", "0") == "1"


def prompt_signature(prompt: str) -> str:
    """
    Compute a paraphrase-robust signature for `prompt`. Returns "" for an
    effectively empty prompt (so callers can short-circuit).
    """
    if not prompt:
        return ""
    norm = re.sub(r"[^a-z0-9\s]+", " ", prompt.lower())
    words = [w for w in norm.split() if w]
    if not words:
        return ""
    # Dedupe while preserving discovery order, then sort the first N.
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        unique.append(w)
    return " ".join(sorted(unique[:MEMORY_MAX_WORDS]))


def _memory_path(args_path: str | None = None) -> Path:
    return Path(args_path or os.environ.get("ROUTER_MEMORY_FILE", str(SCRIPT_DIR / "memory.json")))


def load_memory(path: Path) -> dict:
    """
    Returns {"version": 1, "entries": {sig: {bucket, support, ...}}}.
    Missing / corrupt file => empty memory (no overrides).
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "entries": {}}
    if not isinstance(data, dict) or "entries" not in data:
        return {"version": 1, "entries": {}}
    return data


def memory_lookup(memory: dict, prompt: str, valid_buckets: set[str]) -> tuple[str | None, dict | None]:
    """
    Returns (suggested_bucket, entry_dict) on a confident hit. (None, None)
    otherwise. We re-validate the bucket is real on each lookup so memory
    survives bucket renames in models.yaml without exploding (orphan
    entries just become no-ops).
    """
    sig = prompt_signature(prompt)
    if not sig:
        return None, None
    entry = memory.get("entries", {}).get(sig)
    if not entry:
        return None, None
    bucket = entry.get("bucket")
    if not isinstance(bucket, str) or bucket not in valid_buckets:
        return None, None
    return bucket, entry


def _learn_from_judge_log(
    judge_log_path: Path,
    out_path: Path,
    min_agree: int,
) -> dict:
    """
    Build (or update) memory.json from judge.log. Returns a stats dict
    suitable for printing.

    Algorithm:
      For every judge entry where verdict is "wrong_bucket" or "under" or
      "over" with a `suggested_bucket`, accumulate
        votes[sig][suggested_bucket] += weight
      where weight = max(0.5, confidence). Also count `votes[sig].total`.
      An entry becomes an override iff:
        - the leading bucket has >= min_agree votes
        - the leading bucket has > 60% share of total votes
      Existing memory.json entries are PRESERVED unless this run produces
      a stronger override for the same sig (more votes).
    """
    stats = {
        "judge_lines_read": 0,
        "judge_lines_usable": 0,
        "signatures_seen": 0,
        "overrides_written": 0,
        "overrides_kept": 0,
        "min_agree": min_agree,
    }
    if not judge_log_path.exists():
        return stats

    # signature -> {bucket -> [vote_weight, ...]}
    votes: dict[str, dict[str, list[float]]] = {}
    samples: dict[str, list[str]] = {}

    with judge_log_path.open() as fh:
        for line in fh:
            stats["judge_lines_read"] += 1
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            verdict = ev.get("verdict")
            if not isinstance(verdict, dict):
                continue
            v = verdict.get("verdict")
            sb = verdict.get("suggested_bucket")
            head = ev.get("prompt_head") or ""
            picked_bucket = ev.get("bucket")
            if v not in ("wrong_bucket", "over", "under") or not sb or not head:
                continue
            if sb == picked_bucket:
                # Judge "disagreed" but suggested the same bucket — useless.
                continue
            sig = prompt_signature(head)
            if not sig:
                continue
            try:
                conf = float(verdict.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            weight = max(0.5, min(1.0, conf))
            votes.setdefault(sig, {}).setdefault(sb, []).append(weight)
            samples.setdefault(sig, []).append(head[:120])
            stats["judge_lines_usable"] += 1

    stats["signatures_seen"] = len(votes)

    # Load + merge.
    existing = load_memory(out_path)
    entries = dict(existing.get("entries") or {})

    for sig, by_bucket in votes.items():
        total_votes = sum(len(v) for v in by_bucket.values())
        if total_votes < min_agree:
            continue
        # Find leading bucket.
        leader = max(by_bucket.items(), key=lambda kv: (sum(kv[1]), len(kv[1])))
        lead_bucket, lead_weights = leader
        lead_count = len(lead_weights)
        lead_score = sum(lead_weights)
        if lead_count < min_agree:
            continue
        if lead_score / max(total_votes, 1) <= 0.6:
            continue

        new_entry = {
            "bucket": lead_bucket,
            "support": lead_count,
            "score": round(lead_score, 3),
            "total_votes": total_votes,
            "examples": samples[sig][:3],
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        prev = entries.get(sig)
        # Keep the previous entry if it's stronger (more votes).
        if prev and prev.get("support", 0) > lead_count:
            stats["overrides_kept"] += 1
            continue
        entries[sig] = new_entry
        stats["overrides_written"] += 1

    out = {
        "version": 1,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "min_agree": min_agree,
        "entries": entries,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
        tmp.replace(out_path)
    except OSError as e:
        stats["error"] = f"{type(e).__name__}: {e}"
    return stats


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
    ap.add_argument("--fastpath-file", default=os.environ.get("ROUTER_FASTPATH_FILE", str(SCRIPT_DIR / "fastpath.yaml")))
    ap.add_argument("--no-fastpath", action="store_true", help="disable regex intent fast-path for this invocation")
    ap.add_argument("--roles-dir", default=os.environ.get("ROUTER_ROLES_DIR", str(SCRIPT_DIR / "agents")))
    ap.add_argument("--no-roles", action="store_true", help="ignore agents/*.yaml; do not attach role personas to plan tasks")
    ap.add_argument("--memory-file", default=os.environ.get("ROUTER_MEMORY_FILE", str(SCRIPT_DIR / "memory.json")))
    ap.add_argument("--no-memory", action="store_true", help="skip memory.json lookup for this invocation")
    ap.add_argument("--learn", action="store_true", help="distill judge.log into memory.json overrides and exit")
    ap.add_argument("--memory-min-agree", type=int, default=int(os.environ.get("ROUTER_MEMORY_MIN_AGREE", str(MEMORY_MIN_AGREE_DEFAULT))),
                    help="minimum number of judge agreements required to learn an override")
    ap.add_argument("--judge-log-file", default=os.environ.get("ROUTER_JUDGE_LOG_FILE", str(SCRIPT_DIR / "judge.log")),
                    help="judge.log path used by --learn")
    ap.add_argument("--health-file", default=os.environ.get("ROUTER_HEALTH_FILE", str(SCRIPT_DIR / "health.json")),
                    help="per-model block list consulted by the multi-policy scorer")
    ap.add_argument("--score", action="store_true",
                    help="dump the scored candidate list for the prompt's resolved bucket and exit")
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
        # Also lint fastpath.yaml: every rule must have a valid regex and
        # reference a real bucket. Unreachable rules (pattern already covered
        # by a prior rule on a curated corpus) are a warning, not an error.
        fp_path = Path(args.fastpath_file)
        if fp_path.exists():
            rules = load_fastpath(fp_path)
            seen_ids: set[str] = set()
            for rid, _, decision in rules:
                if rid in seen_ids:
                    problems.append(f"fastpath rule id {rid!r} is duplicated")
                seen_ids.add(rid)
                if decision["bucket"] not in valid_buckets:
                    problems.append(f"fastpath rule {rid!r} references unknown bucket {decision['bucket']!r}")
            print(f"router: {len(rules)} fastpath rule(s) loaded from {fp_path.name}")
        else:
            print(f"router: fastpath file not found ({fp_path}); fast-path disabled")
        # Lint agents/*.yaml: every role's preferred_bucket (if set) must
        # be a real bucket. system_prompt non-empty is enforced by the
        # loader (it just drops the role).
        roles_dir = Path(args.roles_dir)
        if roles_dir.exists():
            roles = load_roles(roles_dir)
            for rname, role in sorted(roles.items()):
                pb = role.get("preferred_bucket")
                if pb and pb not in valid_buckets:
                    problems.append(f"role {rname!r} preferred_bucket={pb!r} is not a real bucket")
                pe = role.get("preferred_effort")
                if pe and pe not in ("normal", "high", "max"):
                    problems.append(f"role {rname!r} preferred_effort={pe!r} must be normal|high|max")
            print(f"router: {len(roles)} role(s) loaded from {roles_dir.name}/")
        else:
            print(f"router: roles dir not found ({roles_dir}); role personas disabled")
        if problems:
            for p in problems:
                print(f"  ! {p}")
            return 1
        print("  ok: every ladder entry is a real Cursor model id")
        return 0

    if args.learn:
        judge_path = Path(args.judge_log_file)
        mem_path = Path(args.memory_file)
        if not judge_path.exists():
            print(f"router: no judge log to learn from ({judge_path})", file=sys.stderr)
            return 0
        stats = _learn_from_judge_log(judge_path, mem_path, args.memory_min_agree)
        if "error" in stats:
            print(f"router: learn failed: {stats['error']}", file=sys.stderr)
            return 1
        print(f"router: learn — read {stats['judge_lines_read']} judge lines, "
              f"{stats['judge_lines_usable']} usable across {stats['signatures_seen']} signature(s); "
              f"wrote {stats['overrides_written']} override(s), "
              f"kept {stats['overrides_kept']} stronger existing override(s) "
              f"(min_agree={stats['min_agree']}). memory file: {mem_path}")
        return 0

    if args.score:
        # Debug: dump scored candidate table for a (bucket, latency, effort)
        # tuple. Reads the bucket from --prompt as "bucket[/latency[/effort]]"
        # for offline inspection without invoking Ollama.
        spec = (args.prompt or "").strip() or sys.stdin.read().strip()
        parts = [p for p in re.split(r"[/\s,]+", spec) if p]
        if not parts or parts[0] not in valid_buckets:
            print(f"router: --score expects 'bucket[/latency[/effort]]'. Got {spec!r}.\n"
                  f"        Buckets: {sorted(valid_buckets)}", file=sys.stderr)
            return 2
        bucket = parts[0]
        latency = parts[1] if len(parts) > 1 and parts[1] in ("fast", "normal") else "normal"
        effort = parts[2] if len(parts) > 2 and parts[2] in ("normal", "high", "max") else "normal"
        health = load_health(Path(args.health_file))
        cands = score_candidates(cfg, bucket, latency, effort, health=health, catalogue=catalogue)
        print(f"router: scored {len(cands)} candidate(s) for "
              f"bucket={bucket} latency={latency} effort={effort}")
        if not cands:
            print("  (no candidates after catalogue/health filtering)")
            return 0
        # Aligned table.
        widths = max((len(c["id"]) for c in cands), default=20)
        print(f"  {'#':>2}  {'id':<{widths}}  cap   cost  health  total  blocked")
        for i, c in enumerate(cands, 1):
            print(f"  {i:>2}  {c['id']:<{widths}}  "
                  f"{c['capability']:.2f}  {c['cost']:.2f}  {c['health']:.2f}   "
                  f"{c['total']:.3f}  {('yes' if c['blocked'] else 'no')}")
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
        roles = load_roles(Path(args.roles_dir)) if (_roles_enabled() and not args.no_roles) else {}
        plan, meta = plan_prompt(
            cfg,
            catalogue,
            prompt,
            args.planner_url,
            args.planner_model,
            args.planner_timeout,
            roles=roles,
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

    # Fast-path: regex-based short-circuit for high-confidence prompts.
    # Runs only on cache-miss; skipped entirely if the user passed
    # --no-fastpath or set ROUTER_FASTPATH=0.
    fastpath_rule: str | None = None
    if picked_id is None and _fastpath_enabled() and not args.no_fastpath:
        rules = load_fastpath(Path(args.fastpath_file))
        fp_decision, fp_rule = try_fastpath(prompt, rules)
        if fp_decision is not None and fp_decision["bucket"] in valid_buckets:
            fp_picked, fp_note = resolve_id(
                cfg, fp_decision["bucket"], fp_decision["latency"], fp_decision["effort"],
                catalogue=catalogue,
            )
            if fp_picked and (not catalogue or fp_picked in catalogue):
                picked_id = fp_picked
                parsed = fp_decision
                fastpath_rule = fp_rule
                reason = f"fastpath:{fp_rule} | {fp_note}"

    # Learned memory: consult memory.json for a prior judge-driven override
    # for this prompt's signature. Memory is OFF by default; the user must
    # ROUTER_MEMORY=1 (and run --learn at least once) to opt in.
    memory_sig: str | None = None
    if picked_id is None and _memory_enabled() and not args.no_memory:
        mem = load_memory(Path(args.memory_file))
        sig = prompt_signature(prompt)
        suggested, mem_entry = memory_lookup(mem, prompt, valid_buckets)
        if suggested is not None:
            # Memory only carries a bucket. We default latency/effort to
            # normal/normal and let resolve_id pick the bottom of the
            # ladder — this is the cheapest defensible model for that
            # bucket. Effort/latency stay configurable for the future.
            mem_picked, mem_note = resolve_id(cfg, suggested, "normal", "normal", catalogue=catalogue)
            if mem_picked and (not catalogue or mem_picked in catalogue):
                picked_id = mem_picked
                parsed = {
                    "bucket": suggested,
                    "latency": "normal",
                    "effort": "normal",
                    "reason": f"memory override (support={mem_entry.get('support')}, "
                              f"score={mem_entry.get('score')})",
                }
                memory_sig = sig
                reason = f"memory:{sig[:40]} | {mem_note}"

    if picked_id is None:
        try:
            llm_prompt = build_prompt(cfg, prompt)
            raw = call_ollama(args.ollama_url, args.ollama_model, llm_prompt, args.timeout)
            parsed, parse_err = parse_router_output(raw, valid_buckets)
            if parsed is None:
                reason = parse_err
            else:
                picked_id, note = resolve_id(cfg, parsed["bucket"], parsed["latency"], parsed["effort"], catalogue=catalogue)
                if picked_id and (not catalogue or picked_id in catalogue):
                    reason = f"{parsed['reason']} | {note}"
                else:
                    picked_id = None
                    reason = f"resolver produced invalid id (bucket={parsed['bucket']}); {note}"
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            err = f"{type(e).__name__}: {e}"

    if not picked_id:
        # Fall back: use default_bucket with normal latency/effort, then finally default_id.
        fb_id, fb_note = resolve_id(cfg, default_bucket, "normal", "normal", catalogue=catalogue)
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
        "fastpath": fastpath_rule,
        "memory": memory_sig,
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
        if cache_status == "hit":
            tag = " (cache hit)"
        elif fastpath_rule:
            tag = " (fastpath)"
        elif memory_sig:
            tag = " (memory)"
        else:
            tag = ""
        print(f"router: picked {picked_id} — {reason}{tag} ({dur_ms}ms)", file=sys.stderr)
    if not args.dry_run:
        sys.stdout.write(picked_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
