# agents/ — role registry

Each YAML file defines one **persona** that can be attached to a task. Personas add a system-prompt preamble to `cursor-agent --print`; they do **not** change which model is picked (that's `models.yaml`'s job).

## Shape

```yaml
# agents/<name>.yaml
name: reviewer                 # required — matches the value tasks reference
description: >                 # short one-line description (for docs/planner)
  Careful code reviewer: spots bugs, missing edge cases, naming issues,
  test gaps. Does NOT rewrite code — comments only.
preferred_bucket: coding_medium   # optional — hint to planner. Buckets still
                                  # win if the planner picks differently.
preferred_effort: normal          # optional — normal | high | max
system_prompt: |                  # required — prepended to each task.
  You are a senior code reviewer. ...
```

## Why

A single prompt like *"Add dark mode toggle"* can decompose into several sub-tasks that differ in both **what model they need** (`bucket`) and **how they should behave** (`role`):

| Task | Bucket | Role |
|---|---|---|
| Write the toggle component | `coding_medium` | `coder` |
| Review the component | `coding_medium` | `reviewer` |
| Write tests | `coding_medium` | `tester` |
| Audit for XSS | `coding_hard` | `security-architect` |

The same `coding_medium` model runs tasks 1-3, but gets a different system-prompt each time so it plays the right role.

## Opt-out

- `ROUTER_ROLES=0` disables role-based prompt prepending globally (env).
- Planner may still emit `role` fields; the orchestrator ignores them.
- A task without a `role` field is executed with an empty preamble — the behaviour before T2.B.

## Validation

`python3 router.py --validate` lints every agents/*.yaml:
- `name` field present and unique across the directory.
- `system_prompt` non-empty.
- `preferred_bucket` (if set) references a real bucket in models.yaml.
