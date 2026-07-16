---
description: Use when running or designing the closed-loop optimization process: ChangeProposal -> patch -> compile -> correctness -> benchmark -> A/B decision -> logs/version promotion.
---

# Closed Loop A/B Skill

This skill is the default workflow for operator optimization in this project.
It supports three coder tiers:

```text
explore   -> macro -> structural candidate
stabilize -> meso  -> local rewrite/repair
tune      -> micro -> small parameter tweak
```

## Contract

One round accepts one `ChangeProposal`.

Micro tuning example:

```json
{
  "proposal_id": "r1_num_splits_12_to_8",
  "target": "NUM_SPLITS",
  "change_type": "param_tune",
  "one_line_summary": "Tune split count for small-batch decode",
  "before": "static constexpr int NUM_SPLITS = 12;",
  "after": "static constexpr int NUM_SPLITS = 8;",
  "hypothesis": "Reduce reduction overhead while keeping enough parallelism",
  "risk": "low"
}
```

Rules:
- For `scale=micro`, `before` must match exactly once and changed lines should stay `<= 6`.
- For `scale=meso`, changed lines should normally stay `<= 40`.
- For `scale=macro`, full `patched_source` is allowed, but only as an isolated candidate with rollback.
- Every macro/meso proposal needs `rollback_plan`.

## Fast Pipeline Check

```bash
python scripts/run_closed_loop.py --phase explore --rounds 1
python scripts/run_closed_loop.py --phase stabilize --rounds 1
python scripts/run_closed_loop.py --phase tune --rounds 1
```

Expected artifacts:

```text
runs/<run_id>/run_manifest.json
runs/<run_id>/events.jsonl
runs/<run_id>/rounds/round_001/proposal.json
runs/<run_id>/rounds/round_001/cand_<proposal>.cu
runs/<run_id>/rounds/round_001/decision.json
runs/<run_id>/logs/summary.md
results/latest_run.txt
```

Dry-run verdict is usually `SKIP`; it proves candidate construction and logging, not speed.

## Real A/B

Run on the MACA GPU machine:

```bash
python scripts/run_closed_loop.py --real --rounds 1 --batch 1 --seq-kv 4096 --headdim 128
```

KEEP requires:
- compile success
- correctness success
- benchmark median timing
- candidate time below baseline by more than `noise_margin`
- promoted version in `runs/<run_id>/versions/manifest_v2.json`

## MCP Equivalents

- `run_closed_loop`
- `latest_run`
- `list_runs`
- `record_iteration`

## Failure Handling

Patch failure:
- category: `error_patch`
- file: `runs/<run_id>/logs/errors.md`
- fix: make `before` longer and unique

Compile/runtime failure:
- category: `error_compile` or `error_runtime`
- file: `runs/<run_id>/logs/errors.md`
- Reflector writes failure memory

Correctness failure:
- verdict: `REJECT`
- file: `runs/<run_id>/logs/rejected_changes.md`

No real speedup:
- verdict: `NOCHANGE`
- file: `runs/<run_id>/logs/rejected_changes.md`
