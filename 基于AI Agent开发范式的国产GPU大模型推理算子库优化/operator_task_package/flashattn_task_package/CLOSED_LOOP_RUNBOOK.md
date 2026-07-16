# Closed-loop Multi-agent Runbook

## Goal

The system optimizes operators through a repeatable A/B loop:

```text
OperatorSpec + roofline + memory
  -> one ChangeProposal
  -> deterministic patch application
  -> compile
  -> correctness
  -> benchmark
  -> A/B decision
  -> version promotion or rollback
  -> structured logs and memory
```

The LLM does not directly overwrite the best kernel. It generates one bounded
candidate per round. Early rounds may be large `macro` candidates; later rounds
should become `meso` stabilization and `micro` tuning.

## Main Entry Points

Dry-run, no GPU required:

```bash
python scripts/run_closed_loop.py --phase explore --rounds 1
python scripts/run_closed_loop.py --phase stabilize --rounds 1
python scripts/run_closed_loop.py --phase tune --rounds 1
```

Real A/B on MACA GPU:

```bash
python scripts/run_closed_loop.py --real --phase tune --rounds 1 --batch 1 --seq-kv 4096 --headdim 128
```

## Coder Tiers

```text
Explore   -> scale=macro -> structural candidate, template swap, new kernel path
Stabilize -> scale=meso  -> local rewrite or correctness/perf repair
Tune      -> scale=micro -> <= 6 line parameter or launch tweak
```

Macro candidates are allowed to provide `patched_source`, but they must be
isolated candidates. They are not promoted unless the deterministic A/B runner
passes compile, correctness, benchmark and KEEP.

MCP equivalents:

- `run_closed_loop`
- `latest_run`
- `list_runs`
- `record_iteration`

## Artifact Contract

```text
runs/<run_id>/
  run_manifest.json
  events.jsonl
  versions/
  logs/
    optimization_log.md
    optimization_log.jsonl
    kept_changes.md
    rejected_changes.md
    errors.md
    summary.md
  memory/
  rounds/
    round_001/
      baseline_a.cu
      analysis.json
      proposal.json
      cand_<proposal>.cu
      decision.json

results/
  latest_run.txt
  summary.md
  opencode_events.jsonl
```

## Why This Avoids Both Timeout And Tiny-only Tuning

The previous multi-agent loop could ask Analyst/Coder to replace `Q@K^T` with a
full mctlass implementation inside one long chat. That requires large
template-heavy C++ generation and can exceed subagent time limits.

The new loop separates scale from validation. Macro work is allowed, but it is a
candidate artifact with rollback. Micro work remains the default for late tuning.

Micro example:

```json
{
  "proposal_id": "r1_num_splits_12_to_8",
  "target": "NUM_SPLITS",
  "change_type": "param_tune",
  "one_line_summary": "Tune Split-K fanout",
  "before": "static constexpr int NUM_SPLITS = 12;",
  "after": "static constexpr int NUM_SPLITS = 8;",
  "hypothesis": "Reduce reduction overhead",
  "risk": "low"
}
```

`apply_change_proposal` applies the snippet to the current source. The real runner then produces independent compile/test/benchmark evidence.

Macro example:

```json
{
  "proposal_id": "explore_mctlass_mainloop_001",
  "target": "qk_softmax_pv_mainloop",
  "change_type": "template_swap",
  "scale": "macro",
  "phase": "explore",
  "template_id": "mctlass_flash_decode_v1",
  "before": "",
  "after": "",
  "patched_source": "...full isolated candidate source...",
  "hypothesis": "Prototype a fused mctlass mainloop",
  "risk": "high",
  "validation_scope": "smoke_then_full_matrix",
  "rollback_plan": "discard candidate unless compile/correctness/A-B pass"
}
```

## Agent Roles

- Analyst: bounded bottleneck report from OperatorSpec, roofline and prior runs.
- Coder: one `ChangeProposal`.
- Profiler: runs `run_closed_loop`.
- Judge: reads `decision.json` and log JSONL, decides from evidence only.
- Reflector: writes failure cases and hardware beliefs after the run.
- Logger: passive event logging only.

## KEEP Rule

A candidate is promoted only when:

1. patch applies
2. compile succeeds
3. correctness passes
4. benchmark uses median timing
5. `candidate_ms < baseline_ms * (1 - noise_margin)`
6. `versions/manifest_v2.json` records the promoted version

Dry-run verdict `SKIP` is never a performance KEEP.
