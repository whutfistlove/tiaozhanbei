# AGENTS.md - Operator Optimization Agent Contract

This project builds a generalized multi-agent system for operator optimization.
The target operator is currently:

- `operator_id`: `flashattention_kvcache_decode`
- seed kernel: `kernel/splitk_h128.cu`
- cross-run best index: `results/current_best.json`
- cross-run best source: `results/best/<operator_id>_best.cu`

Every real optimization run must use this deterministic loop:

```text
resolve current best
-> Analyst report
-> Coder writes one ChangeProposal artifact
-> validate proposal artifact
-> run_closed_loop
-> compile/correctness/benchmark
-> A/B decision
-> KEEP promotes version or rollback
-> logs and memory
```

## Baseline Rule

Agents must analyze the best available kernel, not blindly restart from the seed
file.

Use MCP `resolve_best_kernel` with:

```json
{
  "operator_id": "flashattention_kvcache_decode",
  "kernel_path": "kernel/splitk_h128.cu",
  "baseline_source": "auto"
}
```

`baseline_source="auto"` means:

1. Use `results/current_best.json` when a previous real KEEP exists.
2. Fall back to `kernel/splitk_h128.cu` only when no global best exists.

Use `baseline_source="kernel"` only for an intentional reset experiment.

## Subagent Output Rule

The main agent must never replace a failed subagent.

If Analyst or Coder returns empty:

1. Retry the same subagent once with a shorter prompt.
2. If it is still empty, stop with `ERROR_AGENT_OUTPUT`.
3. Do not let the main agent write the Analyst report or Coder proposal.

For Coder, a chat response is not enough. The main agent should first call MCP
`prepare_proposal_artifact` and pass the returned `proposal_path` to Coder.
Coder must write that artifact. If the caller gives no path, Coder must call
`prepare_proposal_artifact` itself and use the returned unique path.

After Coder runs, the main agent must call MCP `validate_change_proposal`.
If validation fails, retry Coder once with the validation error. If it still
fails, stop; do not generate a replacement proposal.

## Coder Tiers

```text
Explore   -> scale=macro -> structural candidate
Stabilize -> scale=meso  -> local rewrite or correctness/perf repair
Tune      -> scale=micro -> small parameter/launch tweak
```

Macro candidates are allowed in early exploration, but only as isolated
candidates. They must not overwrite the current best directly.

## Required ChangeProposal Fields

Coder must output exactly one `ChangeProposal` object per requested round:

```json
{
  "proposal_id": "unique_id",
  "target": "NUM_SPLITS",
  "change_type": "param_tune",
  "scale": "micro",
  "phase": "tune",
  "one_line_summary": "Tune split count",
  "before": "static constexpr int NUM_SPLITS = 12;",
  "after": "static constexpr int NUM_SPLITS = 8;",
  "hypothesis": "Reduce reduction overhead",
  "risk": "low",
  "validation_scope": "single_case",
  "rollback_plan": "discard unless compile, correctness, and A/B pass"
}
```

For `macro`, `patched_source` is allowed, but it must still be an isolated
candidate passed through `run_closed_loop`.

## Agent Roles

Analyst:
- Calls `get_operator_spec`, `resolve_best_kernel`, `roofline_analyze`,
  `latest_run`, and `query_memory`.
- Emits a non-empty bottleneck report and 1 to 3 bounded directions.
- Does not write code.

Coder:
- Reads the resolved baseline source.
- Writes one `ChangeProposal` artifact.
- Does not run benchmark and does not decide KEEP.

Profiler:
- Calls `validate_change_proposal` and `run_closed_loop`.
- Runs real mode only on the target GPU machine.
- Reports `run_dir`, `decision.json`, `summary.md`, and best-version metadata.

Judge:
- Reads deterministic artifacts only: `decision.json`, log JSONL, and manifests.
- Decides from evidence, not from chat claims.

Reflector:
- Records failure patterns and transferable hardware beliefs into run-local
  memory/logs.

Logger:
- Passive event/log role. It does not optimize code.

## KEEP Rule

A candidate is promoted only when:

1. proposal applies to the current baseline
2. compile succeeds
3. correctness passes
4. benchmark uses median timing
5. `candidate_ms < baseline_ms * (1 - noise_margin)`
6. `runs/<run_id>/versions/manifest_v2.json` records the promoted version
7. `results/current_best.json` is updated with the promoted source

Dry-run `SKIP` is never a performance KEEP.
