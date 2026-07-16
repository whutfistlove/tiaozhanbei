---
description: Suggest the next bounded ChangeProposal from latest run logs and memory.
---

Act as Analyst + Reflector. Use the latest deterministic run artifacts:

1. Locate latest run:
!`cat results/latest_run.txt 2>/dev/null || echo "no latest run"`

2. Read latest summary:
!`LATEST_RUN="$(cat results/latest_run.txt 2>/dev/null || true)"; if [ -n "$LATEST_RUN" ]; then cat "$LATEST_RUN/logs/summary.md" 2>/dev/null | tail -40; fi`

3. Read recent decisions:
!`LATEST_RUN="$(cat results/latest_run.txt 2>/dev/null || true)"; if [ -n "$LATEST_RUN" ]; then find "$LATEST_RUN/rounds" -name decision.json -maxdepth 2 -type f -print -exec cat {} \\; 2>/dev/null | tail -80; fi`

4. Read run-local memory:
!`LATEST_RUN="$(cat results/latest_run.txt 2>/dev/null || true)"; if [ -n "$LATEST_RUN" ]; then cat "$LATEST_RUN/memory/hardware_belief.json" 2>/dev/null | tail -40; cat "$LATEST_RUN/memory/failure_cases/cases.json" 2>/dev/null | tail -40; fi`

Then output one next-step recommendation:
- bottleneck
- evidence
- one `ChangeProposal` target
- risk
- reason to avoid any large rewrite
