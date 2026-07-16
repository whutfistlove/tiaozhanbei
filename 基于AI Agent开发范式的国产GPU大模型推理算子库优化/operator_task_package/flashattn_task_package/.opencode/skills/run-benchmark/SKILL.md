---
name: run-benchmark
description: Use when benchmarking a kernel or comparing performance against baseline. Trigger keywords: benchmark, bench, timing, bandwidth, gap_to_roofline, speedup, compare, 性能测试, 计时. 用 benchmark_engine 跑 GPU 计时并分析性能。
---

# Benchmark 执行流程

默认入口：

```bash
python scripts/run_closed_loop.py --real --rounds 1 --batch 1 --seq-kv 4096 --headdim 128
```

该脚本会自动执行 compile -> correctness -> benchmark -> A/B decision，并将结果写入 `runs/<run_id>/logs/` 和 `rounds/round_###/decision.json`。只有需要调试底层步骤时，才使用下面的手动流程。

## 标准流程（Profiler 角色）
```python
from agent_system.kernel_compiler import compile_source
from agent_system.kernel_loader import load_kernel, call_run_kernel, make_output_tensor
from agent_system.benchmark_engine import benchmark_config
from agent_system.correctness import make_test_inputs, generate_reference, check
from agent_system.roofline_engine import KernelConfig

# 1. 编译
result = compile_source(code, '/tmp/kernel.so')
assert result.success

# 2. 加载+运行+校验
cfg = KernelConfig(batch_size=1, seqlen_kv=4096, headdim=128, num_heads=8)
lres = load_kernel('/tmp/kernel.so')
q, k, v, lens, bt = make_test_inputs(cfg, device='cuda')
output = make_output_tensor(cfg, 'cuda')
out = call_run_kernel(lres.run_kernel_fn, q, k, v, output, lens, bt, cfg, k.shape[0])
ref = generate_reference(q, k, v, lens, bt, cfg)
assert check(out, ref, rtol=1e-2, atol=1e-2).passed  # 先过正确性

# 3. benchmark（CUDA event 精确计时）
def run():
    call_run_kernel(lres.run_kernel_fn, q, k, v, make_output_tensor(cfg,'cuda'), lens, bt, cfg, k.shape[0])
bench = benchmark_config(run, cfg, warmup=10, repeats=100)
print(bench.summary())
# 输出: batch=1 seq_kv=4096 | 0.79ms | bw=21.1 GB/s (1.2%) | gap=68x [memory-bound]
```

## 对比官方 flash_attn baseline
```python
from agent_system.benchmark_engine import benchmark_official_flash_attn
official = benchmark_official_flash_attn(cfg)
speedup = official.time_ms / bench.time_ms
```

## 关键指标解读
- **time_ms**：kernel 平均耗时（越低越好）
- **achievable_bw_gb_s**：有效带宽 = bytes / time
- **bandwidth_utilization**：有效带宽 / 1800(C500峰值)，0~1
- **gap_to_roofline**：实测/理论下限，1.0=已达物理极限，越大优化空间越大
