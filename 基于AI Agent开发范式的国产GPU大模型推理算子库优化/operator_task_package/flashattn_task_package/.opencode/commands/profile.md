---
description: Profiler 角色：编译 kernel 并采集 roofline 分析（$1=文件名）
---
请作为 Profiler 角色分析 kernel/$1.cu 的性能特征：

1. 先读取 kernel 源码，静态分析：
   - grid/block 配置
   - shared memory 用量
   - 访存模式（是否合并、是否向量化）

2. 用 roofline 分析理论极限：
!`python -c "
from agent_system.roofline_engine import KernelConfig, analyze, suggest_split_k
for b in [1,4,16]:
  for s in [1024,4096,8192,16384]:
    cfg = KernelConfig(batch_size=b, seqlen_kv=s, headdim=128, num_heads=8)
    r = analyze(cfg)
    print(f'b={b},seq={s}: {r.bound_type}, T_lower={r.t_lower_bound_s*1e3:.4f}ms, split={suggest_split_k(cfg)}')
" 2>&1`

3. 输出性能分析报告：
   - 当前 kernel 在各配置下的理论瓶颈
   - 哪些配置优化空间最大
   - 具体的优化建议（向量化/tile/流水线）
