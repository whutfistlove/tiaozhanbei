---
description: 编译并 benchmark 指定 kernel（$1=文件名，默认 splitk_h128）
---
对 kernel/$1.cu 跑完整 benchmark 流程：

1. 用 mxcc 编译：
!`MACA_PATH=/opt/maca && $MACA_PATH/mxgpu_llvm/bin/mxcc -std=c++17 -fPIC -shared -DMACA_ARCH=1000 -I$MACA_PATH/include kernel/$1.cu -o /tmp/$1.so 2>&1 | head -5`

2. 用 python 跑 benchmark（正确性+性能）：
!`python -c "
from agent_system.kernel_loader import load_kernel, call_run_kernel, make_output_tensor
from agent_system.benchmark_engine import benchmark_config
from agent_system.correctness import make_test_inputs, generate_reference, check
from agent_system.roofline_engine import KernelConfig
cfg = KernelConfig(batch_size=1, seqlen_kv=4096, headdim=128, num_heads=8)
l = load_kernel('/tmp/$1.so')
if not l.success: print('加载失败:', l.error_msg); exit()
q,k,v,ln,bt = make_test_inputs(cfg, device='cuda', seed=42)
o = call_run_kernel(l.run_kernel_fn,q,k,v,make_output_tensor(cfg,'cuda'),ln,bt,cfg,k.shape[0])
r = generate_reference(q,k,v,ln,bt,cfg)
c = check(o,r)
print('正确性:', 'OK' if c.passed else 'FAIL', c.detail[:60])
def run(): call_run_kernel(l.run_kernel_fn,q,k,v,make_output_tensor(cfg,'cuda'),ln,bt,cfg,k.shape[0])
b = benchmark_config(run, cfg, warmup=5, repeats=30)
print(b.summary())
" 2>&1 | tail -5`

分析结果：带宽利用率多少？gap_to_roofline 多大？离官方 flash_attn 多远？
