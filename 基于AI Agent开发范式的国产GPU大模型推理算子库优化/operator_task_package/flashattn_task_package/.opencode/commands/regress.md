---
description: 回归正确性测试：对 kernel/$1.cu 跑多配置 allclose 校验（$1=文件名，默认 splitk_h128）
---
请作为 Judge 角色对 kernel/$1.cu 做完整正确性回归：

1. 编译：
!`MACA_PATH=/opt/maca && $MACA_PATH/mxgpu_llvm/bin/mxcc -std=c++17 -fPIC -shared -DMACA_ARCH=1000 -I$MACA_PATH/include kernel/$1.cu -o /tmp/$1.so 2>&1 | head -3`

2. 多配置正确性校验（OJ 全部配置）：
!`python -c "
import torch
from agent_system.kernel_loader import load_kernel, call_run_kernel, make_output_tensor
from agent_system.correctness import make_test_inputs, generate_reference, check
from agent_system.roofline_engine import KernelConfig
l = load_kernel('/tmp/$1.so')
if not l.success: print('加载失败'); exit()
print('配置           正确性  max_abs')
for b in [1,4,16]:
  for s in [1024,4096,8192,16384]:
    cfg = KernelConfig(batch_size=b, seqlen_kv=s, headdim=128, num_heads=8)
    try:
      q,k,v,ln,bt = make_test_inputs(cfg, device='cuda', seed=42)
      o = call_run_kernel(l.run_kernel_fn,q,k,v,make_output_tensor(cfg,'cuda'),ln,bt,cfg,k.shape[0])
      r = generate_reference(q,k,v,ln,bt,cfg)
      c = check(o,r)
      print(f'b={b},seq={s:<6} {\"OK\" if c.passed else \"FAIL\"}  {c.max_abs_diff:.4f}')
    except Exception as e:
      print(f'b={b},seq={s:<6} ERROR  {str(e)[:40]}')
" 2>&1 | tail -15`

3. 汇总通过率，失败的配置分析原因。
