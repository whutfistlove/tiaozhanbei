#!/bin/bash
# reproduce.sh —— 评审一键复现脚本
#
# 复现整个 Agent 驱动的算子优化流程：
# 1. 环境检查
# 2. 运行全部测试
# 3. 编译 baseline + 优化 kernel
# 4. 正确性校验
# 5. benchmark 对比
# 6. 输出优化日志摘要
#
# 用法：bash scripts/reproduce.sh
# 前置：MACA_PATH 已设置（默认 /opt/maca），GPU 可用

set -e

export MACA_PATH=${MACA_PATH:-/opt/maca}
export PATH=$MACA_PATH/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}
MXCC=$MACA_PATH/mxgpu_llvm/bin/mxcc
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "================================================"
echo "  FlashAttention 算子优化 Agent 系统 · 复现"
echo "================================================"
echo ""

# 1. 环境检查
echo "[1/6] 环境检查..."
echo "  MACA_PATH: $MACA_PATH"
echo "  mxcc: $($MXCC --version 2>&1 | head -1)"
python --version 2>&1
echo "  GPU: $(timeout 5 mx-smi 2>&1 | grep -o 'MetaX C500' | head -1 || echo '不可用（CPU模式可跑测试）')"
echo ""

# 2. 运行全部测试
echo "[2/6] 运行全部测试（132个）..."
python -m pytest tests/ --tb=no -q 2>&1 | tail -3
echo ""

# 3. 编译 kernel
echo "[3/6] 编译 kernel..."
for src in kernel/baseline_kernel.cu kernel/splitk_h128.cu; do
    name=$(basename "$src" .cu)
    echo "  编译 $name..."
    $MXCC -std=c++17 -fPIC -shared -DMACA_ARCH=1000 -I$MACA_PATH/include "$src" -o "/tmp/${name}.so" 2>&1 | head -3
    if [ -f "/tmp/${name}.so" ]; then
        echo "    ✅ /tmp/${name}.so"
    else
        echo "    ❌ 编译失败"
    fi
done
echo ""

# 4. 正确性校验（需GPU）
echo "[4/6] 正确性校验..."
if timeout 5 mx-smi 2>&1 | grep -q "MetaX C500"; then
    python -c "
from agent_system.kernel_loader import load_kernel, call_run_kernel, make_output_tensor
from agent_system.correctness import make_test_inputs, generate_reference, check
from agent_system.roofline_engine import KernelConfig
l = load_kernel('/tmp/splitk_h128.so')
cfg = KernelConfig(batch_size=1, seqlen_kv=512, headdim=128, num_heads=8)
q,k,v,ln,bt = make_test_inputs(cfg, device='cuda', seed=42)
o = call_run_kernel(l.run_kernel_fn,q,k,v,make_output_tensor(cfg,'cuda'),ln,bt,cfg,k.shape[0])
r = generate_reference(q,k,v,ln,bt,cfg)
c = check(o,r)
print(f'  splitk_h128 正确性: {\"✅通过\" if c.passed else \"❌失败\"} {c.detail}')
" 2>&1 | tail -2
else
    echo "  ⏭️ GPU 不可用，跳过正确性校验"
fi
echo ""

# 5. Roofline 分析（免GPU）
echo "[5/6] Roofline 分析（免GPU）..."
python -c "
from agent_system.roofline_engine import KernelConfig, analyze, suggest_split_k
print('  配置              bound_type      理论下限(ms)  建议split')
for b in [1,4,16]:
  for s in [1024,4096,8192,16384]:
    cfg = KernelConfig(batch_size=b, seqlen_kv=s, headdim=128, num_heads=8)
    r = analyze(cfg)
    print(f'  b={b},seq={s:<6}  {r.bound_type:<15} {r.t_lower_bound_s*1e3:<13.4f} {suggest_split_k(cfg)}')
" 2>&1
echo ""

# 6. 优化日志
echo "[6/6] 优化日志摘要..."
LATEST_RUN="$(cat results/latest_run.txt 2>/dev/null || true)"
if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/logs/summary.md" ]; then
  head -30 "$LATEST_RUN/logs/summary.md"
else
  echo "  （暂无 runs/<run_id>/logs/summary.md；可先运行 python scripts/run_closed_loop.py --rounds 1）"
fi
echo ""

echo "================================================"
echo "  复现完成。Agent 会话日志见 docs/agent_logs/"
echo "  完整方案见 AGENT_DESIGN_PLAN.md"
echo "================================================"
