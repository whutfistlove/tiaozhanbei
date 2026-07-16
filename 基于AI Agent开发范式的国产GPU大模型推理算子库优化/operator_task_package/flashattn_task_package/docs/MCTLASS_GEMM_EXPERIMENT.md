# mctlass GEMM 实验报告

> 实验日期：2026-07-16 | 硬件：MetaX C500 (MXC500) | 工具链：MACA 3.7.1.5

## 一、mctlass GEMM 是什么

mctlass = **M**etaX **C**UTLASS，是沐曦移植到 MACA 平台的 CUTLASS 2.x 矩阵计算库。
它让 BF16 矩阵乘走 C500 的 **Tensor Core**（硬件 MMA 单元），而非 CUDA Core 的标量 ALU。

- 底层指令：`__builtin_mxc_mma_16x16x16bf16`（一条指令算 16×16×16 bf16 矩阵乘加）
- C++ 入口：`mctlass::gemm::device::MacaGemmUniversal<...>`（设备级 GEMM 模板）
- 头文件位置：`/opt/maca/include/mctlass/`（本机实测存在完整原语链）

## 二、算子源码现状（为什么需要 mctlass）

仓库 `kernel/` 三个 `.cu` **全部是纯标量 CUDA 循环，没有一个用 TensorCore**：

| 文件 | Q@K^T 实现 | 带宽利用率 | 说明 |
|------|-----------|----------|------|
| baseline_kernel.cu | `for(d) dot += q[d]*k[d]` 标量 | — | 仅支持 headdim≤32 |
| splitk_h128.cu | 同上标量循环 + Split-K | **1.2%** | 当前最优，d=128 正确 |
| splitk_v1.cu | 标量循环变体 | — | 正确性未通过 |

标量循环用 CUDA Core 算矩阵乘，C500 的 280 TFLOPS BF16 算力（全在 Tensor Core）几乎没用上。

## 三、TensorCore 可用性实证（本机真实测量）

### 方阵 bf16 GEMM（torch.matmul，走 TensorCore）

| M=N=K | time_ms | TFLOPS | 峰值(280T)占比 |
|-------|---------|--------|--------------|
| 1024 | 0.022 | 96.3T | 34.4% |
| 2048 | 0.131 | 131.5T | 46.9% |
| 4096 | 0.826 | 166.4T | 59.4% |
| 8192 | 5.099 | **215.7T** | **77.0%** |

**结论：TensorCore 完全可用，大方阵 GEMM 达峰值算力的 77%。**

### decode attention 对比（b=1, d=128）

| seq_kv | 手写 splitk_h128 | 官方 flash_attn(TC) | 加速比 |
|--------|----------------|-------------------|--------|
| 1024 | 0.247ms (17GB/s) | 0.148ms (28GB/s) | 1.66x |
| 4096 | 0.805ms (21GB/s) | 0.492ms (34GB/s) | 1.64x |
| 16384 | 2.863ms (23GB/s) | 1.866ms (36GB/s) | 1.53x |

**官方 flash_attn（TensorCore 路径）比手写标量快 1.5~1.66x。** 即使在 b=1 paged 场景（带宽利用率都偏低），TensorCore 仍有明显优势。

## 四、mctlass C++ 模板调通情况（未完成）

### 已确认的真实 API（来自头文件实证，与 skill 文档有出入）

1. `MacaGemmUniversal` 完整模板参数顺序：
   `ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages, AlignmentA, AlignmentB, ...`

2. **`LinearCombination` 第 2 参数是 `int Count`（非类型）**，正确写法：
   `LinearCombination<bfloat16_t, 128/sizeof_bits<bfloat16_t>::value, float, float>`
   （skill 文档示例 `LinearCombination<float, bfloat16_t, float, bfloat16_t>` 是错的）

3. bf16 MMA 真实类型是 **`maca_bfloat16`**（不是 `bfloat16_t`）：
   `arch::MacaMma<maca_bfloat16, RowMajor, maca_bfloat16, ColumnMajor, float, ColumnMajor, Sm80>`
   底层 `__builtin_mxc_mma_16x16x16bf16`

4. C500 用 `__MACA_ARCH__ == 1000`，但 arch tag 仍是 `Sm80`（mma_sm80.h 在 1000 下生效）

### 卡点（待解决）

最小 GEMM 模板实例化到 Epilogue 层时报：
- `no member named 'kMacaEpilogueReduceTag' in Epilogue`
- `reference to __host__ variable 'kAccumVecSize' in __device__ function`

这是 `maca_default_gemm_universal.h` 选用的默认 Epilogue 类型与当前 mctlass 版本不兼容，
需显式指定完整的 Epilogue threadblock 类型（非用 default）。这是多日工程任务。

## 五、结论与后续可行路径

### 已证明
- ✅ C500 TensorCore 完全可用（bf16 GEMM 达峰值 77%）
- ✅ TensorCore 路径在 decode attention 上比标量循环快 1.5~1.66x
- ✅ 当前手写算子（1.2% 带宽）的优化空间巨大

### 为什么 mctlass GEMM 难调通（赛题核心痛点）
- mctlass 是魔改 CUTLASS 2.x，默认参数链多处断裂
- 连人工按 skill 文档写都踩坑（文档本身有误，如 LinearCombination 签名）
- 这正是"通用 LLM 在国产硬件正确率≈0%"的实证，也是赛题"用领域记忆 + Agent 迭代"的价值

### 后续路径（按可行性排序）
1. **显式指定 Epilogue 类型**：不用 default，手动给出 `DefaultEpilogueComplex` 等完整类型链
2. **用 frontend_op 高层封装**：`mctlass_moe_gemm.h` 等已处理好默认参数（但接口面向 MoE）
3. **退而求其次**：先用 `arch::MacaMma` 底层 MMA 指令手写一个 warp 级 GEMM（绕过 device 模板）
4. **Triton/TileLang 路径**：更高层抽象，LLM 生成成功率高（赛题一推荐）
