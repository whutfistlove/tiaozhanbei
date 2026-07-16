---
name: mctlass-usage
description: Use when writing or modifying CUDA Maca kernel code that calls mctlass (MetaX CUTLASS) APIs. Trigger keywords: mctlass, MacaGemmUniversal, MacaMma, EpilogueVisitorSoftmax, NumericConverter, MacaConvertAndPack, InstructionShape, split-k. 沐曦 MACA 平台矩阵计算库的正确用法（防止 API 误用，占失败案例 51.9%）。
---

# mctlass 正确用法（从本地头文件实证）

> 来源：`$MACA_PATH/include/mctlass/` 头文件逐字确认。
> mctlass 是 **CUTLASS 2.x 移植**（v2.10.0），**不是 3.x**，无 collective 抽象。

## ⚠️ 最易错的 5 个陷阱（务必规避）

### 1. bf16 InstructionShape 必须是 16x16x16
```cpp
// ❌ 错误：bf16 用 16x16x8 会数值错位
InstructionShape<16, 8, 16>  // 这是逻辑指令，底层映射 16x16x16
// ✅ 正确：C500 (Sm80) bf16 路径
arch::MacaMma<bfloat16_t, ...>  // 内部用 __builtin_mxc_mma_16x16x16bf16
```

### 2. bf16 fragment 必须经 MacaConvertAndPack 重排
```cpp
// f32 累加器灌回 bf16 时，必须带 index 重排，否则数值错位
// idx = (((i<<1)&2) | ((i>>1)&1) | (i&0xfffffffc))
warp::MacaConvertAndPack<bfloat16_t, float, N, Round>  // 必须用这个，不能直接 NumericArrayConverter
```

### 3. softmax 必须在 fp32
```cpp
EpilogueVisitorSoftmax<..., ElementSoftmaxCompute=float, ElementNorm=float, ElementSum=float>
// P 可写 bf16 给下一段，但 softmax 计算本身必须 f32
```

### 4. 128-bit 对齐
- bf16：至少 8 元素（16 字节）
- f32：至少 4 元素
- 否则 cache op 退化为 CacheOperation::Always（性能下降）

### 5. 运行时用 mc_runtime_api，不是 cuda_runtime
```cpp
#include "mctlass/mctlass.h"  // 内含 mc_runtime_api.h
mcMalloc / mcFree / mcStreamSynchronize / mcFuncSetAttribute
// 类型：mcStream_t（不是 cudaStream_t）
```

## FlashAttention decode 的 mctlass 原语链

```
Q@K^T 段：
  arch::MacaMma<bfloat16_t, RowMajor, bfloat16_t, ColumnMajor, float, RowMajor, Sm80>
    → 16x16x16 bf16 MMA，FragmentC = __NATIVE_VECTOR__(4, float)
  warp::MacaMmaTensorOp（一个 warp 的 MmaIterations 条指令）
    → MacaConvertAndPack 重排 bf16
  threadblock::DefaultMmaSoftmaxMainloopFusion（scale=1/√D 融入 mainloop）
  epilogue::EpilogueVisitorSoftmax<UseMasking=true>
    → 一个 pass 内：row max + scale + exp + 输出 P(bf16) + row_max + row_sum

P@V 段：
  device::MacaGemmUniversal（标准 GEMM，ElementAccumulator=float，输出 bf16）

Split-K：
  threadblock::MacaMmaSplitKParallel + reduction::ReduceSplitK
```

## 设备级 GEMM 启动模板

```cpp
using Gemm = mctlass::gemm::device::MacaGemmUniversal<
    bfloat16_t, mctlass::layout::RowMajor,    // A (Q)
    bfloat16_t, mctlass::layout::ColumnMajor, // B (K^T)
    bfloat16_t, mctlass::layout::RowMajor,    // C/D
    float,                                     // ElementAccumulator
    mctlass::arch::OpClassTensorOp,
    mctlass::arch::Sm80,                       // C500
    mctlass::gemm::GemmShape<128, 256, 64>,    // ThreadblockShape
    mctlass::gemm::GemmShape<64, 64, 64>,      // WarpShape
    mctlass::gemm::GemmShape<16, 8, 16>,       // InstructionShape
    mctlass::epilogue::thread::LinearCombination<float, bfloat16_t, float, bfloat16_t>,
    mctlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3, 8, 8>;                                  // Stages, AlignA, AlignB

// Arguments
Gemm::Arguments args{
    mctlass::gemm::GemmUniversalMode::kBatched,
    {M, N, K}, batch_count, epilogue_params,
    A, B, C, D, 0,0,0,0, lda, ldb, ldc, ldd
};
Gemm gemm;
gemm.initialize(args, workspace, stream);
gemm(stream);
```

## 编译命令
```bash
mxcc -std=c++17 -DMACA_ARCH=1000 \
     -I$MACA_PATH/include \
     kernel.cu -o kernel.so -shared
```

## 类型对照
| 概念 | mctlass | CUTLASS 对应 |
|------|---------|-------------|
| 命名空间 | `mctlass::` | `cutlass::` |
| bf16 | `bfloat16_t` | 同 |
| GEMM 设备 | `MacaGemmUniversal` | `GemmUniversal` |
| MMA 原语 | `arch::MacaMma` | `arch::Mma` |
| 运行时 | `mc_runtime_api.h` | `cuda_runtime.h` |
| Stream | `mcStream_t` | `cudaStream_t` |
