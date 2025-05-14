# 这是DUT2025年春lxc老师《计算机组成原理》课程的大作业
## 项目概述

LeNet 是 Yann LeCun 提出的早期卷积神经网络模型，广泛应用于手写数字识别等任务。  
本项目基于华为鲲鹏 920 通用处理器平台搭建 LeNet-5 神经网络，实现对手写数字图像的高效识别，并在保持 97.61% 准确率的前提下，通过多种优化手段显著提升推理性能，包括：

- **GCC 编译优化**  
  使用 `-O3`、`-ffast-math`、`-funroll-loops` 等编译选项，启用编译器级优化，提升指令执行效率和流水线利用率，借助编译器自动展开循环和向量化能力获得基础性能提升。
- **循环展开（Unroll）优化**  
  对卷积操作中的循环手动展开，去除内层控制判断，提升计算密度与 CPU 指令执行效率，降低分支预测开销。
- **NEON 向量化优化**  
  利用 ARMv8 架构提供的 NEON SIMD 指令集，将输入和权重数据打包为 4 元向量，使用 `vfmaq_f32` 等融合指令进行 4 路并行乘加操作，显著提升每周期完成的运算量。
- **手写 NEON 汇编微核**  
  进一步对 NEON 优化进行底层控制，通过汇编语言编写固定结构的 5×5 卷积微核（如 `kern25_fma4`），最大限度压榨指令并行度和寄存器使用率，进一步缩短卷积核心操作耗时。
- **GEMM 转换与优化**  
  使用 `im2col` 方法将卷积操作转化为标准的矩阵乘法（SGEMM），通过手写的 `sgemm4x4` 向量化矩阵乘法内核，利用数据块划分与 NEON 并行乘加，进一步提升整体推理吞吐率。

整体优化效果显著，将基线 C 版本约 49 秒的总推理时间压缩至 1.1 秒以内，验证了在通用 ARM 处理器平台上，通过指令级优化和内核级手工加速手段即可获得数量级的性能提升。

## 项目结构

项目目录结构如下：
```
lenet_kunpeng920/
├── lenet_baseline/ # 基础版本的 LeNet 实现
├── lenet_gemm/ # 使用 GEMM 优化的版本
├── lenet_neon/ # 使用 NEON 向量化优化的版本
├── lenet_unroll/ # 使用循环展开优化的版本
└── README.md # 项目说明文件
```

## 编译与运行

1. **克隆项目**：

   ```bash
   git clone https://github.com/YiShangxuan-DUT/lenet_kunpeng920.git
   cd lenet_kunpeng920
   ```
2. **编译并运行不同优化版本**
   
   编译运行 baseline（纯 C 实现）与gcc指令优化
   ```bash
   cd lenet_baseline
   gcc -O0 -g -march=armv8-a -fno-tree-vectorize -fno-unroll-loops -fno-inline -DPERF -Iinclude src/*.c -lm -o lenet_baseline
   gcc -O3 -march=armv8.2-a+simd -ffast-math -funroll-loops -DPERF -Iinclude src/*.c -lm -o lenet_fp32_O3
   ./lenet_base
   ./lenet_fp32_O3
   ```

   编译 unroll（循环展开优化）
   ```
   cd lenet_unroll
   gcc -O3 -ffast-math -funroll-loops -march=armv8-a+simd -DPERF -Iinclude src/*.c -lm -o lenet_unroll
   ./lenet_unroll
   ```
   
   编译NEON 向量化优化和NEON 汇编微核优化
   ```
   cd lenet_neon
   gcc -O3 -ffast-math -funroll-loops -march=armv8.2-a+simd  -DPERF -DUSE_NEON -Iinclude src/*.c -lm -o lenet_neon #neon 向量化优化
   gcc -O3 -ffast-math -funroll-loops -march=armv8.2-a+simd -DPERF -DUSE_NEON -DUSE_NEON_ASM -Iinclude src/*.c src/conv5x5_neon_asm.S -lm -o lenet_neon_asm #neon 汇编微核优化
   ./lenet_neon
   ./lenet_asm
   ```

   编译 gemm（im2col + sgemm 优化）
   ```
   cd lenet_gemm
   gcc -O3 -ffast-math -funroll-loops -march=armv8.2-a+simd -DPERF -DUSE_NEON -Iinclude src/*.c -lm -o lenet_gemm
   ./lenet_gemm
   ```

## 性能评估
各优化版本在鲲鹏 920 平台上的性能对比如下：

| Version         | Total Time(ms, 10 k imgs) | Speed-upvs Baseline | Images / s   | Images / s提升倍数 | Latency(ms / img) | GFLOPS     | 理论峰值利用率* |
| --------------- | ------------------------- | ------------------- | ------------ | ------------------ | ----------------- | ---------- | --------------- |
| Baseline (-O0)  | 49 312.94                 | 1.00×               | 202.8        | 1.00×              | 4.931             | 2.36       | 1.8 %           |
| Compiler -O3    | 6 050.79                  | **8.15×**           | 1 652.7      | **8.15×**          | 0.605             | 19.20      | 14.5 %          |
| Loop Unroll     | 2 314.46                  | **21.3×**           | 4 320.7      | **21.3×**          | 0.231             | 50.21      | 38.0 %          |
| NEON Intrinsics | 1 318.02                  | **37.4×**           | 7 587.1      | **37.4×**          | 0.132             | 88.16      | 66.8 %          |
| NEON ASM        | 1 101.97                  | **44.7×**           | 9 074.6      | **44.7×**          | 0.110             | 105.45     | 79.8 %          |
| im2col + SGEMM  | **936.71**                | **52.7×**           | **10 875.6** | **53.6×**          | **0.094**         | **124.05** | **94.0 %**      |

## 参考资料
- **Yann LeCun 等人提出的 LeNet 论文**
- **ARM NEON 技术文档**
- **华为鲲鹏920相关文档**
