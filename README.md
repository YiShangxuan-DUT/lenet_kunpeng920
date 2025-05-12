# 这是DUT2025年春lxc老师计组课程的大作业
## 项目概述

LeNet 是 Yann LeCun 提出的早期卷积神经网络模型，广泛应用于手写数字识别等任务。  
本项目在鲲鹏 920 平台上实现 LeNet，并通过多种优化策略提升其性能，包括：

- **GEMM 优化**：将卷积操作转化为矩阵乘法，利用高效的矩阵运算提升计算速度。
- **NEON 向量化**：利用 ARM NEON 指令集进行数据并行处理，加速计算过程。
- **循环展开（Unroll）**：通过展开循环减少分支，提高指令流水线效率。

## 项目结构

项目目录结构如下：
lenet_kunpeng920/
├── lenet_baseline/ # 基础版本的 LeNet 实现
├── lenet_gemm/ # 使用 GEMM 优化的版本
├── lenet_neon/ # 使用 NEON 向量化优化的版本
├── lenet_unroll/ # 使用循环展开优化的版本
└── README.md # 项目说明文件


## 环境要求

- **硬件**：华为鲲鹏 920 服务器或其他支持 ARMv8 架构的设备
- **操作系统**：Ubuntu 20.04 或兼容的 Linux 发行版
- **编译器**：支持 ARM 架构的 GCC（建议版本 ≥ 9.3）
- **依赖库**：标准 C 库，无需额外依赖

## 编译与运行

1. **克隆项目**：

   ```bash
   git clone https://github.com/YiShangxuan-DUT/lenet_kunpeng920.git
   cd lenet_kunpeng920
   ```
2. **编译代码**
   ```bash
   git clone https://github.com/YiShangxuan-DUT/lenet_kunpeng920.git
   cd lenet_kunpeng920
   ```
3. **运行程序**
   ```bash
   git clone https://github.com/YiShangxuan-DUT/lenet_kunpeng920.git
   cd lenet_kunpeng920
   ```

## 性能评估
各优化版本在鲲鹏 920 平台上的性能对比如下：

| 版本           | 运行时间（ms） | 加速比（相对于 Baseline） |
|----------------|----------------|----------------------------|
| Baseline       | XXX            | 1.00x                      |
| GEMM 优化      | XXX            | X.XXx                      |
| NEON 向量化    | XXX            | X.XXx                      |
| 循环展开优化   | XXX            | X.XXx                      |

## 参考资料
- **Yann LeCun 等人提出的 LeNet 论文**
- **ARM NEON 技术文档**
- **相关的高性能计算优化资料**
