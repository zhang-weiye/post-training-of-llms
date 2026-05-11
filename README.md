# Post-Training of LLMs

本课程系统介绍大型语言模型（LLM）的后训练技术，涵盖从工具库使用到前沿论文、从动手练习到完整项目的全链路学习路径。

## 目录结构

```
post-training-of-llms/
├── 01-code/        # 核心工具库学习（PyTorch、HuggingFace 生态、DeepSpeed 等）
├── 02-paper/       # 经典与前沿论文导读（SFT、RL、Safety）
├── 03-exercise/    # 数据集、评测基准与小型项目练习
└── final-project/  # 完整 LLM 后训练流水线实现
```

## 学习路线

1. **工具基础**（01-code）：熟悉 PyTorch 训练范式、HuggingFace 全家桶、分布式训练框架
2. **理论理解**（02-paper）：精读 SFT / RLHF / DPO / 安全对齐核心论文
3. **动手练习**（03-exercise）：在真实数据集和基准上验证理解
4. **综合项目**（final-project）：端到端实现 SFT → 奖励建模 → RLHF/DPO → 安全过滤

## 技术栈

- Python 3.10+
- PyTorch 2.x
- Transformers / Datasets / PEFT / TRL / Accelerate / DeepSpeed
- Weights & Biases（实验跟踪）
