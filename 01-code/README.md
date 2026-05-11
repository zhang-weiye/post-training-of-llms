# 01 - 工具库

本模块覆盖 LLM 后训练所需的全部核心工具库，按依赖层级由底向上排列。

## 子模块

| 目录 | 库 | 核心用途 |
|------|-----|---------|
| 1.1-pytorch | PyTorch | 张量计算、自动微分、训练循环 |
| 1.2-datasets | HuggingFace Datasets | 数据加载、流式处理、格式转换 |
| 1.3-transformers | HuggingFace Transformers | 模型加载、推理、Trainer API |
| 1.4-tokenizers | HuggingFace Tokenizers | 快速分词、自定义词表 |
| 1.5-peft | PEFT | LoRA / QLoRA / Prefix Tuning |
| 1.6-bitsandbytes | bitsandbytes | 4-bit / 8-bit 量化 |
| 1.7-trl | TRL | SFT / PPO / DPO / GRPO Trainer |
| 1.8-accelerate | Accelerate | 多 GPU / 混合精度训练 |
| 1.9-deepspeed | DeepSpeed | ZeRO 优化、千亿参数训练 |
| 1.10-wandb | Weights & Biases | 实验跟踪、超参搜索 |

## 学习建议

- 按编号顺序学习，后面的库依赖前面的概念
- 每个子模块包含示例代码，建议边读边跑
- 重点掌握 1.3、1.5、1.7，这是后训练的核心三件套
