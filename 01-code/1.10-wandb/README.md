# 1.10 - Weights & Biases

W&B 是机器学习实验管理平台，用于记录训练指标、超参、模型 artifact 和数据集版本。

## 核心功能

- **实验跟踪**：自动记录 loss、accuracy、lr 等指标曲线
- **超参搜索（Sweeps）**：贝叶斯 / 随机 / Grid 搜索
- **Artifact**：版本化管理模型和数据集
- **报告（Reports）**：可分享的可视化分析文档

## 基本集成

```python
import wandb

wandb.init(
    project="llm-post-training",
    name="sft-qwen2.5-7b-lora",
    config={"lr": 2e-4, "epochs": 3, "r": 16},
)

# 在训练循环中记录
wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]})

wandb.finish()
```

## 与 Trainer 集成

```python
from transformers import TrainingArguments

args = TrainingArguments(
    report_to="wandb",           # 自动集成
    run_name="sft-experiment-1",
    ...
)
```

## 后训练实验建议记录的指标

- `train/loss`、`eval/loss`
- `eval/reward_margin`（奖励建模）
- `ppo/reward_mean`、`ppo/kl_div`（PPO 训练）
- `eval/win_rate`（DPO 评估）
- 样本生成示例（用 `wandb.Table` 记录）
