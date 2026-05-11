# 1.7 - TRL（Transformer Reinforcement Learning）

TRL 是 HuggingFace 出品的后训练专用库，封装了 SFT、奖励建模、PPO、DPO、GRPO 等全套 Trainer。

## 支持的训练方法

| Trainer | 用途 |
|---------|------|
| `SFTTrainer` | 监督微调，自动处理 chat template 和 label masking |
| `RewardTrainer` | 训练奖励模型（偏好对比学习）|
| `PPOTrainer` | 基于 PPO 的 RLHF 训练 |
| `DPOTrainer` | Direct Preference Optimization（无需奖励模型）|
| `GRPOTrainer` | Group Relative Policy Optimization（DeepSeek-R1 同款）|
| `ORPOTrainer` | Odds Ratio Preference Optimization |

## SFTTrainer 示例

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./sft-output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        max_seq_length=2048,
    ),
    train_dataset=ds,
    peft_config=lora_config,
    processing_class=tokenizer,
)
trainer.train()
```

## DPO 示例

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # 参考模型（冻结）
    args=DPOConfig(beta=0.1, ...),
    train_dataset=ds,     # 需含 prompt / chosen / rejected 字段
    processing_class=tokenizer,
)
```

## 学习重点

- [ ] 理解 `SFTTrainer` 如何自动构造 label mask
- [ ] 掌握 DPO loss 的推导：`log σ(β log(π_θ/π_ref) chosen - rejected)`
- [ ] 了解 PPO 与 DPO 的优劣取舍
