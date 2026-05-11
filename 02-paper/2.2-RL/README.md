# 2.2 - 强化学习对齐论文

RL 方法利用人类偏好信号或可验证奖励进一步优化 SFT 模型，是 ChatGPT / Claude / DeepSeek 背后的关键技术。

## 核心论文

| 论文 | 机构 | 年份 | 贡献 |
|------|------|------|------|
| InstructGPT | OpenAI | 2022 | RLHF = RM + PPO 的完整范式 |
| Constitutional AI | Anthropic | 2022 | 用 AI 反馈替代部分人类标注（RLAIF）|
| DPO | Stanford | 2023 | 直接偏好优化，绕过奖励模型 |
| RLHF-V | 清华 | 2023 | 细粒度视觉 RLHF |
| ORPO | KAIST | 2024 | 单阶段：在 SFT loss 中融入偏好 |
| SimPO | Princeton | 2024 | 无参考模型，用生成长度归一化奖励 |
| DeepSeek-R1 | DeepSeek | 2025 | GRPO + 规则奖励激发长链推理 |
| RLVR | 多机构 | 2025 | 可验证奖励（代码/数学）的 RL 训练 |

## 方法对比

```
RLHF (PPO)
  优点：灵活，奖励信号强
  缺点：训练复杂，需要 4 个模型（actor/critic/reward/ref）

DPO
  优点：稳定，只需 2 个模型（policy/ref）
  缺点：对 chosen/rejected 对质量敏感

GRPO（DeepSeek-R1）
  优点：无需 critic 模型，组内相对奖励更稳定
  缺点：需要可验证的奖励函数
```

## PPO 训练中的关键超参

- `kl_coef`：KL 惩罚系数，防止策略偏离参考模型过远
- `cliprange`：PPO clip 范围（通常 0.2）
- `reward_baseline`：奖励基线，减小方差
