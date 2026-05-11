# 2.1 - SFT 论文

监督微调（Supervised Fine-Tuning）是后训练的第一步，将预训练模型转变为能遵循指令的对话助手。

## 核心论文

| 论文 | 机构 | 年份 | 贡献 |
|------|------|------|------|
| Training language models to follow instructions with human feedback | OpenAI | 2022 | InstructGPT，RLHF 完整流程首次公开 |
| Self-Instruct | Stanford | 2022 | 用 LLM 自动生成指令数据 |
| Stanford Alpaca | Stanford | 2023 | 52K GPT-4 指令数据 + LLaMA SFT |
| Vicuna | UC Berkeley | 2023 | ShareGPT 多轮对话数据的重要性 |
| LIMA | Meta | 2023 | 1000 条精选数据媲美大规模 SFT（Less is More）|
| WizardLM | Microsoft | 2023 | Evol-Instruct：指令复杂度进化 |
| Orca 2 | Microsoft | 2023 | 推理过程监督，教模型"如何思考" |

## 关键发现

- **数据质量 > 数据数量**：LIMA 证明 1K 精选数据可超过 50K 随机数据
- **多样性很重要**：任务类型、风格、长度的多样性显著影响泛化
- **格式一致性**：训练时的 prompt 格式需与推理时完全一致

## SFT 数据格式

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "解释一下什么是量子纠缠"},
    {"role": "assistant", "content": "量子纠缠是..."}
  ]
}
```
