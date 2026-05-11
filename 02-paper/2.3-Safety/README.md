# 2.3 - 安全对齐论文

安全对齐研究如何让模型拒绝有害请求、避免歧视、保持诚实，同时尽量不损害模型的有用性。

## 核心论文

| 论文 | 机构 | 年份 | 贡献 |
|------|------|------|------|
| Constitutional AI | Anthropic | 2022 | 用 AI 批评与修订实现无害性 |
| Llama 2 | Meta | 2023 | 安全 SFT + RLHF 的工业级实践 |
| Llama Guard | Meta | 2023 | 基于 LLM 的输入/输出安全分类器 |
| Beaver | PKU-Alignment | 2023 | 安全约束的 RLHF（RLHF + Safe RL）|
| WildGuard | Allenai | 2024 | 多类型有害内容的统一分类 |
| Representation Engineering | UT Austin | 2023 | 通过激活向量直接控制模型行为 |

## 对齐税（Alignment Tax）

对齐训练可能导致模型在某些能力上退化：
- 代码生成能力下降（过度拒绝）
- 创意写作受限
- 指令遵循变得机械

**缓解方法**：混合安全数据与通用有用性数据；使用 RLHF 而非纯拒绝式 SFT。

## 攻防概念

- **越狱（Jailbreak）**：绕过安全护栏的对抗性 prompt
- **红队（Red Teaming）**：主动测试模型安全边界
- **对齐假象**：模型表面遵守规则但内部表示不一致

## 安全评估基准

- **ToxiGen**：有毒内容生成评估
- **TruthfulQA**：诚实性评估
- **BBQ**：社会偏见评估
- **HarmBench**：越狱攻击成功率
