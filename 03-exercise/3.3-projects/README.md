# 3.3 - 小项目

在进入 final-project 之前，通过这些小型项目验证每个模块的理解。

## 项目列表

### 项目 A：中文指令微调

**目标**：在 Qwen2.5-1.5B 上用 Alpaca-Chinese 做 SFT，验证中文指令遵循效果。

**关键步骤**：
1. 加载数据集并转换为 chat 格式
2. 配置 QLoRA（NF4 + r=16）
3. 用 `SFTTrainer` 训练 3 epoch
4. 主观评测：对比训练前后的输出质量

**预期训练时间**：单张 RTX 3090 约 1 小时

---

### 项目 B：DPO vs SFT 对比实验

**目标**：在同一基础模型上分别跑 SFT 和 DPO，用 MT-Bench 量化比较。

**数据**：`Anthropic/hh-rlhf`（helpful 子集）

**评测**：MT-Bench 总分 + 逐维度分析

---

### 项目 C：奖励模型训练与评估

**目标**：用 `RewardTrainer` 训练一个二元偏好奖励模型。

**指标**：chosen > rejected 的准确率（reward accuracy）

**进阶**：可视化奖励分数分布，分析模型偏好的模式

---

## 提交格式建议

每个项目建议包含：
- `train.py`：训练脚本
- `eval.py`：评测脚本
- `config.yaml`：超参配置
- `results/`：评测结果和日志
