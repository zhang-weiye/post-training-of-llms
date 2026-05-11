# 3.2 - 评测基准

评测是判断后训练效果的关键，本节介绍主流基准的使用方法和结果解读。

## 通用能力基准

| 基准 | 评测内容 | 工具 |
|------|---------|------|
| MMLU | 57 学科知识问答 | `lm-evaluation-harness` |
| HellaSwag | 常识推理 | `lm-evaluation-harness` |
| ARC-Challenge | 科学题 | `lm-evaluation-harness` |
| TruthfulQA | 诚实性 | `lm-evaluation-harness` |
| MT-Bench | 多轮对话质量（GPT-4 打分）| `fastchat` |

## 使用 lm-evaluation-harness

```bash
pip install lm-eval

lm_eval \
  --model hf \
  --model_args pretrained=./sft-output,dtype=bfloat16 \
  --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results
```

## MT-Bench（对话质量）

```bash
# 生成模型回答
python gen_model_answer.py --model-path ./sft-output --model-id my-model

# GPT-4 评分
python gen_judgment.py --model-list my-model --judge-model gpt-4

# 查看分数
python show_result.py --model-list my-model
```

## 练习任务

- [ ] 对基础模型跑 MMLU，记录 baseline 分数
- [ ] SFT 后重新评测，观察分数变化（有时会下降！）
- [ ] 生成 MT-Bench 回答，主观评估对话质量
- [ ] 对比 LoRA rank=8 vs rank=64 的效果差异
