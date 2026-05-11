# 1.9 - DeepSpeed

DeepSpeed 是微软开源的分布式训练框架，其 ZeRO 优化系列可将百亿参数模型的训练成本降低数倍。

## ZeRO 优化阶段

| 阶段 | 分片内容 | 显存节省 |
|------|---------|---------|
| ZeRO-1 | 优化器状态 | ~4x |
| ZeRO-2 | 优化器状态 + 梯度 | ~8x |
| ZeRO-3 | 优化器状态 + 梯度 + 模型参数 | ~64x+ |
| ZeRO-Infinity | ZeRO-3 + NVMe offload | 近乎无限 |

## 配置文件示例（ZeRO-2）

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu" },
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true
  },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

## 与 Transformers 集成

```bash
# 在 TrainingArguments 中指定 deepspeed 配置
python train.py \
  --deepspeed ds_config_zero2.json \
  --num_train_epochs 3
```

## 学习重点

- [ ] 理解 ZeRO-1/2/3 各自分片的内容和通信开销
- [ ] 掌握 CPU offload 的使用场景（显存不足时）
- [ ] 了解 ZeRO-3 与模型并行的区别
