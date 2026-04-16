# SourceTorch v2.0 - 纯 PyTorch 强化学习实现

## 🎯 设计理念

**算法侧同学只需关注损失函数，工程优化已全部完成！**

```
┌─────────────────────────────────────────────┐
│         算法侧（你的工作）                    │
│  - A2C/PPO 损失函数                          │
│  - 奖励函数设计                              │
│  - 超参数调优                                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│       工程侧（已完成，无需关心）               │
│  ✓ 批量 GPU 环境（64并行，5x 加速）           │
│  ✓ 零 CPU-GPU 数据传输                       │
│  ✓ 预计算查找表                              │
│  ✓ 向量化操作                                │
│  ✓ 内存优化                                  │
│  ✓ GPU 监控                                  │
└─────────────────────────────────────────────┘
```

## 📁 目录结构

```
sourceTorch/
├── agent/               # ← 算法侧关注
│   ├── base_algorithm.py    # 算法基类接口
│   ├── a2c.py              # A2C 实现
│   └── ppo.py              # PPO 实现
│
├── trainers/            # ← 工程侧已优化
│   └── batched_gpu_trainer.py  # 批量GPU训练器
│
├── env/                 # ← 工程侧已优化
│   └── batched_gpu_env.py    # 批量GPU环境（64并行）
│
├── nn/                  # ← 工程侧已优化
│   └── policy_value/    # Policy-Value 网络
│
└── utils/               # ← 工程侧已优化
    └── gpu_training_monitor.py  # GPU 监控
```

## 🚀 快速开始

### 1. 使用 A2C 算法（批量 GPU 训练）

```python
from sourceTorch import A2CAlgorithm, BatchedGPUTrainer
from sourceTorch.nn.policy_value.fully_connected import FCPolicyValueNet
from sourceTorch.nn.network_config import NetConfig
import torch

# 创建网络
config = NetConfig(
    input_shape=(7, 7, 3),
    n_actions=132,
    fc_hidden_dims=[256, 128]
)
network = FCPolicyValueNet(config)
network.to('cuda')  # GPU 加速

# 创建算法（配置超参数即可）
algorithm = A2CAlgorithm(
    network=network,
    actor_loss_weight=1.0,
    critic_loss_weight=0.5,
    entropy_weight=0.01
)

# 创建训练器（自动使用64个并行环境）
trainer = BatchedGPUTrainer(
    algorithm=algorithm,
    n_iter=200,
    n_steps_per_env=32,  # 每环境32步，总共 64*32=2048 条轨迹/轮
    agent_results_filepath="results/a2c.pt",
    learning_rate=3e-5,
    batch_size=256
)

# 开始训练
trainer.train()
```

**性能实测**：
- ✅ 100轮耗时: 25秒 (4.00 it/s)
- ✅ 14000轮估算: **约1小时**

### 2. 使用 PPO 算法

```python
from sourceTorch import PPOAlgorithm, BatchedGPUTrainer

algorithm = PPOAlgorithm(
    network=network,
    clip_epsilon=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01
)

trainer = BatchedGPUTrainer(
    algorithm=algorithm,
    n_iter=200,
    n_steps_per_env=32,
    agent_results_filepath="results/ppo.pt",
    learning_rate=3e-5,
    batch_size=256,
    n_optim_steps=4  # PPO 需要多次优化
)

trainer.train()
```

## 🔧 算法侧如何修改

### 修改损失函数权重

```python
# A2C
algorithm = A2CAlgorithm(
    network=network,
    actor_loss_weight=1.0,    # 调整这个
    critic_loss_weight=0.5,   # 调整这个
    entropy_weight=0.01       # 调整这个
)
```

### 自定义优势函数处理

```python
from sourceTorch.algorithms import A2CAlgorithm

class MyA2C(A2CAlgorithm):
    def compute_loss(self, states, actions, action_masks, 
                     advantages, value_targets, **kwargs):
        # 修改优势函数（例如 clip）
        clipped_adv = torch.clamp(advantages, -10, 10)
        
        return super().compute_loss(
            states=states,
            actions=actions,
            action_masks=action_masks,
            advantages=clipped_adv,  # 使用修改后的
            value_targets=value_targets,
            **kwargs
        )
```

### 完全自定义算法

只需继承 `BaseAlgorithm` 并实现 `compute_loss()`：

```python
from sourceTorch.algorithms import BaseAlgorithm

class MyCustomAlgorithm(BaseAlgorithm):
    def compute_loss(self, states, actions, action_masks, 
                     advantages, value_targets, **kwargs):
        # 你的损失函数逻辑
        logits, values = self.network(states)
        
        # ... 计算 loss ...
        
        return {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss
        }
```

## ⚡ 性能对比

| 版本 | 速度 (it/s) | 相对性能 |
|------|------------|----------|
| 原始 (NumPy + Lightning) | 0.40 | 1x |
| 纯 PyTorch (v1) | 1.71 | 4.3x |
| **批量GPU环境 (v2)** | **4.00** | **10x** ✅ |

**实测数据**：100轮耗时25秒，14000轮预计约1小时

## 🤝 协作者指南

### 算法侧同学
- ✅ 修改 `agent/` 下的损失函数
- ✅ 调整超参数
- ✅ 设计新的奖励函数
- ❌ 不需要关心训练循环、GPU 管理、数据加载

### 工程侧同学
- ✅ 优化 `env/batched_gpu_env.py` 向量化
- ✅ 改进 `trainers/batched_gpu_trainer.py` 训练效率
- ✅ 增强 `utils/` 监控功能
- ❌ 不修改算法核心逻辑

## 📊 监控和日志

训练器自动记录：
- GPU 利用率
- 训练速度 (it/s)
- 奖励曲线
- 检查点保存

日志位置：`checkpoints-and-logs/local/<experiment_name>/`

## 🎓 学习资源

- `algorithms/base_algorithm.py` - 理解算法接口
- `algorithms/a2c.py` - A2C 实现参考
- `algorithms/ppo.py` - PPO 实现参考
- `trainers/algorithm_trainer.py` - 训练循环实现

---

**版本**: 2.0.0  
**作者**: Grand-Final Team  
**许可**: MIT
