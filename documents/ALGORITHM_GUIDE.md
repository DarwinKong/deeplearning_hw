# RL 原理与算法实现

## 🎯 问题建模

孔明棋求解被建模为**马尔可夫决策过程（MDP）**：

```
状态 s_t → 策略 π(a|s_t) → 动作 a_t → 环境 → 奖励 r_t + 新状态 s_{t+1}
                    ↓
              价值 V(s_t)
```

### 核心组件

| 模块 | 职责 | 文件位置 |
|------|------|----------|
| **Env** | 棋盘状态管理、动作执行、奖励计算 | `sourceTorch/env/batched_gpu_env.py` |
| **Agent** | 数据收集、回报计算、优势估计 | `sourceTorch/agent/*.py` |
| **NN** | 策略和价值网络、损失计算、参数更新 | `sourceTorch/nn/policy_value/` |

---

## 📊 状态与动作

### 状态表示（7×7×3 张量）

- **通道 1**: 棋子存在性（0/1）
- **通道 2**: 剩余棋子比例 `(n_pegs - 1) / 31`
- **通道 3**: 已移除棋子比例 `(32 - n_pegs) / 31`

### 动作空间

- **132 种可能动作**: 33 个位置 × 4 个方向
- **Action Mask**: 过滤非法动作（只能跳过相邻棋子到空位）

---

## 💰 奖励函数设计

```python
# 每步奖励：成功移除一个棋子
reward = 1 / 31  # ≈ 0.032

# 完美解：只剩1个棋子
if n_pegs == 1:
    return 1.0, state, done  # 总奖励归一化到 [0, 1]

# 失败：无合法动作但棋子 > 1
return removed_ratio, state, done  # 给予部分奖励
```

**设计优势**：
- ✅ **稠密奖励**：每步都有正反馈，避免稀疏奖励问题
- ✅ **归一化**：不同长度游戏可比，训练稳定
- ✅ **鼓励探索**：即使未完美解决也能获得进度奖励

---

## 🔧 A2C 算法

### 数据收集流程

```python
# 1. 与环境交互收集轨迹
for step in range(n_steps):
    action = policy.sample()  # 基于 π(a|s) 采样
    reward, next_state, done = env.step(action)
    
# 2. 逆向计算折扣回报
U_t = R_t + γ * U_{t+1}  # 从终止状态反向传播

# 3. 计算优势函数
A(s_t, a_t) = U_t - V(s_t)  # 实际回报 - 价值估计
```

### 损失函数

```
L_total = L_actor + c1 * L_critic - c2 * H(π)
```

**策略损失**（最大化期望优势）：
```
L_actor = -E[log π(a|s) * A(s,a)]
```

**价值损失**（最小化均方误差）：
```
L_critic = MSE(V(s), U)
```

**熵正则化**（鼓励探索）：
```
H(π) = -E[π(a|s) * log π(a|s)]
```

### 使用示例

```python
from sourceTorch import A2CAlgorithm, BatchedGPUTrainer

algorithm = A2CAlgorithm(
    network=network,
    actor_loss_weight=1.0,
    critic_loss_weight=0.5,
    entropy_weight=0.01,
    normalize_advantages=True  # 稳定训练
)

trainer = BatchedGPUTrainer(
    algorithm=algorithm,
    n_iter=200,
    n_steps_per_env=32,
    learning_rate=3e-5
)
trainer.train()
```

---

## 🚀 PPO 算法

### 核心创新：Clipped Surrogate Objective

防止策略更新过大，提高训练稳定性：

```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  # 重要性采样比率

L_clip = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```

其中 `ε = 0.2` 是裁剪范围。

### 使用示例

```python
from sourceTorch import PPOAlgorithm, BatchedGPUTrainer

algorithm = PPOAlgorithm(
    network=network,
    clip_epsilon=0.2,           # PPO 裁剪参数
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5           # 梯度裁剪
)

trainer = BatchedGPUTrainer(
    algorithm=algorithm,
    n_iter=200,
    n_steps_per_env=32,
    n_optim_steps=4,            # PPO 需要多次优化
    learning_rate=3e-5
)
trainer.train()
```

---

## ⚙️ 关键超参数

| 参数 | A2C | PPO | 说明 |
|------|-----|-----|------|
| `learning_rate` | 3e-5 | 3e-5 | 学习率 |
| `actor_loss_weight` | 1.0 | - | Actor 权重 |
| `critic_loss_weight` | 0.5 | 0.5 | Critic 权重 |
| `entropy_weight` | 0.01 | 0.01 | 探索系数 |
| `clip_epsilon` | - | 0.2 | PPO 裁剪范围 |
| `n_optim_steps` | 1 | 4 | 每轮优化次数 |
| `normalize_advantages` | True | True | 优势归一化 |

---

## 🛠️ 自定义算法

只需继承 `BaseAlgorithm` 并实现 `compute_loss()`：

```python
from sourceTorch.agent.base_algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def compute_loss(self, states, actions, action_masks, 
                     advantages, value_targets, **kwargs):
        # 你的损失函数逻辑
        logits, values = self.get_logits_and_values(states)
        
        # ... 自定义计算 ...
        
        return {
            'total_loss': total_loss,
            'policy_loss': ...,
            'value_loss': ...
        }
```

---

## 📈 性能对比

| 算法 | 速度 (it/s) | 收敛质量 | 适用场景 |
|------|------------|---------|---------|
| A2C | ~4.0 | 良好 | 快速原型验证 |
| PPO | ~3.5 | 优秀 | 最终模型训练 |

*基于 64 并行环境测试*
