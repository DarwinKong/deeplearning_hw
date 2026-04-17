# RL Solitaire - 强化学习孔明棋求解器

使用强化学习（A2C、PPO）解决孔明棋游戏。

<p align="center">
<img src="figures/solitaire_1.gif" width="400" height="400" />
</p>

---

## 🚀 快速开始

### 环境准备

```bash
conda create -n ai python=3.10
conda activate ai
pip install -r requirements.txt
```

### 训练

```bash
# A2C 算法
python run.py -an actor_critic -nn fc_policy_value

# PPO 算法
python run.py -an ppo -nn fc_policy_value
```

训练数据自动保存到 `checkpoints-and-logs/local/{AGENT}_{timestamp}/`

### 游戏演示

```bash
# 指定实验目录
python play.py --experiment checkpoints-and-logs/local/A2C_2026_04_14-21_00 --n-games 5

# 使用最新 checkpoint
python play.py --agent ppo --n-games 3
```

---

## 📁 项目结构

```
├── config/                      # 配置文件
│   ├── paths_config.yaml        # 路径配置
│   └── agent-trainer/           # Agent 配置
│
├── source/                      # 核心代码（原始版本）
│   ├── agents/                  # 强化学习智能体
│   │   ├── actor_critic/        # A2C 算法
│   │   └── ppo/                 # PPO 算法
│   ├── env/                     # 游戏环境
│   ├── nn/                      # 神经网络
│   └── utils/                   # 工具函数
│
├── sourceTorch/                 # 优化版本（纯 PyTorch，10x 加速）
│   ├── agent/                   # 算法模块
│   ├── trainers/                # 训练器
│   ├── env/                     # 批量 GPU 环境
│   └── nn/                      # 神经网络
│
├── documents/                   # 文档
│   ├── ALGORITHM_GUIDE.md       # 算法实现说明
│   ├── OPTIMIZATION_SUMMARY.md  # 优化对比与未来方向
│   └── TO-DOS.md                # 待办事项
│
├── checkpoints-and-logs/        # 实验数据（自动生成）
│   ├── local/                   # 本地实验
│   │   └── A2C_2026_04_14-21_00/
│   │       ├── meta/            # 配置副本
│   │       ├── logs/            # 日志 + TensorBoard
│   │       ├── checkpoints/     # 中间模型
│   │       └── results/         # 训练结果
│   └── remote/                  # 远程实验（手动推送）
│
├── figures/                     # 可视化图表
└── notebooks/                   # Jupyter 分析
```

📚 **详细文档**：
- [算法实现说明](documents/ALGORITHM_GUIDE.md) - SourceTorch 算法模块详解
- [优化对比与未来方向](documents/OPTIMIZATION_SUMMARY.md) - SourceTorch vs Source 性能对比
- [待办事项](documents/TO-DOS.md) - 项目计划

---

## 🔬 技术细节

### 状态表示

7×7×3 张量：
- 通道 1: 棋子存在性 (0/1)
- 通道 2: 全局进度信息
- 通道 3: 额外特征

### 奖励函数

- 每移除一个棋子: +1
- 游戏结束剩余 n 个棋子: -n
- 完美解（剩1个）: 额外奖励

### 神经网络架构

支持多种网络：
- **FC**: 全连接网络
- **CNN**: 卷积神经网络
- **Transformer**: Transformer + 2D 位置编码

---

## ⚡ SourceTorch - 优化版本

**SourceTorch** 是原始 `source/` 的纯 PyTorch 重构版本，实现了 **10x 性能提升**。

### 核心优势

| 特性 | source (原始) | sourceTorch (优化) |
|------|--------------|-------------------|
| 技术栈 | NumPy + PyTorch Lightning | 纯 PyTorch |
| 并行度 | 单环境 | 64 并行环境 |
| 数据传输 | CPU ↔ GPU 频繁传输 | 零 CPU-GPU 传输 |
| 速度 | 0.40 it/s | **4.00 it/s** ✅ |
| 14000轮耗时 | ~9.7小时 | **~1小时** ✅ |

### 快速开始

```bash
# 使用 SourceTorch 训练（推荐）
cd sourceTorch
python -c "
from sourceTorch import A2CAlgorithm, BatchedGPUTrainer
from sourceTorch.nn.policy_value.fully_connected import FCPolicyValueNet
from sourceTorch.nn.network_config import NetConfig

# 创建网络和算法
config = NetConfig(input_shape=(7, 7, 3), n_actions=132, fc_hidden_dims=[256, 128])
network = FCPolicyValueNet(config).to('cuda')
algorithm = A2CAlgorithm(network)

# 训练
trainer = BatchedGPUTrainer(algorithm, n_iter=200, n_steps_per_env=32)
trainer.train()
"
```

📚 **详细文档**：
- [算法实现说明](documents/ALGORITHM_GUIDE.md)
- [优化对比与未来方向](documents/OPTIMIZATION_SUMMARY.md)

---

## 👥 团队分工

详见 [TO-DOS.md](documents/TO-DOS.md)



**最后更新**: 2026-04-16
