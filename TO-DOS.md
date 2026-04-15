# TO-DOS

> 📖 项目介绍见 [README.md](README.md)

## 任务分配

| 成员 | 任务 | 核心文件 |
|------|------|----------|
| **孙欣雨** | MCTS实现 | `source/agents/mcts/` (待创建) |
| **苏俊儒** | PPO改进 (GAE + clip loss + 熵bonus) | `source/agents/ppo/`, `config/agent-trainer/ppo-trainer-config.yaml` |
| **郑菁菁** | 奖励函数设计 | `source/env/env.py` |
| **马雨辰** | 神经网络架构 (CNN修复 + Transformer 2D编码) | `source/nn/policy_value/`, `config/nn/*.yaml` |
| **孔韫知** | A2C baseline + 实验整合 | `source/agents/actor_critic/`, `notebooks/` |

---

## 关键说明

### 路径配置
所有路径从 `config/paths_config.yaml` 读取，不要硬编码。

### 运行命令
```bash
# 训练
python run.py -an actor_critic -nn fc_policy_value

# 游戏演示（指定实验目录）
python play.py --experiment checkpoints-and-logs/local/A2C_2026_04_14-21_00 --n-games 5
```

### Git 规则
- ✅ 提交: 代码、配置文件、figures/
- ❌ 忽略: `checkpoints-and-logs/` (整个目录)
