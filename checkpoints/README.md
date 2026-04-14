# Checkpoints 管理说明

## 目录结构

```
checkpoints/
├── actor_critic/          # A2C 模型的 checkpoints
│   ├── local/             # 本地 checkpoints（不提交到 Git）
│   │   └── *.ckpt         # 训练过程中自动保存的模型
│   └── remote/            # 远程 checkpoints（会提交到 Git）
│       └── *.ckpt         # 精选的最佳模型
└── ppo/                   # PPO 模型的 checkpoints
    ├── local/
    │   └── *.ckpt
    └── remote/
        └── *.ckpt
```

## 使用方式

### 训练时保存 Checkpoints

**默认保存到本地目录**（不会被 Git 跟踪）：
```bash
python run.py -an actor_critic -nn fc_policy_value
python run.py -an ppo -nn fc_policy_value
```

**保存到远程目录**（会被 Git 跟踪，适合分享最佳模型）：
```bash
python run.py -an actor_critic -nn fc_policy_value --remote
python run.py -an ppo -nn fc_policy_value --remote
```

### 游戏演示时加载 Checkpoints

**从本地目录加载**（默认）：
```bash
python play.py --agent actor_critic
python play.py --agent ppo
```

**从远程目录加载**（被 Git 提交的模型）：
```bash
python play.py --agent actor_critic --remote
python play.py --agent ppo --remote
```

**指定具体 checkpoint 文件**：
```bash
python play.py --agent ppo --checkpoint checkpoints/ppo/local/epoch=100_step=800.ckpt
```

## 最佳实践

1. **训练阶段**：使用默认的本地目录保存所有中间 checkpoints
2. **模型选择**：训练完成后，将表现最好的 checkpoint 复制到 remote 目录
3. **版本控制**：只将精选的最佳模型提交到 Git，避免仓库过大
4. **团队协作**：从 remote 目录拉取他人训练的最佳模型进行对比测试

## 示例：保存最佳模型到远程

```bash
# 假设训练完成后，发现 epoch=100 的模型表现最好
cp checkpoints/actor_critic/local/epoch=100_step=796.ckpt \
   checkpoints/actor_critic/remote/best_model.ckpt

# 提交到 Git
git add checkpoints/actor_critic/remote/best_model.ckpt
git commit -m "Add best A2C model (epoch=100, reward=0.806)"
git push
```

## 注意事项

- ✅ `checkpoints/*/local/` 目录已在 `.gitignore` 中忽略
- ✅ `checkpoints/*/remote/` 目录会被 Git 跟踪
- ⚠️ 不要将所有 checkpoints 都提交到 remote，会导致仓库过大
- 💡 建议在 remote 目录中使用描述性文件名，如 `best_model.ckpt`、`final_model.ckpt`
