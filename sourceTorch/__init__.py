"""
纯 PyTorch 实现 - 性能优化版本 v2.0

架构设计：
- algorithms/  : 算法实现（A2C, PPO等）- 算法侧关注
- trainers/    : 训练器（已优化）- 工程侧完成
- env/         : 环境（向量化）- 工程侧完成
- nn/          : 网络（GPU加速）- 工程侧完成
- utils/       : 工具（监控等）- 工程侧完成

算法侧同学只需关注：
1. algorithms/ 目录下的损失函数实现
2. 超参数配置
3. 奖励函数设计

所有工程优化（向量化、GPU加速、内存管理）已完成。
"""
from .agent import A2CAgent, PPOAgent, BaseAgent
from .trainers import BatchedGPUTrainer
from .env import BatchedGPUEnv

__version__ = "2.0.0"
__author__ = "Grand-Final Team"

__all__ = [
    # 算法层（算法侧关注）
    'BaseAgent',
    'A2CAgent', 
    'PPOAgent',
    
    # 训练器（工程侧已优化）
    'BatchedGPUTrainer',
    
    # 环境（工程侧已优化）
    'BatchedGPUEnv'
]
