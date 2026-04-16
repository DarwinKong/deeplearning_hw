"""
算法基类 - 定义所有RL算法的统一接口

算法侧同学只需实现：
1. compute_loss() - 损失函数计算
2. update_policy() - 策略更新逻辑（可选）

工程侧已优化：
- 数据收集（向量化env）
- 训练循环（纯PyTorch）
- GPU管理
- 日志和监控
"""
from abc import ABC, abstractmethod
import torch
from typing import Dict, Any


class BaseAgent(ABC):
    """
    强化学习算法基类
    
    算法侧同学继承此类并实现核心方法即可。
    所有工程优化（向量化、GPU加速等）已在父类和工具中完成。
    """
    
    def __init__(self, network: torch.nn.Module, name: str = "BaseAlgorithm", **kwargs):
        """
        Args:
            network: Policy-Value网络
            name: 算法名称
            **kwargs: 算法特定参数
        """
        self.network = network
        self.name = name
        self.device = next(network.parameters()).device
        self.config = kwargs
        
    @abstractmethod
    def compute_loss(self, 
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     action_masks: torch.Tensor,
                     advantages: torch.Tensor,
                     value_targets: torch.Tensor,
                     old_logits: torch.Tensor = None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算损失函数 - 算法侧核心实现点
        
        Args:
            states: 状态张量 (batch_size, 7, 7, 3)
            actions: 动作张量 (batch_size,)
            action_masks: 动作掩码 (batch_size, 132)
            advantages: 优势函数 (batch_size, 1)
            value_targets: 价值目标 (batch_size, 1)
            old_logits: 旧策略logits（PPO需要）
            
        Returns:
            dict包含:
                - 'total_loss': 总损失
                - 'actor_loss': Actor损失
                - 'critic_loss': Critic损失
                - 其他算法特定的损失项
        """
        pass
    
    def get_policy(self, states: torch.Tensor) -> torch.Tensor:
        """获取策略分布（通常不需要修改）"""
        with torch.no_grad():
            return self.network.get_policy(states.to(self.device))
    
    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        """获取状态价值（通常不需要修改）"""
        with torch.no_grad():
            return self.network.get_value(states.to(self.device))
    
    def get_logits_and_values(self, states: torch.Tensor):
        """获取logits和价值（用于训练时）"""
        return self.network(states.to(self.device))
    
    def set_training_mode(self):
        """设置为训练模式"""
        self.network.train()
    
    def set_evaluation_mode(self):
        """设置为评估模式"""
        self.network.eval()
    
    def to_device(self, device: torch.device = None):
        """移动网络到指定设备"""
        if device is None:
            device = self.device
        self.network.to(device)
        self.device = device
