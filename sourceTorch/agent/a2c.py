"""
A2C (Advantage Actor-Critic) 算法实现

算法侧同学可以修改：
1. 损失函数权重（actor_loss_weight, critic_loss_weight）
2. 优势函数归一化
3. 熵正则化系数

工程优化已完成：
- 向量化计算
- GPU加速
- 批量处理
"""
import torch
import torch.nn.functional as F
from typing import Dict
from .base_agent import BaseAgent


class A2CAgent(BaseAgent):
    """
    A2C算法实现
    
    示例用法：
        algorithm = A2CAlgorithm(
            network=policy_value_net,
            actor_loss_weight=1.0,
            critic_loss_weight=0.5,
            entropy_weight=0.01
        )
    """
    
    def __init__(self,
                 network: torch.nn.Module,
                 actor_loss_weight: float = 1.0,
                 critic_loss_weight: float = 0.5,
                 entropy_weight: float = 0.01,
                 normalize_advantages: bool = True,
                 **kwargs):
        """
        Args:
            network: Policy-Value网络
            actor_loss_weight: Actor损失权重
            critic_loss_weight: Critic损失权重
            entropy_weight: 熵正则化权重（鼓励探索）
            normalize_advantages: 是否归一化优势函数
        """
        super().__init__(network, name="A2C", **kwargs)
        
        self.actor_loss_weight = actor_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.entropy_weight = entropy_weight
        self.normalize_advantages = normalize_advantages
    
    def compute_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     action_masks: torch.Tensor,
                     advantages: torch.Tensor,
                     value_targets: torch.Tensor,
                     old_logits: torch.Tensor = None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算A2C损失函数
        
        Loss = -log_prob(action) * advantage + critic_loss - entropy * entropy_weight
        
        Args:
            states: (batch_size, 7, 7, 3)
            actions: (batch_size,)
            action_masks: (batch_size, 132)
            advantages: (batch_size, 1)
            value_targets: (batch_size, 1)
            
        Returns:
            dict包含各项损失
        """
        # 前向传播
        logits, values = self.get_logits_and_values(states)
        
        # ==================== Actor Loss ====================
        # 应用动作掩码
        masked_logits = logits + (1 - action_masks) * (-1e9)
        
        # 计算策略分布
        policy = F.softmax(masked_logits, dim=-1)
        log_policy = F.log_softmax(masked_logits, dim=-1)
        
        # 选择采取的动作的log概率
        action_log_probs = log_policy.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # 优势函数归一化（可选，通常能提升稳定性）
        if self.normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            normalized_advantages = (advantages - adv_mean) / adv_std
        else:
            normalized_advantages = advantages
        
        # Actor损失：-log_prob * advantage
        actor_loss = -(action_log_probs * normalized_advantages.squeeze(-1)).mean()
        
        # ==================== Critic Loss ====================
        # MSE损失
        critic_loss = F.mse_loss(values.squeeze(-1), value_targets.squeeze(-1))
        
        # ==================== Entropy Bonus ====================
        # 熵 = -sum(p * log(p))，鼓励探索
        entropy = -(policy * log_policy).sum(dim=-1).mean()
        
        # ==================== Total Loss ====================
        total_loss = (
            self.actor_loss_weight * actor_loss +
            self.critic_loss_weight * critic_loss -
            self.entropy_weight * entropy
        )
        
        return {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
            'mean_advantage': normalized_advantages.mean(),
            'std_advantage': normalized_advantages.std()
        }
