"""
PPO (Proximal Policy Optimization) 算法实现

算法侧同学可以修改：
1. Clip范围（clip_epsilon）
2. Value loss系数（value_loss_coef）
3. Entropy系数（entropy_coef）
4. GAE参数（如果使用GAE）

工程优化已完成：
- 向量化计算
- GPU加速
- 批量处理
- Old policy缓存
"""
import torch
import torch.nn.functional as F
from typing import Dict
from .base_algorithm import BaseAlgorithm


class PPOAlgorithm(BaseAlgorithm):
    """
    PPO算法实现（带Clip）
    
    示例用法：
        algorithm = PPOAlgorithm(
            network=policy_value_net,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01
        )
    """
    
    def __init__(self,
                 network: torch.nn.Module,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 **kwargs):
        """
        Args:
            network: Policy-Value网络
            clip_epsilon: PPO clip范围
            value_loss_coef: Value loss系数
            entropy_coef: Entropy bonus系数
            max_grad_norm: 梯度裁剪范数
        """
        super().__init__(network, name="PPO", **kwargs)
        
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def compute_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     action_masks: torch.Tensor,
                     advantages: torch.Tensor,
                     value_targets: torch.Tensor,
                     old_logits: torch.Tensor = None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算PPO损失函数
        
        PPO-Clip Loss:
            L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
            其中 r_t = π_new(a|s) / π_old(a|s)
        
        Args:
            states: (batch_size, 7, 7, 3)
            actions: (batch_size,)
            action_masks: (batch_size, 132)
            advantages: (batch_size, 1)
            value_targets: (batch_size, 1)
            old_logits: (batch_size, 132) - 旧策略的logits
            
        Returns:
            dict包含各项损失
        """
        if old_logits is None:
            raise ValueError("PPO requires old_logits for importance sampling")
        
        # ==================== 前向传播 ====================
        logits, values = self.get_logits_and_values(states)
        
        # ==================== 计算重要性采样比率 ====================
        # 应用动作掩码
        masked_logits = logits + (1 - action_masks) * (-1e9)
        masked_old_logits = old_logits + (1 - action_masks) * (-1e9)
        
        # 计算log概率
        log_probs = F.log_softmax(masked_logits, dim=-1)
        old_log_probs = F.log_softmax(masked_old_logits, dim=-1)
        
        # 选择采取的动作的log概率
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        old_action_log_probs = old_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # 重要性采样比率: r_t = π_new / π_old
        ratio = torch.exp(action_log_probs - old_action_log_probs.detach())
        
        # ==================== PPO Clip Loss ====================
        # 优势函数（已计算好，传入即可）
        adv = advantages.squeeze(-1)
        
        # 未clip的损失
        surr1 = ratio * adv
        
        # Clip后的损失
        ratio_clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surr2 = ratio_clipped * adv
        
        # 取最小值（保守更新）
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # ==================== Value Loss ====================
        # Clipped value loss（可选，提升稳定性）
        values_pred = values.squeeze(-1)
        
        # 如果使用了value clipping
        if 'old_values' in kwargs:
            old_values = kwargs['old_values'].squeeze(-1)
            values_clipped = old_values + torch.clamp(
                values_pred - old_values, 
                -self.clip_epsilon, 
                self.clip_epsilon
            )
            value_loss_unclipped = F.mse_loss(values_pred, value_targets.squeeze(-1), reduction='none')
            value_loss_clipped = F.mse_loss(values_clipped, value_targets.squeeze(-1), reduction='none')
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = F.mse_loss(values_pred, value_targets.squeeze(-1))
        
        # ==================== Entropy Bonus ====================
        policy_probs = F.softmax(masked_logits, dim=-1)
        log_policy_probs = F.log_softmax(masked_logits, dim=-1)
        entropy = -(policy_probs * log_policy_probs).sum(dim=-1).mean()
        
        # ==================== Total Loss ====================
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'clip_ratio': ((ratio > (1 + self.clip_epsilon)) | (ratio < (1 - self.clip_epsilon))).float().mean(),
            'approx_kl': ((old_action_log_probs - action_log_probs).detach()).mean(),
            'mean_advantage': adv.mean(),
            'std_advantage': adv.std()
        }
