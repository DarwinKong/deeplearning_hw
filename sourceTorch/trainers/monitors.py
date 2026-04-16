"""
训练监控模块 - 可插拔的指标收集器

设计原则：
1. 每个监控器独立负责一类指标
2. 支持动态添加/移除监控器
3. 统一的接口规范
"""
import torch
import time
from typing import Dict, Any


class BaseMonitor:
    """监控器基类"""
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.metrics = {}
    
    def on_train_begin(self):
        """训练开始时调用"""
        pass
    
    def on_epoch_begin(self, epoch: int):
        """每轮迭代开始时调用"""
        pass
    
    def on_backward(self, network: torch.nn.Module):
        """反向传播后调用（用于梯度监控）"""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """每轮迭代结束时调用"""
        pass
    
    def on_train_end(self):
        """训练结束时调用"""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return self.metrics.copy()


class GradientMonitor(BaseMonitor):
    """
    梯度监控器 - 记录每层参数的梯度范数
    
    Attributes:
        record_every: 每隔多少轮记录一次
        gradient_norms: 存储梯度范数历史
    """
    
    def __init__(self, record_every: int = 10):
        super().__init__(name="gradient")
        self.record_every = record_every
        self.gradient_history = []
    
    def on_backward(self, network: torch.nn.Module):
        """计算并记录梯度范数（批量处理后转CPU）"""
        gradient_norms = {}
        
        # 先在 GPU 上累积所有梯度范数
        grad_norms_gpu = {}
        for name, param in network.named_parameters():
            if param.grad is not None:
                grad_norms_gpu[name] = param.grad.norm(2)  # 保持为 tensor
        
        # 一次性将所有梯度范数转到 CPU
        if grad_norms_gpu:
            # 批量转换，减少通信次数
            for name, norm_tensor in grad_norms_gpu.items():
                gradient_norms[name] = norm_tensor.item()
            
            # 计算统计信息
            norms_values = list(gradient_norms.values())
            stats = {
                'grad_norm_mean': sum(norms_values) / len(norms_values),
                'grad_norm_max': max(norms_values),
                'grad_norm_min': min(norms_values),
                'gradient_norms': gradient_norms  # 详细数据
            }
            self.metrics.update(stats)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """定期保存梯度历史"""
        if epoch % self.record_every == 0 and 'gradient_norms' in self.metrics:
            self.gradient_history.append({
                'epoch': epoch,
                **self.metrics
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """返回梯度监控摘要"""
        return {
            'gradient_monitor': {
                'total_records': len(self.gradient_history),
                'latest': self.metrics.get('grad_norm_mean', 0),
                'history': self.gradient_history[-5:] if self.gradient_history else []  # 最近5次
            }
        }


class LossMonitor(BaseMonitor):
    """
    损失监控器 - 记录各类损失值
    
    Attributes:
        loss_history: 损失历史
    """
    
    def __init__(self):
        super().__init__(name="loss")
        self.loss_history = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """记录损失指标"""
        loss_data = {
            'epoch': epoch,
            'total_loss': metrics.get('total_loss', 0),
            'actor_loss': metrics.get('actor_loss', 0),
            'critic_loss': metrics.get('critic_loss', 0),
        }
        self.loss_history.append(loss_data)
        self.metrics.update(loss_data)
    
    def get_summary(self) -> Dict[str, Any]:
        """返回损失监控摘要"""
        return {
            'loss_monitor': {
                'total_epochs': len(self.loss_history),
                'latest_total_loss': self.metrics.get('total_loss', 0),
                'trend': self._compute_trend()
            }
        }
    
    def _compute_trend(self) -> str:
        """计算损失趋势"""
        if len(self.loss_history) < 2:
            return "insufficient_data"
        
        recent = [h['total_loss'] for h in self.loss_history[-5:]]
        if all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
            return "decreasing"
        elif all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
            return "increasing"
        else:
            return "fluctuating"


class EntropyMonitor(BaseMonitor):
    """
    熵监控器 - 跟踪策略熵变化
    
    熵的意义：
    - 高熵：策略随机性强，探索充分
    - 低熵：策略确定性强，趋于收敛
    """
    
    def __init__(self):
        super().__init__(name="entropy")
        self.entropy_history = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """记录熵值"""
        entropy = metrics.get('entropy', 0)
        entropy_data = {
            'epoch': epoch,
            'entropy': entropy
        }
        self.entropy_history.append(entropy_data)
        self.metrics['entropy'] = entropy
    
    def get_summary(self) -> Dict[str, Any]:
        """返回熵监控摘要"""
        if not self.entropy_history:
            return {'entropy_monitor': 'no_data'}
        
        entropies = [h['entropy'] for h in self.entropy_history]
        return {
            'entropy_monitor': {
                'current_entropy': self.metrics.get('entropy', 0),
                'mean_entropy': sum(entropies) / len(entropies),
                'min_entropy': min(entropies),
                'max_entropy': max(entropies),
                'convergence_indicator': 'converging' if entropies[-1] < entropies[0] * 0.8 else 'exploring'
            }
        }


class RewardMonitor(BaseMonitor):
    """
    奖励监控器 - 跟踪评估奖励
    
    Attributes:
        best_reward: 最佳奖励
        best_pegs_left: 最佳剩余棋子数
    """
    
    def __init__(self):
        super().__init__(name="reward")
        self.best_reward = -float('inf')
        self.best_pegs_left = float('inf')
        self.reward_history = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """记录奖励并更新最优值"""
        reward = metrics.get('eval_mean_reward', 0)
        pegs_left = metrics.get('eval_mean_pegs_left', 0)
        
        reward_data = {
            'epoch': epoch,
            'reward': reward,
            'pegs_left': pegs_left
        }
        self.reward_history.append(reward_data)
        
        # 更新最优值
        is_better = (reward > self.best_reward) or \
                   (reward == self.best_reward and pegs_left < self.best_pegs_left)
        
        if is_better:
            self.best_reward = reward
            self.best_pegs_left = pegs_left
            self.metrics['is_best'] = True
        else:
            self.metrics['is_best'] = False
        
        self.metrics.update({
            'current_reward': reward,
            'current_pegs_left': pegs_left,
            'best_reward': self.best_reward,
            'best_pegs_left': self.best_pegs_left
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """返回奖励监控摘要"""
        return {
            'reward_monitor': {
                'best_reward': self.best_reward,
                'best_pegs_left': self.best_pegs_left,
                'current_reward': self.metrics.get('current_reward', 0),
                'improvement': self.best_reward > 0
            }
        }


class MonitorManager:
    """
    监控管理器 - 统一管理所有监控器
    
    Usage:
        manager = MonitorManager(log_dir='path/to/logs')
        manager.add_monitor(GradientMonitor(record_every=10))
        manager.add_monitor(LossMonitor())
        manager.add_monitor(EntropyMonitor())
        manager.add_monitor(RewardMonitor())
        
        # 在训练循环中调用
        manager.on_train_begin()
        for epoch in range(n_epochs):
            manager.on_epoch_begin(epoch)
            # ... 训练逻辑 ...
            manager.on_backward(network)
            manager.on_epoch_end(epoch, metrics)
        manager.on_train_end()
    """
    
    def __init__(self, log_dir: str = None):
        self.monitors: Dict[str, BaseMonitor] = {}
        self.log_dir = log_dir
        self.epoch_history = []  # 存储每轮的完整指标
    
    def add_monitor(self, monitor: BaseMonitor):
        """添加监控器"""
        self.monitors[monitor.name] = monitor
    
    def remove_monitor(self, name: str):
        """移除监控器"""
        if name in self.monitors:
            del self.monitors[name]
    
    def on_train_begin(self):
        """通知所有监控器"""
        for monitor in self.monitors.values():
            monitor.on_train_begin()
    
    def on_epoch_begin(self, epoch: int):
        for monitor in self.monitors.values():
            monitor.on_epoch_begin(epoch)
    
    def on_backward(self, network: torch.nn.Module):
        for monitor in self.monitors.values():
            monitor.on_backward(network)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """每轮迭代结束时调用，并实时保存日志"""
        # 存储到历史
        self.epoch_history.append({
            'epoch': epoch,
            **metrics
        })
        
        # 通知所有监控器
        for monitor in self.monitors.values():
            monitor.on_epoch_end(epoch, metrics)
        
        # 实时保存到 CSV（每个 epoch 都保存）
        if self.log_dir and len(self.epoch_history) > 0:
            self._save_history_csv()
    
    def on_train_end(self):
        for monitor in self.monitors.values():
            monitor.on_train_end()
    
    def _save_history_csv(self):
        """实时保存训练历史到 CSV"""
        if not self.epoch_history:
            return
        
        import pandas as pd
        import os
        
        # 转换为 DataFrame
        df = pd.DataFrame(self.epoch_history)
        
        # 保存到 CSV
        csv_path = os.path.join(self.log_dir, 'training_history_full.csv')
        df.to_csv(csv_path, index=False)
    
    def get_full_summary(self) -> Dict[str, Any]:
        """获取所有监控器的摘要"""
        summary = {}
        for name, monitor in self.monitors.items():
            summary.update(monitor.get_summary())
        return summary
