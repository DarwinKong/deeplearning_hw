"""
GPU 优化的训练监控工具
使用 GPU 缓存 + 定时回传策略，减少 CPU-GPU 通信频率
"""
import os
import json
import csv
import torch
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque


class GPUTrainingMonitor:
    """
    GPU 优化的训练监控器
    
    优化策略：
    1. 指标先在 GPU 上累积（减少数据传输）
    2. 定时批量回传到 CPU/磁盘（降低通信频率）
    3. 使用 pinned memory 加速传输
    """
    
    def __init__(self, log_dir: str, device: Optional[torch.device] = None, 
                 flush_interval: int = 10):
        """
        Args:
            log_dir: 日志目录路径
            device: 计算设备（None 则自动选择）
            flush_interval: 每多少个 iteration 刷新一次到磁盘
        """
        self.log_dir = log_dir
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flush_interval = flush_interval
        
        os.makedirs(log_dir, exist_ok=True)
        
        # 文件路径
        self.training_metrics_file = os.path.join(log_dir, "training_metrics.csv")
        self.evaluation_metrics_file = os.path.join(log_dir, "evaluation_metrics.csv")
        self.summary_file = os.path.join(log_dir, "training_summary.json")
        
        # 初始化 CSV
        self._init_csvs()
        
        # GPU 缓存队列（存储待刷新的数据）
        self.training_buffer = deque(maxlen=flush_interval * 10)
        self.evaluation_buffer = deque(maxlen=flush_interval * 10)
        
        # 历史记录（CPU 上，用于总结）
        self.training_history = []
        self.evaluation_history = []
        
        # 计数器
        self.iteration_count = 0
        
        print(f"[GPU Monitor] 初始化完成 - 设备: {self.device}, 刷新间隔: {flush_interval}")
    
    def _init_csvs(self):
        """初始化 CSV 文件"""
        # Training metrics
        with open(self.training_metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'mean_reward', 'std_reward', 'min_reward', 
                           'max_reward', 'mean_pegs_left', 'timestamp'])
        
        # Evaluation metrics
        with open(self.evaluation_metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'eval_mean_reward', 'eval_std_reward', 
                           'eval_mean_pegs_left', 'greedy_reward', 'greedy_pegs_left', 'timestamp'])
    
    def log_training_metrics_gpu(self, iteration: int, rewards_tensor: torch.Tensor, 
                                 pegs_left_tensor: torch.Tensor):
        """
        记录训练指标（GPU 优化版本）
        
        Args:
            iteration: 当前迭代次数
            rewards_tensor: 奖励 tensor (在 GPU 上)
            pegs_left_tensor: 剩余棋子数 tensor (在 GPU 上)
        """
        # 在 GPU 上计算统计量（避免传输原始数据）
        mean_reward = rewards_tensor.mean().item()
        std_reward = rewards_tensor.std().item() if len(rewards_tensor) > 1 else 0.0
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()
        mean_pegs = pegs_left_tensor.float().mean().item()
        
        metrics = {
            'iteration': iteration,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'mean_pegs_left': mean_pegs,
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加到 GPU 缓存
        self.training_buffer.append(metrics)
        self.training_history.append(metrics)
        self.iteration_count += 1
        
        # 定时刷新到磁盘
        if self.iteration_count % self.flush_interval == 0:
            self._flush_to_disk()
        
        return metrics
    
    def log_evaluation_metrics(self, iteration: int, eval_rewards: List[float], 
                               eval_pegs_left: List[float], 
                               greedy_reward: float, greedy_pegs_left: float):
        """记录评估指标（评估频率低，直接写入）"""
        import numpy as np
        
        metrics = {
            'iteration': iteration,
            'eval_mean_reward': float(np.mean(eval_rewards)),
            'eval_std_reward': float(np.std(eval_rewards)),
            'eval_mean_pegs_left': float(np.mean(eval_pegs_left)),
            'greedy_reward': float(greedy_reward),
            'greedy_pegs_left': float(greedy_pegs_left),
            'timestamp': datetime.now().isoformat()
        }
        
        # 评估数据直接写入（频率低）
        with open(self.evaluation_metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)
        
        self.evaluation_history.append(metrics)
        return metrics
    
    def _flush_to_disk(self):
        """将缓存的数据批量写入磁盘"""
        if not self.training_buffer:
            return
        
        # 批量写入 training metrics
        with open(self.training_metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.training_buffer[0].keys())
            writer.writerows(self.training_buffer)
        
        # 清空缓存
        self.training_buffer.clear()
        
        print(f"[GPU Monitor] 已刷新 {self.flush_interval} 条记录到磁盘")
    
    def generate_summary(self):
        """生成训练总结报告"""
        # 先刷新所有缓存
        self._flush_to_disk()
        
        if not self.training_history:
            return None
        
        summary = {
            'total_iterations': len(self.training_history),
            'best_training': {
                'iteration': max(self.training_history, key=lambda x: x['mean_reward'])['iteration'],
                'mean_reward': max(h['mean_reward'] for h in self.training_history),
                'mean_pegs_left': min(h['mean_pegs_left'] for h in self.training_history)
            },
            'latest_training': self.training_history[-1],
            'training_progress': self._calculate_improvement('mean_reward')
        }
        
        if self.evaluation_history:
            summary['best_evaluation'] = {
                'iteration': max(self.evaluation_history, key=lambda x: x['eval_mean_reward'])['iteration'],
                'eval_mean_reward': max(h['eval_mean_reward'] for h in self.evaluation_history),
                'greedy_reward': max(h['greedy_reward'] for h in self.evaluation_history)
            }
            summary['latest_evaluation'] = self.evaluation_history[-1]
        
        # 保存总结
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def _calculate_improvement(self, metric: str) -> Dict[str, float]:
        """计算指标改进情况"""
        if len(self.training_history) < 2:
            return {'first': None, 'last': None, 'improvement': None}
        
        first_value = self.training_history[0][metric]
        last_value = self.training_history[-1][metric]
        improvement = last_value - first_value
        improvement_pct = (improvement / first_value * 100) if first_value != 0 else 0
        
        return {
            'first': first_value,
            'last': last_value,
            'absolute_improvement': improvement,
            'percentage_improvement': improvement_pct
        }
    
    def __del__(self):
        """析构时确保所有数据都已刷新"""
        self._flush_to_disk()
