"""
训练监控工具
记录训练过程中的关键指标到本地文件
"""
import os
import json
import csv
from datetime import datetime
from typing import Dict, List


class TrainingMonitor:
    """
    训练监控器，记录强化学习训练的关键指标
    
    监控的指标包括：
    1. 训练指标（每个 iteration）
       - mean_reward: 平均奖励
       - mean_pegs_left: 平均剩余棋子数
       - std_reward: 奖励标准差
       - min_reward / max_reward: 最小/最大奖励
       
    2. 评估指标（定期评估）
       - eval_mean_reward: 评估平均奖励
       - eval_mean_pegs_left: 评估平均剩余棋子数
       - greedy_reward: 贪婪策略奖励
       - greedy_pegs_left: 贪婪策略剩余棋子数
       
    3. 损失函数指标（从 TensorBoard 日志中提取）
       - policy_loss: 策略损失
       - value_loss: 价值损失
       - entropy: 熵（探索程度）
    """
    
    def __init__(self, log_dir: str):
        """
        :param log_dir: 日志目录路径
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 训练指标文件
        self.training_metrics_file = os.path.join(log_dir, "training_metrics.csv")
        self.evaluation_metrics_file = os.path.join(log_dir, "evaluation_metrics.csv")
        self.summary_file = os.path.join(log_dir, "training_summary.json")
        
        # 初始化 CSV 文件
        self._init_training_csv()
        self._init_evaluation_csv()
        
        # 存储所有指标用于生成总结
        self.training_history = []
        self.evaluation_history = []
    
    def _init_training_csv(self):
        """初始化训练指标 CSV 文件"""
        with open(self.training_metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration',
                'mean_reward',
                'std_reward',
                'min_reward',
                'max_reward',
                'mean_pegs_left',
                'timestamp'
            ])
    
    def _init_evaluation_csv(self):
        """初始化评估指标 CSV 文件"""
        with open(self.evaluation_metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration',
                'eval_mean_reward',
                'eval_std_reward',
                'eval_mean_pegs_left',
                'greedy_reward',
                'greedy_pegs_left',
                'timestamp'
            ])
    
    def log_training_metrics(self, iteration: int, rewards: List[float], pegs_left: List[float]):
        """
        记录训练指标
        
        :param iteration: 当前迭代次数
        :param rewards: 奖励列表
        :param pegs_left: 剩余棋子数列表
        """
        import numpy as np
        
        metrics = {
            'iteration': iteration,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'mean_pegs_left': float(np.mean(pegs_left)),
            'timestamp': datetime.now().isoformat()
        }
        
        # 写入 CSV
        with open(self.training_metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)
        
        # 保存到历史
        self.training_history.append(metrics)
        
        return metrics
    
    def log_evaluation_metrics(self, iteration: int, eval_rewards: List[float], 
                               eval_pegs_left: List[float], 
                               greedy_reward: float, greedy_pegs_left: float):
        """
        记录评估指标
        
        :param iteration: 当前迭代次数
        :param eval_rewards: 评估奖励列表
        :param eval_pegs_left: 评估剩余棋子数列表
        :param greedy_reward: 贪婪策略奖励
        :param greedy_pegs_left: 贪婪策略剩余棋子数
        """
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
        
        # 写入 CSV
        with open(self.evaluation_metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)
        
        # 保存到历史
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def generate_summary(self):
        """生成训练总结报告"""
        if not self.training_history:
            return
        
        summary = {
            'total_iterations': len(self.training_history),
            'best_training': {
                'iteration': max(self.training_history, key=lambda x: x['mean_reward'])['iteration'],
                'mean_reward': max(h['mean_reward'] for h in self.training_history),
                'mean_pegs_left': min(h['mean_pegs_left'] for h in self.training_history)
            },
            'latest_training': self.training_history[-1],
            'training_progress': {
                'reward_improvement': self._calculate_improvement('mean_reward'),
                'pegs_reduction': self._calculate_improvement('mean_pegs_left', reverse=True)
            }
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
    
    def _calculate_improvement(self, metric: str, reverse: bool = False) -> Dict[str, float]:
        """
        计算指标改进情况
        
        :param metric: 指标名称
        :param reverse: 是否越小越好（如 pegs_left）
        :return: 改进统计
        """
        if len(self.training_history) < 2:
            return {'first': None, 'last': None, 'improvement': None}
        
        first_value = self.training_history[0][metric]
        last_value = self.training_history[-1][metric]
        
        if reverse:
            improvement = first_value - last_value  # 越小越好
        else:
            improvement = last_value - first_value  # 越大越好
        
        improvement_pct = (improvement / first_value * 100) if first_value != 0 else 0
        
        return {
            'first': first_value,
            'last': last_value,
            'absolute_improvement': improvement,
            'percentage_improvement': improvement_pct
        }
    
    def print_current_status(self, training_metrics: Dict = None, evaluation_metrics: Dict = None):
        """
        打印当前训练状态到控制台
        
        :param training_metrics: 训练指标
        :param evaluation_metrics: 评估指标
        """
        print("\n" + "="*60)
        print("📊 训练监控报告")
        print("="*60)
        
        if training_metrics:
            print(f"\n🎯 训练指标 (Iteration {training_metrics['iteration']:,}):")
            print(f"   平均奖励: {training_metrics['mean_reward']:.3f} ± {training_metrics['std_reward']:.3f}")
            print(f"   奖励范围: [{training_metrics['min_reward']:.3f}, {training_metrics['max_reward']:.3f}]")
            print(f"   平均剩余棋子: {training_metrics['mean_pegs_left']:.2f}")
        
        if evaluation_metrics:
            print(f"\n✅ 评估指标:")
            print(f"   评估平均奖励: {evaluation_metrics['eval_mean_reward']:.3f}")
            print(f"   贪婪策略奖励: {evaluation_metrics['greedy_reward']:.3f}")
            print(f"   评估平均剩余棋子: {evaluation_metrics['eval_mean_pegs_left']:.2f}")
            print(f"   贪婪策略剩余棋子: {evaluation_metrics['greedy_pegs_left']:.2f}")
        
        print("\n" + "="*60)
