"""
批量 GPU 训练器 - 使用向量化环境实现极致加速

核心优化：
1. 同时运行 N 个游戏（n_envs）
2. 所有数据在 GPU 上，零 CPU-GPU 传输
3. 批量前向传播
4. 高效的数据收集
5. 模块化监控系统
"""
import logging
import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from sourceTorch.agent.base_agent import BaseAgent
from sourceTorch.env.batched_gpu_env import BatchedGPUEnv
from sourceTorch.utils.gpu_training_monitor import GPUTrainingMonitor
from sourceTorch.trainers.monitors import (
    MonitorManager,
    GradientMonitor,
    LossMonitor,
    EntropyMonitor,
    RewardMonitor
)

logger = logging.getLogger()


class BatchedGPUTrainer:
    """
    批量 GPU 训练器
    
    使用 BatchedGPUEnv 实现极致的并行化和 GPU 加速。
    
    Args:
        n_envs: 并行环境数量（建议 64-256）
        algorithm: 算法实例
        n_iter: 训练迭代次数
        n_steps_per_env: 每个环境每轮收集的步数
        ...
    """
    
    def __init__(
        self,
                 n_envs=64,
                 algorithm: BaseAgent = None,
                 n_iter=200,
                 n_steps_per_env=32,
                 agent_results_filepath="results/batched.pt",
                 learning_rate=3e-5,
                 batch_size=256,
                 n_optim_steps=1,
                 log_dir=None,
                 checkpoints_dir=None,
                 meta_dir=None,
                 results_dir=None,
                 enable_monitors=True):  # 新增：是否启用监控
        
        self.n_envs = n_envs
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_steps_per_env = n_steps_per_env
        self.agent_results_filepath = agent_results_filepath
        self.batch_size = batch_size
        self.n_optim_steps = n_optim_steps
        
        # 创建批量环境
        device = next(algorithm.network.parameters()).device
        self.env = BatchedGPUEnv(n_envs=n_envs, device=device)
        
        # 优化器
        self.optimizer = optim.Adam(algorithm.network.parameters(), lr=learning_rate)
        
        # 监控器
        log_dir = log_dir or os.path.dirname(agent_results_filepath) or "temp/logs_batched"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.monitor = GPUTrainingMonitor(log_dir, device=device, flush_interval=10)
        
        # 模块化监控系统
        self.monitor_manager = MonitorManager(log_dir=log_dir)  # 传递 log_dir
        if enable_monitors:
            self.monitor_manager.add_monitor(GradientMonitor(record_every=10))
            self.monitor_manager.add_monitor(LossMonitor())
            self.monitor_manager.add_monitor(EntropyMonitor())
            self.monitor_manager.add_monitor(RewardMonitor())
            logger.info("✓ Monitors enabled: Gradient, Loss, Entropy, Reward")
        else:
            logger.info("✗ Monitors disabled for performance testing")
        
        # 目录
        self.checkpoints_dir = checkpoints_dir or "temp/checkpoints_batched"
        self.meta_dir = meta_dir or "temp/meta_batched"
        self.results_dir = results_dir or "temp/results_batched"
        
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    @staticmethod
    def find_latest_checkpoint(checkpoints_dir: str):
        """
        查找最新的 checkpoint 文件
        
        Args:
            checkpoints_dir: checkpoint 目录路径
        
        Returns:
            tuple: (checkpoint_path, latest_iter) 或 None
        """
        if not os.path.exists(checkpoints_dir):
            return None
        
        # 查找所有 iter_*.pt 文件
        checkpoint_files = [
            f for f in os.listdir(checkpoints_dir) 
            if f.startswith('iter_') and f.endswith('.pt')
        ]
        
        if not checkpoint_files:
            return None
        
        # 提取迭代次数并找到最大的
        def get_iter_num(filename):
            try:
                return int(filename.replace('iter_', '').replace('.pt', ''))
            except ValueError:
                return -1
        
        latest_file = max(checkpoint_files, key=get_iter_num)
        latest_iter = get_iter_num(latest_file)
        
        return os.path.join(checkpoints_dir, latest_file), latest_iter
    
    def resume_from_checkpoint(self, checkpoint_path: str = None) -> int:
        """
        从 checkpoint 恢复训练
        
        Args:
            checkpoint_path: checkpoint 文件路径，如果为 None 则自动查找最新的
        
        Returns:
            start_iter: 恢复后的起始迭代次数
        """
        # 如果没有指定路径，自动查找最新的
        if checkpoint_path is None:
            result = self.find_latest_checkpoint()
            if result is None:
                logger.info("No checkpoint found, starting from scratch")
                return 0
            checkpoint_path, start_iter = result
        else:
            # 从文件名提取迭代次数
            basename = os.path.basename(checkpoint_path)
            try:
                start_iter = int(basename.replace('iter_', '').replace('.pt', '')) + 1
            except ValueError:
                start_iter = 0
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 加载网络权重
        self.algorithm.network.load_state_dict(checkpoint['network_state_dict'])
        logger.info("✓ Network weights loaded")
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ Optimizer state loaded")
        
        # 加载监控器历史（如果有）
        if 'monitor_history' in checkpoint and hasattr(self.monitor_manager, 'epoch_history'):
            self.monitor_manager.epoch_history = checkpoint['monitor_history']
            logger.info(f"✓ Monitor history loaded ({len(self.monitor_manager.epoch_history)} epochs)")
        
        logger.info(f"Resuming training from iteration {start_iter}")
        return start_iter
        
        logger.info(f"Initialized BatchedGPU Trainer")
        logger.info(f"  n_envs: {n_envs}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Learning rate: {learning_rate}")
    
    def collect_batched_data(self):
        """
        批量收集数据
        
        Returns:
            dict with all tensors on GPU:
                - states: (total_steps, 7, 7, 3)
                - actions: (total_steps,)
                - action_masks: (total_steps, 132)
                - rewards: (total_steps,)
                - dones: (total_steps,)
        """
        self.algorithm.set_evaluation_mode()
        
        total_steps = self.n_envs * self.n_steps_per_env
        
        # 预分配缓冲区（GPU）
        states_buf = torch.zeros(total_steps, 7, 7, 3, device=self.env.device)
        actions_buf = torch.zeros(total_steps, dtype=torch.long, device=self.env.device)
        masks_buf = torch.zeros(total_steps, 132, device=self.env.device)
        rewards_buf = torch.zeros(total_steps, device=self.env.device)
        dones_buf = torch.zeros(total_steps, dtype=torch.bool, device=self.env.device)
        
        # 重置环境
        self.env.reset()
        
        step_idx = 0
        for step in range(self.n_steps_per_env):
            # 获取当前状态和可行动作
            states = self.env.state  # (n_envs, 7, 7, 3)
            feasible = self.env.feasible_actions  # (n_envs, 132)
            
            # 批量选择动作
            with torch.no_grad():
                policies = self.algorithm.get_policy(states)  # (n_envs, 132)
                masked_policies = policies * feasible.float()
                
                # 采样动作
                action_dist = torch.distributions.Categorical(masked_policies)
                actions = action_dist.sample()  # (n_envs,)
            
            # 执行动作
            result = self.env.step(actions)
            
            # 存储数据
            start_idx = step * self.n_envs
            end_idx = start_idx + self.n_envs
            
            states_buf[start_idx:end_idx] = states
            actions_buf[start_idx:end_idx] = actions
            masks_buf[start_idx:end_idx] = feasible
            rewards_buf[start_idx:end_idx] = result['rewards']
            dones_buf[start_idx:end_idx] = result['dones']
            
            step_idx += self.n_envs
            
            # 重置已完成的环境
            if result['dones'].any():
                self.env.reset(mask=result['dones'])
        
        return {
            'states': states_buf,
            'actions': actions_buf,
            'action_masks': masks_buf,
            'rewards': rewards_buf,
            'dones': dones_buf
        }
    
    def compute_returns_and_advantages(self, data):
        """
        计算 returns 和 advantages（完全 GPU 操作）
        
        Args:
            data: dict from collect_batched_data
        
        Returns:
            data with added 'value_targets' and 'advantages'
        """
        states = data['states']
        rewards = data['rewards']
        dones = data['dones']
        
        # 批量价值估计
        values = self.algorithm.get_value(states).squeeze(-1)  # (total_steps,)
        
        # 重塑为 (n_steps, n_envs)
        n_steps = self.n_steps_per_env
        values_reshaped = values.view(n_steps, self.n_envs)
        rewards_reshaped = rewards.view(n_steps, self.n_envs)
        dones_reshaped = dones.view(n_steps, self.n_envs)
        
        # 逆向计算 returns
        returns = torch.zeros_like(values_reshaped)
        advantages = torch.zeros_like(values_reshaped)
        
        # 最后一步的 bootstrap value
        next_values = torch.zeros(self.n_envs, device=self.env.device)
        
        for t in reversed(range(n_steps)):
            # 如果 done，则 next_value = 0
            effective_next_values = next_values * (~dones_reshaped[t]).float()
            
            # TD target
            returns[t] = rewards_reshaped[t] + self.algorithm.config.get('discount', 1.0) * effective_next_values
            
            # Advantage
            advantages[t] = returns[t] - values_reshaped[t]
            
            # Update next_values
            next_values = values_reshaped[t]
        
        # 展平
        data['value_targets'] = returns.flatten()
        data['advantages'] = advantages.flatten()
        
        return data
    
    def update_agent(self, data):
        """
        更新算法参数（批量训练）
        
        Returns:
            Dict: 训练指标
                - loss_metrics: 损失、熵、KL散度等
                - gradient_norms: 每层梯度范数字典
        """
        self.algorithm.set_training_mode()
        
        # 创建 DataLoader
        dataset = torch.utils.data.TensorDataset(
            data['states'],
            data['actions'],
            data['action_masks'],
            data['advantages'].unsqueeze(-1),
            data['value_targets'].unsqueeze(-1)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # 初始化指标收集
        loss_metrics = {
            'total_loss': 0.0,
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
        }
        gradient_norms = {}  # 每层梯度范数
        n_batches = 0
        
        for epoch in range(self.n_optim_steps):
            for batch_states, batch_actions, batch_masks, batch_advs, batch_targets in dataloader:
                # 计算损失
                loss_dict = self.algorithm.compute_loss(
                    states=batch_states,
                    actions=batch_actions,
                    action_masks=batch_masks,
                    advantages=batch_advs,
                    value_targets=batch_targets
                )
                
                total_loss = loss_dict['total_loss']
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 通知监控器（计算梯度范数）
                self.monitor_manager.on_backward(self.algorithm.network)
                
                # 计算梯度范数（在裁剪之前，只记录最后一次优化的第一个batch）
                if epoch == self.n_optim_steps - 1 and n_batches == 0:
                    # 先在 GPU 上累积
                    grad_norms_gpu = {}
                    for name, param in self.algorithm.network.named_parameters():
                        if param.grad is not None:
                            grad_norms_gpu[name] = param.grad.norm(2)  # 保持为 tensor
                    
                    # 批量转换到 CPU
                    for name, norm_tensor in grad_norms_gpu.items():
                        gradient_norms[name] = norm_tensor.item()
                
                # 梯度裁剪
                if hasattr(self.algorithm, 'max_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.algorithm.network.parameters(),
                        max_norm=self.algorithm.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # 累积指标（先在 GPU 上累积，最后统一转 CPU）
                loss_metrics['total_loss'] += loss_dict.get('total_loss', total_loss)
                loss_metrics['actor_loss'] += loss_dict.get('actor_loss', torch.tensor(0.0, device=total_loss.device))
                loss_metrics['critic_loss'] += loss_dict.get('critic_loss', torch.tensor(0.0, device=total_loss.device))
                loss_metrics['entropy'] += loss_dict.get('entropy', torch.tensor(0.0, device=total_loss.device))
                n_batches += 1
        
        # 平均化并转换到 CPU（批量操作）
        for key in loss_metrics:
            loss_metrics[key] = (loss_metrics[key] / n_batches).item()
        
        return {
            'loss_metrics': loss_metrics,
            'gradient_norms': gradient_norms
        }
    
    def evaluate(self):
        """评估（使用贪心策略）"""
        self.algorithm.set_evaluation_mode()
        
        # 使用单个环境进行评估
        single_env = BatchedGPUEnv(n_envs=1, device=self.env.device)
        
        total_rewards = []
        pegs_left_list = []
        
        for _ in range(10):  # 评估 10 次
            single_env.reset()
            done = False
            
            while not done:
                state = single_env.state
                feasible = single_env.feasible_actions
                
                with torch.no_grad():
                    policy = self.algorithm.get_policy(state)[0]
                    masked_policy = policy * feasible[0].float()
                    
                    if masked_policy.sum() == 0:
                        break
                    
                    action = torch.argmax(masked_policy).unsqueeze(0)
                
                result = single_env.step(action)
                done = result['dones'][0].item()
            
            total_rewards.append(single_env.total_reward[0].item())
            pegs_left_list.append(single_env.n_pegs[0].item())
        
        return {
            'eval_mean_reward': sum(total_rewards) / len(total_rewards),
            'eval_std_reward': torch.tensor(total_rewards).std().item(),
            'eval_mean_pegs_left': sum(pegs_left_list) / len(pegs_left_list),
            'timestamp': time.time()
        }
    
    def train(self, start_iter: int = 0):
        """主训练循环
        
        Args:
            start_iter: 起始迭代次数（用于从 checkpoint 恢复）
        """
        logger.info(f"Starting batched GPU training")
        logger.info(f"  Iteration range: {start_iter} -> {self.n_iter}")
        logger.info(f"  Parallel environments: {self.n_envs}")
        logger.info(f"  Steps per env: {self.n_steps_per_env}")
        logger.info(f"  Total steps per iteration: {self.n_envs * self.n_steps_per_env}")
        
        # 初始化监控系统
        self.monitor_manager.on_train_begin()
        
        with logging_redirect_tqdm():
            for i in tqdm(range(start_iter, self.n_iter)):
                # 通知监控器
                self.monitor_manager.on_epoch_begin(i)
                
                # 1. 批量收集数据
                start_time = time.time()
                
                with torch.no_grad():
                    data = self.collect_batched_data()
                    data = self.compute_returns_and_advantages(data)
                
                collect_time = time.time() - start_time
                
                # 2. 更新 Agent
                start_time = time.time()
                update_metrics = self.update_agent(data)
                train_time = time.time() - start_time
                
                # 提取训练指标
                loss_metrics = update_metrics['loss_metrics']
                gradient_norms = update_metrics['gradient_norms']
                
                # 3. 评估
                if i % 10 == 0:
                    eval_metrics = self.evaluate()
                else:
                    eval_metrics = {
                        'eval_mean_reward': 0,
                        'eval_std_reward': 0,
                        'eval_mean_pegs_left': 0,
                        'timestamp': time.time()
                    }
                
                # 4. 记录指标
                metrics = {
                    'iteration': i,
                    'mean_reward': eval_metrics['eval_mean_reward'],
                    'collect_time': collect_time,
                    'train_time': train_time,
                    # 训练损失指标
                    'total_loss': loss_metrics['total_loss'],
                    'actor_loss': loss_metrics['actor_loss'],
                    'critic_loss': loss_metrics['critic_loss'],
                    'entropy': loss_metrics['entropy'],
                    # 梯度范数（取平均值和最大值）
                    'grad_norm_mean': sum(gradient_norms.values()) / len(gradient_norms) if gradient_norms else 0.0,
                    'grad_norm_max': max(gradient_norms.values()) if gradient_norms else 0.0,
                    **eval_metrics
                }
                self.monitor.training_history.append(metrics)
                
                # 通知监控器 epoch 结束
                self.monitor_manager.on_epoch_end(i, metrics)
                
                # 6. 保存检查点
                if i % 50 == 0:
                    checkpoint_path = os.path.join(self.checkpoints_dir, f"iter_{i}.pt")
                    torch.save({
                        'iteration': i,
                        'network_state_dict': self.algorithm.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'monitor_history': getattr(self.monitor_manager, 'epoch_history', []),  # 保存监控历史
                    }, checkpoint_path)
                
                # 7. 保存最优模型（使用 RewardMonitor）
                if i % 10 == 0 and eval_metrics['eval_mean_reward'] > 0:
                    reward_monitor = self.monitor_manager.monitors.get('reward')
                    if reward_monitor and reward_monitor.metrics.get('is_best', False):
                        # 保存最优模型到 results 目录
                        best_model_path = os.path.join(self.results_dir, 'best_model.pt')
                        torch.save({
                            'iteration': i,
                            'network_state_dict': self.algorithm.network.state_dict(),
                            'eval_reward': reward_monitor.best_reward,
                            'eval_pegs_left': reward_monitor.best_pegs_left,
                            'timestamp': time.time()
                        }, best_model_path)
                        logger.info(
                            f"  ★ New best model saved! "
                            f"Reward={reward_monitor.best_reward:.2f}, "
                            f"Pegs={reward_monitor.best_pegs_left:.2f}"
                        )
                
                # 8. 打印进度
                if i % 10 == 0:
                    speed = 1.0 / (collect_time + train_time)
                    logger.info(
                        f"Iter {i}: "
                        f"Reward={eval_metrics['eval_mean_reward']:.2f}, "
                        f"Pegs={eval_metrics['eval_mean_pegs_left']:.2f}, "
                        f"Loss={loss_metrics['total_loss']:.4f}, "
                        f"Entropy={loss_metrics['entropy']:.4f}, "
                        f"GradNorm={metrics['grad_norm_mean']:.4f}, "
                        f"Speed={speed:.1f} it/s"
                    )
        
        logger.info("Training completed!")
        
        # 通知监控器训练结束
        self.monitor_manager.on_train_end()
        
        # 保存监控摘要
        monitor_summary = self.monitor_manager.get_full_summary()
        import json
        summary_path = os.path.join(self.results_dir, 'monitor_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(monitor_summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Monitor summary saved to {summary_path}")
        
        # 保存详细的梯度范数信息
        if gradient_norms:
            import json
            gradient_report = {
                'iteration': self.n_iter - 1,
                'gradient_norms': {name: float(norm) for name, norm in gradient_norms.items()},
                'summary': {
                    'mean': float(metrics['grad_norm_mean']),
                    'max': float(metrics['grad_norm_max']),
                    'min': float(min(gradient_norms.values())),
                }
            }
            gradient_report_path = os.path.join(self.results_dir, 'gradient_report.json')
            with open(gradient_report_path, 'w', encoding='utf-8') as f:
                json.dump(gradient_report, f, indent=2, ensure_ascii=False)
            logger.info(f"Gradient report saved to {gradient_report_path}")
        
        return self.monitor.training_history
