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

# 可选：Wandb 监控
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

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
                 enable_monitors=True,  # 新增：是否启用监控
                 network_config=None,  # 新增：网络配置（用于读取优化器参数）
                 enable_wandb=False,  # 新增：是否启用 Wandb 监控
                 wandb_project="peg-solitaire-rl",  # Wandb 项目名称
                 wandb_run_name=None,  # Wandb 运行名称
                 wandb_entity=None):  # Wandb 实体名称
        
        self.n_envs = n_envs
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_steps_per_env = n_steps_per_env
        self.agent_results_filepath = agent_results_filepath
        self.batch_size = batch_size
        self.n_optim_steps = n_optim_steps
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.wandb_entity = wandb_entity
        
        # 初始化 Wandb
        if self.enable_wandb:
            wandb_args = {
                "project": wandb_project,
                "name": wandb_run_name or f"{algorithm.name}_{n_envs}envs_{n_iter}iter",
                "config": {
                    "n_envs": n_envs,
                    "n_iter": n_iter,
                    "n_steps_per_env": n_steps_per_env,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "algorithm": algorithm.name,
                    "network_config": network_config.config_dict if network_config else None,
                }
            }
            if self.wandb_entity:  # 跳过 None / ""，避免显式 entity 与 API slug 不一致导致 404
                wandb_args["entity"] = self.wandb_entity
            try:
                wandb.init(**wandb_args)
                logger.info("✓ Wandb logging enabled")
            except Exception as e:
                logger.warning(f"✗ Wandb init failed: {e}")
                logger.warning("✗ Wandb disabled for this run")
                self.enable_wandb = False
        else:
            if enable_wandb and not WANDB_AVAILABLE:
                logger.warning("✗ Wandb requested but not installed. Install with: pip install wandb")
            else:
                logger.info("✗ Wandb disabled")
        
        # 创建批量环境
        device = next(algorithm.network.parameters()).device
        self.env = BatchedGPUEnv(n_envs=n_envs, device=device)
        
        # 优化器（从网络配置中读取）
        optimizer_config = network_config.config_dict.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam').lower()
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                algorithm.network.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(
                algorithm.network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                algorithm.network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=optimizer_config.get('momentum', 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
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
        
        # PPO 需要保存旧策略的 logits 和 values
        if hasattr(self.algorithm, 'clip_epsilon'):  # 检测是否为 PPO
            old_logits_buf = torch.zeros(total_steps, 132, device=self.env.device)
            old_values_buf = torch.zeros(total_steps, device=self.env.device)
        else:
            old_logits_buf = None
            old_values_buf = None
        
        # 重置环境
        self.env.reset()
        
        step_idx = 0
        for step in range(self.n_steps_per_env):
            # 获取当前状态和可行动作
            states = self.env.state  # (n_envs, 7, 7, 3)
            feasible = self.env.feasible_actions  # (n_envs, 132)
            
            # 计算索引范围
            start_idx = step * self.n_envs
            end_idx = start_idx + self.n_envs
            
            # 批量选择动作
            with torch.no_grad():
                logits, values = self.algorithm.get_logits_and_values(states)  # (n_envs, 132), (n_envs, 1)
                masked_logits = logits + (1 - feasible.float()) * (-1e9)
                policies = torch.softmax(masked_logits, dim=-1)
                
                # 采样动作
                action_dist = torch.distributions.Categorical(policies)
                actions = action_dist.sample()  # (n_envs,)
                
                # 保存旧策略的 logits 和 values（用于 PPO）
                if old_logits_buf is not None:
                    old_logits_buf[start_idx:end_idx] = logits
                    old_values_buf[start_idx:end_idx] = values.squeeze(-1)
            
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
        
        result = {
            'states': states_buf,
            'actions': actions_buf,
            'action_masks': masks_buf,
            'rewards': rewards_buf,
            'dones': dones_buf
        }
        
        # 添加 PPO 需要的旧策略信息
        if old_logits_buf is not None:
            result['old_logits'] = old_logits_buf
            result['old_values'] = old_values_buf
        
        return result
    
    def compute_returns_and_advantages(self, data):
        """
        计算 returns 和 advantages（完全 GPU 操作）
        
        支持两种模式：
        1. GAE (Generalized Advantage Estimation) - 推荐
        2. Simple TD(0) - 快速 baseline
        
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
        
        # 从 config 读取参数
        discount = self.algorithm.config.get('discount', 0.99)
        use_gae = self.algorithm.config.get('use_gae', True)
        gae_lambda = self.algorithm.config.get('gae_lambda', 0.95)
        
        if use_gae:
            # ==================== GAE 计算 ====================
            # GAE 公式：A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
            # 其中 delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            
            returns = torch.zeros_like(values_reshaped)
            advantages = torch.zeros_like(values_reshaped)
            
            # 最后一步的 bootstrap value
            next_values = torch.zeros(self.n_envs, device=self.env.device)
            gae = torch.zeros(self.n_envs, device=self.env.device)
            
            for t in reversed(range(n_steps)):
                # 如果 done，则 next_value = 0
                effective_next_values = next_values * (~dones_reshaped[t]).float()
                
                # TD error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
                delta = rewards_reshaped[t] + discount * effective_next_values - values_reshaped[t]
                
                # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
                gae = delta + discount * gae_lambda * gae * (~dones_reshaped[t]).float()
                
                # Advantage
                advantages[t] = gae
                
                # Return = Advantage + Value
                returns[t] = gae + values_reshaped[t]
                
                # Update next_values
                next_values = values_reshaped[t]
        else:
            # ==================== 简单 TD(0) 计算 ====================
            returns = torch.zeros_like(values_reshaped)
            advantages = torch.zeros_like(values_reshaped)
            
            # 最后一步的 bootstrap value
            next_values = torch.zeros(self.n_envs, device=self.env.device)
            
            for t in reversed(range(n_steps)):
                # 如果 done，则 next_value = 0
                effective_next_values = next_values * (~dones_reshaped[t]).float()
                
                # TD target
                returns[t] = rewards_reshaped[t] + discount * effective_next_values
                
                # Advantage
                advantages[t] = returns[t] - values_reshaped[t]
                
                # Update next_values
                next_values = values_reshaped[t]
        
        # 优势函数标准化（可选，提升训练稳定性）
        normalize_advantages = self.algorithm.config.get('normalize_advantages', True)
        if normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std
        
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
        dataset_tensors = [
            data['states'],
            data['actions'],
            data['action_masks'],
            data['advantages'].unsqueeze(-1),
            data['value_targets'].unsqueeze(-1)
        ]
        
        # PPO 需要额外的 old_logits 和 old_values
        if 'old_logits' in data and 'old_values' in data:
            dataset_tensors.extend([data['old_logits'], data['old_values']])
        
        dataset = torch.utils.data.TensorDataset(*dataset_tensors)
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
            # PPO 指标
            'clip_ratio': 0.0,
            'approx_kl': 0.0,
            'mean_advantage': 0.0,
        }
        gradient_norms = {}  # 每层梯度范数
        n_batches = 0
        
        for epoch in range(self.n_optim_steps):
            for batch_data in dataloader:
                # 根据批次数据的数量决定如何解包
                if len(batch_data) == 5:  # A2C 格式
                    batch_states, batch_actions, batch_masks, batch_advs, batch_targets = batch_data
                    batch_old_logits = None
                    batch_old_values = None
                elif len(batch_data) == 7:  # PPO 格式
                    batch_states, batch_actions, batch_masks, batch_advs, batch_targets, batch_old_logits, batch_old_values = batch_data
                else:
                    raise ValueError(f"Unexpected number of tensors in batch: {len(batch_data)}")
                
                # 计算损失
                loss_kwargs = {
                    'states': batch_states,
                    'actions': batch_actions,
                    'action_masks': batch_masks,
                    'advantages': batch_advs,
                    'value_targets': batch_targets
                }
                
                # 如果是 PPO，添加旧策略信息
                if batch_old_logits is not None:
                    loss_kwargs['old_logits'] = batch_old_logits
                    loss_kwargs['old_values'] = batch_old_values
                
                loss_dict = self.algorithm.compute_loss(**loss_kwargs)
                
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
                # 兼容不同的命名方式
                loss_metrics['actor_loss'] += loss_dict.get('actor_loss', loss_dict.get('policy_loss', torch.tensor(0.0, device=total_loss.device)))
                loss_metrics['critic_loss'] += loss_dict.get('critic_loss', loss_dict.get('value_loss', torch.tensor(0.0, device=total_loss.device)))
                
                # 记录熵并检测崩溃
                entropy = loss_dict.get('entropy', torch.tensor(0.0, device=total_loss.device))
                loss_metrics['entropy'] += entropy
                
                # 记录额外的 PPO 指标（如果存在）
                if 'clip_ratio' in loss_dict:
                    loss_metrics['clip_ratio'] += loss_dict['clip_ratio']
                if 'approx_kl' in loss_dict:
                    loss_metrics['approx_kl'] += loss_dict['approx_kl']
                if 'mean_advantage' in loss_dict:
                    loss_metrics['mean_advantage'] += loss_dict['mean_advantage']
                
                n_batches += 1
        
        # 平均化并转换到 CPU（批量操作）
        if n_batches > 0:
            for key in loss_metrics:
                if isinstance(loss_metrics[key], torch.Tensor):
                    loss_metrics[key] = (loss_metrics[key] / n_batches).item()
                else:
                    loss_metrics[key] = loss_metrics[key] / n_batches
        else:
            # 如果没有 batch，设置为 0
            for key in loss_metrics:
                loss_metrics[key] = 0.0
        
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
                    # 与训练一致：在 logits 上做掩码（加 -inf）再 argmax，
                    # 避免无约束的不可行 logits 漂移污染 softmax 分布
                    logits, _ = self.algorithm.get_logits_and_values(state)
                    mask = feasible.bool()
                    if not mask.any():
                        break
                    masked_logits = logits.masked_fill(~mask, float('-inf'))
                    action = torch.argmax(masked_logits[0]).unsqueeze(0)
                
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
        
        # Rolling mean 统计（用于进度条显示）
        reward_history = []
        pegs_left_history = []
        window_size = 50
        
        with logging_redirect_tqdm():
            pbar = tqdm(range(start_iter, self.n_iter))
            for i in pbar:
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
                
                # 🔍 Entropy 崩溃检测
                current_entropy = loss_metrics.get('entropy', 0.0)
                if current_entropy < 1e-6:  # 熵接近 0
                    logger.warning(f"\n{'='*80}")
                    logger.warning(f"⚠️  WARNING: Entropy 崩溃检测！")
                    logger.warning(f"   Epoch: {i}")
                    logger.warning(f"   Entropy: {current_entropy:.6f} (< 1e-6)")
                    logger.warning(f"   Reward: {loss_metrics.get('eval_mean_reward', 0):.3f}")
                    logger.warning(f"   Pegs Left: {loss_metrics.get('eval_mean_pegs_left', 0):.1f}")
                    logger.warning(f"{'='*80}\n")
                    
                    # 保存当前 checkpoint
                    self.save_checkpoint(i)
                    
                    # 抛出异常停止训练
                    raise RuntimeError(
                        f"Entropy 崩溃！Epoch {i}, Entropy={current_entropy:.6f}. "
                        f"建议：增大 entropy_coef 或降低 learning_rate"
                    )
                
                # 3. 评估（每 10 轮一次）
                if i % 10 == 0 or i == self.n_iter - 1:  # 最后一轮也评估
                    eval_metrics = self.evaluate()
                else:
                    # 非评估轮次，使用上一轮的评估结果或默认值
                    eval_metrics = {
                        'eval_mean_reward': 0,
                        'eval_std_reward': 0,
                        'eval_mean_pegs_left': 0,
                        'timestamp': time.time()
                    }
                
                # 4. 记录指标
                metrics = {
                    'epoch': i,  # 使用 epoch 而不是 iteration（更标准）
                    'mean_reward': eval_metrics['eval_mean_reward'],
                    'collect_time': collect_time,
                    'train_time': train_time,
                    # 训练损失指标
                    'total_loss': loss_metrics['total_loss'],
                    'actor_loss': loss_metrics['actor_loss'],
                    'critic_loss': loss_metrics['critic_loss'],
                    'entropy': loss_metrics['entropy'],
                    # PPO 特有指标（如果存在）
                    'clip_ratio': loss_metrics.get('clip_ratio', 0.0),
                    'approx_kl': loss_metrics.get('approx_kl', 0.0),
                    'mean_advantage': loss_metrics.get('mean_advantage', 0.0),
                    # 梯度范数（取平均值和最大值）
                    'grad_norm_mean': sum(gradient_norms.values()) / len(gradient_norms) if gradient_norms else 0.0,
                    'grad_norm_max': max(gradient_norms.values()) if gradient_norms else 0.0,
                    **eval_metrics
                }
                self.monitor.training_history.append(metrics)
                
                # Wandb 记录
                if self.enable_wandb:
                    wandb.log(metrics, step=i)
                
                # 更新 rolling mean 历史
                if eval_metrics['eval_mean_reward'] > 0:  # 只有评估过的轮次
                    reward_history.append(eval_metrics['eval_mean_reward'])
                    pegs_left_history.append(eval_metrics['eval_mean_pegs_left'])
                
                # 计算 rolling mean
                if len(reward_history) > 0:
                    recent_rewards = reward_history[-window_size:]
                    recent_pegs = pegs_left_history[-window_size:]
                    rolling_reward = sum(recent_rewards) / len(recent_rewards)
                    rolling_pegs = sum(recent_pegs) / len(recent_pegs)
                    
                    # 更新进度条描述（使用 set_postfix）
                    pbar.set_postfix(
                        Reward=f"{rolling_reward:.2f}",
                        Pegs=f"{rolling_pegs:.2f}"
                    )
                
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
        
        # Wandb 结束
        if self.enable_wandb:
            wandb.finish()
            logger.info("✓ Wandb run finished")
        
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
    
    def save_checkpoint(self, epoch: int):
        """
        保存检查点（用于异常情况下紧急保存）
        
        Args:
            epoch: 当前轮次
        """
        try:
            checkpoint_path = os.path.join(self.checkpoints_dir, f'emergency_iter_{epoch}.pt')
            torch.save({
                'iteration': epoch,
                'network_state_dict': self.algorithm.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'monitor_history': getattr(self.monitor_manager, 'epoch_history', []),
                'timestamp': time.time()
            }, checkpoint_path)
            logger.info(f"✅ Emergency checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save emergency checkpoint: {e}")
        
        return self.monitor.training_history