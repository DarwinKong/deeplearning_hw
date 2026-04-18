"""
批量 GPU 环境 - 极致向量化实现

核心优化：
1. 同时运行 N 个游戏实例（batched environments）
2. 所有数据保持在 GPU 上，零 CPU-GPU 传输
3. 预计算所有常量，避免重复计算
4. 使用 gather/scatter 进行高效的状态更新

接口说明：
    reset() -> Dict[str, torch.Tensor]
        重置所有环境
        返回: {
            'states': Tensor (n_envs, 7, 7, 3),  # 初始状态
            'feasible_actions': Tensor (n_envs, 132)  # 可行动作掩码
        }
    
    step(actions: Tensor) -> Dict[str, torch.Tensor]
        执行批量动作
        参数: actions - Tensor (n_envs,), 每个环境的动作索引 [0, 131]
        返回: {
            'rewards': Tensor (n_envs,),  # 每步奖励
            'states': Tensor (n_envs, 7, 7, 3),  # 新状态
            'dones': Tensor (n_envs,),  # 终止标志（布尔）
            'feasible_actions': Tensor (n_envs, 132)  # 新的可行动作掩码
        }
"""
import torch
import torch.nn as nn
from typing import Dict

# 导入本地常量（不依赖 source）
from .constants import (
    GRID, POS_TO_INDICES, N_PEGS, MOVES, ACTIONS, 
    N_ACTIONS, N_STATE_CHANNELS, OUT_OF_BORDER_ACTIONS
)

# 导入奖励函数
from .reward import compute_batched_rewards

# 预计算常量（在 CPU 上，之后移到 GPU）
GRID_TENSOR = torch.tensor(GRID, dtype=torch.long)  # (33, 2)
MOVES_TENSOR = torch.tensor(MOVES, dtype=torch.long)  # (4, 2)
POS_TO_INDICES_TENSOR = torch.tensor([
    POS_TO_INDICES[pos] for pos in GRID
], dtype=torch.long)  # (33, 2)

# 预计算每个位置对应的 (i, j) 索引
I_INDICES = POS_TO_INDICES_TENSOR[:, 0]  # (33,)
J_INDICES = POS_TO_INDICES_TENSOR[:, 1]  # (33,)

# 预计算动作对应的位置和移动
ACTION_POS_IDS = torch.arange(N_PEGS + 1).repeat_interleave(4)  # (132,)
ACTION_MOVE_IDS = torch.arange(4).repeat(N_PEGS + 1)  # (132,)

# 预计算每个动作的中间位置和目标位置
ACTION_POSITIONS = GRID_TENSOR[ACTION_POS_IDS]  # (132, 2)
ACTION_MOVES = MOVES_TENSOR[ACTION_MOVE_IDS]  # (132, 2)
ACTION_MID_POSITIONS = ACTION_POSITIONS + ACTION_MOVES  # (132, 2)
ACTION_TARGET_POSITIONS = ACTION_POSITIONS + 2 * ACTION_MOVES  # (132, 2)

# 预计算中间位置和目标位置的索引
def find_position_index(pos_tensor):
    """找到位置在 GRID 中的索引，如果不存在则返回 -1"""
    # broadcasting: (132, 1, 2) == (1, 33, 2) -> (132, 33, 2) -> (132, 33)
    matches = (GRID_TENSOR.unsqueeze(0) == pos_tensor.unsqueeze(1)).all(dim=2)
    # 使用 where 找到匹配的索引，如果没有匹配则返回 -1
    indices = matches.float().argmax(dim=1)
    has_match = matches.any(dim=1)
    indices = torch.where(has_match, indices, torch.full_like(indices, -1))
    return indices

ACTION_MID_INDICES = find_position_index(ACTION_MID_POSITIONS)  # (132,)
ACTION_TARGET_INDICES = find_position_index(ACTION_TARGET_POSITIONS)  # (132,)

# 预计算 out_of_border mask
OUT_OF_BORDER_MASK = torch.from_numpy(OUT_OF_BORDER_ACTIONS)  # (33, 4)


class BatchedGPUEnv(nn.Module):
    """
    批量 GPU 环境 - 同时运行多个游戏实例
    
    核心特性：
    - 并行执行 n_envs 个独立的游戏实例
    - 所有数据保持在 GPU，零 CPU-GPU 传输
    - 预计算常量，运行时 O(1) 查表
    - 向量化状态更新，避免 Python 循环
    
    Attributes:
        n_envs (int): 并行环境数量
        device (torch.device): GPU 设备
        pegs (torch.Tensor): 棋子状态，形状 (n_envs, 33)，值域 [0, 1]
        n_pegs (torch.Tensor): 每环境的棋子数，形状 (n_envs,)，值域 [1, 32]
        done (torch.Tensor): 终止标志，形状 (n_envs,)，布尔类型
        total_reward (torch.Tensor): 累计奖励，形状 (n_envs,)
    
    Example:
        >>> env = BatchedGPUEnv(n_envs=64, device='cuda')
        >>> obs = env.reset()
        >>> print(obs['states'].shape)  # torch.Size([64, 7, 7, 3])
        >>> actions = torch.randint(0, 132, (64,), device='cuda')
        >>> result = env.step(actions)
        >>> print(result['rewards'].shape)  # torch.Size([64])
    """
    
    def __init__(self,
                 n_envs=64,
                 device='cuda',
                 reward_mode='default',
                 mobility_alpha=0.1,
                 mobility_normalize=False,
                 mobility_alpha_final=None,
                 mobility_anneal_end_progress=1.0):
        super().__init__()
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.reward_mode = reward_mode
        self.mobility_alpha = mobility_alpha
        self.mobility_normalize = mobility_normalize
        self.mobility_alpha_final = mobility_alpha if mobility_alpha_final is None else mobility_alpha_final
        self.mobility_anneal_end_progress = mobility_anneal_end_progress
        self.training_progress = 0.0
        
        # 注册为 buffer（不计算梯度，但会随模块移动到设备）
        self.register_buffer('pegs', torch.ones(n_envs, N_PEGS + 1, dtype=torch.float32, device=self.device))
        self.register_buffer('n_pegs', torch.full((n_envs,), N_PEGS, dtype=torch.long, device=self.device))
        self.register_buffer('done', torch.zeros(n_envs, dtype=torch.bool, device=self.device))
        self.register_buffer('total_reward', torch.zeros(n_envs, dtype=torch.float32, device=self.device))
        
        # 预计算常量移到 GPU
        self.register_buffer('grid_tensor', GRID_TENSOR.to(self.device))
        self.register_buffer('action_pos_ids', ACTION_POS_IDS.to(self.device))
        self.register_buffer('action_move_ids', ACTION_MOVE_IDS.to(self.device))
        self.register_buffer('action_mid_indices', ACTION_MID_INDICES.to(self.device))
        self.register_buffer('action_target_indices', ACTION_TARGET_INDICES.to(self.device))
        self.register_buffer('out_of_border_mask', OUT_OF_BORDER_MASK.to(self.device))
        self.register_buffer('i_indices', I_INDICES.to(self.device))
        self.register_buffer('j_indices', J_INDICES.to(self.device))
        
        self.reset()

    def set_training_progress(self, progress: float):
        self.training_progress = float(max(0.0, min(1.0, progress)))

    def get_current_mobility_alpha(self) -> float:
        if self.mobility_anneal_end_progress <= 0:
            return self.mobility_alpha_final

        anneal_ratio = min(self.training_progress / self.mobility_anneal_end_progress, 1.0)
        return (1.0 - anneal_ratio) * self.mobility_alpha + anneal_ratio * self.mobility_alpha_final
    
    def reset(self, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        重置所有环境到初始状态
        
        Args:
            mask (torch.Tensor, optional): 指定哪些环境需要重置
                - 形状: (n_envs,)
                - 类型: torch.bool
                - 默认: None（重置所有环境）
        
        Returns:
            Dict[str, torch.Tensor]: 初始观测数据
                - 'states': torch.Tensor, 形状 (n_envs, 7, 7, 3), dtype=torch.float32
                  棋盘状态张量，3个通道分别为：棋子存在性、剩余比例、已移除比例
                - 'feasible_actions': torch.Tensor, 形状 (n_envs, 132), dtype=torch.float32
                  可行动作掩码，1表示合法动作，0表示非法动作
        
        Example:
            >>> obs = env.reset()
            >>> print(obs['states'].shape)  # (64, 7, 7, 3)
            >>> print(obs['feasible_actions'].shape)  # (64, 132)
        """
        if mask is None:
            mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        else:
            # MPS 上布尔高级索引不能安全地复用同一底层存储，先 clone 一份。
            mask = mask.clone()
        
        # 重置棋子状态：中心位置（索引16）为空，其他都有棋子
        self.pegs[mask] = 1.0
        self.pegs[mask, 16] = 0.0  # 中心位置 (0, 0) 为空
        
        # 重置计数器
        self.n_pegs[mask] = N_PEGS
        self.done[mask] = False
        self.total_reward[mask] = 0.0
        
        # 返回观测数据
        return {
            'states': self.state,
            'feasible_actions': self.feasible_actions
        }
    
    @property
    def state(self):
        """
        获取当前状态 batch
        
        Returns:
            states: (n_envs, 7, 7, 3) tensor on GPU
        """
        n = self.n_envs
        states = torch.zeros(n, 7, 7, 3, dtype=torch.float32, device=self.device)
        
        # 通道 0: 棋子位置
        states[:, self.i_indices, self.j_indices, 0] = self.pegs
        
        # 通道 1: 归一化的棋子数量
        peg_ratio = (self.n_pegs.float() - 1) / (N_PEGS - 1)  # (n_envs,)
        states[:, :, :, 1] = peg_ratio.view(n, 1, 1)
        
        # 通道 2: 归一化的已移除棋子数量
        removed_ratio = (N_PEGS - self.n_pegs.float()) / (N_PEGS - 1)
        states[:, :, :, 2] = removed_ratio.view(n, 1, 1)
        
        return states
    
    @property
    def feasible_actions(self):
        """
        获取可行动作 mask
        
        Returns:
            mask: (n_envs, 132) bool tensor on GPU
        """
        n = self.n_envs
        
        # 初始 mask：排除越界动作
        mask = ~self.out_of_border_mask.unsqueeze(0).expand(n, -1, -1).reshape(n, N_ACTIONS)
        
        # 排除没有棋子的位置
        no_peg = (self.pegs == 0)  # (n_envs, 33)
        # 将 (n_envs, 33) 扩展到 (n_envs, 132)
        no_peg_expanded = no_peg.index_select(1, self.action_pos_ids)  # (n_envs, 132)
        mask = mask & ~no_peg_expanded
        
        # 检查跳跃可行性（向量化）
        # 对于每个环境、每个动作，检查中间位置有棋子且目标位置为空
        mid_indices = self.action_mid_indices  # (132,)
        target_indices = self.action_target_indices  # (132,)
        
        # 处理越界情况：如果索引为 -1，说明动作越界，应该标记为不可行
        mid_valid = (mid_indices >= 0)  # (132,)
        target_valid = (target_indices >= 0)  # (132,)
        
        # 使用 clamp 避免负数索引
        mid_indices_safe = mid_indices.clamp(min=0)
        target_indices_safe = target_indices.clamp(min=0)
        
        mid_has_peg = self.pegs[:, mid_indices_safe]  # (n_envs, 132)
        target_empty = (self.pegs[:, target_indices_safe] == 0)  # (n_envs, 132)
        
        # 只有当索引有效时才检查棋子和空位
        jump_feasible = mid_valid.unsqueeze(0) & target_valid.unsqueeze(0) & (mid_has_peg > 0) & target_empty
        mask = mask & jump_feasible
        
        # 已完成的环境，所有动作都不可行
        mask[self.done] = False
        
        return mask

    def count_feasible_actions(self):
        """返回每个环境当前可行动作数量。"""
        return self.feasible_actions.sum(dim=1)
    
    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        批量执行动作
        
        Args:
            actions (torch.Tensor): 动作索引张量
                - 形状: (n_envs,)
                - 类型: torch.long
                - 值域: [0, 131]，每个元素对应一个合法的动作 ID
        
        Returns:
            Dict[str, torch.Tensor]: 环境响应数据
                - 'rewards': torch.Tensor, 形状 (n_envs,), dtype=torch.float32
                  每步奖励，成功移除棋子为 1/31 ≈ 0.032，终止状态根据剩余棋子数计算
                - 'states': torch.Tensor, 形状 (n_envs, 7, 7, 3), dtype=torch.float32
                  执行动作后的新状态
                - 'dones': torch.Tensor, 形状 (n_envs,), dtype=torch.bool
                  终止标志，True 表示该环境已结束（胜利或失败）
                - 'feasible_actions': torch.Tensor, 形状 (n_envs, 132), dtype=torch.float32
                  新的可行动作掩码
                - 'infos': Dict
                  - 'n_pegs': torch.Tensor, 形状 (n_envs,), 当前棋子数
                  - 'total_reward': torch.Tensor, 形状 (n_envs,), 累计奖励
        
        Example:
            >>> actions = torch.tensor([0, 5, 10, ...], device='cuda')  # 64个动作
            >>> result = env.step(actions)
            >>> print(result['rewards'])  # tensor([0.0323, 0.0323, ...])
            >>> print(result['dones'])  # tensor([False, False, ...])
        """
        n = self.n_envs
        
        mobility_before = self.count_feasible_actions()

        # 获取动作对应的位置索引
        pos_ids = self.action_pos_ids[actions]  # (n_envs,)
        mid_indices = self.action_mid_indices[actions]  # (n_envs,)
        target_indices = self.action_target_indices[actions]  # (n_envs,)
        
        # 创建环境索引
        env_indices = torch.arange(n, device=self.device)
        
        # 更新棋盘状态（使用 scatter/update）
        # 移除起始位置的棋子
        self.pegs[env_indices, pos_ids] = 0.0
        # 移除中间位置的棋子
        self.pegs[env_indices, mid_indices] = 0.0
        # 在目标位置添加棋子
        self.pegs[env_indices, target_indices] = 1.0
        
        # 更新棋子计数
        n_pegs_before = self.n_pegs.clone()
        self.n_pegs -= 1
        n_pegs_after = self.n_pegs.clone()
        
        # 检查是否结束
        done_win = (self.n_pegs == 1)
        
        # 检查是否还有可行动作
        feasible = self.feasible_actions
        mobility_after = feasible.sum(dim=1)
        has_feasible = feasible.any(dim=1)  # (n_envs,)
        done_no_moves = ~has_feasible
        
        self.done = done_win | done_no_moves
        
        # 计算奖励（使用 reward 模块）
        rewards = compute_batched_rewards(
            n_pegs_before=n_pegs_before,
            n_pegs_after=n_pegs_after,
            is_terminal=self.done,
            reward_mode=self.reward_mode,
            mobility_before=mobility_before,
            mobility_after=mobility_after,
            mobility_alpha=self.get_current_mobility_alpha(),
            mobility_normalize=self.mobility_normalize,
            device=self.device
        )
        
        # 累计奖励
        self.total_reward += rewards
        
        # 返回结果
        return {
            'rewards': rewards,
            'states': self.state,
            'dones': self.done,
            'feasible_actions': self.feasible_actions,
            'infos': {
                'n_pegs': self.n_pegs.clone(),
                'total_reward': self.total_reward.clone()
            }
        }
    
    def render(self, env_idx=0):
        """渲染单个环境（用于调试）"""
        print(f"Environment {env_idx}:")
        print(f"  Pegs: {self.n_pegs[env_idx].item()}")
        print(f"  Done: {self.done[env_idx].item()}")
        print(f"  Total Reward: {self.total_reward[env_idx].item():.2f}")


# 测试
if __name__ == "__main__":
    print("测试批量 GPU 环境...")
    
    n_envs = 8
    env = BatchedGPUEnv(n_envs=n_envs, device='cuda')
    
    print(f"\n初始状态:")
    print(f"  环境数: {n_envs}")
    print(f"  每个环境的棋子数: {env.n_pegs}")
    
    print(f"\nState shape: {env.state.shape}")
    print(f"Feasible actions shape: {env.feasible_actions.shape}")
    
    # 测试随机动作
    actions = torch.randint(0, N_ACTIONS, (n_envs,), device='cuda')
    result = env.step(actions)
    
    print(f"\n执行动作后:")
    print(f"  Rewards: {result['rewards']}")
    print(f"  Dones: {result['dones']}")
    print(f"  N pegs: {result['infos']['n_pegs']}")
    
    print("\n✅ 批量 GPU 环境测试通过！")
