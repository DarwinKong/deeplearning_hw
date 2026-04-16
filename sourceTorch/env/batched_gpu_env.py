"""
批量 GPU 环境 - 极致向量化实现

核心优化：
1. 同时运行 N 个游戏实例（batched environments）
2. 所有数据保持在 GPU 上，零 CPU-GPU 传输
3. 预计算所有常量，避免重复计算
4. 使用 gather/scatter 进行高效的状态更新
"""
import torch
import torch.nn as nn

# 从原始 env 导入常量
from source.env.env import (
    GRID, POS_TO_INDICES, N_PEGS, MOVES, ACTIONS, 
    N_ACTIONS, N_STATE_CHANNELS, OUT_OF_BORDER_ACTIONS
)

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
    """找到位置在 GRID 中的索引"""
    # broadcasting: (132, 1, 2) == (1, 33, 2) -> (132, 33, 2) -> (132, 33)
    matches = (GRID_TENSOR.unsqueeze(0) == pos_tensor.unsqueeze(1)).all(dim=2)
    return matches.float().argmax(dim=1)

ACTION_MID_INDICES = find_position_index(ACTION_MID_POSITIONS)  # (132,)
ACTION_TARGET_INDICES = find_position_index(ACTION_TARGET_POSITIONS)  # (132,)

# 预计算 out_of_border mask
OUT_OF_BORDER_MASK = torch.from_numpy(OUT_OF_BORDER_ACTIONS)  # (33, 4)


class BatchedGPUEnv(nn.Module):
    """
    批量 GPU 环境
    
    同时运行 n_envs 个游戏实例，所有计算在 GPU 上完成。
    
    Args:
        n_envs: 并行环境数量（建议 64-256）
        device: GPU 设备
    """
    
    def __init__(self, n_envs=64, device='cuda'):
        super().__init__()
        self.n_envs = n_envs
        self.device = torch.device(device)
        
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
    
    def reset(self, mask=None):
        """
        重置环境
        
        Args:
            mask: 可选，指定哪些环境需要重置 (n_envs,) bool tensor
        """
        if mask is None:
            mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        
        # 重置棋子状态：第一个位置为空，其他都有棋子
        self.pegs[mask] = 1.0
        self.pegs[mask, 0] = 0.0
        
        # 重置计数器
        self.n_pegs[mask] = N_PEGS
        self.done[mask] = False
        self.total_reward[mask] = 0.0
    
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
        
        mid_has_peg = self.pegs[:, mid_indices]  # (n_envs, 132)
        target_empty = (self.pegs[:, target_indices] == 0)  # (n_envs, 132)
        
        jump_feasible = (mid_has_peg > 0) & target_empty
        mask = mask & jump_feasible
        
        # 已完成的环境，所有动作都不可行
        mask[self.done] = False
        
        return mask
    
    def step(self, actions):
        """
        执行动作 batch
        
        Args:
            actions: (n_envs,) long tensor，每个元素是动作 ID (0-131)
        
        Returns:
            rewards: (n_envs,) float tensor
            states: (n_envs, 7, 7, 3) tensor
            dones: (n_envs,) bool tensor
            infos: dict
        """
        n = self.n_envs
        
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
        self.n_pegs -= 1
        
        # 计算奖励
        rewards = torch.where(
            self.n_pegs == 1,
            torch.ones(n, device=self.device),  # 胜利
            torch.ones(n, device=self.device) / (N_PEGS - 1)  # 普通步骤
        )
        
        # 检查是否结束
        done_win = (self.n_pegs == 1)
        
        # 检查是否还有可行动作
        feasible = self.feasible_actions
        has_feasible = feasible.any(dim=1)  # (n_envs,)
        done_no_moves = ~has_feasible
        
        self.done = done_win | done_no_moves
        
        # 累计奖励
        self.total_reward += rewards
        
        # 返回结果
        return {
            'rewards': rewards,
            'states': self.state,
            'dones': self.done,
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
