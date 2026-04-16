"""
奖励函数模块 - 定义孔明棋的奖励机制

算法侧同学可以在此修改奖励函数设计。
"""
import torch


def compute_step_reward(n_pegs_before: int, n_pegs_after: int, device: torch.device = None) -> torch.Tensor:
    """
    计算单步奖励
    
    Args:
        n_pegs_before (int): 动作前的棋子数量
        n_pegs_after (int): 动作后的棋子数量
        device (torch.device, optional): 设备类型
        
    Returns:
        torch.Tensor: 奖励值（标量）
    
    奖励规则：
    - 成功移除一个棋子: +1/31 ≈ 0.032
    - 完美解（只剩1个棋子）: 总奖励归一化到 1.0
    - 失败（无合法动作但棋子 > 1）: 给予已移除比例的奖励
    """
    if device is None:
        device = torch.device('cpu')
    
    N_PEGS = 32  # 初始棋子数
    
    # 每步基础奖励：移除一个棋子的归一化奖励
    step_reward = 1.0 / (N_PEGS - 1)  # ≈ 0.032
    
    return torch.tensor(step_reward, device=device)


def compute_terminal_reward(n_pegs: int, device: torch.device = None) -> torch.Tensor:
    """
    计算终止状态奖励
    
    Args:
        n_pegs (int): 终止时的棋子数量
        device (torch.device, optional): 设备类型
        
    Returns:
        torch.Tensor: 终止奖励值
        
    终止奖励规则：
    - 完美解（n_pegs == 1）: 补足到总奖励 1.0
    - 失败（n_pegs > 1）: 给予已移除棋子的比例奖励
    """
    if device is None:
        device = torch.device('cpu')
    
    N_PEGS = 32  # 初始棋子数
    
    if n_pegs == 1:
        # 完美解：确保总奖励为 1.0
        # 已获得的奖励 = (32 - 1) / 31 = 1.0
        # 无需额外奖励
        return torch.tensor(0.0, device=device)
    else:
        # 失败：给予已移除棋子的比例奖励
        # 已移除 = 32 - n_pegs
        removed_ratio = (N_PEGS - n_pegs) / (N_PEGS - 1)
        return torch.tensor(removed_ratio, device=device)


def compute_batched_rewards(
    n_pegs_before: torch.Tensor,
    n_pegs_after: torch.Tensor,
    is_terminal: torch.Tensor,
    device: torch.device = None
) -> torch.Tensor:
    """
    批量计算奖励（用于 BatchedGPUEnv）
    
    Args:
        n_pegs_before (torch.Tensor): 动作前的棋子数量，形状 (n_envs,)
        n_pegs_after (torch.Tensor): 动作后的棋子数量，形状 (n_envs,)
        is_terminal (torch.Tensor): 是否为终止状态，形状 (n_envs,)，布尔类型
        device (torch.device, optional): 设备类型
        
    Returns:
        torch.Tensor: 批量奖励，形状 (n_envs,)
        
    示例：
        >>> n_pegs_before = torch.tensor([32, 32, 32])
        >>> n_pegs_after = torch.tensor([31, 31, 31])
        >>> is_terminal = torch.tensor([False, False, False])
        >>> rewards = compute_batched_rewards(n_pegs_before, n_pegs_after, is_terminal)
        >>> print(rewards)  # tensor([0.0323, 0.0323, 0.0323])
    """
    if device is None:
        device = n_pegs_before.device
    
    N_PEGS = 32
    
    # 初始化奖励张量
    rewards = torch.zeros_like(n_pegs_before, dtype=torch.float32, device=device)
    
    # 非终止状态：每步奖励
    non_terminal_mask = ~is_terminal
    rewards[non_terminal_mask] = 1.0 / (N_PEGS - 1)
    
    # 终止状态：根据剩余棋子数计算
    terminal_mask = is_terminal
    
    # 完美解（只剩1个棋子）
    perfect_mask = terminal_mask & (n_pegs_after == 1)
    # 完美解的奖励已经在之前步骤中累积，这里不需要额外奖励
    rewards[perfect_mask] = 0.0
    
    # 失败情况（棋子 > 1）
    fail_mask = terminal_mask & (n_pegs_after > 1)
    if fail_mask.any():
        removed_ratio = (N_PEGS - n_pegs_after[fail_mask]).float() / (N_PEGS - 1)
        rewards[fail_mask] = removed_ratio
    
    return rewards
