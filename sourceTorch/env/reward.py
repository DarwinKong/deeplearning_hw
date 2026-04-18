"""
奖励函数模块 - 定义孔明棋的奖励机制。

当前支持：
- default: 原始奖励
- mobility: 原始奖励 + mobility shaping
"""
import torch

from .constants import N_ACTIONS

SUPPORTED_REWARD_MODES = {"default", "mobility"}


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
        
    终止奖励规则（与原版对齐）：
    - 完美解（n_pegs == 1）: +1.0
    - 失败（n_pegs > 1）: +1/31（与每步奖励相同）
    """
    if device is None:
        device = torch.device('cpu')
    
    N_PEGS = 32  # 初始棋子数
    
    if n_pegs == 1:
        # 完美解：额外奖励 1.0（总奖励 = 31 * 1/31 + 1.0 = 2.0）
        return torch.tensor(1.0, device=device)
    else:
        # 失败：给予固定奖励 1/31（与每步奖励相同，与原版对齐）
        return torch.tensor(1.0 / (N_PEGS - 1), device=device)


def compute_batched_rewards(
    n_pegs_before: torch.Tensor,
    n_pegs_after: torch.Tensor,
    is_terminal: torch.Tensor,
    reward_mode: str = "default",
    mobility_before: torch.Tensor = None,
    mobility_after: torch.Tensor = None,
    mobility_alpha: float = 0.1,
    mobility_normalize: bool = False,
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
        
    奖励规则（与原版对齐）：
    - 非终止状态：每步 1/31
    - 终止且完美解（n_pegs == 1）：1.0
    - 终止且失败（n_pegs > 1）：1/31
        
    示例：
        >>> n_pegs_before = torch.tensor([32, 32, 32])
        >>> n_pegs_after = torch.tensor([31, 31, 31])
        >>> is_terminal = torch.tensor([False, False, False])
        >>> rewards = compute_batched_rewards(n_pegs_before, n_pegs_after, is_terminal)
        >>> print(rewards)  # tensor([0.0323, 0.0323, 0.0323])
    """
    if device is None:
        device = n_pegs_before.device
    
    if reward_mode not in SUPPORTED_REWARD_MODES:
        raise ValueError(f"Unsupported reward mode {reward_mode}. Supported modes: {SUPPORTED_REWARD_MODES}")

    N_PEGS = 32
    STEP_REWARD = 1.0 / (N_PEGS - 1)  # ≈ 0.032
    
    # 初始化奖励张量
    rewards = torch.zeros_like(n_pegs_before, dtype=torch.float32, device=device)
    
    # 非终止状态：每步奖励
    non_terminal_mask = ~is_terminal
    rewards[non_terminal_mask] = STEP_REWARD
    
    # 终止状态：根据剩余棋子数计算
    terminal_mask = is_terminal
    
    # 完美解（只剩1个棋子）：额外奖励 1.0
    perfect_mask = terminal_mask & (n_pegs_after == 1)
    rewards[perfect_mask] = 1.0
    
    # 失败情况（棋子 > 1）：固定奖励 1/31
    fail_mask = terminal_mask & (n_pegs_after > 1)
    rewards[fail_mask] = STEP_REWARD
    
    if reward_mode == "mobility":
        if mobility_before is None or mobility_after is None:
            raise ValueError("mobility reward requires both mobility_before and mobility_after")
        mobility_delta = mobility_after.float() - mobility_before.float()
        if mobility_normalize:
            mobility_delta = mobility_delta / N_ACTIONS
        rewards = rewards + mobility_alpha * mobility_delta

    return rewards
