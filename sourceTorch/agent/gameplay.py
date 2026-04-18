"""
Agent 游戏交互模块

提供 Agent 与环境交互的统一接口，包括：
- select_action(): 动作选择
- play_game(): 单局游戏
"""
import torch
from typing import Tuple, Optional
import numpy as np

from ..env.constants import GRID, MOVES, POS_TO_INDICES
from ..env.rendering import render_board

MIN_ACTION_PROBA = 1.0e-7


def extract_pegs_from_state(state: torch.Tensor) -> np.ndarray:
    """
    从状态张量中提取棋子数组（按照 GRID 顺序）
    
    Args:
        state: 状态张量，形状 (7, 7, 3)
    
    Returns:
        pegs: 棋子数组，形状 (33,)，值域 {0, 1}
    """
    # 确保在 CPU 上
    if state.is_cuda:
        state = state.cpu()
    
    # 转换为 numpy
    state_np = state.numpy()  # (7, 7, 3)
    
    # 按照 GRID 顺序提取棋子
    pegs = np.zeros(len(GRID), dtype=np.float32)
    for idx, (pos_i, pos_j) in enumerate(GRID):
        # POS_TO_INDICES 映射: (i, j) -> (row, col) in 7x7 grid
        row, col = POS_TO_INDICES[(pos_i, pos_j)]
        pegs[idx] = state_np[row, col, 0]
    
    return pegs


def select_action(agent, 
                  state: torch.Tensor, 
                  feasible_actions: torch.Tensor, 
                  greedy: bool = False,
                  temperature: float = 1.0) -> int:
    """
    选择动作（对齐 source 版本的接口风格）
    
    Args:
        agent: 智能体
        state: 状态张量，形状 (7, 7, 3)
        feasible_actions: 可行动作掩码，形状 (132,)
        greedy: 是否使用贪婪策略
        temperature: 温度参数（用于控制随机性，仅当 greedy=False 时有效）
    
    Returns:
        action_index: 动作索引 (0-131)，如果没有合法动作则返回 -1
    """
    # 获取策略（batch 维度）
    policy = agent.get_policy(state.unsqueeze(0).to(agent.device))[0]  # (132,)
    
    # 添加最小概率阈值（避免数值问题）
    policy = torch.clamp(policy, min=MIN_ACTION_PROBA)
    
    # 应用动作掩码并归一化
    masked_policy = policy * feasible_actions.float()
    if masked_policy.sum() > 0:
        masked_policy = masked_policy / masked_policy.sum()
    else:
        # 没有合法动作
        return -1
    
    # 选择动作
    if greedy:
        action_index = torch.argmax(masked_policy).item()
    else:
        # 应用温度参数
        if temperature != 1.0:
            logits = torch.log(masked_policy + 1e-10) / temperature
            probs = torch.softmax(logits, dim=0)
            action_index = torch.multinomial(probs, 1).item()
        else:
            action_index = torch.multinomial(masked_policy, 1).item()
    
    return action_index


def action_index_to_pos_move(action_index: int) -> Tuple[int, int]:
    """
    将动作索引转换为 (pos_id, move_id)
    
    Args:
        action_index: 动作索引 (0-131)
    
    Returns:
        (pos_id, move_id): 位置ID和移动方向ID
    """
    pos_id = action_index // 4
    move_id = action_index % 4
    return pos_id, move_id


def play_game(agent, 
              env, 
              render: bool = False, 
              game_id: int = 1, 
              greedy: bool = False,
              temperature: float = 1.0,
              step_delay_before: float = 1.0,  # 动作前延迟（显示轨迹）
              step_delay_after: float = 0.8) -> Tuple[float, int]:  # 动作后延迟（显示结果）
    """
    玩一局游戏（对齐 source 版本的接口风格）
    
    Args:
        agent: 智能体
        env: 环境（BatchedGPUEnv，n_envs=1）
        render: 是否渲染
        game_id: 游戏ID
        greedy: 是否使用贪婪策略
        temperature: 温度参数
        step_delay: 每步渲染延迟（秒）
    
    Returns:
        total_return: 累计奖励
        n_pegs_left: 剩余棋子数
    """
    import matplotlib.pyplot as plt
    
    obs = env.reset()
    states = obs['states'][0].to(agent.device)  # (7, 7, 3)
    feasible_actions = obs['feasible_actions'][0].to(agent.device)  # (132,)
    
    total_return = 0.0
    step_count = 0
    
    # 初始化渲染窗口
    fig, ax = None, None
    if render:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        pegs = extract_pegs_from_state(states)
        render_board(pegs, title=f"Game {game_id} - Step 0", ax=ax)
        plt.pause(1.5)
    
    while True:
        # 选择动作
        action_index = select_action(agent, states, feasible_actions, greedy, temperature)
        
        # 检查是否有合法动作
        if action_index == -1:
            break
        
        # 执行动作前渲染（显示动作轨迹）
        if render:
            ax.clear()
            pegs = extract_pegs_from_state(states)
            pos_id, move_id = action_index_to_pos_move(action_index)
            render_board(pegs, action=(pos_id, move_id), show_action=True,
                        title=f"Game {game_id} - Step {step_count + 1}", ax=ax)
            plt.pause(step_delay_before)
        
        # 执行动作
        action_tensor = torch.tensor([action_index], device=states.device)
        result = env.step(action_tensor)
        
        reward = result['rewards'][0].item()
        states = result['states'][0].to(agent.device)
        feasible_actions = result['feasible_actions'][0].to(agent.device)
        done = result['dones'][0].item()
        
        total_return += reward
        step_count += 1
        
        # 执行动作后渲染
        if render:
            ax.clear()
            pegs = extract_pegs_from_state(states)
            n_pegs = int(pegs.sum())
            render_board(pegs, title=f"Game {game_id} - Step {step_count} (Pegs: {n_pegs})", ax=ax)
            plt.pause(step_delay_after)
        
        if done:
            break
    
    # 最终状态显示
    if render:
        ax.clear()
        pegs = extract_pegs_from_state(states)
        n_pegs = int(pegs.sum())
        render_board(pegs, title=f"Game {game_id} - Final (Pegs: {n_pegs})", ax=ax)
        plt.pause(2)
        plt.close(fig)
    
    n_pegs_left = int(extract_pegs_from_state(states).sum())
    
    return total_return, n_pegs_left
