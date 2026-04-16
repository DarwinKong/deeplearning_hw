"""环境模块 - 批量 GPU 环境实现"""
from .batched_gpu_env import BatchedGPUEnv
from .reward import compute_step_reward, compute_terminal_reward, compute_batched_rewards
from .rendering import render_board, render_state_batch
from .constants import (
    GRID, POS_TO_INDICES, N_PEGS, MOVES, ACTIONS,
    N_ACTIONS, N_STATE_CHANNELS, OUT_OF_BORDER_ACTIONS
)

__all__ = [
    'BatchedGPUEnv',
    'compute_step_reward',
    'compute_terminal_reward', 
    'compute_batched_rewards',
    'render_board',
    'render_state_batch',
    'GRID',
    'POS_TO_INDICES',
    'N_PEGS',
    'MOVES',
    'ACTIONS',
    'N_ACTIONS',
    'N_STATE_CHANNELS',
    'OUT_OF_BORDER_ACTIONS'
]
