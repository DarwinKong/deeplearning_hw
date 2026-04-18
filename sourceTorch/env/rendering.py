"""
环境渲染模块 - 可视化孔明棋棋盘

提供交互式棋盘显示功能，用于 play.py 等脚本。
"""
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Optional
import numpy as np

from .constants import GRID, N_PEGS


def render_board(
    pegs: np.ndarray,
    action: Optional[Tuple[int, int]] = None,
    show_action: bool = False,
    title: str = "Peg Solitaire",
    ax=None
) -> None:
    """
    渲染棋盘状态
    
    Args:
        pegs (np.ndarray): 棋子状态数组，形状 (33,)，值域 {0, 1}
            - 1 表示有棋子
            - 0 表示空位
        action (Tuple[int, int], optional): 当前动作 (pos_id, move_id)
            如果提供且 show_action=True，会高亮显示动作相关的棋子
        show_action (bool): 是否显示动作轨迹（棕色=起始位置，黑色=被跳过的棋子）
        title (str): 图表标题
        ax: matplotlib axes 对象，如果为 None 则创建新的
    
    Example:
        >>> pegs = np.ones(33)
        >>> pegs[16] = 0  # 中心空位
        >>> render_board(pegs, title="Initial State")
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 解析动作（如果提供且需要显示）
    if show_action and action is not None:
        pos_id, move_id = action
        pos_i, pos_j = GRID[pos_id]
        move_i, move_j = [(0, 1), (0, -1), (1, 0), (-1, 0)][move_id]
        mid_i, mid_j = pos_i + move_i, pos_j + move_j
        target_i, target_j = pos_i + 2 * move_i, pos_j + 2 * move_j
    else:
        pos_id = None
        mid_i = mid_j = None
    
    # 绘制每个位置
    for idx, (i, j) in enumerate(GRID):
        if pegs[idx] == 1:
            # 有棋子
            if show_action and action is not None and idx == pos_id:
                color = 'black'  # 被移动的棋子（黑色）
            elif show_action and action is not None and (i, j) == (mid_i, mid_j):
                color = 'red'  # 被跳过的棋子（红色，将被删除）
            else:
                color = 'burlywood'  # 普通棋子
            circle = matplotlib.patches.Circle(xy=(j, -i), radius=0.495, color=color, fill=True)
        else:
            # 空位
            circle = matplotlib.patches.Circle(xy=(j, -i), radius=0.495, color='burlywood', 
                                              fill=False, linewidth=1.5)
        ax.add_patch(circle)
    
    # 如果显示动作，绘制箭头
    if show_action and action is not None:
        # 起始位置
        start_x, start_y = GRID[pos_id][1], -GRID[pos_id][0]
        # 目标位置
        target_x, target_y = target_j, -target_i
        
        # 绘制箭头（从起始位置指向目标位置）
        arrow = matplotlib.patches.FancyArrowPatch(
            (start_x, start_y),
            (target_x, target_y),
            arrowstyle='->', 
            mutation_scale=30,  # 箭头大小
            linewidth=3,
            color='blue',
            alpha=0.7,
            zorder=10  # 确保箭头在棋子上方
        )
        ax.add_patch(arrow)
    
    # 设置坐标轴
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 添加网格线（可选）
    ax.grid(True, linestyle='--', alpha=0.3)


def render_state_batch(
    states: np.ndarray,
    n_cols: int = 4,
    title_prefix: str = "Game"
) -> None:
    """
    批量渲染多个游戏状态
    
    Args:
        states (np.ndarray): 状态数组，形状 (n_games, 7, 7, 3)
        n_cols (int): 每行显示的列数
        title_prefix (str): 标题前缀
    
    Example:
        >>> states = np.random.rand(8, 7, 7, 3)
        >>> render_state_batch(states, n_cols=4)
        >>> plt.tight_layout()
        >>> plt.show()
    """
    n_games = states.shape[0]
    n_rows = (n_games + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_games > 1 else [axes]
    
    for idx in range(n_games):
        # 从状态张量提取棋子信息（通道0）
        pegs = states[idx, :, :, 0].flatten()[:33]  # 取前33个位置
        render_board(pegs, title=f"{title_prefix} {idx+1}", ax=axes[idx])
    
    # 隐藏多余的子图
    for idx in range(n_games, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
