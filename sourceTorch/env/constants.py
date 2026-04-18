"""
孔明棋环境常量定义

这些常量定义了棋盘结构、动作空间等基本信息。
"""

# ============================================================================
# 棋盘网格定义
# ============================================================================

# 7x7 十字形棋盘的33个有效位置 (i, j) 坐标
# i: 行索引 (-3 到 3), j: 列索引 (-3 到 3)
GRID = [(i, j) for j in [-3, -2] for i in [-1, 0, 1]] + \
       [(i, j) for j in [-1, 0, 1] for i in [-3, -2, -1, 0, 1, 2, 3]] + \
       [(i, j) for j in [2, 3] for i in [-1, 0, 1]]

# 位置到索引的映射字典 {(i, j): index}
POS_TO_INDICES = {(x_, y_): (3 - y_, x_ + 3) for x_, y_ in GRID}

# 初始棋子数（排除中心空位）
N_PEGS = len(GRID) - 1  # = 32

# ============================================================================
# 动作空间定义
# ============================================================================

# 4个移动方向：上、下、右、左
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 所有可能的动作：(起始位置, 移动方向)
ACTIONS = [(pos, move) for pos in GRID for move in MOVES]

# 总动作数 = 33个位置 × 4个方向 = 132
N_ACTIONS = len(ACTIONS)  # = 132

# ============================================================================
# 状态表示
# ============================================================================

# 状态张量的通道数
N_STATE_CHANNELS = 3  # [棋子存在性, 剩余比例, 已移除比例]

# ============================================================================
# 边界检查
# ============================================================================

def _compute_out_of_border_actions(grid):
    """
    预计算哪些动作会超出边界
    
    Returns:
        numpy.ndarray: 形状 (33, 4) 的布尔数组，True 表示该动作越界
    """
    import numpy as np
    
    out_of_border = np.zeros((len(grid), len(MOVES)), dtype=bool)
    
    for pos_idx, (pos_i, pos_j) in enumerate(grid):
        for move_idx, (move_i, move_j) in enumerate(MOVES):
            # 计算中间位置和目标位置
            mid_i, mid_j = pos_i + move_i, pos_j + move_j
            target_i, target_j = pos_i + 2 * move_i, pos_j + 2 * move_j
            
            # 检查是否在 GRID 内
            if (mid_i, mid_j) not in grid or (target_i, target_j) not in grid:
                out_of_border[pos_idx, move_idx] = True
    
    return out_of_border


# 预计算出界动作掩码
OUT_OF_BORDER_ACTIONS = _compute_out_of_border_actions(GRID)
