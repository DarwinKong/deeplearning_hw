"""
纯 PyTorch 向量化版本的孔明棋环境
完全移除 numpy 依赖，所有操作在 GPU 上向量化执行
"""
import torch
from .rendering import *

# ============================================================================
# 常量定义（纯 PyTorch）
# ============================================================================

# 棋盘位置网格 (33个位置)
GRID = [(i, j) for j in [-3, -2] for i in [-1, 0, 1]] + \
       [(i, j) for j in [-1, 0, 1] for i in range(-3, 4)] + \
       [(i, j) for j in [2, 3] for i in [-1, 0, 1]]

N_PEGS = len(GRID) - 1  # 32
N_STATE_CHANNELS = 3
ACTION_NAMES = ["up", "down", "right", "left"]
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
N_MOVES = len(MOVES)
N_ACTIONS = len(GRID) * N_MOVES  # 33 * 4 = 132

# 位置到索引的映射
POS_TO_INDICES = {(x_, y_): (3 - y_, x_ + 3) for x_, y_ in GRID}

# 将 GRID 转换为 tensor (33, 2)
GRID_TENSOR = torch.tensor(GRID, dtype=torch.long)  # (33, 2)

# 将 MOVES 转换为 tensor (4, 2)
MOVES_TENSOR = torch.tensor(MOVES, dtype=torch.long)  # (4, 2)

# ============================================================================
# 预计算边界掩码（向量化）
# ============================================================================

def _compute_out_of_border_actions_tensor():
    """向量化计算边界外动作掩码"""
    n_positions = len(GRID)
    out_of_border = torch.zeros((n_positions, N_MOVES), dtype=torch.bool)
    
    positions = GRID_TENSOR  # (33, 2)
    x = positions[:, 0]  # (33,)
    y = positions[:, 1]  # (33,)
    
    # Up (move_id=0): (0, 1)
    # 如果 y >= 0 且 (x < -1 或 x > 1)，或者 y >= 2
    condition_up = (y >= 0) & ((x < -1) | (x > 1) | (y >= 2))
    out_of_border[:, 0] = condition_up
    
    # Down (move_id=1): (0, -1)
    # 如果 y <= 0 且 (x < -1 或 x > 1)，或者 y <= -2
    condition_down = (y <= 0) & ((x < -1) | (x > 1) | (y <= -2))
    out_of_border[:, 1] = condition_down
    
    # Right (move_id=2): (1, 0)
    # 如果 x >= 0 且 (y < -1 或 y > 1)，或者 x >= 2
    condition_right = (x >= 0) & ((y < -1) | (y > 1) | (x >= 2))
    out_of_border[:, 2] = condition_right
    
    # Left (move_id=3): (-1, 0)
    # 如果 x <= 0 且 (y < -1 或 y > 1)，或者 x <= -2
    condition_left = (x <= 0) & ((y < -1) | (y > 1) | (x <= -2))
    out_of_border[:, 3] = condition_left
    
    return out_of_border

def _get_board_mask_tensor():
    """获取棋盘掩码"""
    board_mask = torch.zeros((7, 7), dtype=torch.bool)
    for pos, (i, j) in POS_TO_INDICES.items():
        board_mask[i, j] = True
    return board_mask.reshape(-1)  # (49,)

# 预计算常量
OUT_OF_BORDER_ACTIONS = _compute_out_of_border_actions_tensor()  # (33, 4)
BOARD_MASK = _get_board_mask_tensor()  # (49,)


class Env(object):
    """
    纯 PyTorch 向量化孔明棋环境
    
    核心优化：
    1. 使用 tensor 存储棋盘状态，避免 dict 遍历
    2. 所有操作用向量化实现，无 Python 循环
    3. feasible_actions 完全向量化计算
    4. state 直接由 tensor 构建，无需遍历
    """
    N_MAX_STEPS = N_PEGS
    _BOARD_MASK = BOARD_MASK

    def __init__(self, verbose=False, init_fig=False, interactive_plot=False, device=None):
        """
        初始化环境
        
        Args:
            verbose: 是否显示详细信息
            init_fig: 是否初始化渲染图形
            interactive_plot: 是否交互式绘图
            device: 计算设备（None 则自动选择）
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 棋盘状态：使用 tensor 代替 dict，(33,) 每个位置对应 GRID 中的一个位置
        self.pegs_tensor = torch.ones(N_PEGS + 1, dtype=torch.float32, device=self.device)  # 33个位置
        self.pegs_tensor[0] = 0  # 第一个位置为空 (0, 0)
        
        self.n_pegs = N_PEGS
        
        if init_fig:
            self.init_fig(interactive_plot)
        else:
            self.interactive_plot = False
        self.verbose = verbose

    def reset(self):
        """重置环境"""
        self.n_pegs = N_PEGS
        self.pegs_tensor = torch.ones(N_PEGS + 1, dtype=torch.float32, device=self.device)
        self.pegs_tensor[0] = 0  # 第一个位置为空

    def step(self, action):
        """
        执行动作并返回奖励、新状态和终止标志（向量化）
        
        Args:
            action: tuple (pos_id, move_id)
            
        Returns:
            reward: float
            next_state: torch.Tensor (7, 7, 3)
            end: bool
        """
        pos_id, move_id = action
        
        # 获取位置和移动方向
        pos = GRID_TENSOR[pos_id]  # (2,)
        move = MOVES_TENSOR[move_id]  # (2,)
        
        # 计算中间位置和目标位置
        mid_pos = pos + move
        target_pos = pos + 2 * move
        
        # 找到这些位置在 GRID 中的索引
        # 使用 broadcasting 查找匹配
        mid_match = torch.all(GRID_TENSOR == mid_pos.unsqueeze(0), dim=1)  # (33,)
        target_match = torch.all(GRID_TENSOR == target_pos.unsqueeze(0), dim=1)  # (33,)
        
        mid_idx = torch.argmax(mid_match.float())
        target_idx = torch.argmax(target_match.float())
        
        # 更新棋盘状态（向量化操作）
        self.pegs_tensor[pos_id] = 0
        self.pegs_tensor[mid_idx] = 0
        self.pegs_tensor[target_idx] = 1
        self.n_pegs -= 1

        # 检查游戏结束
        if self.n_pegs == 1:
            if self.verbose:
                print('End of the game, you solved the puzzle !')
            return 1.0, self.state, True
        else:
            # 检查是否还有可行动作
            feasible = self.feasible_actions
            if torch.sum(feasible) == 0:
                if self.verbose:
                    print(f'End of the game. You lost : {self.n_pegs} pegs remaining')
                return 1 / (N_PEGS - 1), self.state, True
            else:
                return 1 / (N_PEGS - 1), self.state, False

    @staticmethod
    def convert_action_id_to_action(action_index):
        """将动作ID转换为(pos_id, move_id)"""
        return divmod(action_index, N_MOVES)

    @property
    def board_mask(self):
        return self._BOARD_MASK

    @property
    def state(self) -> torch.Tensor:
        """
        返回状态（完全向量化，无循环）
        
        Returns:
            torch.Tensor of shape (7, 7, 3) on self.device
        """
        state = torch.zeros((7, 7, N_STATE_CHANNELS), dtype=torch.float32, device=self.device)
        
        # 向量化：直接使用索引赋值
        indices_list = [POS_TO_INDICES[pos] for pos in GRID]
        indices_tensor = torch.tensor(indices_list, dtype=torch.long, device=self.device)  # (33, 2)
        
        i_indices = indices_tensor[:, 0]
        j_indices = indices_tensor[:, 1]
        
        # 通道0: 棋子位置
        state[i_indices, j_indices, 0] = self.pegs_tensor

        # 通道1: 剩余棋子比例
        state[:, :, 1] = (self.n_pegs - 1) / (N_PEGS - 1)
        # 通道2: 已移除棋子比例
        state[:, :, 2] = (N_PEGS - self.n_pegs) / (N_PEGS - 1)
        
        return state

    @property
    def feasible_actions(self) -> torch.Tensor:
        """
        返回可行动作掩码（完全向量化，无 Python 循环）
        
        Returns:
            torch.Tensor of shape (33, 4) on CPU
        """
        # 初始掩码：排除边界外动作
        feasible_actions = ~OUT_OF_BORDER_ACTIONS.to(self.device)  # (33, 4)
        
        # 排除没有棋子的位置
        no_peg_mask = (self.pegs_tensor == 0)  # (33,)
        feasible_actions[no_peg_mask, :] = False
        
        # 获取所有可能的位置和动作
        pos_ids, move_ids = torch.where(feasible_actions)  # 两个 1D tensor
        
        if len(pos_ids) == 0:
            return feasible_actions.cpu()
        
        # 向量化计算所有动作的跳跃可行性
        positions = GRID_TENSOR[pos_ids.cpu()].to(self.device)  # (N, 2)
        moves = MOVES_TENSOR[move_ids.cpu()].to(self.device)  # (N, 2)
        
        # 计算中间位置和目标位置
        mid_positions = positions + moves  # (N, 2)
        target_positions = positions + 2 * moves  # (N, 2)
        
        # 完全向量化：使用 broadcasting 比较所有位置
        # GRID_TENSOR: (33, 2), mid_positions: (N, 2)
        # 扩展维度后比较：(N, 33, 2)
        mid_match = (GRID_TENSOR.to(self.device).unsqueeze(0) == mid_positions.unsqueeze(1)).all(dim=2)  # (N, 33)
        target_match = (GRID_TENSOR.to(self.device).unsqueeze(0) == target_positions.unsqueeze(1)).all(dim=2)  # (N, 33)
        
        # 找到匹配的索引（argmax 不支持 bool，需要转换）
        mid_indices = mid_match.float().argmax(dim=1)  # (N,)
        target_indices = target_match.float().argmax(dim=1)  # (N,)
        
        # 检查中间位置有棋子且目标位置为空
        mid_has_peg = self.pegs_tensor[mid_indices] == 1
        target_empty = self.pegs_tensor[target_indices] == 0
        jump_feasible = mid_has_peg & target_empty
        
        # 更新可行动作掩码
        feasible_actions[pos_ids, move_ids] = jump_feasible
        
        return feasible_actions.cpu()

    def action_jump_feasible(self, pos_index, move_id):
        """检查跳跃动作是否可行（保留接口兼容性）"""
        x, y = GRID[pos_index]
        d_x, d_y = MOVES[move_id]
        mid_pos = (x + d_x, y + d_y)
        target_pos = (x + 2 * d_x, y + 2 * d_y)
        return self.pegs.get(mid_pos, 0) == 1 and self.pegs.get(target_pos, 0) == 0

    def render(self, action=None, show_action=False, show_axes=False):
        """渲染当前状态（从 tensor 重构 pegs dict 用于渲染）"""
        # 为了兼容渲染代码，临时构建 pegs dict
        pegs_dict = {GRID[i]: self.pegs_tensor[i].item() for i in range(len(GRID))}
        
        ax = plt.gca()
        for p in reversed(ax.patches):
            p.remove()
        ax.axes.get_xaxis().set_visible(show_axes)
        ax.axes.get_yaxis().set_visible(show_axes)
        if show_action:
            assert action is not None
            pos_id, move_id = action
            x, y = GRID[pos_id]
            dx, dy = MOVES[move_id]
            jumped_pos = (x + dx, y + dy)
            for pos, value in pegs_dict.items():
                if value == 1:
                    if pos == (x, y):
                        ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='brown', fill=True))
                    elif pos == jumped_pos:
                        ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='black', fill=True))
                    else:
                        ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=True))
                if value == 0:
                    ax.add_patch(
                        matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=False, linewidth=1.5))
        else:
            assert action is None
            for pos, value in pegs_dict.items():
                if value == 1:
                    ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=True))
                if value == 0:
                    ax.add_patch(
                        matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=False, linewidth=1.5))

        plt.ylim(-4, 4)
        plt.xlim(-4, 4)
        plt.draw()
        plt.pause(0.01)
