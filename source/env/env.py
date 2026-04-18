import numpy as np

from .rendering import *

GRID = [(i, j) for j in [-3, -2] for i in [-1, 0, 1]] + \
       [(i, j) for j in [-1, 0, 1] for i in np.arange(-3, 4)] + \
       [(i, j) for j in [2, 3] for i in [-1, 0, 1]]

# positions (x,y) in the grid are related to indexes (i,j) of a 7x7 array by
# i = 3 - y
# j = x + 3
POS_TO_INDICES = {(x_, y_): (3 - y_, x_ + 3) for x_, y_ in GRID}

# actions are ordered as follows : we take positions as they are ordered in GRID and list in order "up", "down",
# "right", "left", i.e. the 4 possible actions
N_PEGS = len(GRID) - 1  # = 32, center point in the grid does not contain any peg
ACTION_NAMES = ["up", "down", "right", "left"]
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]
ACTIONS = [(pos, move) for pos in GRID for move in MOVES]
N_ACTIONS = len(ACTIONS)  # = len(GRID) * len(MOVES) = 33 * 4 = 132

N_STATE_CHANNELS = 3
DEFAULT_STEP_REWARD = 1 / (N_PEGS - 1)
SUPPORTED_REWARD_MODES = {"default", "mobility", "potential", "terminal_diff", "hybrid_curriculum"}
SUPPORTED_POTENTIAL_FUNCTIONS = {"remaining_pegs", "dispersion", "combined"}


def _compute_out_of_border_actions(grid):
    '''
    Returns a 2d-array whose shape is (n,m) where n is the number of positions in the grid, and m=4 for every possible move (up, down,
    right, left).

    Parameters
    ----------
    grid : list of tuples (x,y) of ints
        List of positions in the grid.

    Returns
    -------
    out : 2d-array of bools
        An array specifying, for each position, if moves will end up out of the borders of the game (True) or not (False).
    '''
    out_of_border = np.zeros((len(grid), 4), dtype=bool)
    for i, pos in enumerate(grid):
        x, y = pos

        # check up
        if y >= 0:
            if x < -1 or x > 1:
                out_of_border[i, 0] = True
            else:
                if y >= 2:
                    out_of_border[i, 0] = True

        # check down
        if y <= 0:
            if x < -1 or x > 1:
                out_of_border[i, 1] = True
            else:
                if y <= -2:
                    out_of_border[i, 1] = True

        # check right
        if x >= 0:
            if y < -1 or y > 1:
                out_of_border[i, 2] = True
            else:
                if x >= 2:
                    out_of_border[i, 2] = True

        # check left
        if x <= 0:
            if y < -1 or y > 1:
                out_of_border[i, 3] = True
            else:
                if x <= -2:
                    out_of_border[i, 3] = True

    return out_of_border


def _get_board_mask() -> np.array:
    """
    Returns the flattened version of a 2d bool array containing True at index (i,j) iff the position (i,j) is not out of
    the border of the board.
    :return: a bool array of shape (len(GRID)).
    """
    board_mask = np.zeros((7, 7), dtype=bool)
    for i, j in POS_TO_INDICES.values():
        board_mask[i, j] = True
    return board_mask.reshape(-1)


OUT_OF_BORDER_ACTIONS = _compute_out_of_border_actions(GRID)


class Env(object):
    """A class implementing the solitaire environment"""
    N_MAX_STEPS = N_PEGS
    _BOARD_MASK = _get_board_mask()

    def __init__(self,
                 verbose=False,
                 init_fig=False,
                 interactive_plot=False,
                 reward_mode="default",
                 mobility_alpha=0.1,
                 terminal_bonus_alpha=0.2,
                 potential_alpha=0.1,
                 potential_gamma=1.0,
                 potential_function="dispersion",
                 terminal_penalty_scale=1.0,
                 curriculum_phase1_end=0.3,
                 curriculum_phase2_end=0.7,
                 curriculum_phase1_mobility_alpha=0.1,
                 curriculum_phase1_terminal_bonus_alpha=0.0,
                 curriculum_phase2_mobility_alpha=0.05,
                 curriculum_phase2_terminal_bonus_alpha=0.1,
                 curriculum_phase3_mobility_alpha=0.0,
                 curriculum_phase3_terminal_bonus_alpha=0.2):
        '''
        Instantiates an object of the class Env by initializing the number of pegs in the game as well as their
        positions on the grid.

        Parameters
        ----------
        verbose : bool (default False)
            Whether or not to display messages.
        init_fig : bool (default False)
            Whether or not to initialize a figure for rendering.

        Attributes
        ----------
        n_pegs : int Number of pegs remaining on the board pegs : dict of tuples of ints Keys are the positions in the
        grid, and values are binary ints indicating the presence (1) or absence (0) of pegs. fig :
        matplotlib.figure.Figure The figure to render plots ax : matplotlib.axes.Axes Axes of the plot.
        '''
        self.n_pegs = N_PEGS
        self._init_pegs()
        if init_fig:
            self.init_fig(interactive_plot)
            pass
        else:
            self.interactive_plot = False
        self.verbose = verbose
        if reward_mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(f"Unsupported reward_mode {reward_mode}. Supported modes: {SUPPORTED_REWARD_MODES}")
        if potential_function not in SUPPORTED_POTENTIAL_FUNCTIONS:
            raise ValueError(
                f"Unsupported potential_function {potential_function}. "
                f"Supported functions: {SUPPORTED_POTENTIAL_FUNCTIONS}"
            )
        self.reward_mode = reward_mode
        self.mobility_alpha = mobility_alpha
        self.terminal_bonus_alpha = terminal_bonus_alpha
        self.potential_alpha = potential_alpha
        self.potential_gamma = potential_gamma
        self.potential_function = potential_function
        self.terminal_penalty_scale = terminal_penalty_scale
        self.curriculum_phase1_end = curriculum_phase1_end
        self.curriculum_phase2_end = curriculum_phase2_end
        self.curriculum_phase1_mobility_alpha = curriculum_phase1_mobility_alpha
        self.curriculum_phase1_terminal_bonus_alpha = curriculum_phase1_terminal_bonus_alpha
        self.curriculum_phase2_mobility_alpha = curriculum_phase2_mobility_alpha
        self.curriculum_phase2_terminal_bonus_alpha = curriculum_phase2_terminal_bonus_alpha
        self.curriculum_phase3_mobility_alpha = curriculum_phase3_mobility_alpha
        self.curriculum_phase3_terminal_bonus_alpha = curriculum_phase3_terminal_bonus_alpha
        self.training_progress = 0.0

    def _init_pegs(self):
        '''
        Initializes the positions of the pegs in the grid : puts a peg on each position except the center one (0,0).
        '''
        self.pegs = dict()
        for pos in GRID:
            self.pegs[pos] = 1
        self.pegs[(0, 0)] = 0

    def init_fig(self, interactive_plot=True):
        '''
        Initializes the figure and axes for the rendering.

        Parameters
        ----------
        interactive_plot : bool (default True)
            Whether the plot is interactive or not.
        '''
        if interactive_plot:
            plt.ion()
        self.fig = plt.figure(figsize=(10, 10))

    def reset(self):
        '''
        Resets the environment to its initial state.
        '''
        self.n_pegs = N_PEGS
        self._init_pegs()

    def set_training_progress(self, progress: float):
        self.training_progress = float(np.clip(progress, 0.0, 1.0))

    def step(self, action):
        '''
        Returns a length-3 tuple (reward, next_state, end), where reward is a float representing the reward by taking action `action`
        in the current state, nex_state is a representation of the next state after taking the action, and end is a boolean indicating
        whether or not the game has ended after the action (no more moves available).

        Parameters
        ----------
        action : tuple of ints (position_id, move_id)
            position_id indicates the id of the position on the grid according to the variable GRID. move_id is in {0,1,2,3} and
            indicates whether the move to make is up, down, right or left according to the variable ACTION_NAMES.

        Returns
        -------
        out : tuple (reward, next_state, end)
            reward is a float, next_state is a 2d-array, and end is a bool.
        '''
        mobility_before = self.count_feasible_actions()
        potential_before = self.get_potential()

        # update state of the env
        pos_id, move_id = action
        pos = GRID[pos_id]
        x, y = pos
        d_x, d_y = MOVES[move_id]
        self.pegs[pos] = 0  # peg moves from its current position
        self.pegs[(x + d_x, y + d_y)] = 0  # jumps over an adjacent peg, which is removed
        self.pegs[(x + 2 * d_x, y + 2 * d_y)] = 1  # initial peg ends up in new position
        self.n_pegs -= 1  # adjacent peg was removed

        end = False
        solved = False
        if self.n_pegs == 1:
            solved = True
            end = True
            if self.verbose:
                print('End of the game, you solved the puzzle !')
        elif self.count_feasible_actions() == 0:
            end = True
            if self.verbose:
                print('End of the game. You lost : {} pegs remaining'.format(self.n_pegs))

        mobility_after = self.count_feasible_actions()
        potential_after = self.get_potential()
        reward = self._compute_reward(mobility_before, mobility_after, potential_before, potential_after, end, solved)
        return reward, self.state, end

    @staticmethod
    def convert_action_id_to_action(action_index):
        return divmod(action_index, len(MOVES))

    @property
    def board_mask(self):
        return self._BOARD_MASK

    @property
    def state(self):
        '''
        Returns the state of the env as a 2d-array of ints. The state is represented as a 7x7 grid where values are 1
        if there is a peg at this position, and 0 otherwise (and 0 outside the board).
        '''
        state = np.zeros((7, 7, N_STATE_CHANNELS), dtype=np.float32)
        for pos, value in self.pegs.items():
            i, j = POS_TO_INDICES[pos]
            state[i, j, 0] = value

        state[:, :, 1] = (self.n_pegs - 1) / (N_PEGS - 1)  # percentage of pegs left to solve the game
        state[:, :, 2] = (N_PEGS - self.n_pegs) / (N_PEGS - 1)  # percentage of pegs already removed
        return state

    @property
    def feasible_actions(self):
        '''
        Returns a 2d-array of bools indicating, for each position on the grid, whether each action (up, down, right,
        left) is feasible (True) or not (False).
        '''
        feasible_actions = ~OUT_OF_BORDER_ACTIONS  # 2d bool array (len(GRID), len(MOVES))
        where_no_peg = np.array([self.pegs[pos] == 0 for pos in GRID])
        feasible_actions[where_no_peg, :] = False
        feasible = np.argwhere(feasible_actions)
        action_jump_feasible_np = np.vectorize(self.action_jump_feasible)
        feasible_actions[feasible[:, 0], feasible[:, 1]] = action_jump_feasible_np(feasible[:, 0], feasible[:, 1])
        return feasible_actions

    def count_feasible_actions(self) -> int:
        return int(np.sum(self.feasible_actions))

    def get_potential(self) -> float:
        remaining_progress = (N_PEGS - self.n_pegs) / (N_PEGS - 1)
        remaining_penalty = self.n_pegs / N_PEGS
        dispersion = self._peg_dispersion()

        if self.potential_function == "remaining_pegs":
            return remaining_progress
        if self.potential_function == "dispersion":
            return -dispersion
        if self.potential_function == "combined":
            return remaining_progress - remaining_penalty - dispersion
        raise ValueError(f"Unsupported potential function {self.potential_function}")

    def _compute_reward(self,
                        mobility_before: int,
                        mobility_after: int,
                        potential_before: float,
                        potential_after: float,
                        end: bool,
                        solved: bool) -> float:
        reward = self._base_reward(end=end, solved=solved)

        if self.reward_mode == "default":
            return reward
        if self.reward_mode == "mobility":
            mobility_delta = (mobility_after - mobility_before) / N_ACTIONS
            return reward + self.mobility_alpha * mobility_delta
        if self.reward_mode == "potential":
            shaping = self.potential_gamma * potential_after - potential_before
            return reward + self.potential_alpha * shaping
        if self.reward_mode == "terminal_diff":
            return self._terminal_diff_reward(end=end, solved=solved, base_reward=reward)
        if self.reward_mode == "hybrid_curriculum":
            mobility_alpha, terminal_bonus_alpha = self._curriculum_coefficients()
            mobility_delta = (mobility_after - mobility_before) / N_ACTIONS
            shaped_reward = reward + mobility_alpha * mobility_delta
            if end:
                shaped_reward += terminal_bonus_alpha * self._terminal_score()
            return shaped_reward
        raise ValueError(f"Unsupported reward mode {self.reward_mode}")

    @staticmethod
    def _base_reward(end: bool, solved: bool) -> float:
        if solved:
            return 1.0
        return DEFAULT_STEP_REWARD

    def _terminal_diff_reward(self, end: bool, solved: bool, base_reward: float) -> float:
        if solved:
            return 1.0
        if not end:
            return base_reward
        return self.terminal_penalty_scale * self._terminal_score()

    def _terminal_score(self) -> float:
        return (N_PEGS - self.n_pegs) / (N_PEGS - 1)

    def _curriculum_coefficients(self) -> tuple[float, float]:
        if self.training_progress < self.curriculum_phase1_end:
            return self.curriculum_phase1_mobility_alpha, self.curriculum_phase1_terminal_bonus_alpha
        if self.training_progress < self.curriculum_phase2_end:
            return self.curriculum_phase2_mobility_alpha, self.curriculum_phase2_terminal_bonus_alpha
        return self.curriculum_phase3_mobility_alpha, self.curriculum_phase3_terminal_bonus_alpha

    def _peg_dispersion(self) -> float:
        peg_positions = np.array([pos for pos, value in self.pegs.items() if value == 1], dtype=np.float32)
        if len(peg_positions) <= 1:
            return 0.0
        center = np.mean(peg_positions, axis=0, keepdims=True)
        sq_distances = np.sum((peg_positions - center) ** 2, axis=1)
        max_sq_distance = 18.0  # max squared distance on the board from the center is 3^2 + 3^2
        return float(np.mean(sq_distances) / max_sq_distance)

    def action_jump_feasible(self, pos_index, move_id):
        '''
        Returns a boolean indicating whether or not the move of id move_id (in [0,1,2,3]) of jumping over a peg is
        feasible at position pos in the grid. The move is feasible if there is a peg to jump over and if there is no
        peg on the potential arrival position in the grid.

        Parameters
        ----------
        pos_index : int
            The index of the considered position in the grid.
        move_id : int
            Gives the id of the move considered.

        Returns
        -------
        out : bool
            A boolean indicating if the move is feasible or not.
        '''
        x, y = GRID[pos_index]
        d_x, d_y = MOVES[move_id]

        return self.pegs[(x + d_x, y + d_y)] == 1 and self.pegs[(x + 2 * d_x, y + 2 * d_y)] == 0

    def render(self, action=None, show_action=False, show_axes=False):
        '''
        Renders the current state of the environment.

        Parameters
        ----------
        action : tuple of ints or None (default None)
            If not None, a tuple (pos_id, move_id) of ints indicating the id of the position in the grid, and the id of the move.
        show_action : bool (default False)
            Indicates whether or not to change the color of the peg being moved and the peg being jumped over in the rendering, in
            order to be able to visualise the action selected.
        show_axes : bool (default False)
            Whether to display the axes in the rendering or not.
        '''
        ax = plt.gca()
        # 先清除上一帧的 patches，再绘制新的，避免闪烁
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
            for pos, value in self.pegs.items():
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
            for pos, value in self.pegs.items():
                if value == 1:
                    ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=True))
                if value == 0:
                    ax.add_patch(
                        matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=False, linewidth=1.5))

        plt.ylim(-4, 4)
        plt.xlim(-4, 4)
        plt.axis('scaled')
        plt.title('Current State of the Board')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
