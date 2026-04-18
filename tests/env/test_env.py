import numpy as np
import unittest

from source.env.env import ACTIONS, DEFAULT_STEP_REWARD, Env, GRID, MOVES, N_ACTIONS, OUT_OF_BORDER_ACTIONS
from source.agents.random_agent import RandomAgent


class TestEnv(unittest.TestCase):
    def setUp(self) -> None:
        pass

    @staticmethod
    def _set_board(env: Env, occupied_positions):
        env.pegs = {pos: 0 for pos in GRID}
        for pos in occupied_positions:
            env.pegs[pos] = 1
        env.n_pegs = len(occupied_positions)

    def test_constant_definitions(self):
        self.assertTrue(N_ACTIONS == len(ACTIONS) == len(GRID) * len(MOVES) == OUT_OF_BORDER_ACTIONS.size)

    def test_feasible_actions(self):
        def test_feasible_actions_(env, feasible_actions):
            feasible_actions_ = np.ones(shape=(len(GRID), len(MOVES)), dtype=bool)
            for i, pos in enumerate(GRID):
                if env.pegs[pos] == 0:
                    feasible_actions_[i, :] = False
                else:
                    for move_id in range(len(MOVES)):
                        if OUT_OF_BORDER_ACTIONS[i, move_id]:
                            feasible_actions_[i, move_id] = False
                        else:
                            if not env.action_jump_feasible(i, move_id):
                                feasible_actions_[i, move_id] = False

            np.testing.assert_array_equal(feasible_actions, feasible_actions_)

        random_agent = RandomAgent()
        env = Env()
        end = False
        cmpt = 0
        while not end:
            feasible_actions = env.feasible_actions
            test_feasible_actions_(env, feasible_actions)
            action_index = random_agent.select_action(env.state, feasible_actions)
            action = env.convert_action_id_to_action(action_index)
            reward, next_state, end = env.step(action)
            cmpt += 1

    def test_default_reward_matches_previous_behavior(self):
        env = Env()
        action = tuple(np.argwhere(env.feasible_actions)[0])

        reward, _, end = env.step(action)

        self.assertFalse(end)
        self.assertAlmostEqual(reward, DEFAULT_STEP_REWARD)

    def test_mobility_reward_adds_normalized_action_delta(self):
        env = Env(reward_mode="mobility", mobility_alpha=0.5)
        action = tuple(np.argwhere(env.feasible_actions)[0])
        mobility_before = env.count_feasible_actions()

        reward, _, _ = env.step(action)

        mobility_after = env.count_feasible_actions()
        expected_reward = DEFAULT_STEP_REWARD + 0.5 * ((mobility_after - mobility_before) / N_ACTIONS)
        self.assertAlmostEqual(reward, expected_reward)

    def test_potential_reward_uses_potential_difference(self):
        env = Env(reward_mode="potential",
                  potential_alpha=0.25,
                  potential_gamma=0.9,
                  potential_function="remaining_pegs")
        action = tuple(np.argwhere(env.feasible_actions)[0])
        potential_before = env.get_potential()

        reward, _, _ = env.step(action)

        potential_after = env.get_potential()
        expected_reward = DEFAULT_STEP_REWARD + 0.25 * (0.9 * potential_after - potential_before)
        self.assertAlmostEqual(reward, expected_reward)

    def test_terminal_diff_reward_distinguishes_failed_terminal_states(self):
        env = Env(reward_mode="terminal_diff", terminal_penalty_scale=1.0)
        self._set_board(env, occupied_positions=[(-3, 0), (-2, 0), (3, 0)])
        action = (GRID.index((-3, 0)), MOVES.index((1, 0)))

        reward, _, end = env.step(action)

        self.assertTrue(end)
        self.assertEqual(env.n_pegs, 2)
        self.assertAlmostEqual(reward, (32 - 2) / 31)

    def test_terminal_diff_keeps_success_reward_at_one(self):
        env = Env(reward_mode="terminal_diff")
        self._set_board(env, occupied_positions=[(-3, 0), (-2, 0)])
        action = (GRID.index((-3, 0)), MOVES.index((1, 0)))

        reward, _, end = env.step(action)

        self.assertTrue(end)
        self.assertEqual(env.n_pegs, 1)
        self.assertEqual(reward, 1.0)

    def test_hybrid_curriculum_uses_phase_one_mobility_bonus(self):
        env = Env(reward_mode="hybrid_curriculum",
                  curriculum_phase1_mobility_alpha=0.1,
                  curriculum_phase1_terminal_bonus_alpha=0.0,
                  curriculum_phase2_mobility_alpha=0.0,
                  curriculum_phase2_terminal_bonus_alpha=0.0,
                  curriculum_phase3_mobility_alpha=0.0,
                  curriculum_phase3_terminal_bonus_alpha=0.0)
        env.set_training_progress(0.1)
        action = tuple(np.argwhere(env.feasible_actions)[0])
        mobility_before = env.count_feasible_actions()

        reward, _, _ = env.step(action)

        mobility_after = env.count_feasible_actions()
        expected_reward = DEFAULT_STEP_REWARD + 0.1 * ((mobility_after - mobility_before) / N_ACTIONS)
        self.assertAlmostEqual(reward, expected_reward)

    def test_hybrid_curriculum_adds_terminal_bonus_in_late_phase(self):
        env = Env(reward_mode="hybrid_curriculum",
                  curriculum_phase1_mobility_alpha=0.0,
                  curriculum_phase1_terminal_bonus_alpha=0.0,
                  curriculum_phase2_mobility_alpha=0.0,
                  curriculum_phase2_terminal_bonus_alpha=0.0,
                  curriculum_phase3_mobility_alpha=0.0,
                  curriculum_phase3_terminal_bonus_alpha=0.2)
        env.set_training_progress(0.9)
        self._set_board(env, occupied_positions=[(-3, 0), (-2, 0), (3, 0)])
        action = (GRID.index((-3, 0)), MOVES.index((1, 0)))

        reward, _, end = env.step(action)

        self.assertTrue(end)
        self.assertAlmostEqual(reward, DEFAULT_STEP_REWARD + 0.2 * ((32 - 2) / 31))


if __name__ == '__main__':
    unittest.main()
