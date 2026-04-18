import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sourceTorch.env.constants import N_ACTIONS
from sourceTorch.env.reward import compute_batched_rewards


def test_mobility_reward_matches_legacy_normalized_formula():
    rewards = compute_batched_rewards(
        n_pegs_before=torch.tensor([10]),
        n_pegs_after=torch.tensor([9]),
        is_terminal=torch.tensor([False]),
        reward_mode="mobility",
        mobility_before=torch.tensor([8]),
        mobility_after=torch.tensor([10]),
        mobility_alpha=0.1,
        mobility_normalize=True,
    )

    expected = (1.0 / 31.0) + 0.1 * ((10 - 8) / N_ACTIONS)
    assert torch.allclose(rewards, torch.tensor([expected], dtype=torch.float32))


def test_mobility_reward_keeps_existing_unnormalized_behavior_by_default():
    rewards = compute_batched_rewards(
        n_pegs_before=torch.tensor([10]),
        n_pegs_after=torch.tensor([9]),
        is_terminal=torch.tensor([False]),
        reward_mode="mobility",
        mobility_before=torch.tensor([8]),
        mobility_after=torch.tensor([10]),
        mobility_alpha=0.1,
    )

    expected = (1.0 / 31.0) + 0.1 * (10 - 8)
    assert torch.allclose(rewards, torch.tensor([expected], dtype=torch.float32))
