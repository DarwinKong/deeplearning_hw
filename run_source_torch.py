import os
from datetime import datetime
from pathlib import Path

import click
import torch
import yaml

from sourceTorch.agent import A2CAgent, PPOAgent
from sourceTorch.nn.network_config import NetConfig
from sourceTorch.nn.policy_value.fully_connected import FCPolicyValueNet
from sourceTorch.nn.policy_value.conv import ConvPolicyValueNet
from sourceTorch.nn.policy_value.transformer import TransformerPolicyValueNet
from sourceTorch.trainers import BatchedGPUTrainer
from sourceTorch.utils.path_config import path_config
from sourceTorch.utils.tools import read_yaml, set_random_seeds, set_up_logger, strp_datetime


NETWORKS = {
    "fc_policy_value": FCPolicyValueNet,
    "conv_policy_value": ConvPolicyValueNet,
    "transformer_policy_value": TransformerPolicyValueNet,
}


def resolve_device(device_name: str = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_algorithm(agent_name: str, network: torch.nn.Module, trainer_config: dict):
    if agent_name == "actor_critic":
        return A2CAgent(
            network=network,
            discount=trainer_config.get("discount", 0.99),
            use_gae=trainer_config.get("use_gae", True),
            gae_lambda=trainer_config.get("gae_lambda", 0.95),
            normalize_advantages=trainer_config.get("normalize_advantages", True),
        )
    if agent_name == "ppo":
        ppo_config = trainer_config.get("ppo", {}) or {}
        return PPOAgent(
            network=network,
            clip_epsilon=ppo_config.get("clip_epsilon", 0.2),
            value_loss_coef=ppo_config.get("value_loss_coef", 0.5),
            entropy_coef=ppo_config.get("entropy_coef", 0.01),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            discount=ppo_config.get("discount", 0.99),
            use_gae=ppo_config.get("use_gae", True),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            normalize_advantages=ppo_config.get("normalize_advantages", True),
        )
    raise ValueError(f"Unsupported agent {agent_name}")


def resolve_network(network_name: str, network_config_path: str, device: torch.device):
    if network_name not in NETWORKS:
        raise ValueError(f"Unsupported network {network_name}. Supported: {list(NETWORKS)}")
    network_config_dict = read_yaml(network_config_path)
    network = NETWORKS[network_name](NetConfig(config_dict=network_config_dict)).to(device)
    return network, network_config_dict


def copy_yaml(target_dir: str, filename: str, data: dict):
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, filename), "w") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def build_experiment_name(agent_name: str, trainer_config_path: str, reward_mode: str, timestamp: str) -> str:
    agent_abbr = {
        "actor_critic": "A2C",
        "ppo": "PPO",
    }.get(agent_name, agent_name.upper())
    config_stem = Path(trainer_config_path).stem
    suffix = config_stem
    suffix = suffix.replace("actor-critic-trainer-", "")
    suffix = suffix.replace("ppo-trainer-", "")
    suffix = suffix.replace("-config", "")
    suffix = suffix.replace("-reward", "")
    suffix = suffix.replace("-", "_")
    if not suffix:
        suffix = (reward_mode or "default").replace("-", "_")
    return f"{agent_abbr}_{suffix}_{timestamp}"


@click.command()
@click.option("-an", "--agent_name", required=True, type=click.STRING)
@click.option("-nn", "--network_name", default="fc_policy_value", type=click.STRING)
@click.option("--trainer_config", required=True, type=click.Path(exists=True))
@click.option("--network_config", default=None, type=click.Path(exists=True))
@click.option("--device", default=None, type=click.STRING)
@click.option("--n_iter_override", default=None, type=click.INT)
def main(agent_name: str,
         network_name: str,
         trainer_config: str,
         network_config: str,
         device: str,
         n_iter_override: int):
    trainer_config_dict = read_yaml(trainer_config)
    network_config_path = network_config or path_config.get_nn_config_path(network_name)
    device_obj = resolve_device(device)

    seed = trainer_config_dict.get("seed")
    set_random_seeds(42 if seed is None else seed)

    reward_config = trainer_config_dict.get("env", {}) or {}
    reward_mode = reward_config.get("reward_mode", "default")
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    experiment_name = build_experiment_name(agent_name, trainer_config, reward_mode, timestamp)

    network, network_config_dict = resolve_network(network_name, network_config_path, device_obj)
    algorithm = resolve_algorithm(agent_name, network, trainer_config_dict)

    experiment_dir = os.path.join(path_config.ckps_base_dir, path_config.ckps_local_subdir, experiment_name)
    meta_dir = os.path.join(experiment_dir, path_config.ckps_meta_dir)
    logs_dir = os.path.join(experiment_dir, path_config.ckps_logs_dir)
    checkpoints_dir = os.path.join(experiment_dir, path_config.ckps_checkpoints_dir)
    results_dir = os.path.join(experiment_dir, path_config.ckps_results_dir)
    os.makedirs(logs_dir, exist_ok=True)

    logger = set_up_logger(os.path.join(logs_dir, "log.txt"))

    copy_yaml(meta_dir, os.path.basename(trainer_config), trainer_config_dict)
    copy_yaml(meta_dir, os.path.basename(network_config_path), network_config_dict)

    optimizer_cfg = network_config_dict.get("optimizer", {}) or {}
    trainer = BatchedGPUTrainer(
        n_envs=trainer_config_dict.get("n_envs", trainer_config_dict.get("n_games_train", 64)),
        algorithm=algorithm,
        n_iter=n_iter_override or trainer_config_dict.get("n_iter", 200),
        n_steps_per_env=trainer_config_dict.get("n_steps_per_env", trainer_config_dict.get("n_steps_update", 32) or 32),
        agent_results_filepath=os.path.join(results_dir, "training_history.pt"),
        learning_rate=trainer_config_dict.get("learning_rate", optimizer_cfg.get("lr", 3e-5)),
        batch_size=trainer_config_dict.get("batch_size", 256),
        n_optim_steps=trainer_config_dict.get("n_optim_steps", 4 if agent_name == "ppo" else 1) or (4 if agent_name == "ppo" else 1),
        log_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        meta_dir=meta_dir,
        results_dir=results_dir,
        enable_monitors=trainer_config_dict.get("enable_monitors", True),
        network_config=NetConfig(config_dict=network_config_dict),
        reward_config=reward_config,
    )

    logger.info(f"Running sourceTorch experiment at {experiment_dir}")
    logger.info(f"Agent={agent_name}, Network={network_name}, Device={device_obj}")
    logger.info(f"Reward config: {trainer.reward_config}")

    trainer.train()


if __name__ == "__main__":
    main()
