import os
from datetime import datetime
from pytorch_lightning import seed_everything
import click
import yaml

# 将项目根目录加入路径（兼容从任意位置运行）
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from source.env.env import Env
from source.utils.tools import strp_datetime, set_up_logger, read_yaml
from source.utils.path_config import path_config
from source.nn.utils import get_network_class_from_name
from source.nn.network_config import NetConfig
from source.agents.utils import get_class_from_name


ROOT = "./"

RUNS_DIRNAME = "runs"
RUN_CONFIG_FILENAME = "run_config.yaml"
LOG_FILENAME = "log.txt"
RESULTS_FILENAME = "agent_results.pickle"
SEED = 42
DEFAULT_DISCOUNT_FACTOR = 1.0


@click.command()
@click.option('-an', '--agent_name', required=True, type=click.STRING, help='Agent name: "actor_critic" (A2C) or "ppo"')
@click.option('-nn', '--network_name', required=False, type=click.STRING, default='fc_policy_value',help='Network architecture: "fc_policy_value", "conv_policy_value", or "transformer_policy_value"')
@click.option('--remote', is_flag=True, default=False,help='Save checkpoints to remote directory (committed to git). Default: save to local directory.')
@click.option('--trainer_config', required=False, type=click.Path(path_type=str),
              help='Optional trainer config path. Supports custom reward experiment configs.')
def main(agent_name: str, network_name: str = None, remote: bool = False, trainer_config: str = None):
    """RL Solitaire 训练入口"""
    run(agent_name=agent_name,
        network_name=network_name,
        use_remote_checkpoints=remote,
        trainer_config_path=trainer_config)


def run(agent_name: str,
        network_name: str = None,
        use_remote_checkpoints: bool = False,
        trainer_config_path: str = None):
    # 生成实验名称和时间戳
    timestamp = strp_datetime(datetime.now())
    experiment_name = path_config.get_experiment_name(agent_name, timestamp)
    
    # 创建所有子目录（local）
    meta_dir = path_config.get_meta_dir(agent_name, use_remote=False, timestamp=timestamp)
    logs_dir = path_config.get_logs_dir(agent_name, use_remote=False, timestamp=timestamp)
    checkpoints_dir = path_config.get_checkpoints_dir(agent_name, use_remote=False, timestamp=timestamp)
    results_dir = path_config.get_results_dir(agent_name, use_remote=False, timestamp=timestamp)
    
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 日志文件路径
    log_filepath = os.path.join(logs_dir, LOG_FILENAME)
    results_filepath = os.path.join(results_dir, RESULTS_FILENAME)
    logger = set_up_logger(path=log_filepath)

    # trainer config - 从 path_config 读取路径
    trainer_config_filepath = resolve_trainer_config_path(agent_name, trainer_config_path)
    trainer_config = read_yaml(trainer_config_filepath)
    with open(os.path.join(meta_dir, os.path.basename(trainer_config_filepath)), 'w') as file:
        yaml.safe_dump(trainer_config, file)
    env_config = trainer_config.pop("env", {}) or {}

    # set seed
    seed = get_seed(trainer_config)
    seed_everything(seed)

    # define network
    if network_name is None:
        network = None
    else:
        # 从 path_config 读取网络配置路径
        network_config_filepath = path_config.get_nn_config_path(network_name)
        network_config_dict = read_yaml(network_config_filepath)
        config_filename = os.path.basename(network_config_filepath)
        with open(os.path.join(meta_dir, config_filename), 'w') as file:
            yaml.safe_dump(network_config_dict, file)
        network_config = NetConfig(config_dict=network_config_dict)
        network_class = get_network_class_from_name(network_name)
        network = network_class(network_config)

    # define agent
    agent_class = get_class_from_name(agent_name, class_type="agent")
    discount_factor = get_discount_factor(trainer_config)
    if network is None:
        agent = agent_class(discount=discount_factor)
    else:
        full_agent_name = f"{network.name}-{agent_class.__name__}"
        agent = agent_class(network=network, name=full_agent_name, discount=discount_factor)

    # define trainer
    trainer_class = get_class_from_name(agent_name, class_type="trainer")
    trainer = trainer_class(
        env=Env(**env_config),
        agent=agent, 
        agent_results_filepath=results_filepath,
        log_dir=logs_dir,  # 使用 logs_dir
        checkpoints_dir=checkpoints_dir,
        remote_checkpoints_dir=None,  # 不再需要单独的 remote dir
        meta_dir=meta_dir,
        results_dir=results_dir,
        **trainer_config
    )

    # log run parameters
    logger.info(f"---------  Running experiment with agent {agent_name} and network {network_name} ---------")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Saving run results and logs at {path_config.get_experiment_dir(agent_name, use_remote=False, timestamp=timestamp)}")
    logger.info(f"Running with random seed {seed}")
    logger.info(f"Running with discount factor {discount_factor}")
    logger.info(f"Environment reward config: {env_config if env_config else {'reward_mode': 'default'}}")

    trainer.train()



def resolve_trainer_config_path(agent_name: str, trainer_config_path: str = None) -> str:
    if trainer_config_path is None:
        return path_config.get_agent_trainer_config_path(agent_name)

    if os.path.isabs(trainer_config_path):
        return trainer_config_path

    candidate_paths = [
        os.path.abspath(trainer_config_path),
        os.path.join(path_config.agent_trainer_config_dir, trainer_config_path),
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Trainer config not found: {trainer_config_path}")


def get_seed(config_dict: dict) -> int:
    if "seed" not in config_dict:
        return SEED
    elif config_dict["seed"] is None:
        config_dict.pop("seed")
        return SEED
    else:
        return config_dict.pop("seed")


def get_discount_factor(config_dict: dict) -> float:
    if "discount" not in config_dict.keys():
        return DEFAULT_DISCOUNT_FACTOR
    elif config_dict["discount"] is None:
        config_dict.pop("discount")
        return DEFAULT_DISCOUNT_FACTOR
    else:
        return config_dict.pop("discount")


if __name__ == "__main__":
    # 使用方法 1: 命令行运行（推荐）
    #   python run.py -an actor_critic -nn fc_policy_value
    #   python run.py -an ppo -nn fc_policy_value
    #   python run.py -an ppo --remote  # 保存到远程目录（会被 git 提交）
    #
    # 使用方法 2: 直接修改下面的参数并运行
    #   python run.py
    
    # 默认配置（当直接运行时使用）
    network_name = "fc_policy_value"  # 网络架构
    agent_name = 'actor_critic'        # 算法: 'actor_critic' (A2C) 或 'ppo'
    use_remote = False                 # 是否保存到远程目录
    
    # 如果是通过命令行调用，Click 会处理；否则直接运行
    import sys
    if len(sys.argv) == 1:
        # 没有命令行参数，使用默认配置
        run(agent_name=agent_name, network_name=network_name, use_remote_checkpoints=use_remote)
    else:
        # 有命令行参数，使用 Click
        main()
