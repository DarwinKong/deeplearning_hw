"""
从已保存的 checkpoint 加载 agent 并运行游戏演示。
支持 A2C (actor_critic) 和 PPO 两种算法。

用法:
    python3 play.py                                         # 使用最新 checkpoint，渲染游戏画面
    python3 play.py --agent ppo                            # 使用 PPO 算法
    python3 play.py --no-render                             # 不渲染，只打印结果
    python3 play.py --n-games 5                            # 运行 5 局游戏
    python3 play.py --experiment checkpoints-and-logs/local/A2C_2026_04_14-21_00  # 指定实验目录
    python3 play.py --checkpoint <路径>                    # 指定 checkpoint 文件
    python3 play.py --remote                               # 从远程目录加载（被 git 提交的）
"""
import os
import sys
import glob
import argparse
import numpy as np

# 将项目根目录加入路径（兼容从任意位置运行）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 跨平台兼容的 matplotlib backend 设置
import platform
import matplotlib
system = platform.system()
if system == 'Darwin':  # macOS
    matplotlib.use('macosx')
elif system == 'Windows':  # Windows
    matplotlib.use('TkAgg')
else:  # Linux
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from source.env.env import Env
from source.nn.network_config import NetConfig
from source.nn.policy_value.fully_connected import FCPolicyValueNet
from source.agents.actor_critic.actor_critic_agent import ActorCriticAgent
from source.agents.ppo.ppo_agent import PPOAgent
from source.utils.tools import read_yaml
from source.utils.path_config import path_config


# Checkpoints directory structure - 从 path_config 读取
# 不再硬编码，统一由 path_config 管理

# 支持的算法及其配置
SUPPORTED_AGENTS = {
    'actor_critic': {
        'agent_class': ActorCriticAgent,
        'agent_name': 'ActorCriticAgent'
    },
    'ppo': {
        'agent_class': PPOAgent,
        'agent_name': 'PPOAgent'
    }
}

# 默认网络配置路径 - 从 path_config 读取
NETWORK_CONFIG_PATH = path_config.get_nn_config_path('fc_policy_value')


def find_latest_checkpoint(agent_name: str = 'actor_critic', use_remote: bool = False, experiment_dir: str = None) -> str:
    """在指定算法的 checkpoints 目录中找到最新的 checkpoint 文件"""
    if agent_name not in SUPPORTED_AGENTS:
        raise ValueError(f"不支持的算法: {agent_name}。支持的算法: {list(SUPPORTED_AGENTS.keys())}")
    
    # 如果指定了实验目录，直接使用该目录下的 checkpoints
    if experiment_dir:
        if not os.path.exists(experiment_dir):
            raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")
        checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    else:
        # 从 path_config 获取 checkpoints 目录
        checkpoints_dir = path_config.get_checkpoints_dir(agent_name, use_remote=use_remote)
    
    pattern = os.path.join(checkpoints_dir, "*.ckpt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        dir_type = "远程" if use_remote else "本地"
        raise FileNotFoundError(f"在 {checkpoints_dir} ({dir_type}) 下未找到任何 checkpoint 文件，请先运行训练。")
    # 按文件修改时间排序，取最新的
    latest = max(checkpoints, key=os.path.getmtime)
    return latest


def load_agent(checkpoint_path: str, agent_name: str = 'actor_critic'):
    """从 checkpoint 加载 Agent（支持 A2C 和 PPO）"""
    if agent_name not in SUPPORTED_AGENTS:
        raise ValueError(f"不支持的算法: {agent_name}。支持的算法: {list(SUPPORTED_AGENTS.keys())}")
    
    agent_class = SUPPORTED_AGENTS[agent_name]['agent_class']
    base_agent_name = SUPPORTED_AGENTS[agent_name]['agent_name']
    
    # 优先从 checkpoint 所在目录读取 network config，否则用默认配置
    # 新结构: checkpoints-and-logs/local/{AGENT}_{timestamp}/checkpoints/*.ckpt
    # 需要向上两级到实验根目录，然后进入 meta/
    experiment_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    local_config = glob.glob(os.path.join(experiment_dir, "meta", "*policy-value-config.yaml"))
    config_path = local_config[0] if local_config else NETWORK_CONFIG_PATH

    network_config_dict = read_yaml(config_path)
    network_config = NetConfig(config_dict=network_config_dict)

    # 从 checkpoint 加载网络权重
    network = FCPolicyValueNet.load_from_checkpoint(checkpoint_path, config=network_config)
    network.eval()

    agent = agent_class(network=network, name=f"fc_policy_value-{base_agent_name}")
    agent.set_evaluation_mode()
    return agent


def play_games(agent: ActorCriticAgent, n_games: int = 1, render: bool = True, greedy: bool = True):
    """运行指定局数的游戏"""
    env = Env()
    rewards = []
    pegs_left_list = []

    for i in range(n_games):
        env.reset()
        reward, n_pegs = agent.play(env, render=render, greedy=greedy)
        rewards.append(reward)
        pegs_left_list.append(n_pegs)
        print(f"局 {i+1:>3d}/{n_games}: reward = {reward:.3f}, pegs left = {n_pegs}")

    print("\n===== 汇总结果 =====")
    print(f"平均 reward    : {np.mean(rewards):.3f}")
    print(f"平均剩余棋子数  : {np.mean(pegs_left_list):.2f}")
    print(f"最少剩余棋子数  : {np.min(pegs_left_list)}")
    print(f"最多剩余棋子数  : {np.max(pegs_left_list)}")


def play(experiment: str = None, 
         agent: str = 'actor_critic',
         checkpoint: str = None,
         n_games: int = 1,
         render: bool = True,
         greedy: bool = True,
         remote: bool = False):
    """
    从 checkpoint 加载 Agent 并运行游戏（脚本模式）
    
    Args:
        experiment: 实验目录路径（例如: checkpoints-and-logs/local/A2C_2026_04_14-21_00）
        agent: 使用的算法 ('actor_critic' 或 'ppo')
        checkpoint: checkpoint 文件路径（不指定则自动使用最新）
        n_games: 游戏局数
        render: 是否渲染游戏画面
        greedy: 是否使用贪心策略
        remote: 是否从远程目录加载
    """
    # 确定 checkpoint 路径
    if checkpoint:
        checkpoint_path = checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"错误：checkpoint 文件不存在: {checkpoint_path}")
            return
    else:
        checkpoint_path = find_latest_checkpoint(
            agent_name=agent, 
            use_remote=remote,
            experiment_dir=experiment
        )

    location = "远程" if remote else ("实验目录" if experiment else "本地")
    print(f"算法: {agent.upper()}")
    print(f"Checkpoint 位置: {location}")
    print(f"加载 checkpoint: {checkpoint_path}")

    # 加载 agent
    agent_obj = load_agent(checkpoint_path, agent_name=agent)
    print(f"Agent 加载成功：{agent_obj.name}\n")

    # 运行游戏
    play_games(agent_obj, n_games=n_games, render=render, greedy=greedy)


def main():
    parser = argparse.ArgumentParser(description="从 checkpoint 加载 Agent 并运行游戏")
    parser.add_argument("--agent", type=str, default='actor_critic',
                        choices=['actor_critic', 'ppo'],
                        help="使用的算法: actor_critic (A2C) 或 ppo（默认 actor_critic）")
    parser.add_argument("--experiment", type=str, default=None,
                        help="实验目录路径（例如: checkpoints-and-logs/local/A2C_2026_04_14-21_00）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 文件路径（不指定则自动使用最新）")
    parser.add_argument("--n-games", type=int, default=1,
                        help="游戏局数（默认 1）")
    parser.add_argument("--no-render", action="store_true",
                        help="不渲染游戏画面，只打印数值结果")
    parser.add_argument("--greedy", action="store_true", default=True,
                        help="是否使用贪心策略选择动作（默认 True）")
    parser.add_argument("--remote", action="store_true", default=False,
                        help="从远程目录加载 checkpoint（被 git 提交的）")
    args = parser.parse_args()

    # 调用 play 函数
    play(
        experiment=args.experiment,
        agent=args.agent,
        checkpoint=args.checkpoint,
        n_games=args.n_games,
        render=not args.no_render,
        greedy=args.greedy,
        remote=args.remote
    )


if __name__ == "__main__":
    # 使用方法 1: 命令行运行（推荐）
    #   python play.py --experiment checkpoints-and-logs/remote/A2C_2026_04_14-21_00 --n-games 5
    #   python play.py --agent ppo --n-games 3
    #   python play.py --agent actor_critic --no-render
    #
    # 使用方法 2: 脚本模式（在 Python 代码中调用）
    #   play(experiment="checkpoints-and-logs/remote/A2C_2026_04_14-21_00", n_games=5)
    #   play(agent="ppo", n_games=3, render=False)
    
    # 如果是通过命令行调用，argparse 会处理；否则直接运行
    import sys
    if len(sys.argv) == 1:
        # 没有命令行参数，使用默认配置
        play(agent='actor_critic',experiment="checkpoints-and-logs/remote/A2C_2026_04_14-21_00", n_games=1, render=True)
    else:
        # 有命令行参数，使用 argparse
        main()
