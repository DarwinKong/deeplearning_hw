"""
从已保存的 checkpoint 加载 agent 并运行游戏演示。
支持 A2C (actor_critic) 和 PPO 两种算法。

用法:
    python3 play.py                         # 使用最新 checkpoint，渲染游戏画面
    python3 play.py --agent ppo            # 使用 PPO 算法
    python3 play.py --no-render             # 不渲染，只打印结果
    python3 play.py --n-games 5            # 运行 5 局游戏
    python3 play.py --checkpoint <路径>    # 指定 checkpoint 文件
    python3 play.py --remote               # 从远程目录加载（被 git 提交的）
"""
import os
import sys
import glob
import argparse
import numpy as np

# 将 rl-solitaire 目录加入路径（兼容从根目录运行）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl-solitaire'))

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

from env.env import Env
from nn.network_config import NetConfig
from nn.policy_value.fully_connected import FCPolicyValueNet
from agents.actor_critic.actor_critic_agent import ActorCriticAgent
from agents.ppo.ppo_agent import PPOAgent
from utils.tools import read_yaml


# Checkpoints directory structure
CHECKPOINTS_BASE_DIR = "./checkpoints"
LOCAL_CHECKPOINTS_SUBDIR = "local"
REMOTE_CHECKPOINTS_SUBDIR = "remote"

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

NETWORK_CONFIG_PATH = "./rl-solitaire/nn/policy_value/fc_policy_value_config.yaml"


def find_latest_checkpoint(agent_name: str = 'actor_critic', use_remote: bool = False) -> str:
    """在指定算法的 checkpoints 目录中找到最新的 checkpoint 文件"""
    if agent_name not in SUPPORTED_AGENTS:
        raise ValueError(f"不支持的算法: {agent_name}。支持的算法: {list(SUPPORTED_AGENTS.keys())}")
    
    checkpoint_subdir = REMOTE_CHECKPOINTS_SUBDIR if use_remote else LOCAL_CHECKPOINTS_SUBDIR
    checkpoints_dir = os.path.join(CHECKPOINTS_BASE_DIR, agent_name, checkpoint_subdir)
    
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
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    local_config = glob.glob(os.path.join(run_dir, "fc_policy_value_config.yaml"))
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


def main():
    parser = argparse.ArgumentParser(description="从 checkpoint 加载 Agent 并运行游戏")
    parser.add_argument("--agent", type=str, default='actor_critic',
                        choices=['actor_critic', 'ppo'],
                        help="使用的算法: actor_critic (A2C) 或 ppo（默认 actor_critic）")
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

    # 确定 checkpoint 路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"错误：checkpoint 文件不存在: {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path = find_latest_checkpoint(agent_name=args.agent, use_remote=args.remote)

    location = "远程" if args.remote else "本地"
    print(f"算法: {args.agent.upper()}")
    print(f"Checkpoint 位置: {location}")
    print(f"加载 checkpoint: {checkpoint_path}")

    # 加载 agent
    agent = load_agent(checkpoint_path, agent_name=args.agent)
    print(f"Agent 加载成功：{agent.name}\n")

    # 运行游戏
    render = not args.no_render
    play_games(agent, n_games=args.n_games, render=render, greedy=args.greedy)


if __name__ == "__main__":
    main()
