"""
从已保存的 checkpoint 加载 actor_critic agent 并运行游戏演示。
用法:
    python3 play.py                         # 使用最新 checkpoint，渲染游戏画面
    python3 play.py --no-render             # 不渲染，只打印结果
    python3 play.py --n-games 5            # 运行 5 局游戏
    python3 play.py --checkpoint <路径>    # 指定 checkpoint 文件
"""
import os
import sys
import glob
import argparse
import numpy as np

# macOS 上必须在 import pyplot 之前设置后端，否则后台进程无法弹窗
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt

# 将 rl-solitaire 目录加入路径（兼容从任意目录运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.env import Env
from nn.network_config import NetConfig
from nn.policy_value.fully_connected import FCPolicyValueNet
from agents.actor_critic.actor_critic_agent import ActorCriticAgent
from utils.tools import read_yaml


RUNS_DIR = "./agents/actor_critic/runs"
NETWORK_CONFIG_PATH = "./nn/policy_value/fc_policy_value_config.yaml"


def find_latest_checkpoint() -> str:
    """在所有 runs 目录中找到最新的 checkpoint 文件"""
    pattern = os.path.join(RUNS_DIR, "*", "checkpoints", "*.ckpt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        raise FileNotFoundError(f"在 {RUNS_DIR} 下未找到任何 checkpoint 文件，请先运行训练。")
    # 按文件修改时间排序，取最新的
    latest = max(checkpoints, key=os.path.getmtime)
    return latest


def load_agent(checkpoint_path: str) -> ActorCriticAgent:
    """从 checkpoint 加载 ActorCriticAgent"""
    # 优先从 checkpoint 所在目录读取 network config，否则用默认配置
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    local_config = glob.glob(os.path.join(run_dir, "fc_policy_value_config.yaml"))
    config_path = local_config[0] if local_config else NETWORK_CONFIG_PATH

    network_config_dict = read_yaml(config_path)
    network_config = NetConfig(config_dict=network_config_dict)

    # 从 checkpoint 加载网络权重
    network = FCPolicyValueNet.load_from_checkpoint(checkpoint_path, config=network_config)
    network.eval()

    agent = ActorCriticAgent(network=network, name="fc_policy_value-ActorCriticAgent")
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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 文件路径（不指定则自动使用最新）")
    parser.add_argument("--n-games", type=int, default=1,
                        help="游戏局数（默认 1）")
    parser.add_argument("--no-render", action="store_true",
                        help="不渲染游戏画面，只打印数值结果")
    parser.add_argument("--greedy", action="store_true", default=True,
                        help="是否使用贪心策略选择动作（默认 True）")
    args = parser.parse_args()

    # 确定 checkpoint 路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"错误：checkpoint 文件不存在: {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path = find_latest_checkpoint()

    print(f"加载 checkpoint: {checkpoint_path}")

    # 加载 agent
    agent = load_agent(checkpoint_path)
    print(f"Agent 加载成功：{agent.name}\n")

    # 运行游戏
    render = not args.no_render
    play_games(agent, n_games=args.n_games, render=render, greedy=args.greedy)


if __name__ == "__main__":
    main()
