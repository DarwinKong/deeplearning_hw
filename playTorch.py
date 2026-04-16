"""
SourceTorch 游戏演示脚本（简化版）

用法:
    python play_torch_simple.py --agent a2c --n-games 5
    python play_torch_simple.py --checkpoint path/to/checkpoint.ckpt --n-games 3 --render
"""
import argparse
import os
import torch
import numpy as np

# SourceTorch imports
from sourceTorch import A2CAgent, PPOAgent
from sourceTorch.nn.policy_value.fully_connected import FCPolicyValueNet
from sourceTorch.nn.network_config import NetConfig
from sourceTorch.env.batched_gpu_env import BatchedGPUEnv
from sourceTorch.agent.gameplay import play_game  # 使用重构后的模块
from sourceTorch.utils.tools import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(description='SourceTorch Play Script')
    parser.add_argument('--agent', type=str, default='a2c',
                        choices=['a2c', 'ppo'], help='Algorithm name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--n-games', type=int, default=5,
                        help='Number of games to play')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('--greedy', action='store_true', default=False,
                        help='Use greedy action selection (argmax)')
    parser.add_argument('--sample', action='store_false', dest='greedy',
                        help='Use sampling action selection (multinomial, default)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    return parser.parse_args()


def load_agent(agent_name, checkpoint_path=None, device='cuda'):
    """加载训练好的智能体"""
    print(f"Loading {agent_name.upper()} agent...")
    
    # 加载网络配置
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'nn', 
                               f"{agent_name.replace('_', '-')}-policy-value-config.yaml" if agent_name == 'a2c' else "fc-policy-value-config.yaml")
    
    if agent_name == 'a2c':
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'nn', 'fc-policy-value-config.yaml')
    
    if os.path.exists(config_path):
        network_config_dict = read_yaml(config_path)
        net_config = NetConfig(config_dict=network_config_dict)
        network = FCPolicyValueNet(net_config)
    else:
        # 默认配置
        from sourceTorch.nn.utils import create_fc_network
        network = create_fc_network()
    
    # 创建 agent
    if agent_name == 'a2c':
        agent = A2CAgent(
            network=network,
            actor_loss_weight=1.0,
            critic_loss_weight=0.5,
            entropy_weight=0.01
        )
    elif agent_name == 'ppo':
        agent = PPOAgent(
            network=network,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01
        )
    
    # 加载检查点
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['network_state_dict'])
        print("Checkpoint loaded successfully!")
    
    # 移动到指定设备并更新 agent.device
    agent.to_device(device)
    agent.set_evaluation_mode()
    
    return agent


def main():
    args = parse_args()
    
    print("=" * 80)
    print("SourceTorch Game Demo")
    print("=" * 80)
    print(f"Agent: {args.agent.upper()}")
    print(f"Games: {args.n_games}")
    print(f"Render: {args.render}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # 加载智能体
    agent = load_agent(args.agent, args.checkpoint, args.device)
    
    # 创建环境
    env = BatchedGPUEnv(n_envs=1, device=args.device)
    
    # 玩游戏
    print("\n开始游戏...\n")
    
    total_returns = []
    n_pegs_list = []
    
    for game_id in range(1, args.n_games + 1):
        total_return, n_pegs = play_game(
            agent=agent,
            env=env,
            render=args.render,
            game_id=game_id,
            greedy=args.greedy
        )
        
        total_returns.append(total_return)
        n_pegs_list.append(n_pegs)
        
        status = "✓ Perfect!" if n_pegs == 1 else f"✗ Failed ({n_pegs} pegs left)"
        print(f"Game {game_id:2d}: Return={total_return:.2f}, Pegs Left={n_pegs:2d} {status}")
    
    # 统计
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)
    print(f"Average Return: {np.mean(total_returns):.2f} ± {np.std(total_returns):.2f}")
    print(f"Average Pegs Left: {np.mean(n_pegs_list):.2f}")
    print(f"Perfect Solutions: {sum(1 for n in n_pegs_list if n == 1)}/{args.n_games}")
    print("=" * 80)


if __name__ == '__main__':
    main()
