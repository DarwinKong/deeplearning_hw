"""
SourceTorch 训练脚本 - 基于优化的批量 GPU 实现

用法:
    python runTorch.py -an a2c -nn fc_policy_value
    python runTorch.py -an ppo -nn fc_policy_value
"""
import argparse
import os
from datetime import datetime

# SourceTorch imports（不依赖 source）
from sourceTorch import A2CAgent, PPOAgent, BatchedGPUTrainer
from sourceTorch.nn.policy_value.fully_connected import FCPolicyValueNet
from sourceTorch.nn.network_config import NetConfig
from sourceTorch.utils.tools import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(description='SourceTorch Training Script')
    parser.add_argument('-an', '--agent_name', type=str, default='a2c',
                        choices=['a2c', 'ppo'], help='Algorithm name')
    parser.add_argument('-nn', '--network_name', type=str, default='fc_policy_value',
                        choices=['fc_policy_value', 'conv_policy_value', 'transformer_policy_value'],
                        help='Network architecture')
    parser.add_argument('--n-iter', type=int, default=200, help='Number of training iterations')
    parser.add_argument('--n-envs', type=int, default=64, help='Number of parallel environments')
    parser.add_argument('--n-steps', type=int, default=32, help='Steps per environment')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--enable-monitors', action='store_true', default=True, help='Enable training monitors')
    parser.add_argument('--disable-monitors', action='store_false', dest='enable_monitors', help='Disable monitors for speed test')
    
    # 新增：从 checkpoint 恢复训练
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from experiment directory or checkpoint file. '
                             'If directory, will find latest checkpoint automatically.')
    
    # PPO 超参数
    parser.add_argument('--ppo-entropy-coef', type=float, default=0.01,
                        help='PPO entropy coefficient (default: 0.01, recommended: 0.05-0.1 for long training)')
    parser.add_argument('--ppo-clip-epsilon', type=float, default=0.2,
                        help='PPO clip epsilon (default: 0.2)')
    parser.add_argument('--ppo-value-loss-coef', type=float, default=0.5,
                        help='PPO value loss coefficient (default: 0.5)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity/team name for uploads')
    
    return parser.parse_args()


def get_agent_class(agent_name):
    """获取算法类"""
    if agent_name.lower() == 'a2c':
        return A2CAgent
    elif agent_name.lower() == 'ppo':
        return PPOAgent
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def get_network_class(network_name):
    """获取网络类"""
    if network_name == 'fc_policy_value':
        return FCPolicyValueNet
    # TODO: 添加其他网络类型
    else:
        raise ValueError(f"Unknown network: {network_name}")


def load_config_from_yaml(agent_name, network_name):
    """
    从 YAML 配置文件加载配置
    
    Args:
        agent_name: 算法名称 ('a2c' or 'ppo')
        network_name: 网络名称 ('fc_policy_value', etc.)
    
    Returns:
        tuple: (network_config_dict, trainer_kwargs)
    """
    # 构建配置文件路径
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    
    # 加载网络配置
    network_config_file = f"{network_name.replace('_', '-')}-config.yaml"
    network_config_path = os.path.join(config_dir, 'nn', network_config_file)
    
    if os.path.exists(network_config_path):
        network_config = read_yaml(network_config_path)
        print(f"✓ Loaded network config from: {network_config_path}")
    else:
        print(f"⚠ Network config not found: {network_config_path}, using defaults")
        network_config = {'name': 'FCPolicyValueNet'}
    
    # 加载训练器配置
    trainer_config_file = f"{'actor-critic' if agent_name == 'a2c' else 'ppo'}-trainer-config.yaml"
    trainer_config_path = os.path.join(config_dir, 'agent-trainer', trainer_config_file)
    
    if os.path.exists(trainer_config_path):
        trainer_config = read_yaml(trainer_config_path)
        print(f"✓ Loaded trainer config from: {trainer_config_path}")
    else:
        print(f"⚠ Trainer config not found: {trainer_config_path}, using defaults")
        trainer_config = {}
    
    return network_config, trainer_config


def main():
    args = parse_args()
    
    # 从 YAML 加载配置
    network_config_dict, trainer_config = load_config_from_yaml(args.agent_name, args.network_name)
    
    # 命令行参数优先，否则从 YAML 读取
    n_iter = args.n_iter
    enable_wandb = trainer_config.get('enable_wandb', False)  # 从 trainer config 读取 wandb 开关
    wandb_project = trainer_config.get('wandb_project', 'sourcetorch-project')  # 从 trainer config 读取 wandb 项目名称
    wandb_run_name = trainer_config.get('wandb_run_name', f"{args.agent_name.upper()}_{args.network_name}")  # 从 trainer config 读取 wandb 运行名称
    wandb_entity = args.wandb_entity or trainer_config.get('wandb_entity', None)
    
    # Learning Rate: 如果用户没有显式指定（使用默认值 3e-5），则从 YAML 读取
    if args.lr == 3e-5 and 'optimizer' in network_config_dict:
        learning_rate = network_config_dict['optimizer'].get('lr', args.lr)
    else:
        learning_rate = args.lr
    
    batch_size = trainer_config.get('batch_size', 256)
    n_optim_steps = trainer_config.get('n_optim_steps', 1) or 1
    
    print("=" * 80)
    print("SourceTorch Training")
    print("=" * 80)
    print(f"Algorithm: {args.agent_name.upper()}")
    print(f"Network: {args.network_name}")
    print(f"Iterations: {n_iter}")
    print(f"Parallel Envs: {args.n_envs}")
    print(f"Steps per Env: {args.n_steps}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Device: {args.device}")
    if args.resume_from:
        print(f"Resume from: {args.resume_from}")
    print("=" * 80)
    
    # 处理 resume 逻辑
    start_iter = 0
    if args.resume_from:
        # 判断是目录还是文件
        if os.path.isdir(args.resume_from):
            # 是实验目录，自动查找最新 checkpoint
            exp_dir = args.resume_from
            checkpoints_dir = os.path.join(exp_dir, "checkpoints")
            print(f"\nSearching for latest checkpoint in: {checkpoints_dir}")
            
            # 创建临时 trainer 来查找 checkpoint
            result = BatchedGPUTrainer.find_latest_checkpoint(checkpoints_dir)
            
            if result is None:
                print(f"⚠ No checkpoint found in {checkpoints_dir}, starting from scratch")
                resume_checkpoint = None
            else:
                checkpoint_path, latest_iter = result
                print(f"✓ Found latest checkpoint: iter_{latest_iter}.pt")
                resume_checkpoint = checkpoint_path
                start_iter = latest_iter + 1
                
                # 复用该实验目录
                base_dir = os.path.dirname(exp_dir)
                exp_name = os.path.basename(exp_dir)
                log_dir = os.path.join(exp_dir, "logs")
                checkpoint_dir = checkpoints_dir
                meta_dir = os.path.join(exp_dir, "meta")
                results_dir = os.path.join(exp_dir, "results")
        else:
            # 是 checkpoint 文件路径
            resume_checkpoint = args.resume_from
            if not os.path.exists(resume_checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint}")
            
            # 从文件名提取迭代次数
            basename = os.path.basename(resume_checkpoint)
            try:
                latest_iter = int(basename.replace('iter_', '').replace('.pt', ''))
                start_iter = latest_iter + 1
            except ValueError:
                start_iter = 0
            
            print(f"✓ Loading checkpoint: {resume_checkpoint}")
            
            # 使用当前时间创建新目录
            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            exp_name = f"{args.agent_name.upper()} (resumed) {timestamp}"
            base_dir = "checkpoints-and-logs/local"
            log_dir = os.path.join(base_dir, exp_name, "logs")
            checkpoint_dir = os.path.join(base_dir, exp_name, "checkpoints")
            meta_dir = os.path.join(base_dir, exp_name, "meta")
            results_dir = os.path.join(base_dir, exp_name, "results")
            
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(meta_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
    else:
        # 从头开始训练，创建新目录
        resume_checkpoint = None
        
        # 创建输出目录（使用美观的命名格式）
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        exp_name = f"{args.agent_name.upper()} {timestamp}"
        
        # 使用相对路径
        base_dir = "checkpoints-and-logs/local"
        log_dir = os.path.join(base_dir, exp_name, "logs")
        checkpoint_dir = os.path.join(base_dir, exp_name, "checkpoints")
        meta_dir = os.path.join(base_dir, exp_name, "meta")
        results_dir = os.path.join(base_dir, exp_name, "results")
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    
    # 创建网络
    net_config = NetConfig(config_dict=network_config_dict)
    network_class = get_network_class(args.network_name)
    network = network_class(net_config)
    network.to(args.device)
    
    # 创建算法
    agent_class = get_agent_class(args.agent_name)
    
    # PPO 超参数（可通过命令行调整）
    ppo_clip_epsilon = getattr(args, 'ppo_clip_epsilon', 0.2)
    ppo_value_loss_coef = getattr(args, 'ppo_value_loss_coef', 0.5)
    ppo_entropy_coef = getattr(args, 'ppo_entropy_coef', 0.01)  # 默认 0.01，建议 0.05-0.1
    
    if args.agent_name.lower() == 'a2c':
        # 从网络配置读取损失权重
        loss_config = network_config_dict.get('loss', {})
        actor_coef = loss_config.get('actor_loss', {}).get('coef', 1.0)
        critic_coef = loss_config.get('critic_loss', {}).get('coef', 0.5)
        entropy_coef = loss_config.get('regularization', {}).get('coef', 0.01)
        
        algorithm = agent_class(
            network=network,
            actor_loss_weight=actor_coef,
            critic_loss_weight=critic_coef,
            entropy_weight=entropy_coef
        )
    elif args.agent_name.lower() == 'ppo':
        # 从 trainer config 读取 PPO 参数
        ppo_config = trainer_config.get('ppo', {})
        clip_epsilon = ppo_config.get('clip_epsilon', ppo_clip_epsilon)
        value_loss_coef = ppo_config.get('value_loss_coef', ppo_value_loss_coef)
        entropy_coef = ppo_config.get('entropy_coef', ppo_entropy_coef)
        max_grad_norm = ppo_config.get('max_grad_norm', 0.5)
        
        algorithm = agent_class(
            network=network,
            clip_epsilon=clip_epsilon,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm
        )
        
        # 将 GAE 等参数传递给 algorithm.config（供 compute_returns_and_advantages 使用）
        algorithm.config['use_gae'] = ppo_config.get('use_gae', True)
        algorithm.config['gae_lambda'] = ppo_config.get('gae_lambda', 0.95)
        algorithm.config['discount'] = ppo_config.get('discount', 0.99)
        algorithm.config['normalize_advantages'] = ppo_config.get('normalize_advantages', True)
    
    # 创建训练器
    trainer = BatchedGPUTrainer(
        n_envs=args.n_envs,
        algorithm=algorithm,
        n_iter=n_iter,
        n_steps_per_env=args.n_steps,
        agent_results_filepath=os.path.join(results_dir, "agent_results.pt"),
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_optim_steps=n_optim_steps,
        log_dir=log_dir,
        checkpoints_dir=checkpoint_dir,
        meta_dir=meta_dir,
        results_dir=results_dir,
        enable_monitors=args.enable_monitors,
        network_config=net_config,  # 传递网络配置（包含优化器参数）  # 传递监控开关
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity

    )
    
    # 如果需要，从 checkpoint 恢复
    if args.resume_from and resume_checkpoint:
        start_iter = trainer.resume_from_checkpoint(resume_checkpoint)
    else:
        start_iter = 0
    
    # 开始训练
    print("\n开始训练...")
    trainer.train(start_iter=start_iter)
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"实验目录: {os.path.join(base_dir, exp_name)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
