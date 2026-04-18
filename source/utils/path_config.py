"""
路径管理工具
从 config/paths_config.yaml 读取所有路径配置
"""
import os
from source.utils.tools import read_yaml


class PathConfig:
    """全局路径配置管理器"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载路径配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'paths_config.yaml')
        self._config = read_yaml(config_path)
    
    @property
    def ckps_base_dir(self):
        return self._config['checkpoints_and_logs']['base_dir']
    
    @property
    def ckps_local_subdir(self):
        return self._config['checkpoints_and_logs']['local_subdir']
    
    @property
    def ckps_remote_subdir(self):
        return self._config['checkpoints_and_logs']['remote_subdir']
    
    @property
    def ckps_meta_dir(self):
        return self._config['checkpoints_and_logs']['meta_dir']
    
    @property
    def ckps_logs_dir(self):
        return self._config['checkpoints_and_logs']['logs_dir']
    
    @property
    def ckps_checkpoints_dir(self):
        return self._config['checkpoints_and_logs']['checkpoints_dir']
    
    @property
    def ckps_results_dir(self):
        return self._config['checkpoints_and_logs']['results_dir']
    
    def get_experiment_name(self, agent_name: str, timestamp: str = None) -> str:
        """
        生成实验名称
        
        :param agent_name: agent 名称 (actor_critic 或 ppo)
        :param timestamp: 时间戳，如果为 None 则使用当前时间
        :return: 实验名称，格式: {AGENT_NAME}_{TIMESTAMP}
        """
        if timestamp is None:
            from source.utils.tools import strp_datetime
            from datetime import datetime
            timestamp = strp_datetime(datetime.now())
        
        # 将 agent_name 转换为大写缩写
        agent_abbr = {
            'actor_critic': 'A2C',
            'ppo': 'PPO'
        }.get(agent_name, agent_name.upper())
        
        return f"{agent_abbr}_{timestamp}"
    
    def get_experiment_dir(self, agent_name: str, use_remote: bool = False, timestamp: str = None) -> str:
        """
        获取实验根目录路径
        
        :param agent_name: agent 名称
        :param use_remote: 是否使用 remote 目录
        :param timestamp: 时间戳
        :return: 实验根目录完整路径
        """
        subdir = self.ckps_remote_subdir if use_remote else self.ckps_local_subdir
        experiment_name = self.get_experiment_name(agent_name, timestamp)
        return os.path.join(self.ckps_base_dir, subdir, experiment_name)
    
    def get_experiment_subdir(self, agent_name: str, subdir_name: str, use_remote: bool = False, timestamp: str = None) -> str:
        """
        获取实验子目录路径
        
        :param agent_name: agent 名称
        :param subdir_name: 子目录名称 (meta/logs/checkpoints/results)
        :param use_remote: 是否使用 remote 目录
        :param timestamp: 时间戳
        :return: 子目录完整路径
        """
        experiment_dir = self.get_experiment_dir(agent_name, use_remote, timestamp)
        return os.path.join(experiment_dir, subdir_name)
    
    def get_meta_dir(self, agent_name: str, use_remote: bool = False, timestamp: str = None) -> str:
        """获取 meta 目录"""
        return self.get_experiment_subdir(agent_name, self.ckps_meta_dir, use_remote, timestamp)
    
    def get_logs_dir(self, agent_name: str, use_remote: bool = False, timestamp: str = None) -> str:
        """获取 logs 目录"""
        return self.get_experiment_subdir(agent_name, self.ckps_logs_dir, use_remote, timestamp)
    
    def get_checkpoints_dir(self, agent_name: str, use_remote: bool = False, timestamp: str = None) -> str:
        """获取 checkpoints 目录"""
        return self.get_experiment_subdir(agent_name, self.ckps_checkpoints_dir, use_remote, timestamp)
    
    def get_results_dir(self, agent_name: str, use_remote: bool = False, timestamp: str = None) -> str:
        """获取 results 目录"""
        return self.get_experiment_subdir(agent_name, self.ckps_results_dir, use_remote, timestamp)
    
    def get_remote_best_model_path(self, agent_name: str, timestamp: str = None) -> str:
        """获取 remote 最优模型路径"""
        remote_dir = self.get_experiment_dir(agent_name, use_remote=True, timestamp=timestamp)
        return os.path.join(remote_dir, "best_model.ckpt")
    
    @property
    def config_base_dir(self):
        return self._config['config']['base_dir']
    
    @property
    def agent_trainer_config_dir(self):
        return os.path.join(self.config_base_dir, self._config['config']['agent_trainer_dir'])
    
    @property
    def nn_config_dir(self):
        return os.path.join(self.config_base_dir, self._config['config']['nn_dir'])
    
    def get_agent_trainer_config_path(self, agent_name: str) -> str:
        """
        获取 agent trainer 配置文件路径
        
        :param agent_name: agent 名称
        :return: 配置文件完整路径
        """
        filename = {
            "actor_critic": "actor-critic-trainer-default-reward-config.yaml",
            "ppo": "ppo-trainer-default-reward-config.yaml",
        }.get(agent_name, f"{agent_name.replace('_', '-')}-trainer-config.yaml")
        return os.path.join(self.agent_trainer_config_dir, filename)
    
    def get_nn_config_path(self, network_name: str) -> str:
        """
        获取神经网络配置文件路径
        
        :param network_name: 网络名称
        :return: 配置文件完整路径
        """
        filename = f"{network_name.replace('_', '-')}-config.yaml"
        return os.path.join(self.nn_config_dir, filename)


# 全局实例
path_config = PathConfig()
