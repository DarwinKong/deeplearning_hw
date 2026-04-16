"""
并行数据收集器 - 使用多进程加速数据收集

核心优化：
1. 多个 worker 进程并行运行环境
2. 每个 worker 独立收集数据
3. 主进程汇总所有数据
4. 适用于 CPU-bound 的环境模拟

注意：当前 BatchedGPUEnv 已经在 GPU 上高度并行，
此模块主要用于未来扩展到更复杂的环境或分布式训练。
"""
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Any
from multiprocessing import Queue


class ParallelDataCollector:
    """
    并行数据收集器
    
    Args:
        n_workers: worker 进程数量
        env_class: 环境类
        env_kwargs: 环境初始化参数
        device: 设备类型
    """
    
    def __init__(self, n_workers: int = 4, env_class=None, env_kwargs: dict = None, device: str = 'cuda'):
        self.n_workers = n_workers
        self.env_class = env_class
        self.env_kwargs = env_kwargs or {}
        self.device = device
        
        # 数据队列
        self.data_queues = [Queue() for _ in range(n_workers)]
        self.result_queue = Queue()
        
        # Worker 进程
        self.workers = []
    
    def _worker_process(self, worker_id: int, data_queue: Queue, result_queue: Queue):
        """
        Worker 进程函数
        
        Args:
            worker_id: worker ID
            data_queue: 接收任务数据的队列
            result_queue: 返回结果的队列
        """
        try:
            # 在每个 worker 中创建独立的环境
            env = self.env_class(**self.env_kwargs)
            
            while True:
                # 从队列获取任务
                task = data_queue.get()
                
                if task is None:  # 终止信号
                    break
                
                # 执行任务
                if task['type'] == 'collect':
                    n_steps = task['n_steps']
                    agent_state_dict = task['agent_state_dict']
                    
                    # 加载 agent 状态（如果需要）
                    # 这里简化处理，实际应该传递完整的 agent
                    
                    # 收集数据
                    collected_data = self._collect_data(env, n_steps)
                    
                    # 返回结果
                    result_queue.put({
                        'worker_id': worker_id,
                        'data': collected_data
                    })
        
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            result_queue.put({'worker_id': worker_id, 'error': str(e)})
    
    def _collect_data(self, env, n_steps: int) -> Dict[str, torch.Tensor]:
        """
        收集数据（简化版本，实际需要与 agent 集成）
        
        Args:
            env: 环境实例
            n_steps: 收集步数
        
        Returns:
            收集的数据字典
        """
        # 这里是占位符，实际实现需要：
        # 1. 重置环境
        # 2. 使用 agent 选择动作
        # 3. 执行动作并记录 (state, action, reward, next_state, done)
        # 4. 返回批量数据
        
        raise NotImplementedError("需要与具体的 Agent 集成")
    
    def start_workers(self):
        """启动 worker 进程"""
        for i in range(self.n_workers):
            p = mp.Process(
                target=self._worker_process,
                args=(i, self.data_queues[i], self.result_queue)
            )
            p.start()
            self.workers.append(p)
    
    def stop_workers(self):
        """停止 worker 进程"""
        for queue in self.data_queues:
            queue.put(None)  # 发送终止信号
        
        for p in self.workers:
            p.join(timeout=5)
        
        self.workers.clear()
    
    def collect_batched_data_parallel(self, agent, n_steps_per_env: int) -> Dict[str, torch.Tensor]:
        """
        并行收集批量数据
        
        Args:
            agent: 智能体实例
            n_steps_per_env: 每个环境收集的步数
        
        Returns:
            收集的数据字典
        """
        if not self.workers:
            self.start_workers()
        
        # 分发任务到各个 worker
        for i in range(self.n_workers):
            self.data_queues[i].put({
                'type': 'collect',
                'n_steps': n_steps_per_env,
                'agent_state_dict': agent.network.state_dict()
            })
        
        # 收集结果
        all_data = []
        for _ in range(self.n_workers):
            result = self.result_queue.get(timeout=30)
            if 'error' in result:
                print(f"Worker {result['worker_id']} error: {result['error']}")
            else:
                all_data.append(result['data'])
        
        # 合并数据（简化版本）
        # 实际需要按照 key 拼接 tensor
        merged_data = self._merge_data(all_data)
        
        return merged_data
    
    def _merge_data(self, data_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        合并多个 worker 的数据
        
        Args:
            data_list: 数据列表
        
        Returns:
            合并后的数据
        """
        if not data_list:
            return {}
        
        merged = {}
        for key in data_list[0].keys():
            # 沿 batch 维度拼接
            tensors = [d[key] for d in data_list if key in d]
            if tensors:
                merged[key] = torch.cat(tensors, dim=0)
        
        return merged
    
    def __del__(self):
        """析构时清理资源"""
        self.stop_workers()


# 注意：由于当前 BatchedGPUEnv 已经在 GPU 上实现了高效的批量并行，
# 多进程并行主要用于以下场景：
# 1. 环境模拟非常复杂，CPU 成为瓶颈
# 2. 需要分布式训练，跨多台机器
# 3. 混合 CPU/GPU 工作负载
#
# 对于当前的孔明棋环境，BatchedGPUEnv 已经足够高效，
# 建议优先使用单进程 + GPU 批量的方式。
