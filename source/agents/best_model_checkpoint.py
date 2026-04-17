"""
自定义 Checkpoint Callback，自动保存最优模型到 remote 目录
"""
import os
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint


class BestModelCheckpoint(ModelCheckpoint):
    """
    扩展 ModelCheckpoint，在每次保存新最优模型时，
    自动复制一份到 remote 目录
    """
    
    def __init__(self, remote_checkpoints_dir: str = None, **kwargs):
        """
        :param remote_checkpoints_dir: remote checkpoints 目录路径
        :param kwargs: 传递给 ModelCheckpoint 的其他参数
        """
        super().__init__(**kwargs)
        self.remote_checkpoints_dir = remote_checkpoints_dir
        if remote_checkpoints_dir:
            os.makedirs(remote_checkpoints_dir, exist_ok=True)
    
    def _save_checkpoint(self, trainer, filepath):
        """重写保存方法，同时保存到 remote 目录"""
        # 调用父类方法保存 checkpoint
        super()._save_checkpoint(trainer, filepath)
        
        # 如果配置了 remote 目录，且这是最优模型，复制到 remote
        if self.remote_checkpoints_dir and self.monitor:
            # 检查当前保存的是否是最优模型
            current_score = trainer.callback_metrics.get(self.monitor)
            best_score = self.best_model_score
            
            if current_score is not None and best_score is not None:
                # 判断是否达到了新的最优（考虑模式：min 或 max）
                is_better = False
                if self.mode == 'min':
                    is_better = current_score <= best_score
                else:  # mode == 'max'
                    is_better = current_score >= best_score
                
                if is_better:
                    # 提取文件名
                    filename = os.path.basename(filepath)
                    remote_filepath = os.path.join(self.remote_checkpoints_dir, filename)
                    
                    # 复制到 remote 目录
                    shutil.copy2(filepath, remote_filepath)
                    print(f"✓ 最优模型已保存到 remote: {remote_filepath}")
                    
                    # 删除旧的 remote checkpoint（如果有）
                    self._cleanup_old_remote_checkpoints(remote_filepath)
    
    def _cleanup_old_remote_checkpoints(self, current_remote_path: str):
        """清理 remote 目录中的旧 checkpoint，只保留最新的"""
        if not os.path.exists(self.remote_checkpoints_dir):
            return
        
        # 获取所有 .ckpt 文件
        ckpt_files = [f for f in os.listdir(self.remote_checkpoints_dir) if f.endswith('.ckpt')]
        
        # 删除除了当前文件之外的所有文件
        current_filename = os.path.basename(current_remote_path)
        for filename in ckpt_files:
            if filename != current_filename:
                old_path = os.path.join(self.remote_checkpoints_dir, filename)
                try:
                    os.remove(old_path)
                    print(f"  已删除旧的 remote checkpoint: {filename}")
                except Exception as e:
                    print(f"  警告: 无法删除 {filename}: {e}")
