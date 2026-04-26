import torch

from sourceTorch.nn.blocks.residual import ResidualBlock
from sourceTorch.nn.network_config import NetConfig
from .skeleton import BasePolicyValueNet


class ConvPolicyValueNet(BasePolicyValueNet):
    """
    卷积 policy-value 网络，含残差块与归一化层。

    可通过 YAML `embeddings` 配置：
    - kernel_size: [5,5] 或 [3,3]
    - norm_type: batch | group（GroupNorm 更适应小 batch / 多环境累加梯度场景）
    - norm_groups: GroupNorm 分组数（需整除通道数）
    - residual_preact: 是否使用 PreAct 风格卷积残差
    - se_reduction: >0 时启用 SE 通道注意力（0 关闭）
    """

    def __init__(self, config: NetConfig):
        super().__init__(config)
        self._set_src_mask()

    def _build_model(self, architecture_config: dict):
        emb = architecture_config.get("embeddings", {})
        ks = emb.get("kernel_size", [5, 5])
        self._kernel_size = tuple(ks) if isinstance(ks, (list, tuple)) else (5, 5)
        self._norm_type = emb.get("norm_type", "batch")
        self._norm_groups = int(emb.get("norm_groups", 8))
        self._residual_preact = bool(emb.get("residual_preact", False))
        self._se_reduction = int(emb.get("se_reduction", 0))
        super()._build_model(architecture_config)

    @staticmethod
    def _kernel_kw(kwargs: dict, fallback: tuple) -> tuple:
        ks = kwargs.get("kernel_size", fallback)
        if isinstance(ks, (list, tuple)):
            return tuple(ks)
        return fallback

    def _make_norm(self, num_features: int) -> torch.nn.Module:
        if self._norm_type == "group":
            g = self._norm_groups
            while g > 1 and num_features % g != 0:
                g -= 1
            return torch.nn.GroupNorm(g, num_features)
        return torch.nn.BatchNorm2d(num_features=num_features)

    def _build_state_embeddings(self, n_residual_blocks: int, input_dim: int, hidden_dim: int, n_layers: int = 2,
                                residual_hidden_dim: int = None, bias: bool = True, kernel_size: tuple = None, **kwargs):
        kernel_size = self._kernel_kw(kwargs, self._kernel_size)
        self.state_embeddings = torch.nn.Sequential()
        self.state_embeddings.add_module(name="input_conv",
                                         module=torch.nn.Conv2d(in_channels=input_dim,
                                                                out_channels=hidden_dim,
                                                                padding="same",
                                                                kernel_size=kernel_size,
                                                                bias=bias))
        self.state_embeddings.add_module(name="input_activation", module=self.activation)
        self.state_embeddings.add_module(name="residual_blocks",
                                         module=self._get_residual_blocks(n_residual_blocks=n_residual_blocks,
                                                                          input_dim=hidden_dim,
                                                                          hidden_dim=residual_hidden_dim,
                                                                          n_layers=n_layers,
                                                                          bias=bias,
                                                                          kernel_size=kernel_size))
        self.state_embeddings.add_module(name="ouput_activation", module=self.activation)

    def _build_policy_head(self, n_residual_blocks: int, output_dim: int, input_dim: int, n_layers: int = 2,
                           hidden_dim: int = None, bias: bool = True, kernel_size: tuple = None, **kwargs):
        kernel_size = self._kernel_kw(kwargs, self._kernel_size)
        self.policy_head = torch.nn.Sequential()
        self.policy_head.add_module(name="residual_blocks",
                                    module=self._get_residual_blocks(n_residual_blocks=n_residual_blocks,
                                                                     input_dim=input_dim,
                                                                     hidden_dim=hidden_dim,
                                                                     n_layers=n_layers,
                                                                     bias=bias,
                                                                     kernel_size=kernel_size))
        self.policy_head.add_module(name="flatten", module=torch.nn.Flatten(start_dim=1, end_dim=-1))
        self.policy_head.add_module(name="output_activation", module=self.activation)
        self.policy_head.add_module(name="output_linear",
                                    module=torch.nn.Linear(in_features=7 * 7 * input_dim,
                                                           out_features=output_dim,
                                                           bias=bias))

    def _build_value_head(self, n_residual_blocks: int, output_dim: int, input_dim: int, n_layers: int = 2,
                          hidden_dim: int = None, bias: bool = True, kernel_size: tuple = None, **kwargs):
        kernel_size = self._kernel_kw(kwargs, self._kernel_size)
        self.value_head = torch.nn.Sequential()
        self.value_head.add_module(name="residual_blocks",
                                   module=self._get_residual_blocks(n_residual_blocks=n_residual_blocks,
                                                                    input_dim=input_dim,
                                                                    hidden_dim=hidden_dim,
                                                                    n_layers=n_layers,
                                                                    bias=bias,
                                                                    kernel_size=kernel_size))
        self.value_head.add_module(name="flatten", module=torch.nn.Flatten(start_dim=1, end_dim=-1))
        # Bugfix: 必须为 value_head 添加激活，勿错误挂到 policy_head
        self.value_head.add_module(name="output_activation", module=self.activation)
        self.value_head.add_module(name="output_linear",
                                   module=torch.nn.Linear(in_features=7 * 7 * input_dim,
                                                          out_features=output_dim,
                                                          bias=bias))

    def _get_residual_blocks(self, n_residual_blocks: int, input_dim: int, hidden_dim: int, n_layers: int = 2,
                             bias: bool = True, kernel_size: tuple = (5, 5)):
        residual_blocks = torch.nn.Sequential()
        for i in range(1, n_residual_blocks + 1):
            residual_blocks.add_module(name=f"norm{i}",
                                       module=self._make_norm(input_dim))
            residual_blocks.add_module(name=f"residual{i}",
                                       module=ResidualBlock(input_dim=input_dim,
                                                            hidden_dim=hidden_dim,
                                                            activation=self.activation,
                                                            layer_type="conv",
                                                            n_layers=n_layers,
                                                            bias=bias,
                                                            kernel_size=kernel_size,
                                                            preact=self._residual_preact,
                                                            se_reduction=self._se_reduction))
        return residual_blocks

    @staticmethod
    def _reshape_2d_input(x: torch.Tensor):
        """
        Reshapes a tensor from shape (N, W, H, C) to (N, C, W, H) for the convolution modules.
        :param x: torch.Tensor of shape (N, W, H, C).
        :return: torch.Tensor of shape (N, C, W, H).
        """
        return x.reshape(x.shape[0], x.shape[-1], x.shape[1], x.shape[2])

    def reformat_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._reshape_2d_input(x)
