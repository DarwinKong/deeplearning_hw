import torch
import torch.nn.functional as F


class SqueezeExcitation2d(torch.nn.Module):
    """
    通道注意力（Squeeze-and-Excitation），对卷积特征图按通道重标定。
    适合 policy/value 共享骨干：突出与「可走步、终局价值」相关的通道模式。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = torch.nn.Linear(channels, hidden)
        self.fc2 = torch.nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        w = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ResidualBlock(torch.nn.Module):
    """
    残差块：全连接或卷积子路径 + 恒等映射。

    - 标准顺序（默认）：Conv — BN — Act — …（与原实现一致，BN 在残差栈内由 ConvPolicyValueNet 提供）
    - PreAct 卷积（preact=True）：BN — Act — Conv — …，再与 x 相加（更接近 PreActResNet 主路径）
    - SE（se_reduction>0）：在残差分支输出上施加通道注意力后再与 x 相加
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: torch.nn.Module, layer_type: str, n_layers: int = 2,
                 bias: bool = True, preact: bool = False, se_reduction: int = 0, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.layer_type = layer_type
        self.bias = bias
        self.preact = preact and layer_type == "conv"
        self.se = None
        if se_reduction and se_reduction > 0 and layer_type == "conv":
            self.se = SqueezeExcitation2d(input_dim, reduction=se_reduction)

        if self.preact:
            self._build_conv_layers_preact(**kwargs)
        else:
            self._build_layers(**kwargs)

    def _build_layers(self, **kwargs):
        if self.n_layers <= 1:
            raise ValueError(f"ResidualBlock only accepts at least 2 layers "
                             f"but the number of layers was {self.n_layers}")
        self.layers = torch.nn.Sequential()
        if self.layer_type == "linear":
            self._build_linear_layers()

        elif self.layer_type == "conv":
            self._build_conv_layers(**kwargs)
        else:
            raise ValueError(f"`layer_type` argument must be one of 'linear' or 'conv' but was {self.layer_type}")

    def _build_conv_layers_preact(self, **kwargs):
        """BN → Act → Conv 重复，最后与输入相加（PreAct 风格，仅卷积）。"""
        if self.n_layers <= 1:
            raise ValueError("PreAct conv residual needs at least 2 conv layers")
        self._preact_bns = torch.nn.ModuleList()
        self._preact_convs = torch.nn.ModuleList()
        self._preact_bns.append(torch.nn.BatchNorm2d(self.input_dim))
        self._preact_convs.append(
            torch.nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim,
                            padding="same", bias=self.bias, **kwargs))
        for _ in range(1, self.n_layers - 1):
            self._preact_bns.append(torch.nn.BatchNorm2d(self.hidden_dim))
            self._preact_convs.append(
                torch.nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                padding="same", bias=self.bias, **kwargs))
        self._preact_bns.append(torch.nn.BatchNorm2d(self.hidden_dim))
        self._preact_convs.append(
            torch.nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim,
                            padding="same", bias=self.bias, **kwargs))
        self.layers = None  # 使用专用 forward

    def _build_linear_layers(self):
        self.layers = torch.nn.Sequential()
        self.layers.add_module(name=f"linear1", module=torch.nn.Linear(in_features=self.input_dim,
                                                                       out_features=self.hidden_dim,
                                                                       bias=self.bias))
        self.layers.add_module(name=f"{self.activation}1", module=self.activation)
        for i in range(1, self.n_layers - 1):
            self.layers.add_module(name=f"linear{i + 1}",
                                   module=torch.nn.Linear(in_features=self.hidden_dim,
                                                          out_features=self.hidden_dim,
                                                          bias=self.bias))
            self.layers.add_module(name=f"{self.activation}{i + 1}", module=self.activation)

        self.layers.add_module(name=f"linear{self.n_layers}",
                               module=torch.nn.Linear(in_features=self.hidden_dim,
                                                      out_features=self.input_dim,
                                                      bias=self.bias))

    def _build_conv_layers(self, **kwargs):
        self.layers = torch.nn.Sequential()
        self.layers.add_module(name=f"conv1", module=torch.nn.Conv2d(in_channels=self.input_dim,
                                                                     out_channels=self.hidden_dim,
                                                                     padding="same",
                                                                     bias=self.bias, **kwargs))
        self.layers.add_module(name=f"{self.activation}1", module=self.activation)
        for i in range(1, self.n_layers - 1):
            self.layers.add_module(name=f"conv{i + 1}",
                                   module=torch.nn.Conv2d(in_channels=self.hidden_dim,
                                                          out_channels=self.hidden_dim,
                                                          padding="same",
                                                          bias=self.bias, **kwargs))
            self.layers.add_module(name=f"{self.activation}{i + 1}", module=self.activation)
        self.layers.add_module(name=f"conv{self.n_layers}",
                               module=torch.nn.Conv2d(in_channels=self.hidden_dim,
                                                      out_channels=self.input_dim,
                                                      padding="same",
                                                      bias=self.bias, **kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "conv" and self.preact:
            out = x
            for i in range(len(self._preact_convs)):
                out = self._preact_bns[i](out)
                out = self.activation(out)
                out = self._preact_convs[i](out)
            if self.se is not None:
                out = self.se(out)
            return x + out

        out = self.layers(x)
        if self.se is not None:
            out = self.se(out)
        return x + out
