# 人员 C — policy_value 网络架构实验报告（模板）

> 以下为代码已落地内容 + 需在本地 GPU 跑满实验后补数值。**训练请使用 `deeplearning_hw/runTorch.py`**。

**一键全架构消融（A2C+PPO × FC + CNN a～e + Transformer 1D/2D，共 16 次）**：

```bash
cd deeplearning_hw
bash tools/run_all_arch_ablation_parallel.sh
```

- **CNN**：`cnn_ablation_a`～`d`（卷积/归一化），`cnn_ablation_e`（PreAct+SE 残差，见 `conv-policy-value-ablations.yaml`）。
- **Transformer**：`transformer_pe_1d` / `transformer_pe_2d`（`transformer-policy-value-ablations.yaml`）。

仅 CNN（含 e）：`bash tools/run_cnn_ablation_parallel.sh`（关闭 FC/Transformer）。

---

## 一、Bug 修复报告

**文件**：`sourceTorch/nn/policy_value/conv.py` — `_build_value_head`

**问题**：价值头在展平后误把 `output_activation` 注册到 `self.policy_head`，导致策略头多出一层激活、价值头缺少该层，两分支结构不一致。

**修复前**：

```python
self.value_head.add_module(name="flatten", ...)
self.policy_head.add_module(name="output_activation", module=self.activation)  # 错误
self.value_head.add_module(name="output_linear", ...)
```

**修复后**：

```python
self.value_head.add_module(name="flatten", ...)
self.value_head.add_module(name="output_activation", module=self.activation)
self.value_head.add_module(name="output_linear", ...)
```

**影响分析（简述）**：

- **Policy / value loss**：价值分支容量与正则化路径与预期不符，critic 拟合 TD 目标时梯度经错误拓扑回传，value loss 曲线可表现为震荡或偏置；策略头多一层激活可改变 logits 尺度，间接扭曲 policy loss。
- **行为**：价值估计偏差 → advantage 噪声增大 → 策略更新方向不稳；极端时出现「会走子但估值不准」或收敛变慢。
- **修复后**：双头对称、与架构设计一致，critic 与 actor 梯度语义清晰，预期训练更稳定、评估指标（剩余棋子数）更可信。

---

## 二、消融实验结果（含推荐配置及原因）

四组消融已合并到 **同一文件** `config/nn/conv-policy-value-ablations.yaml`，变体名统一为 `cnn_ablation_a`～`cnn_ablation_d`：

| 组 | `--nn-variant` | kernel | 归一化 |
|----|----------------|--------|--------|
| CNN-A | `cnn_ablation_a` | 5×5 | BatchNorm |
| CNN-B | `cnn_ablation_b` | 3×3 | BatchNorm |
| CNN-C | `cnn_ablation_c` | 5×5 | GroupNorm |
| CNN-D | `cnn_ablation_d` | 3×3 | GroupNorm |

**启动命令示例**（统一条件见第五节）：

```bash
cd deeplearning_hw
python runTorch.py -an a2c -nn conv_policy_value \
  --nn-config conv-policy-value-ablations.yaml --nn-variant cnn_ablation_b \
  --n-iter 14000 --n-envs 64 --n-steps 32 --seed 42 --device cuda
```

省略 `--nn-config` / `--nn-variant` 时，`conv_policy_value` 默认加载该文件并选用 **`cnn_ablation_a`**。

**实验理由（报告用语）**：

- **3×3**：有效棋盘区域约 5×5，单步跳子为局部 3 格关系，小卷积核更贴合「一步邻域」模式，参数量更少，利于在同等迭代下拟合局部走法。
- **GroupNorm**：64 并行环境时优化器 step 对应的统计仍来自大批样本，但 BN 依赖 batch 维统计；若有效 batch 波动大或子 batch 相关性强，GN 按通道分组更稳健。具体优劣以消融曲线为准。

**推荐配置**：跑完四组后，以 **平均剩余棋子最低、完美解率最高** 且训练稳定者为 **CNN-Best**（若 D 最优则选 D）。

---

## 三、Transformer 2D 位置编码实现

**文件**：`sourceTorch/nn/policy_value/transformer.py`

- 类 `PositionalEncoding2D`：行嵌入 `nn.Embedding(H, d/2)` + 列嵌入 `nn.Embedding(W, d/2)`，对 `seq_len=49` 的 token 用 `i → (i//7, i%7)` 查表后拼接，再与线性投影后的 token 相加。
- 配置项：`architecture.embeddings.positional_encoding: "2d"`（默认已在 `transformer-policy-value-config.yaml`），`1d` 则沿用正弦 1D 编码。

**为何更合理**：孔明棋在 2D 网格上具有旋转/对称先验；行列可学习编码显式区分「从哪一格出发、朝哪一侧跳」，比单序列 1D 位置更符合空间结构；跳子有方向性，行列分离有助于注意力对齐几何邻域。

---

## 四、残差块改进方案

**文件**：`sourceTorch/nn/blocks/residual.py`

- **SqueezeExcitation2d**：对卷积输出做通道注意力，强调与决策更相关的特征通道。
- **PreAct 卷积残差**：可选 `residual_preact=True`，残差分支内为 BN→Act→Conv 堆叠，再与恒等相加。
- **配置**：`embeddings.residual_preact`、`embeddings.se_reduction`（0 关闭）。示例见 `config/nn/conv-policy-value-residual-adv.yaml`。

**收益**：policy/value 共享几何特征时，SE 改善通道选择；PreAct 利于深层训练稳定。孔明棋稀疏奖励下，更稳定的梯度与更清晰的价值通道有助于 critic 学习终局回报，但需注意过强 SE 可能过拟合小数据——以验证曲线为准。

---

## 五、三架构对比实验结果

**固定训练条件**（与任务书一致）：

```bash
SEEDS=(42 123 2026)
for s in "${SEEDS[@]}"; do
  python runTorch.py -an a2c -nn fc_policy_value --n-iter 14000 --n-envs 64 --n-steps 32 --device cuda --seed "$s"
  python runTorch.py -an a2c -nn conv_policy_value \
    --nn-config conv-policy-value-ablations.yaml --nn-variant cnn_ablation_d \
    --n-iter 14000 --n-envs 64 --n-steps 32 --device cuda --seed "$s"
  python runTorch.py -an a2c -nn transformer_policy_value \
    --n-iter 14000 --n-envs 64 --n-steps 32 --device cuda --seed "$s"
done
```

将 **CNN 行** 中 `--nn-config` 换为你的 **CNN-Best** 文件名。

**结果汇总**：

```bash
python tools/compare_sourcetorch_experiments.py checkpoints-and-logs/local/<实验1> checkpoints-and-logs/local/<实验2> ...
```

TensorBoard（若写了 `logs/tensorboard` 或项目自定义日志目录）：

```bash
tensorboard --logdir checkpoints-and-logs/local
```

（此处粘贴各 seed 均值、policy_loss/value_loss 趋势、平均剩余棋子、完美解率、达到「平均剩余≤5」的 iteration。）

---

## 六、最优 policy_value 配置推荐

- **默认基线**：FC 最省调参；计算允许且 CNN-Best 优于 FC 时优先 CNN。
- **需要长程依赖与全局统筹**：Transformer-2D。
- **资源有限**：小 CNN（3×3 + GN）或 FC。

（根据第五节实测结论二选一。）

---

## 七、后续优化建议

- 对 Transformer 尝试 **相对位置偏置** 或 **窗口注意力** 降低 49×49 自注意力开销。
- CNN 与 GN 组合时扫描 `norm_groups`（4/8/16）。
- 多 seed 报告 **均值±方差**，避免单次偶然。
