# Spatial 数值诊断探针

`spatial_pretrained_backbone_probe.py --diagnose-numerics` 在原始完整骨干验收之外，复用 CPU 捕获的真实融合 embedding、Qwen3 layer hidden、4D additive mask、`position_embeddings` 和 `cache_position`，在 CPU 与 MUSA 上做相同输入回放。

诊断会分别记录：

- CPU 融合输入直接送入 CPU/MUSA language network 的结果；
- 第 3、7、11 个 Qwen3 layer 的 source eager / MUSA fallback 对照；
- 视觉第 0、12、24、26 个 layer 的同输入回放，以及 MUSA source eager / fallback 对照；
- RMSNorm、Q/K/V、RoPE、QK、softmax、PV、MLP 和残差相关的局部结果；
- `N_cpu(h_cpu)`、`N_musa(h_cpu)`、`N_cpu(h_musa)`、`N_musa(h_musa)` 的误差分解；
- 独立 CPU FP64 算子公式。该公式只用于诊断，不替换生产路径。

它要求完整 Spatial 权重、单卡 MUSA 和原隔离环境；不会保存权重或捕获 tensor。原始数值 gate、容差和失败状态保持不变，`controls_pass` 只表示诊断输入传输和 CPU 回放控制有效。

```bash
$ROUTE2_INTEGRATION_PYTHON routes/route2/model_probes/eagle_integration/spatial_pretrained_backbone_probe.py \
  --source-tree worktrees/gr00t-spatial-eager \
  --weights results/weights/Spatial-73f710e \
  --device musa --diagnose-numerics \
  --output results/spatial-pretrained-backbone-musa-diagnostics.json
```

`spatial_action_functional_probe.py` 进一步把完整585张量 backbone 的 CPU/MUSA feature 接入完整314张量 action head。默认 `embodiment_id=31` 对应 RLinf 的 `libero_franka`，state 为8个有效坐标后补到64，action为7个有效坐标后补到32；噪声是完全共享的标准高斯，图像/文字仍是合成输入。

该探针分别运行 CPU 原生 attention、CPU FP32 fallback 和 MUSA FP32 fallback，分离 action head 设备差异与骨干误差传播。每组执行4步 Euler action、固定 flow-matching loss、完整 backward 和 AdamW 单步更新，并直接比较活动分支及全部共享参数的梯度/更新向量。未参与的机器人分支另验梯度严格为0。它还调用未修改的上游 `forward/get_action`：只在函数私有 globals 中替换随机数提供器、在当前实例上固定 sample_time，以精确核对手动复现公式；Torch globals 和源类方法不修改。清零 feature 的正控制验证动作确实依赖 feature。

```bash
$ROUTE2_INTEGRATION_PYTHON routes/route2/model_probes/eagle_integration/spatial_action_functional_probe.py \
  --source-tree worktrees/gr00t-spatial-eager \
  --weights results/weights/Spatial-73f710e \
  --device musa --with-backbone --embodiment-id 31 \
  --seed 271829 --backbone-seed 271829 --cpu-threads 4 \
  --output results/spatial-action-functional-new.json
```

输出路径必须是新的。`functional_completion/pass` 与退出码只验收本轮 finite/梯度/更新和控制合同；`numerical_action_gate_pass`、原 feature mixed allclose 以及逐参数对照各自保留。它不训练 backbone，不执行真实预处理、动作反归一化、LIBERO episode 或 RLinf PPO。源 DiT 忽略 encoder mask 的行为保持原样。本轮早期v1/v2用的是0号分支与0.1噪声，只作 smoke；v3因空切片all检查保留失败，原始脚本在versions/functional-v3；正式结论采用v4。
