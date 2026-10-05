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
