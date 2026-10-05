# R2-1：从功能探针推进学习基线

Pendulum 三个种子已完成；每种子 102,400 transitions，在独立 20 个 test episode 上的平均回报如下（越高越好）。

| 种子 | 初始 | 训练后 | 提升 |
|---|---:|---:|---:|
| 7 | -1191.12 | -286.31 | +904.81 |
| 17 | -1180.24 | -435.37 | +744.87 |
| 27 | -1186.04 | -606.41 | +579.63 |

3/3 提升，中位提升 744.87，满足下述工程目标；不代表统计显著性或完整 VLA 训练通过。[原始证据与汇总](../evidence/learning/pendulum-summary.json) 保留全部种子/episode。

2026-10-05，本轮预先固定训练 seed 为 7/17/27。先在 Pendulum-v1 训练每 seed 102,400 transitions（8 env × 128 horizon × 100 iterations），Adam lr 3e-4/eps 1e-5、10 epochs、minibatch 256、gamma .99、GAE lambda .95、reward scale .1、entropy coefficient 0。调用固定 RLinf 的 GAE、actor/critic loss；critic 的 value clipping 显式禁用（内部 inf，JSON 为 null），Huber delta 10。Pendulum 首轮不做观测归一化。

本轮仍为自定义普通 nn.Module ActorWorker，rollout/update 共驻，CPU EnvWorker 经显式 Ray Channel 通信。不是官方 EmbodiedRunner/FSDP Actor，也没有多卡或 GR00T。HalfCheetah 在 Pendulum 结果审查后使用同一 runner，明确开启完整观测统计。

## 为什么修改学习版

历史 `routes/route1/mujoco_ppo_worker.py` 原样保留。学习版继承其策略、环境 Worker、动作缩放、模型/优化器恢复代码，重写采样、更新与评估：

- 归一化（开启时）收集全部实际用于采样的 raw observations，每个 horizon 的统计冻结到 PPO update 完毕，之后合并；避免每轮只采第一个观测。评估不更新统计。
- critic 使用明确的可配置 clipping/Huber。inf 表示关闭 value clipping，不是把阈值设为 0。
- 每 20 iterations，在固定 validation seeds 10000–10004 上无探索评估；训练后在独立 test seeds 20000–20019 上评估一次。比较相同 seed 的初始/训练后完整权重＋normalizer bundle，raw undiscounted rewards。若开启 normalization，额外输出 final-normalizer 下的初始权重辅助对照。
- 每个种子均保存同批恢复检查、原始 episode 回报、动作每维裁剪比例、k3 形式近似 KL、normalizer 样本数和参数变化。Gaussian pre-clip 动作用于 PPO likelihood，环境执行缩放后 clipped action。

最终 checkpoint 固定为训练预算终点，不依赖 test 选择。Pendulum 的本项目初步验收目标：至少 2/3 seeds 的 test mean return gain 为正，三个 gain 的中位数 ≥300，并完整报告所有 seed/episode。该阈值是工程目标，不是官方 benchmark，也不是统计显著性结论。5 条 validation episode 用于廉价监测，不能替代最终 20 条 test。此处 seed/episode 定义是本轮实际协议；规划审计中更大的 20/50 episode 建议尚未采用。

## 首次失败记录

CPU smoke 通过后，首次 MUSA smoke 在训练前的单条观测评估失败：muDNN 2.7 `MatMul::RunWithBiasAdd` 报 `invalid parameter lda 0`。Pendulum 禁用 normalizer 后，`obs[None]` 的 NumPy 第一维 stride 为 0，旧 Torch-MUSA 张量拷贝仍触发此布局问题。Identity normalization 改成显式 C-order 深拷贝，相关布局测试与实际 MUSA 重测验证修复。完整结果以 evidence 为准，不将首轮失败覆盖为未发生。

## 重跑

在匹配旧 MUSA 的服务器，保证固定已适配源树及共用 runner 可导入：

```bash
export PYTHONPATH=/path/to/RLinf:/path/to/probes:/path/to/routes/route2/learning
export RLINF_MUSA_TORCH_DETECTION_FALLBACK=1
export MUJOCO_GL=egl
ROUTE2_PYTHON=/path/to/matched-env/bin/python

"$ROUTE2_PYTHON" -m pytest -q /path/to/routes/route2/learning/test_learning_semantics.py
"$ROUTE2_PYTHON" /path/to/routes/route2/learning/ppo_learning.py \
  --source-commit c70606f08cdca259b8dec03d4430926b5b8fac9d \
  --seed 7 --resume-check --output results/learning/pendulum-seed7.json
```

分别对 17 和 27 重跑，相同单卡顺序执行。研究仓库的 `scripts/run_route2_learning.sh` 固定了这组实际服务器命令。单批恢复保存 pending raw observations，重做更新后 normalizer 回到同一状态；不保存 simulator state。Pendulum 历史结果是在增强 normalizer/RNG 对照前运行，参数/Adam 检查通过；增强检查另由 HalfCheetah CPU/MUSA 归一化短测验证，不回填到历史结果。

## HalfCheetah 批量评估

`--batch-evaluation` 让独立 CPU 环境共享一次策略 batch 推理，不改变每个环境的 reset seed 或 episode 终止语义。保持串行仿真 step，不等同于并行 MuJoCo 或多 GPU。

[真实 CPU 对照](evaluation_cpu_validation.md) 的三个 episode 回报/长度精确一致，Python/NumPy/Torch CPU RNG 不变；[MUSA smoke](../evidence/learning/batched-smoke-musa.json) 完成 256 transitions、8 次轨迹更新并通过 normalizer/输出/RNG 恢复。短测不证明学习收益。正式 HalfCheetah 三种子命令见 `scripts/run_route2_halfcheetah_learning.sh`；开启完整观测归一化与批量评估，其余训练预算相同。

首个串行评估长测因速度停止，保留初步结果，并从初始状态重新开始正式三种子批量版本。未来结果记录 runner/evaluator/共用 worker SHA256；Pendulum 旧结果通过历史 Git 版本关联，不宣称旧 JSON 含该字段。

正式 HalfCheetah 三种子已完成，每种子 102,400 transitions、独立 20 局 test；平均回报 seed 7 为 -0.33→693.69，17 为 -0.32→884.94，27 为 -0.33→678.26。全部同批恢复检查通过。跨种子训练后均值 752.30、样本标准差 115.13，不能据三个种子声称统计显著性或任务已解决。原始报告内 runner/evaluator/共用 worker hashes 与本轮提交材料一致，见 [汇总](../evidence/learning/halfcheetah-summary.json)。
