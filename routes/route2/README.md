# 路线 2：当前 RLinf 在 S4000 默认栈上的最小适配

2026-10-08追加：两卡官方Runner完成正式actor/value学习率`5e-6 / 1e-4`、chunk5/240步/2环境的3轮更新，21项独立审计通过；完整10任务×50初始状态的初始策略评测完成，`success_once=46.8%`、`success_at_end=44.6%`，20项独立审计通过。完整suite初始基线已有证据，长期具身PPO学习收益仍待训练前后配对评测。用户已报告实例关机，未独立读取平台确认。详见[最新交接记录](../../reports/route2-handoff.md)。以下旧节点按其日期与范围解释。

2026-10-07追加：两卡完整GR00T FULL_SHARD、Actor DCP新进程精确恢复、官方Runner三轮80步和chunk5/240步两轮已通过；后者192动作块、960有效模拟动作槽、超时截断与下一轮reset。十任务pilot及Runner保存/新进程恢复/续训和末次版本2同步也已独立通过，详见[两卡最新记录](planning/two-gpu-results.md)。以下旧节点按各自日期和范围解释。

2026-10-05：在旧系统栈完成 Pendulum 与 MuJoCo HalfCheetah 三种子 PPO 学习基线，每种子 102,400 transitions，独立 test 回报均提升。HalfCheetah 从约 -0.33 提升到 694/885/678，见 [学习版说明](learning/README.md)、[Pendulum 证据](evidence/learning/pendulum-summary.json) 与 [HalfCheetah 证据](evidence/learning/halfcheetah-summary.json)。

真实 RLinf `FSDPStrategy` 已通过单卡 FP32 `NO_SHARD` 更新和两种 checkpoint 保存/恢复格式的 CPU 数值与 MUSA/MCCL 实测，见 [独立实验说明](fsdp_probes/README.md)。真实 Diffusers Attention 与固定双层 GR00T DiT 已在 S4000 上完成前向、输入 VJP 和参数 VJP；原始严格逐参数 gate 仍因解析零方向保持 false，独立语义 gate 全部通过，见 [action 接口探针](planning/action-interface-probe.md)。RLinf 官方 GR00T N1.5 wrapper 与真实 LIBERO-Spatial 8 步 episode 已通过，且真实 rollout 元数据已完成真实 GAE 的多步 PPO loss/backward/AdamW 更新，见 [真实 episode 报告](../../reports/route2-libero-real.md) 与 [多步证据](evidence/libero/libero-rlinf-ppo-multi-step-v5.json)。这证明单卡上的适配计算闭环，不代表 reward 或长期成功率提升。官方 `EmbodiedFSDPActor` 在基础源树仍拒绝 Torch 2.2；独立 FSDP1 实验源树加窄 DTensor 回退后已通过模块导入审计，尚未完成 Actor 构造、多卡、offload 或完整 runner。

追加功能节点已加载完整 Spatial 骨干585张量和完整 action head 314张量，对 RLinf 的 LIBERO 分支31执行固定噪声4步动作、loss、backward 和 AdamW 更新对照。原骨干严格数值失败保留；该固定合成输入验收独立于真实预处理和官方 PPO，详见 [功能影响报告](../../reports/route2-spatial-eagle.md)。

以下保留 2026-10-04 首轮限定闭环的证据与历史边界；首轮 256 transitions 的更新短测没有建立学习收益。

## 固定来源与环境

| 项目 | 本轮版本或路径 |
|---|---|
| 上游 | https://github.com/RLinf/RLinf |
| 固定 commit | `c70606f08cdca259b8dec03d4430926b5b8fac9d` |
| 本地分支 | `research/s4000-current` |
| 本地 worktree | `worktrees/rlinf-route2` |
| 远程源目录 | `/root/autodl-tmp/s4000-research/route2/RLinf` |
| 隔离解释器 | `/root/autodl-tmp/s4000-research/envs/route2/bin/python` |
| GPU | 单卡 MTT S4000，48 GiB |
| 系统运行栈 | Driver 2.7.0，MUSA 3.1.0，Torch 2.2.0，Torch-MUSA 1.3.0 |
| 调度依赖 | Ray 2.59.0，OmegaConf 2.3.1，Accelerate 1.15.0 |

`importlib.metadata` 报告系统 Torch 包为 `2.2.0a0+git8ac9b20`，`torch.__version__` 为 `2.2.0`。本轮从固定源目录加载 RLinf，不修改上游 `torch>=2.5.0` 的全项目依赖声明。隔离环境继承系统匹配的 Torch/Torch-MUSA；首轮补装缺少的 `regex==2024.11.6` 时使用 no-deps，未覆盖基础运行栈。

## 本轮补丁

`patches/current-minimal-compat.patch` 包含：

1. `rlinf/utils/utils.py` 优先从公开 namespace 导入 `DTensor`，缺失时回退到 `torch.distributed._tensor`。Torch 2.2 的公开 `torch.distributed.tensor` 模块存在，但没有 `DTensor`；回退修复后，当前 advantages/losses 和模型可以使用共享 utils。
2. 容器内 MTML 报错时，增加显式 `RLINF_MUSA_TORCH_DETECTION_FALLBACK=1` 选项，通过真实 Torch-MUSA 查询数量、型号。默认仍使用 MTML；该 fallback 可能初始化运行时，不提供版本兼容保证。当前真实 Worker 实测已通过正常 MTML 设备探测，fallback 另以 SDK fake 验证。
3. `Channel.create/connect` 和 Worker convenience API 增加显式 `transport="ray"`，沿用既有 Ray object put/get 接口传 CPU payload，默认 collective 不变。transport 属于句柄，随序列化保留；按名称重连需要再次指定。当前版本的 Ray get/get_batch 增加可选 consumer 名，保持 dealing dispatcher 的路由。该路径不能用于 `local=True`，也不证明默认 GPU collective 已兼容。
4. 对应回归测试加入上游原有 `test_utils.py`、`test_worker.py`、`test_channel.py`，附 EN/ZH MUSA 与 Channel 文档。新 Channel 测试使用真实 CPU Workers，主动拒绝 Worker send/recv，从公共 API 验证 sync/async/nowait/batch、同名创建、重连、空队列和 dispatcher 行为；远程 8/8 通过。

没有移除 FSDP 的版本检查，没有伪造 FSDP2 类型，也没有把 GR00T 的 vendor FlashAttention 声明为可用。

## 已取得的结果

实测时间：2026-10-04，结果见 `evidence/first-probes.jsonl`。

| 项目 | 结果 |
|---|---|
| 当前 scheduler、utils、GAE、PPO loss、MLPPolicy 导入 | 通过 |
| 本地 CPU MLP/PPO 更新 | 通过，loss `-0.4521328211` |
| 当前 scheduler 创建本地 Ray Cluster | 识别 `MUSA_GPU:MTT S4000`、1 accelerator、15 CPUs |
| 当前真实 Worker 上 MLP/PPO 前向、反向、AdamW | 通过，设备 `musa:0`、16 个梯度张量、loss `-0.4521328509` |
| 参数实际变化 | 最大变化 `0.0010045171` |
| GAE CPU 与 MUSA 数值对照 | 通过 |
| checkpoint 恢复后输出对照 | 通过 |
| 恢复后继续一次更新 | 两个模型全部参数以及 AdamW step、exp_avg、exp_avg_sq 对照通过 |
| 原生 Torch FSDP1 / MCCL Worker | 通过，单进程 `NO_SHARD`、loss `1.0672278404`，最大参数变化 `0.0010024309` |
| HalfCheetah-v5 → Ray CPU Channel → MUSA PPO 更新 | 通过，4 env × 32 horizon × 2 iterations = 256 transitions，共 8 次 optimizer step |
| MuJoCo workload 的同批 checkpoint 恢复更新 | 通过，参数与 optimizer 最大绝对误差都为 0；不含模拟器状态恢复 |
| Ray CPU Channel regression | 8/8 通过，包括真实 CPU Tensor 的 Worker 间传输与 dispatcher consumer 路由 |
| 官方 `EmbodiedFSDPActor` 导入 | 基础源树拒绝 Torch 2.2；FSDP1 实验源树加显式开关和真实 DTensor 回退后导入通过，构造/runner 未验证 |
| GR00T N1.5 MUSA patch 模块导入 | 通过；不等于 GR00T 模型可加载 |
| vendor `flash_attn.flash_attn_interface` | 缺包 |
| Transformers | 本轮隔离环境尚未安装 |

Probe 使用上游 `MLPPolicy`、`compute_gae_advantages_and_returns` 和 `compute_ppo_actor_loss`。Actor/critic 更新通过限定 `Worker` 扩展执行，value loss 使用简单 MSE。它不使用完整官方 embodied runner 或 FSDP actor，因此结果不能等同于完整 RLinf PPO benchmark。原生 FSDP1 probe 的独立证据见 `evidence/native-fsdp.jsonl`，明确 `rlinf_fsdp_package_used=false`；首轮没有测试多卡、`FULL_SHARD`、offload 或 FSDP checkpoint。2026-10-05 的独立 checkpoint 节点见前述入口，仍只覆盖单卡、world size 1、同进程新对象恢复。

MuJoCo 闭环复用路线 1 同一份 runner（`routes/route1/mujoco_ppo_worker.py`），从各自固定的源树导入 scheduler、GAE、PPO actor/critic loss。它使用普通 Gaussian MLP，11,085 个参数，CPU 仿真、MUSA 策略与优化；8 次 optimizer step 的两个 iteration loss 分别为 `0.2330210060`、`0.1646597348`，parameter L1 delta 分别为 `8.3295786097`、`6.5673401153`，与路线 1 相同 seed 的对应数值一致。完整 compact 结果见 `evidence/mujoco-loop.json`。

只有一条 held-out evaluation episode，回报从 `0.1779401849` 到 `0.1545195603`，没有收益提升。该短测验证有限数值、通信与更新闭环、同批恢复，不是收敛 benchmark，也不能据此比较两条路线的长期学习性能。

Channel 测试首次失败原因是 Ray child 无法导入 `test_channel`，需要把 `tests/unit_tests` 同样加入 `PYTHONPATH`。之后同名创建测试暴露既有异步启动竞态：actor 名称已占用但 WorkerManager 尚未注册；测试先用首次句柄的 `qsize()` 等待 actor 完成初始化再重连，最终 8/8 通过。这两次失败与处理记录见 `evidence/channel-tests.json`；没有改变生产端默认失败处理或宣称修复同名创建竞态。

## FSDP 与 GR00T 的具体边界

在远程 Torch 2.2 上读取真实 API：

| API | 是否存在 |
|---|---|
| `torch.distributed._tensor.DTensor` | 存在 |
| `DeviceMesh`、`init_device_mesh` | 存在 |
| DCP `StateDictOptions`、模型/优化器 get/set state dict | 存在 |
| DCP `Stateful` | 存在 |
| 原生 `FullyShardedDataParallel` | 存在，支持 `device_mesh` 参数 |
| `FSDPModule`、`CPUOffloadPolicy`、`MixedPrecisionPolicy`、`fully_shard` | 缺失 |
| `torch.distributed._composable.fsdp` | 模块缺失 |

当前 RLinf 的 FSDP 包将 FSDP1/FSDP2 共用符号统一导入并拒绝 Torch <2.4。即使采用 FSDP1，仍会先触发该门槛。`strategy/base.py`、`fsdp_model_manager.py`、部分 actor worker 还有直接导入新 `DTensor` namespace 的代码。原生 FSDP1 已在 MUSA 上完成单卡前反向更新，因此应把问题定位到 RLinf 当前包的 API 兼容边界，而不能据此断言底层 FSDP1 不支持。继续推进需要明确分离可用 API 与 FSDP2 专属 API，并验证 checkpoint、offload、weight sync、DeviceMesh/MCCL；仅降低版本门槛不够。

GR00T N1.5 当前构建函数显式冻结 `tune_visual=False` 和 `tune_llm=False`，可以先评估 action head 的内存和算子。它仍需要 Transformers 4.51.3、GR00T 模型代码、Diffusers 等依赖以及厂商 MUSA FlashAttention。官方 MUSA patch 在加载时把 vendor FlashAttention 绑定进 Transformers，并为 RADIO 的 CUDA capability 检查提供 sentinel。本机旧运行栈没有 vendor FlashAttention；应先测试 SDPA/eager attention 的数学正确性、梯度和内存，再决定是否移植 GR00T attention 路径。本轮没有下载约 7.6 GB 权重，也没有宣称模型可训练。

## 运行方式

使用本轮隔离解释器，在协调者的单卡 GPU 排期下运行：

```bash
ROUTE2_SOURCE=/root/autodl-tmp/s4000-research/route2/RLinf
ROUTE2_PYTHON=/root/autodl-tmp/s4000-research/envs/route2/bin/python
ROUTE2_PROBES=/root/autodl-tmp/s4000-research/route2/probes

$ROUTE2_PYTHON "$ROUTE2_PROBES/current_rlinf_probe.py" --source "$ROUTE2_SOURCE" --phase imports
$ROUTE2_PYTHON "$ROUTE2_PROBES/current_rlinf_probe.py" --source "$ROUTE2_SOURCE" --phase local --device cpu
$ROUTE2_PYTHON "$ROUTE2_PROBES/cpu_compat_checks.py" --source "$ROUTE2_SOURCE"

cd "$ROUTE2_SOURCE"
PYTHONPATH="$ROUTE2_SOURCE:$ROUTE2_SOURCE/tests/unit_tests" \
  $ROUTE2_PYTHON -m pytest --capture=no -vv -x --tb=short \
  tests/unit_tests/test_channel.py \
  -k 'TestRayChannelTransport or test_ray_get_preserves_dispatcher_consumer_routing'

# GPU jobs：由协调者序列运行。
$ROUTE2_PYTHON "$ROUTE2_PROBES/current_rlinf_probe.py" --source "$ROUTE2_SOURCE" --phase worker --device musa
$ROUTE2_PYTHON "$ROUTE2_PROBES/current_rlinf_probe.py" --source "$ROUTE2_SOURCE" --phase native-fsdp --device musa
```

`native-fsdp` 在真实 Worker 内测试原生 Torch FSDP1/MCCL 单进程 `NO_SHARD`。它不加载当前 RLinf 的 FSDP 包；首轮实测通过，支持继续探索该后端。

默认 collective 先后遇到 Torch-MUSA 1.3 顶层 `is_initialized` 未导出、Torch 2.2 私有 `_register_process_group` 缺失；仅修 lazy-init 不够。当前通过显式 Ray CPU Channel 完成限定闭环，避免移植整个新版 c10d helper；默认 collective 的适配仍单列为待解决项。下一轮可扩大相同 workload 的预算与 evaluation seeds，并开展 FSDP1 分离与 N1.5 attention 适配。
