# R2-2：Torch 2.2 / MUSA 3.1 的 RLinf FSDP1 源码审计

下一步应先让真实 `RLinf FSDPStrategy` 在 S4000 上完成单进程 `NO_SHARD` 更新与恢复，再接独立 Actor/Rollout 权重同步。原生 Torch FSDP1 已通过更新探针；目前的阻塞来自 RLinf 共用导入、DCP 保存签名和尚未验证的后端接口，不能直接归结为显卡不支持 FSDP。

审计日期：2026-10-05。本轮只读源码、核对公开 PyTorch API 并保存设计，没有修改上游源码、启动远程任务或运行 GPU。服务器尚未可连接，以下新节点均为待执行，不是测试结果。

## 1. 固定来源与已知证据

| 项目 | 固定来源 |
|---|---|
| RLinf 上游 | `c70606f08cdca259b8dec03d4430926b5b8fac9d`，https://github.com/RLinf/RLinf |
| 当前适配提交 | `3e7329c629293f1b0f9c331b5bb24eccc3e3f142` |
| 本地源树 | `worktrees/rlinf-route2`；下文 RLinf 行号取自这个适配提交 |
| 已验证运行栈 | Driver 2.7.0 / MUSA 3.1.0 / Torch runtime 2.2.0 / Torch-MUSA runtime `1.3.0+81caf0a` |
| Torch distribution metadata | `2.2.0a0+git8ac9b20` |
| PyTorch API 源码参照 | `8ac9b20d4b090c213799e81acf48a55ea8d437d6`，对应 v2.2.0 |
| Torch-MUSA 源码参照 | 本地 `worktrees/torch-musa-route4` 的公开 v1.3.0，`73c9f5ba21dc6a0dc0ed07ef027c804dc53644a9`；它不是已安装 wheel 的已确认构建源码 |

前一轮的 [原生 FSDP 证据](../evidence/native-fsdp.jsonl) 为：MCCL、world size 1、`NO_SHARD`、loss `1.0672278404`、最大参数变化 `0.0010024309`，并明确 `rlinf_fsdp_package_used=false`。此前的 MuJoCo 与恢复结果来自普通模块 Worker；它们没有覆盖下文的 RLinf strategy、FSDP checkpoint 或独立 Rollout。

前轮远程 API 检查已确认 `DeviceMesh`、`init_device_mesh`、DCP `StateDictOptions`、模型/优化器 get/set state dict、`Stateful` 和原生 FSDP1 存在。FSDP2 的 `FSDPModule`、`fully_shard`、`MixedPrecisionPolicy`、`CPUOffloadPolicy` 以及 `_composable.fsdp` 不存在。公开源码核对可以说明预期签名，下一轮仍须记录已安装运行时的 `inspect.signature` 和模块路径，避免把厂商 wheel 当作未修改的 PyTorch。

## 2. 导入分离需要覆盖的完整路径

当前最小 strategy 的导入链为：

```text
fsdp/__init__.py
  └─ strategy/fsdp.py
       ├─ strategy/base.py
       │    └─ strategy/checkpoint.py ── fsdp/utils.py
       └─ fsdp/utils.py ── rlinf/config.py、accelerate、Transformers helper
```

`FSDPStrategyBase.create()` 已经按配置懒加载具体 strategy，无需重做工厂。真正的问题是包初始化与这些共用模块在选择 FSDP1 前就要求 FSDP2 类型存在。

| 位置 | 当前行为与最小处理建议 |
|---|---|
| `rlinf/hybrid_engines/fsdp/__init__.py:24`、`:37`、`:54` | Torch ≥2.6 导入公开 FSDP2，2.4～2.5 导入 composable FSDP2，低于2.4整包抛错。将 FSDP1 的 `FSDP/CPUOffload/BackwardPrefetch/ShardingStrategy` 与 FSDP2 能力分开。旧栈只开放已审计的 FSDP1 路径；请求 `strategy=fsdp2` 时给出明确的不支持错误，保留新栈现有导出。 |
| `strategy/base.py:30`、`:34`、`:87` | 新 namespace 的 `DTensor` 在 Torch2.2 缺失；统一使用已存在的真实 legacy fallback。`FSDPModule` 在此主要用于注解，使用延迟注解和 `TYPE_CHECKING`；工厂的 FSDP2 分支验证真实 API 再导入，不能把缺少类型替成 `nn.Module` 或假类。 |
| `strategy/checkpoint.py:28`、`:39` | `FSDPModule` 用于注解。运行时必须保留真实 `FSDP`、`DTensor` 和 `Stateful`；延迟 FSDP2 注解，不能跳过真实 checkpoint 代码。 |
| `fsdp/utils.py:47`、`:443`、`:536` | 顶层同时导入 `CPUOffloadPolicy/FSDPModule/MixedPrecisionPolicy/fully_shard`。只在 `apply_fsdp2_to_model()` 实际调用时加载 FSDP2 运行符号，注解延迟求值。这个函数以及 `strategy/fsdp2.py:24` 仍使用真正的 FSDP2，旧栈调用必须失败且说明原因。 |
| `fsdp/utils.py:44`、`:149` | 顶层 Transformers `get_module_class_from_name` 使最小 MLP strategy 依赖尚未安装的 Transformers。可以把该 helper 移到 `_resolve_module_classes_to_wrap()` 的实际使用点。首个 probe 关闭 auto-wrap，保留其它模型的原始行为；否则需在隔离环境补齐与 Torch2.2 匹配的 HF 依赖。 |
| `fsdp/fsdp_model_manager.py:23`、`:38`、`:530` | Manager 同样需要真实 DTensor fallback 与延迟 FSDP2 注解。`:26` 的 AutoConfig/AutoModel 导入是下一阶段 HF 依赖，strategy probe 不必先导入 Manager。 |
| `workers/actor/fsdp_actor_worker.py:21`；`workers/inference/fsdp_inference_worker.py:24` | reasoning/inference 的 DTensor 同样来自新 namespace；若接这些 Worker 必须处理。本轮首个 strategy probe不加载它们，避免把范围扩展到整个 reasoning 栈。 |
| `hybrid_engines/weight_syncer/base.py:22`、`bucket_syncer.py:20`、`patch_syncer.py:22`、`:23` | 独立 Rollout 同步还有一组 DTensor/Replicate 新 namespace 导入，FSDP 导入分离不会自动修复。`:Replicate` 应从真实 legacy placement namespace fallback。weight_syncer 包 `__init__.py:15`～`:23` 会加载 bucket、patch、compressor，即使只选 bucket也会碰到 patch 导入；须处理包导入链或保留这些导出的兼容导入。 |

建议共用 FSDP 模块复用包里已有的真实 `DTensor` fallback，weight_syncer 复用现有 shared utils 的真实 DTensor 或同等窄 fallback，避免多份不一致的兼容判断。本次搜索未发现上述模块运行时使用 `get_type_hints()`；增加延迟注解后仍要检查文档构建和类型工具，不能只测 import。

## 3. 已知 API 差异与需要实测的行为

导入成功后仍须逐层验证，以下分类避免把潜在问题写成已失败的事实。

### 3.1 DCP 保存签名有明确差异

当前 `strategy/base.py:236` 使用：

```python
dcp.save({"fsdp_checkpoint": training_state}, checkpoint_id=dcp_save_path)
```

公开 PyTorch2.2 的 [save 签名](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/state_dict_saver.py#L37) 要求 `storage_writer`，没有 `checkpoint_id`。最小兼容处理是在旧签名分支使用真正的 `dcp.FileSystemWriter(dcp_save_path)`，保留现有 `Stateful` payload；先读取已安装签名，再实现局部分支。可以用显式版本/能力判断，不能捕获任意保存异常后换方式重试，因为真实 I/O 或 schema 错误不应被吞掉。

当前 `strategy/base.py:348`、`:355` 已用 `FileSystemReader` 和 `storage_reader=`，与2.2 [load 签名](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/state_dict_loader.py#L33) 匹配；不能笼统说 save/load 都因 `checkpoint_id` 不兼容。2.2 的 save/load 真实支持 `Stateful`，而非仅支持普通 dict。

2.2 [StateDictOptions](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/state_dict.py#L81) 包含 `full_state_dict/cpu_offload/ignore_frozen_params/keep_submodule_prefixes/strict`。当前 base 的 `full_state_dict/cpu_offload` 参数存在；本轮路径没有要求后来加入的 `broadcast_from_rank0`。可用类型不等于 MUSA 上 DCP 已通过，还需实际执行保存与恢复。

### 3.2 DeviceMesh 与设备选择需要一个明确的起点

`fsdp/utils.py:94` 先执行不带 backend 的 `init_process_group(timeout=...)`，然后 `:95` 构建 MUSA mesh。公开 Torch-MUSA1.3 的 `distributed/device_mesh.py:21`～`:25` patch 会在 mesh 自己初始化默认组时查 `default_device_backend_map` 选择 MCCL；RLinf 预先建立默认组会让这条选择逻辑不再执行。

不能据此宣称已安装环境一定失败。首个 probe 在真实 MUSA Worker 内先设 `musa:0`，显式 `init_process_group(backend="mccl", rank=0, world_size=1, timeout=Cluster.get_collective_timeout())`，再调用真实 `create_device_mesh(1)` 和 `gradient_reduction_group(mesh)`，记录 backend、device type、mesh name和 timeout。这一做法验证 strategy，其后再单测 RLinf 未预初始化组的入口；若需要改生产函数，应从当前 Worker 设备选 backend，保留 CUDA/NPU/CPU 行为与配置超时。

`strategy/fsdp.py:181` 传整数 `device_id`，`:184` 开启 `sync_module_states`。此前原生探针传 `torch.device("musa")`、没有 mesh，也没有同样的 sync 参数。因此这些组合是新增实测点；若整数在已安装 runtime 上解析错误，再改成由 Worker 后端与 `LOCAL_RANK` 组成的显式设备，不预先假定错误。首轮使用已经物化的 MUSA FP32 模型，避免同时测试 meta 初始化。

### 3.3 梯度裁剪和 offload 不能靠删 API 检查解决

`strategy/fsdp.py:363` 要求 FSDP 根对象 `_all_handles`，随后 `NO_SHARD` 分支 `:370` 调用 `torch.nn.utils.clip_grad_norm_`。公开 Torch2.2 [runtime lazy-init](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/fsdp/_runtime_utils.py#L215) 确实创建 `_all_handles`；它是私有接口，首个 forward 后仍须在厂商 runtime 核实，不能把它记为已知缺失。

旧 MUSA 的 norm/foreach 行为尚未由原生 FSDP探针验证。首个 strategy update 必须实际经过 `strategy.clip_grad_norm_()`，记录梯度、norm和裁剪后全局 L2，对照 CPU。若报 unsupported op，先定位到具体算子，再给 NO_SHARD 单独加经数值对照的窄 fallback；不能让 probe自带裁剪绕过这个调用然后宣称 RLinf strategy通过。

offload `strategy/fsdp.py:216`、`:261` 会读写 flat parameter私有 shard/view/cache属性并释放 storage，`optimizer :307/:325` 也移动状态；它们未被前向更新覆盖。首个 probe关闭 offload，后续做一次 offload→onload→下一次更新对照再验收，不在本轮同时修改这块。

### 3.4 Checkpoint 还包含控制流与 RNG

`strategy/checkpoint.py:102` 用真实 DCP get_state_dict，`:181` 用 set_state_dict；`:115` 在默认组执行 `all_gather_object`，保存所有 rank RNG。Torch2.2 会根据组支持的设备选择对象通信设备，不能仅因 RNG字典在CPU就断定必然走CPU/Gloo。必须在MCCL上实测包括序列化的 `uint8`/`int64` 和 DCP内部 collectives，保留实际失败日志；若需要 CPU控制组，明确生命周期和参与者后再设计。

`utils/utils.py:736`、`:755` 的 RNG包含 Torch CPU、NumPy、Python及当前 MUSA设备状态。`fsdp_model_manager.py:631` warmup会创建 Adam状态；其 shared helper `utils/utils.py:720` 会重置步数到0。恢复测试需要对照这一步，避免 warmup改变优化轨迹。

## 4. 收紧后的最小 probe 与验收顺序

建议新增路线二私有 probe `probes/fsdp1_backend_probe.py`，按下表实现阶段开关；该文件本轮尚未创建。先追加上游现有组件测试，再由协调者在单卡时段顺序运行，不把所有条件一次混进官方 runner。

| 阶段 | 真实执行路径 | 通过条件与证据 |
|---|---|---|
| A：API/导入 | 从固定source导入 FSDP package、utils、base、checkpoint、FSDPStrategy；读取 FSDP、DCP和MixedPrecision签名；请求 FSDP2的负例 | FSDP1链正常导入；缺少FSDP2时明确报错；记录真实模块路径和版本，不能在 `sys.modules` 注入假的可用类型 |
| B：strategy update | 一个真实 MUSA Worker；显式MCCL→RLinf create_device_mesh→`FSDPStrategyBase.create(strategy="fsdp")`→`wrap_model`→`before_micro_batch`→backward→`clip_grad_norm_`→真实AdamW.step | FP32 tiny MLP和固定batch有限loss/梯度，参数变化非零，strategy类确实是RLinf FSDPStrategy；先单microbatch，随后两个microbatch验证no_sync/累积缩放 |
| C：checkpoint | 同一个strategy的 `save_checkpoint/load_checkpoint`；先 `local_shard`，再独立 `dcp`；在新Worker或独立进程重建后恢复 | 相同后续batch的输出、loss、全部参数、Adam step/exp_avg/exp_avg_sq、LR scheduler和各RNG与连续训练对照；每种格式单独状态，DCP失败不能靠local_shard通过替代 |
| D：manager/官方Actor | 保留前述strategy，通过限定 `FSDPModelManager + Worker` 扩展注入小模型并走 setup_model_and_optimizer、optimizer_step；之后原始 EmbodiedFSDPActor+MLPPolicy | 分别记录“RLinf strategy”“Manager扩展”“官方Actor”级别；不得把B的成功写成官方Actor训练通过；逐项补HF/数据依赖与真实batch契约 |
| E：独立Rollout | 独立CPU Rollout Worker，用实际BucketWeightSyncer和显式Ray CPU Channel做v0/v1/v2同步；再接官方两个Worker的sync接口 | 权重/buffer完整覆盖，策略版本与输出对照；记录所用transport，官方collective仍要独立验收 |

阶段B优先沿用原生探针的 `Linear(17,32)→Tanh→Linear(32,6)`，固定FP32 batch与CPU初始权重，不下载模型、不启用AMP/BF16/compile/FlashAttention/梯度checkpointing。这样失败可以定位为包装、mesh、裁剪或state管理，而非VLA模型算子。

配置至少显式设置：

```yaml
model:
  model_type: mlp_policy
  is_lora: false
fsdp_config:
  strategy: fsdp
  sharding_strategy: no_shard
  disable: true  # 仅关闭auto-wrap；wrap_model仍真实建立FSDP
  use_orig_params: true
  cpu_offload: false
  forward_prefetch: false
  backward_prefetch: null
  limit_all_gathers: true
  enable_gradient_accumulation: false
  mixed_precision:
    param_dtype: fp32
    reduce_dtype: fp32
    buffer_dtype: fp32
optim:
  clip_grad: 1.0
```

这里只是阶段B的strategy配置，不能直接当作官方Actor完整YAML。阶段D另需补 `precision/amp_autocast/grad_scaler/Adam/LR/global_batch_size/micro_batch_size/placement` 等默认，启用实际Manager构建与step，不复用一份不完整配置硬塞给全runner。

阶段B跑通后可使用上游 `MLPPolicy.default_forward` 和 PPO actor/critic loss替换MSE固定batch，再沿用既有MuJoCo workload接采样。第一阶段MSE更新证明训练后端功能，第二阶段真实PPO数据证明模型/算法接口；均不替代多seed的学习收益验收。

## 5. Checkpoint 的具体对照方法

使用两条轨迹和同一组物化batch X/Y：轨迹A执行X更新→保存→Y更新；轨迹B在新Worker/进程重建模型、优化器与scheduler→恢复→Y更新。恢复之前不要使用保存文件里的下一步输出来“验证”恢复，必须独立执行相同前向/反向。

保存后先记录命名完整参数、buffers、Adam状态、LR与scheduler计数、随机状态以及策略版本。恢复之后核对保存点，随后核对Y更新后的全部参数和状态；用独立随机抽样验证Torch CPU/MUSA、NumPy和Python RNG恢复。FP32记录最大绝对/相对误差及阈值，不要求不同设备天然bitwise一致。纯权重文件 `full_weights.pt` 不足以验收训练恢复。

`local_shard` 只验收同world size、同模型拓扑，本次world1/NO_SHARD不等于跨拓扑恢复。`dcp` 必须走真实Writer/Reader、Stateful、FSDP get/set state dict和磁盘文件。注意Manager当前 `save_checkpoint(step=...)` 没有把step写进Checkpoint payload；`optimizer_steps` 是Manager自己的计数，与Adam每参数step不同。官方runner从目录名恢复global_step，并在下一轮调用 `set_global_step`（`runners/embodied_runner.py:193`、`:498`）。首个probe关闭critic warmup；精确恢复声明必须同时记录所恢复的训练计数与策略版本，不能仅比较Adam state。

## 6. 独立 Actor/Rollout 接口的最小验证

现有MuJoCo runner的Actor同时执行推理和更新，没有独立Rollout模型，所以这一关确实尚未做。

官方路径：`EmbodiedFSDPActor.sync_model_to_rollout()`（`workers/actor/embodied_fsdp_actor_worker.py:138`）取state dict，经Bucket/Patch WeightSyncer调用`:151`的 `Worker.broadcast`、`:163`的 `Worker.recv`。`MultiStepRolloutWorker.sync_model_from_actor()`（`workers/rollout/hf/huggingface_worker.py:626`）对应broadcast与send，应用后`:662`更新version。`runners/embodied_runner.py:200` 同时发起两边调用后等待，避免互相等待。

这些函数不使用现有Ray Channel，所以默认collective仍会遇到此前未适配的MUSA `is_initialized` 与 `multi_channel_pg.py:712` 的Torch私有 `_register_process_group`。本节点先通过WeightSyncer公开 `SendFn/RecvFn` 回调注入显式Ray CPU Channel，不先移植整份c10d helper，也不改已有官方broadcast默认行为。

最小验证用一个S4000 FSDP Actor Worker和一个CPU Rollout Worker，Actor只负责train，Rollout拥有独立未包装的同构小模型：

1. 读取Actor `get_model_state_dict(cpu_offload=True, full_state_dict=True)`，确认named keys对齐未包装模型；模型带一个persistent buffer和一个冻结参数，参数清单明确同步范围。world1普通tensor可用，若出现ShardedTensor/DTensor不能假设Bucket materialize可处理所有类型。
2. 真正 `BucketWeightSyncer(bucket_device="cpu", bucket_dtype=None)`，选很小的bucket_size让模型至少拆成两桶；通过已验证的 `Channel(transport="ray")` 发送和接收回调运行 `sync/apply`。关闭NVComp、delta patch与stream优化。保留首桶的total_buckets/version，检查每个期望key确实到达；仅 `load_state_dict(strict=False)` 成功不说明完整覆盖。
3. 同步v0后用固定observations/actions比较Rollout与Actor的mean、value、logprob。Actor更新后，同步前Rollout仍是v0，Actor必须已经有变化；同步v1后输出与新Actor对照且Rollout version=1；再做v2确认重复同步不会混入上一轮bucket。
4. 若涉及随机动作，用同分布参数和固定actions核对logprob，再单独控制RNG核对采样；不能用Actor/Rollout自然随机动作不相等证明权重同步失败。
5. 小模型回调路径通过后，才给官方两Worker增加显式可选择的transport并跑其sync接口，或另行解决collective。前者的结果称“WeightSyncer + 独立Worker Ray CPU同步”，后者通过才称“官方Actor/Rollout同步”。性能与GPU直接传输另做测量。

首个独立Rollout用CPU是为了在单S4000上避免同时放两份GPU模型与offload竞态。将来MUSA Rollout、多卡、FULL_SHARD需重新验证key materialization、跨rank参与、bucket生命周期和显存，不能自动推广这个结果。

## 7. 下一次实际实施清单

1. 开机后先保存已安装FSDP/DCP签名与module file，在隔离route2环境核实本审计的确定差异。源码继续固定，系统driver/Toolkit/Torch不动。
2. 提交第一份窄补丁：FSDP1/FSDP2真实导入分离、DTensor namespace、MLP路径的按需HF helper。修改现有 `tests/unit_tests/test_fsdp.py`：当前该文件`:48`顶层导入FSDP2，也必须将FSDP2专属测试按能力skip/按需导入，确保旧栈可以收集FSDP1测试；保留新栈原测试。运行时不伪造缺少类型。
3. 实现阶段A/B probe，真实RLinf strategy包装、mesh、裁剪、更新。遇到失败保留精确调用与算子；先修这一个失败，未通过之前不接GR00T或完整runner。
4. B通过后给DCP save增加Writer兼容，分别执行local_shard/DCP的阶段C。未通过DCP时可以记录local_shard进展，但不把DCP标为完成。
5. B/C通过后再开Manager/独立Rollout这两块；Bucket/namespace回归加入原有 `test_weight_syncer.py`。参数/版本/输出证据齐全后，再尝试官方MLPActor与MuJoCo闭环。

本节点阶段报告应保存：上游和适配commit、安装包版本/module path、策略与backend、world size、精确配置、命令、finite/parameter delta、checkpoint格式及误差、sync版本/keys/buckets、峰值内存、失败调用与边界。多卡分片、offload、VLA/BF16与学习收益各自单独验收。
