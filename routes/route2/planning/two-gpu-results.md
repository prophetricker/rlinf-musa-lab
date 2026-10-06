# 两张 S4000：本轮验证记录

日期：2026-10-06。按用户要求暂停并停止实验进程，结果已保存本地。已通过两卡小张量通信、小模型 FULL_SHARD 更新，以及 Actor 卡 0 / Rollout 卡 1 的真实 LIBERO 轨迹更新与同步。GR00T 两 rank FULL_SHARD 已通过官方同步初始化、完整轨迹 GAE和同步前后向；**尚未通过大模型分片参数更新**。下一步仍是梯度范数/通信等待的定位。

## 环境、源码与证据范围

同节点两张 MTT S4000，保留 Driver 2.7.0、MUSA Toolkit 3.1.0、Torch 2.2.0、Torch-MUSA `1.3.0+81caf0a`、MCCL 2.11.4。宿主驱动、`/usr/local/musa` 与默认 Python 环境均不改动，探针使用研究目录的隔离环境与源码。研究盘已扩至 100 GiB，本轮协调者现场检查约 45 GiB 可用；可用空间会随 checkpoint 和结果增加而变化。

源码基线见 [官方 Actor source lock](../model_probes/official-actor-source-lock.json)：

- [RLinf](https://github.com/RLinf/RLinf)：公开上游 `c70606f08cdca259b8dec03d4430926b5b8fac9d`；既有适配基线 `82505b27d4d27f8e9af1d7e52524d524dc0f0d50`。
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)：公开上游 `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`；既有 eager / lazy PyTorch3D 适配基线 `bdd6c7426ff62c1fc07153f4ae41769cac51289b`。
- GR00T N1.5 Spatial-73f710e 权重；冻结 BF16 Eagle，FP32 action/value head；AMP 关闭，critic warmup 为 0，学习率 `1e-8`。

source lock 是既有基线。当前两卡实验的具体探针哈希、配置与逐文件源码指纹，以各次结果 JSON 为准；不能把基线 commit 当作所有后续实验修改的完整记录。v12 探针已冻结为 [update-v12-probe.py](../evidence/multi-gpu/versions/update-v12-probe.py)，SHA256 为 `4c62cdd91f6e2e73b93e74ea57d27e5a6c2d478c867d4590ff17c83f8464e282`。

## 已通过的三个层次

| 层次 | 实测结果 | 能证明的范围 |
|---|---|---|
| 独立 torchrun / MCCL | 每 rank 6/6：两次 barrier、broadcast、SUM all-reduce、list all-gather、send/recv | 同节点两卡、小 FP32 张量、显式 MCCL 通信 |
| 小模型 FSDP1 FULL_SHARD | 两 rank 均完成一次 AdamW 更新；梯度范数 `0.618055522441864`；rank 0 / 1 分别 1 / 4 个本地参数视图变化 | 真正分片的小 FP32 模型前向、反向、梯度归约与更新 |
| Actor / Rollout 分卡 | Actor 卡 0、Rollout 卡 1，各组 world size 1；真实 8 步 LIBERO → 官方 GAE/PPO，版本 0/1 同步后全部 907 个状态哈希一致 | 两组分开部署、CPU/Ray Bucket 传输、官方单 rank Actor 更新与跨卡同步 |

torchrun 数值检查包括 broadcast `[17, 23]`、SUM `[3, 6]`、all-gather `[[0, 10], [1, 11]]`、接收 `[31, 37]`。证据：[rank 0](../evidence/multi-gpu/mccl-multirank-v1.rank0.json)、[rank 1](../evidence/multi-gpu/mccl-multirank-v1.rank1.json)。这两次 JSON 记录的各操作耗时合计为 0.810670 / 0.812276 秒，仅是小张量功能探针。

小 FULL_SHARD 模型为 `Linear(8,16) → Tanh → Linear(16,4)`，使用 `use_orig_params=True`。两 rank 的 loss 均为 `0.13031381368637085`，更新后参数平方和归约均为 `6.591341972351074`。两卡的变化项合计 5 是本地分片视图计数，同一原始 weight 可跨两卡，不能当作 5 个独立原始参数。证据：[rank 0](../evidence/multi-gpu/fsdp-multirank-v2.rank0.json)、[rank 1](../evidence/multi-gpu/fsdp-multirank-v2.rank1.json)。两 rank 使用相同小批次，未在这里建立单卡/两卡数值等价或学习收益。

[Actor / Rollout 分卡 v1](../evidence/multi-gpu/official-actor-disaggregated-v1.json) 中，第一次真实 8 步更新有 312 个参数变化，Adam step `0→1`，梯度范数 `603.681884765625`，训练 ratio `1.0`。随后另收集 8 步轨迹并执行 continuation，310 个参数变化，Adam step `1→2`，梯度范数 `473.5367736816406`，ratio `1.0`。版本 0/1 的固定输入 logprob、old logprob、value 均精确相等，最大差值 0；每次核对覆盖全部 907 个状态张量，Bucket 选择集为 322 项。

最终 Actor 已到版本 2，Rollout 最后一次同步停在版本 1；不能把版本 2 的同步算作通过。v1 保存了版本 1 的训练 checkpoint，但本轮记录没有新的分卡 fresh-process 恢复结果。[分卡 sync v2](../evidence/multi-gpu/official-actor-rollout-disaggregated-v2.json) 另独立复核了初始版本 0 的 907 个状态与固定输出。

分卡 v1 的 Actor allocator peak allocated / reserved 为 27.023738 / 33.103516 GiB，Rollout 为 8.373157 / 8.476563 GiB。训练段记录约 32.18 / 32.75 秒，包含哈希和审计调用，不是纯训练吞吐。两批奖励均为 0，微小学习率用于验证更新；这些结果尚不证明任务学习。此布局仍是 Actor 单 rank NO_SHARD，也没有证明分片 Actor → Rollout 的多 rank 权重同步。

## Ray 隔离设备下的通信问题与已验证配置

默认 RLinf Ray Worker 每 rank 只看一张物理卡：rank 0 的 `MUSA_VISIBLE_DEVICES=0`，rank 1 为 `1`；两者 `device_count=1`、逻辑设备 `musa:0`、`LOCAL_RANK=0`。日志物理 bus ID 分别为 `13000` / `16000`，没有把两个 Worker 误绑到同一张卡。

- [默认 mesh v2](../evidence/multi-gpu/worker-collectives-mesh-v2.json)：两 rank 的首次 SUM 均失败；默认 group 名称 `undefined`，实际 MUSA backend 为 `ProcessGroupMCCL`。
- [显式 mccl v1](../evidence/multi-gpu/worker-collectives-mccl-v1.json)：同样在首次 SUM 失败，排除了“只要显式改 backend 就能解决”的解释。
- [原 transport 日志](../evidence/multi-gpu/worker-collectives-mccl-v1.transport.txt)：建立 `via P2P/IPC` 通道时，`transport/p2p.cc:184` 报 `Musa failure 'invalid argument'`；随后上层报 `ProcessGroupMCCL.cpp:1129, unhandled musa error`。

Torch-MUSA 1.3 对应位置在创建 MCCL communicator 的 `mcclGroupEnd()`，先于真正 collective 执行。因此这些失败不能推导为 FP32 SUM、BF16 或 base/list all-gather 本身不受支持，也不能由 group 显示 `undefined` 推导为用错 backend。

仅给本次作业设置 `MCCL_P2P_DISABLE=1`，保留 SHM 后，[mesh 通信探针](../evidence/multi-gpu/worker-collectives-p2p-disabled-v1.json) **每 rank 8/8 通过**：SUM、MAX、AVG，FP32/BF16 的 list all-gather 与 `all_gather_into_tensor`，以及 FP32 `reduce_scatter_tensor`。输入为 16 元素小张量，均核对数值并同步。首次 SUM 含 communicator 初始化，rank 0 / 1 耗时 1.223708 / 1.244820 秒；其余操作约 0.00069–0.00262 秒，不能据此估算大模型通信性能。

[新 transport 日志](../evidence/multi-gpu/worker-collectives-p2p-disabled-v1.transport.txt) 明确显示 `via direct shared memory`，两 rank 都 `Init COMPLETE`。这是当前固定软硬件与 Ray 隔离进程下已验证的兼容配置；尚未定位 P2P/IPC 失败的最终底层原因，也尚未测量 SHM 大张量吞吐。没有关闭 RLinf 的 GPU 隔离，或修改宿主驱动。

## GR00T 两 rank FULL_SHARD：v12 到达 Adam，但更新失败

[v12 partial JSON](../evidence/multi-gpu/fsdp-official-actor-update-v12.partial.json) 使用 `MCCL_P2P_DISABLE=1` 和官方 `sync_module_states=True`。两 rank 初始化权重有限，各有 1,364,245,473 / 1,364,245,472 个本地参数元素，合计 2,728,490,945。初始化 snapshot 的 peak allocated 均为 11.718463 GiB，peak reserved 为 18.359375 / 18.361328 GiB；这是初始化阶段统计，不是完成训练后的峰值。

这次输入为单卡先前版本 3 采集的既有真实 8 步 Trajectory fixture：`official-actor-continuation-v1.pt`，SHA256 `35d92973138157c1f1db57a7df2f86ed074d753ed657ae235532a27d1eaf84ad`。本次没有重新进行 on-policy rollout，也不是学习实验。

每 rank 先对相同完整 8 步序列执行官方 receive-side 处理、GAE 与优势归一化，再沿时间轴切片：rank 0 取 0–3，rank 1 取 4–7。`dones/terminations/truncations/prev_values` 各留 5 行，其他字段各留 4 行；切片保留完整轨迹的 `loss_mask_sum`，不重算半条轨迹的 GAE 或归一化。全局 batch 8、每卡 micro batch 1、update epoch 1，目标为每 rank 累计四个 micro batch 后各执行一次更新。

两 rank 的完整 GAE 汇总一致：有效 mask 比例 `1.0`，reward `0.0`，优势均值 `0.0`、最小/最大 `-1.0901256799697876 / 1.745521903038025`，return 均值 `0.5657316446304321`。两个半批的 advantages 和 returns 均有限，样本索引合并恰为 0–7。此记录未额外保存切片前全部优势/return 张量的跨 rank 哈希，不能把汇总一致升级为逐元素一致结论。

[v12 失败堆栈](../evidence/multi-gpu/fsdp-official-actor-update-v12.failure.txt) 显示 rank 1 已进入官方 `optimizer_step()`，在 Torch AdamW `_single_tensor_adamw` 的 `exp_avg.lerp_(grad, 1 - beta1)` 报 `MUSA error: unknown error`。执行路径已走到 optimizer；由于 MUSA 异步执行，报错位置仍可能只是之前错误的暴露点。不能把它认定为训练挂起，也不能认定所有前后向 kernel 已同步完成或两卡参数更新成功。未取得 `after` 快照、成功的 training metrics 和两卡更新验收。

初始化时每 rank 已有 322 个 Adam 状态，step 均为 0；官方 optimizer warmup 会预建这些状态。空 orig-param 分片真实反向时可为 `grad=None`，其预建状态可能保持 step 0。因此后续验收应区别空分片与实际参与更新的非空分片，要求后者 `0→1`，而不是要求全部预建状态均为 1。

[历史 init-v8](../evidence/multi-gpu/fsdp-official-actor-init-v8.json) 使用独立加载 opt-in，只验证初始化；其 [冻结脚本](../evidence/multi-gpu/versions/init-v8-probe.py) 保留用于追溯。v12 已通过默认同步初始化，因此独立加载绕过 broadcast 不需要作为最终方案。

## 暂停前新增诊断

[v14阶段日志和堆栈](../evidence/multi-gpu/fsdp-official-actor-update-v14.diagnostic.txt)及[partial JSON](../evidence/multi-gpu/fsdp-official-actor-update-v14.partial.json)表明：两个rank各完成四次`train_micro_batch`，每次返回后显式`torch.musa.synchronize()`通过；随后进入optimizer。四个微批次约38.09秒，不含初始化和后续梯度裁剪。[rank0 events](../evidence/multi-gpu/fsdp-official-actor-update-v14.rank0.events.jsonl)与[rank1 events](../evidence/multi-gpu/fsdp-official-actor-update-v14.rank1.events.jsonl)保留各边界。[冻结v14脚本](../evidence/multi-gpu/versions/update-v14-probe.py)保留当时诊断与验收逻辑，不能当作已通过的训练实现。

faulthandler显示两rank停留在`get_grad_norm_for_mixed_precision`，最终MCCL报`enqueue.cc:371 Musa failure 'wait operation timed out'`，rank1异常退出。不能沿用v12堆栈把首因定为AdamW，也不能据此定为某个归约算子不支持。尚需区分本地梯度范数耗时、流/事件依赖和通信等待。rank0有227个非空梯度视图，rank1有98个；rank1最大的单个梯度是一维150,994,944元素。完整元数据：[rank0](../evidence/multi-gpu/fsdp-official-actor-update-v14.rank0.gradients.json)、[rank1](../evidence/multi-gpu/fsdp-official-actor-update-v14.rank1.gradients.json)。

[补充标量通信](../evidence/multi-gpu/worker-collectives-scalars-v1.json)在相同隔离mesh和`MCCL_P2P_DISABLE=1`下每rank10/10通过，包括真正零维FP32 SUM/MAX。此前8项探针只覆盖16元素；补充结果支持标量通信可用，不能推广为所有训练通信顺序都正确。

[独立大向量norm](../evidence/multi-gpu/large-norm-direct-v1.json)覆盖1,048,577至150,994,944元素的FP32常量向量（offset=3），六种长度全通过；与解析值的最大相对误差约6.72e-8。因此“大向量norm本身必然失败”的假设未获支持。它没有复现真实梯度值、分片offset或多rank流依赖。

[分块norm诊断](../evidence/multi-gpu/large-norm-chunked-v1.json)仅是候选实现测试，用户暂停时被终止，顶层状态仍是预设fail/最后一行running；[暂停记录](../evidence/multi-gpu/two-gpu-stop.json)说明终止原因。已完成行可单独读取，但不能把整组算通过或部署进官方训练。末段小通信测试曾与该独立诊断重叠，不使用其时间建立性能结论。`musa_fsdp_norm.py`未注册到生产模型扩展。

最小Adam探针的[原始v1](../evidence/multi-gpu/adam-shards-v1.json)在空张量有限性断言失败，非空scalar/offset/large检查通过；空张量的optimizer调用本身已返回。当前检查已对空分片使用数学上的空集判定，修正未重跑。v13也因相同的空状态检查问题提前终止，不计为新backend失败。

暂停后的恢复顺序：先复核真实梯度范数与通信等待，再验收两个rank的非空参数/Adam状态变化；随后两rank DCP新进程恢复、分片Actor→Rollout同步及完整Runner。当前脚本的`--phase train/recover`是已准备入口，尚未运行、不能声明两rankcheckpoint通过。

## 复现配置与命令

以下在已准备好源码、权重与依赖的研究实例内执行；不含连接/认证信息。命令是同配置复现入口，输出使用新的 `repro` 路径。若再次运行，换一个不存在的结果目录。复现历史 v12 时，先把本仓库冻结的 `versions/update-v12-probe.py` 部署到研究目录 `shared/update-v12-probe.py`；不要用正在修改的当前脚本冒充 v12。

```bash
lab_root=/root/autodl-tmp/s4000-research
lab_python="$lab_root/envs/route2/bin/python"
lab_result_dir="$lab_root/route2/results/two-gpu-repro"
mkdir -p "$lab_result_dir"
export PYTHONPATH="$lab_root/envs/route2-integration/lib/python3.10/site-packages:$lab_root/envs/route2-integration"
```

独立 torchrun 小通信与小模型分片探针：

```bash
"$lab_python" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$lab_root/shared/probe_route2_mccl_multirank.py" \
  --output "$lab_result_dir/mccl.json"

"$lab_python" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$lab_root/shared/probe_route2_fsdp_multirank.py" \
  --output "$lab_result_dir/tiny-fsdp.json"
```

Ray 隔离环境的默认与显式 group 对照，再验证 SHM 配置：

```bash
env -u MCCL_P2P_DISABLE -u MCCL_SHM_DISABLE MCCL_DEBUG=INFO \
  "$lab_python" "$lab_root/shared/probe_route2_worker_collectives.py" \
  --source "$lab_root/route2/RLinf-official-actor" --mode mesh \
  --output "$lab_result_dir/mesh-default.json"

env -u MCCL_P2P_DISABLE -u MCCL_SHM_DISABLE MCCL_DEBUG=INFO \
  "$lab_python" "$lab_root/shared/probe_route2_worker_collectives.py" \
  --source "$lab_root/route2/RLinf-official-actor" --mode mccl \
  --output "$lab_result_dir/mccl-default.json"

env -u MCCL_SHM_DISABLE MCCL_P2P_DISABLE=1 MCCL_DEBUG=INFO \
  "$lab_python" "$lab_root/shared/probe_route2_worker_collectives.py" \
  --source "$lab_root/route2/RLinf-official-actor" --mode mesh \
  --output "$lab_result_dir/mesh-shm.json"
```

真实 LIBERO Actor / Rollout 分卡训练入口：

```bash
"$lab_python" "$lab_root/route2/model_probes/official_actor_rollout_probe.py" \
  --rlinf-source "$lab_root/route2/RLinf-official-actor" \
  --gr00t-source "$lab_root/route2/Isaac-GR00T-official-actor" \
  --model-path "$lab_root/route2/weights/Spatial-73f710e" \
  --placement-mode disaggregated --phase train --iterations 1 --steps 8 \
  --checkpoint "$lab_root/route2/checkpoints/disaggregated-repro" \
  --continuation-trajectory "$lab_result_dir/disaggregated-continuation.pt" \
  --output "$lab_result_dir/disaggregated.json"
```

GR00T 分片 v12 的既有轨迹 fixture 入口；预期用于复核当时的初始化/GAE和失败位置，成功验收仍待修复：

```bash
env -u MCCL_SHM_DISABLE RLINF_MUSA_FSDP_INDEPENDENT_INIT=0 \
  MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN \
  "$lab_python" "$lab_root/shared/update-v12-probe.py" \
  --rlinf-source "$lab_root/route2/RLinf-official-actor" \
  --gr00t-source "$lab_root/route2/Isaac-GR00T-official-actor" \
  --model-path "$lab_root/route2/weights/Spatial-73f710e" \
  --trajectory "$lab_root/route2/results/official-actor-continuation-v1.pt" \
  --output "$lab_result_dir/full-shard-v12.json"
```

小探针、分卡单 rank 更新、分片初始化和完整分片 PPO 更新是独立验收层次。后续仍需两 rank GR00T 参数更新、分片 checkpoint 新进程恢复、分片 Actor → Rollout 同步、完整官方 Runner，以及独立多任务/多 seed 学习评估。进入顺序见 [多卡计划](multi-gpu-entry.md)。
