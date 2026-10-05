# 跨进程 FSDP1 checkpoint 实验

2026-10-05，S4000/MUSA/MCCL 的 **local_shard 与 DCP 均通过跨进程保存、恢复和继续更新**。两种格式各执行独立 save、restore 命令，四条命令退出码均为 0（由执行实机命令的协调者确认；紧凑 JSONL 本身不包含退出码）。恢复后的第1步完整训练状态、四类 RNG、下一段随机样本与继续更新后的全部参数/Adam/scheduler 完全相同，继续训练参数和 optimizer 最大绝对误差均为 `0.0`。

原始紧凑记录为 [local_shard-save.jsonl](../evidence/checkpoint-process/local_shard-save.jsonl)、[local_shard-restore.jsonl](../evidence/checkpoint-process/local_shard-restore.jsonl)、[dcp-save.jsonl](../evidence/checkpoint-process/dcp-save.jsonl) 和 [dcp-restore.jsonl](../evidence/checkpoint-process/dcp-restore.jsonl)。命令、退出码来源、文件哈希和本地备份核验汇总见 [checkpoint-process-run-validation.json](checkpoint-process-run-validation.json)。

[checkpoint_process_probe.py](checkpoint_process_probe.py) SHA256 仍为 `280e0efd3a2e85c6beb157712ce1d68a3b9f34375d5f7763f6530538f7c1a17f`。本地 AST、py_compile、Ruff、格式检查与 `--help` 通过；旧 [checkpoint_probe.py](checkpoint_probe.py) 保持已验证的 `da07c559…` 字节，生产源码和两份既有补丁不修改。

新探针只复用旧 helper 的模型、更新、active clipping、AdamW、StepLR、state/RNG 比较。一个命令仅执行 save 或 restore；两者必须是独立 Python 命令及独立真实 RLinf Worker。先 local_shard 完成 save→restore，再独立运行 DCP。协调者统一安排单卡时段。

## 1. 上传和运行前提

把新探针与原 `checkpoint_probe.py` 放在远程同一目录 `route2/fsdp_probes`。源码沿用已通过本地对象恢复的 `RLinf-checkpoint`，隔离解释器沿用 `envs/route2/bin/python`。新探针把源目录和探针目录都放入 PYTHONPATH，真实 Ray Worker 必须能读取这两个目录。

探针依赖 Linux `/proc` 获取 driver/Worker 的 boot id、PID 和 kernel start ticks。restore 要求 save driver 已退出，save/restore driver 身份不同、save/restore Worker 身份不同；PID 被复用时以 start ticks 区分。当前只强制检查 save driver 结束，不检查旧 Worker 是否仍存活；`fresh_process=true` 的判定依据两个 Worker 的真实不同身份。

以下四条命令为重跑示例，使用的 `results/fsdp1/process-*-first` 与第4节已经存在的实际运行路径不同；执行前仍须确认示例路径从未存在，否则选择全新路径。reference 必须位于 checkpoint 目录外；它与自动生成的 `.manifest.json` 都不能被覆盖。save 成功并退出后才运行对应 restore，两个阶段之间保持源码、helper、新探针和 runtime 字节/版本一致。

## 2. local_shard 的两个命令

先执行 save：

```bash
RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
  /root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/fsdp_probes/checkpoint_process_probe.py \
  --source /root/autodl-tmp/s4000-research/route2/RLinf-checkpoint \
  --phase save --format local_shard --enable-torch22 \
  --checkpoint-dir /root/autodl-tmp/s4000-research/route2/results/fsdp1/process-local-first \
  --reference-file /root/autodl-tmp/s4000-research/route2/results/fsdp1/process-local-first.reference.pt
```

确认该命令输出 save pass 并退出后，执行 restore：

```bash
RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
  /root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/fsdp_probes/checkpoint_process_probe.py \
  --source /root/autodl-tmp/s4000-research/route2/RLinf-checkpoint \
  --phase restore --format local_shard --enable-torch22 \
  --checkpoint-dir /root/autodl-tmp/s4000-research/route2/results/fsdp1/process-local-first \
  --reference-file /root/autodl-tmp/s4000-research/route2/results/fsdp1/process-local-first.reference.pt
```

## 3. DCP 的两个命令

local_shard 完成后，DCP 使用另一套全新路径。先执行 save：

```bash
RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
  /root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/fsdp_probes/checkpoint_process_probe.py \
  --source /root/autodl-tmp/s4000-research/route2/RLinf-checkpoint \
  --phase save --format dcp --enable-torch22 \
  --checkpoint-dir /root/autodl-tmp/s4000-research/route2/results/fsdp1/process-dcp-first \
  --reference-file /root/autodl-tmp/s4000-research/route2/results/fsdp1/process-dcp-first.reference.pt
```

确认 DCP save 命令输出 pass 并退出后，执行 restore：

```bash
RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
  /root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/fsdp_probes/checkpoint_process_probe.py \
  --source /root/autodl-tmp/s4000-research/route2/RLinf-checkpoint \
  --phase restore --format dcp --enable-torch22 \
  --checkpoint-dir /root/autodl-tmp/s4000-research/route2/results/fsdp1/process-dcp-first \
  --reference-file /root/autodl-tmp/s4000-research/route2/results/fsdp1/process-dcp-first.reference.pt
```

## 4. 实际路径和验收

首次实机运行根目录为 `/root/autodl-tmp/s4000-research/route2/results/checkpoint-process-20261005-first`。local_shard 使用其下的 `local_shard` checkpoint 目录与 `local_shard.reference.pt`；DCP 使用 `dcp` 与 `dcp.reference.pt`，每份 reference 另有 `.manifest.json`。这些历史路径已存在，不能当作新的 save 输出。

| 格式 | save → restore driver PID | save → restore Worker PID | 结果 |
|---|---|---|---|
| local_shard | `151460 → 154592` | `153406 → 156538` | 两命令 exit 0；状态、RNG、继续更新 exact |
| DCP | `157854 → 160966` | `159800 → 162912` | 两命令 exit 0；状态、RNG、继续更新 exact |

进程均在同一 hostname/boot；身份还包含 Linux start ticks，不仅按 PID 比较。两个 restore 记录均为 `fresh_objects=true`、`fresh_process=true`、`save_driver_finished=true`、`driver_and_worker_process_identity_differ=true`。实测 Torch runtime `2.2.0`、Torch-MUSA `1.3.0+81caf0a`、设备 `MTT S4000 / musa:0`、backend `mccl`、world size 1、FP32、NO_SHARD。两次主动裁剪把 norm `12.79115677 / 19.06024361` 降至约 `0.25`；CPU 参数对照最大误差 `1.4901161193847656e-08`。恢复后的第2步 loss、raw/clipped norm、LR 与未中断分支精确相同。

save Worker 设置固定 CPU/MUSA/NumPy/Python seed，使用真实 MCCL、mesh、FSDPStrategy、FP32 NO_SHARD 模型。第1步经真实 active clipping、AdamW、StepLR，再调用真实 strategy.save_checkpoint。随后保存各 RNG 的 expected next samples，完成未中断的固定 batch2。两步都有同权重 CPU active-clipping参数对照，容差仍为 rtol=3e-4 / atol=3e-6。

独立 CPU reference artifact 包含第1步完整训练状态/RNG、固定 batch2、next samples、未中断第2步完整训练状态/RNG和 stats，所有 Tensor 必须在 CPU。reference 完成后生成 sidecar manifest，记录 schema、源码/两份 probe/config/runtime、两种进程身份、reference 和 checkpoint 各文件哈希/大小。文件使用新路径、临时文件与原子 replace；只有完整 checkpoint 和 reference 都发布后才输出 save pass。

restore 在启动 GPU 工作前核对完成清单、格式、源码/helper/probe/config、reference与checkpoint文件哈希，并验证 save driver 已退出。新 Worker 在构造训练对象前把reference读取到CPU，核对 runtime/device model，创建新的真实 FSDP/AdamW/StepLR。warmup 只接受新 optimizer 的空 state 和 None gradients，验证参数无变化、全部 step=0、momentum=0，然后调用真正 strategy.load_checkpoint。

load 后全部验收使用 exact，容差不放宽：

1. 恢复第1步参数、全部 Adam step/exp_avg/exp_avg_sq/param groups、scheduler/LR，及四类原始 RNG state。
2. 抽各 RNG 下一段样本，并对照样本及抽样后的 RNG state。
3. 使用reference中固定 batch2 进行真正策略更新，比较全部训练状态、loss、gradient norm、LR 和四类最后 RNG state。
4. 确认checkpoint文件未改变，两类 driver/Worker 身份不同后输出 `fresh_objects=true, fresh_process=true`。

reference的训练状态仅用于比较，不复制到模型或 optimizer，不手动恢复 RNG，不在 load 后重新设置 seed。恢复端也不重新执行第1步。源文件和 probe hashes是实时读取，未通过的阶段输出 last_stage 和 traceback，失败退出码非零；完整 stdout/stderr由协调者保留，不能仅凭 save pass 宣称跨进程通过。

## 5. 范围

本地检查记录见 [checkpoint-process-static-validation.json](checkpoint-process-static-validation.json)，实机与备份核验见 [checkpoint-process-run-validation.json](checkpoint-process-run-validation.json)。本地没有 Torch/Ray 运行验证；上述结果来自协调者执行的真实 S4000。旧 probe 的同 Worker `fresh_objects=true / fresh_process=false` 历史结果仍见 [checkpoint-README.md](checkpoint-README.md)，不改写为跨进程记录。

本节点限定同 Linux host/boot、相同 runtime/device model/local index、world size 1、FP32、NO_SHARD、无 buffer 的 tiny MLP。相同 GPU model与local index不等于验证了物理设备 UUID；local_shard仍保留MUSA optimizer tensor、同拓扑load。跨进程成功不扩展为多卡分片/reshard、设备迁移、offload、mixed precision、full_weights导出、Manager、官方Actor、PPO或simulator通过。
