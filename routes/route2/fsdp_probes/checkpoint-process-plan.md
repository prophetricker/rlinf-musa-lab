# 后续节点：跨进程 checkpoint 恢复拆分方案

当前 [checkpoint_probe.py](checkpoint_probe.py) 已通过 CPU/Gloo 与 MUSA/MCCL 两种格式的新对象恢复测试，哈希固定为 `da07c5591fe6d50a2eb8983cd706f117762720cfa4b37d44c8527820c8772a79`。保留此文件字节与现有生产实现；跨进程实验新增 `checkpoint_process_probe.py`，不改现有 probe，也不把同 Worker 的 fresh_objects 结果改名为 fresh_process。

## 1. 新文件复用已验证 helper

新增脚本导入 `checkpoint_probe`，复用其 `make_model`、`optimizer_and_scheduler`、`update`、`training_state`、`parameters`、`assert_tree`、`assert_adam_state`、`cpu_copy`、`rng_samples` 和裁剪常量。两个脚本均上传到同一目录，新脚本显式把源码目录与 probe 目录都放入 PYTHONPATH，使 Ray Worker 可以正常导入 helper。

在新文件实现 `build_training_context()`：一个真实 MUSA Worker、MCCL、相同 timeout、真实 RLinf create_device_mesh / gradient_reduction_group / FSDPStrategy，NO_SHARD、FP32、use_orig_params=true。先复制已验证 config 字段并记录 config hash；本节点不为了抽公共 helper 而修改旧 probe。CPU对照仅用于 save 阶段复现现有结果；restore 主验收对象是保存进程的实际 MUSA 连续训练分支。

API 拆分如下：

| 新函数 | 输入 | 输出/责任 |
|---|---|---|
| `build_training_context()` | 相同明确 config | 新真实 FSDP、strategy、AdamW、StepLR、mesh |
| `save_phase(format, checkpoint_dir, reference_file)` | 两个全新输出路径 | 保存第1步checkpoint；运行未中断第2步；写CPU reference artifact；结束Worker |
| `restore_phase(format, checkpoint_dir, reference_file)` | 已完成的两份文件 | 新进程中重建、warmup、真实load，比较原始状态/RNG，再复现第2步 |
| `source_evidence()` | source/core/new probe | 6个生产源、旧helper、新probe及config的真实hash，不借用旧probe的 `__file__` 记录 |

命令只接受显式 `--phase save` 或 `--phase restore`、一种 `--format`，避免默认入口在同进程做完整流程。协调者执行两个独立 Python 命令并确认 save 命令退出后再启动 restore，不在一个 Worker 内顺序调用两个方法。

## 2. save phase 与独立 CPU reference artifact

save Worker 按已验证路径完成 update1，经真实 strategy.save_checkpoint 保存，再抽各 RNG 流下一段样本，最后运行固定 batch2 得到未中断 expected 状态。输出独立 `reference.pt`，包含：

- versioned schema、format、world size、device、source/helper/new-probe/config hashes、Torch/Torch-MUSA版本。
- save driver/Worker PID 与 Linux process start time；本轮精确恢复要求同runtime与同设备拓扑。
- 第1步完整 CPU model/optimizer/scheduler snapshot 与原始 RNG snapshot。
- 固定 batch2 的 CPU inputs/targets；不能到 restore 再随机生成。
- 各 RNG 下一段 expected samples、抽样后/更新2后的 expected RNG。
- 实际 MUSA 未中断 update2 的完整 CPU model/optimizer/scheduler snapshot、loss/gradient-norm/LR stats。
- 已验证 CPU active-clipping参数对照的误差，及实际checkpoint文件清单/大小。

所有 Tensor 经真实 `cpu_copy()` 与保存端训练状态解除别名；NumPy array/tuple、Python RNG tuple保留真实结构。reference是验收oracle，不能替代strategy checkpoint，也不能向新模型手动load其中的参数或Adam。

用临时文件完成 CPU torch.save，再 os.replace 到最终reference路径。只在checkpoint与reference都完成后返回 save-pass；拒绝已存在的checkpoint/reference文件，失败保留现场。序列化与文件命名不得再调用已保存RNG生成路径；可用协调者给出的run id或OS熵。

## 3. restore phase 的关键顺序

先读取 reference 到 CPU，再创建新 Worker、MCCL、全新 FSDP/AdamW/StepLR；此时消耗初始RNG是允许的。使用真实warmup，仅允许 fresh empty optimizer state和None gradients，验证参数不变、4个step=0/momentum=0。随后调用真正 strategy.load_checkpoint。

load完成后依次：

1. 第1步参数、全部Adam momentum/counters/param groups、scheduler/LR与reference精确相同。
2. 四类原始RNG state与保存端checkpoint snapshot精确相同。
3. 调用同一 `rng_samples()`，与expected next samples精确相同。此后不要再读reference或构建随机数据；CPU参考数据应已在load前读取完成。
4. 用reference中的固定batch2执行真实strategy update；比较全部参数、Adam状态、scheduler/LR、loss/norm与保存进程未中断update2，默认保持精确验收。
5. 比较更新后的四类RNG。记录save与restore driver/Worker PID、process start time不同，才输出fresh_process=true。

不能再次执行update1、从reference复制参数代替load，或在load后manual_seed。若跨进程数值不一致，保留最大误差、首次差异路径与实际环境，再定位；不要直接放宽容差把失败改成成功。

## 4. 先后顺序与输出

先local_shard的save→restore，随后DCP使用另一套目录独立save→restore。单卡GPU操作仍由协调者统一排队。两个阶段都保留生产GPU housekeeping、真实MCCL集体通信与格式原路径，没有no_dist、Gloo替换或假FSDP。

紧凑JSON应含phase/format、provenance、checkpoint/reference文件哈希、save/restore PID与start time、fresh_objects/fresh_process、完整state/RNG覆盖布尔值、continuation差异，以及主动裁剪norm。reference不把完整RNG或模型权重打印进日志。

可先新增独立CPU/Gloo双子进程回归，使用普通CPU模型、显式CPU Worker/None平台，限定跳过GPU housekeeping为gc.collect；Gloo/Stateful/DCP仍真实执行。该测试只证明跨进程控制流，不代替真实MUSA FSDP恢复。现有已通过的CPU测试与probe不修改。

world size 1 / NO_SHARD / 同设备跨进程成功仍不证明多卡分片、reshard、异构设备迁移、offload、mixed precision、官方Actor或完整PPO恢复；跨进程节点继续保持独立验收范围。
