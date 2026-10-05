# 下一阶段：Torch 2.2 / MUSA FSDP1 的最小 checkpoint

可以沿用现有 RLinf `Checkpoint(Stateful)` 和 Torch 2.2 的真实 DCP API 实现。公开源码已确认保存需要 `storage_writer`、加载接受 `storage_reader`，且 save/load 会调用 Stateful 对象的 state_dict/load_state_dict。独立增量已修复 Torch 2.2 的保存签名，并在单卡 MUSA/MCCL 上分别通过 local_shard 与 DCP 的保存、恢复和继续更新；跨进程、多 rank、offload 与官方 Actor 仍未验证。

真实 strategy 已通过 [第三轮实机验证](../evidence/fsdp1/strategy-third.jsonl)。本文件保留 strategy 节点完成时的设计方案；后续独立 checkpoint 实现与命令见 [checkpoint-README.md](checkpoint-README.md)。先执行 local_shard 对象重建与继续更新验证，再独立执行 DCP。

## 1. 明确范围

沿用一个真实 MUSA Worker、显式 MCCL 默认组、真实 RLinf mesh / FSDPStrategy、FP32 tiny MLP、NO_SHARD 和 use_orig_params=true。关闭 auto-wrap、offload、混合精度及梯度累积。不加载 Manager、官方 Actor、PPO 或 MuJoCo。

分别执行 `local_shard` 和 `dcp`，使用不同且为空的 checkpoint 目录；一种通过不能替代另一种。保存明确使用 `save_full_model_weights=False`，先验证训练状态，不叠加额外的 full_weights 导出分支。world size 1 的 NO_SHARD 结果不能证明跨 rank 的分片、重分片、world size 改变或分布式 RNG 映射已经正确。

## 2. 最小生产代码变化

仅在 `strategy/base.py` 已开放的 Torch 2.2 路径中，把当前保存调用改为：

```python
dcp.save(
    {"fsdp_checkpoint": training_state},
    storage_writer=dcp.FileSystemWriter(dcp_save_path),
)
```

保留新版 runtime 的 `checkpoint_id=` 路径以及现有完整 Stateful payload。版本识别使用 `packaging.version` 的 release major/minor，兼容厂商的 2.2 prerelease 字符串；不要在任意保存异常后切换路径重试，也不设置 `no_dist=True` 来绕过真实 collective。

load 已用真正 `FileSystemReader` 和 `storage_reader=`，不需要为了 Torch 2.2 改调用签名。`Checkpoint` 的模型/优化器 get_state_dict、set_state_dict 和 StateDictOptions(full_state_dict=False, cpu_offload=True) 也有对应 2.2 API，先保留原实现，再根据实测第一条失败定位。

## 3. 先执行真实 CPU/Gloo 回归

新增独立测试，使用普通 CPU nn.Module、AdamW、StepLR 和真实 Gloo group，经 `FSDPStrategy.save_checkpoint/load_checkpoint` 走现有 `Checkpoint(Stateful)`。按 local_shard/dcp 参数化，写入 tmp_path。这个用例验证 RLinf checkpoint 控制流、schema 和 DCP 旧 API，不声称 CPU 普通模块就是 FSDP 或 MUSA。

至少覆盖保存后重建模型/优化器/调度器，恢复首步的完整参数和 Adam step/exp_avg/exp_avg_sq、optimizer param groups，以及 scheduler state_dict / 当前 LR。完成相同下一批次更新后，与未中断分支比较全部状态。

用当前源码的 `rlinf.utils.utils.warmup_optimizer_state()` 初始化恢复端优化器，验证 warmup 前后参数不变、step counter 为 0。Torch 2.2 自带 DCP helper 在优化器 state 为空时会执行 zero-grad optimizer.step，可能引入 AdamW decay 和步数变化；不能依赖这一步隐式初始化，也不能以一份没有 momentum 的 optimizer 空状态测试恢复。

## 4. 真实 MUSA strategy checkpoint 探针

建议在现有 probe 增加独立 checkpoint phase，显式参数选择一种格式及持久化目录，不把 checkpoint 自动加入每次 strategy 更新。每种格式依次执行：

1. 构建真实 FSDP1、AdamW、StepLR，完成第 1 步更新与 scheduler.step；缓存 CPU 参数/优化器/调度器对照。
2. 调用 strategy.save_checkpoint，保持默认 barrier 与真实 Stateful state_dict；不在 probe 内改写 payload 或手动 torch.save 替代 DCP。
3. 未中断分支执行固定第 2 个 batch，并记录第 2 步后参数、Adam 各状态和 scheduler。
4. 重建新的 FSDP 包装、optimizer 和 scheduler，先用真实 warmup helper 建立 Adam 状态，再调用 strategy.load_checkpoint。
5. 先确认恢复后等于第 1 步，再执行同一第 2 个 batch，比较参数、loss、梯度范数、Adam 状态与 scheduler/LR。模型输入、目标与初始 CPU 权重固定，继续经 strategy.before_micro_batch / clip_grad_norm_ / AdamW.step。

这一轮先在同一 Worker 内做新的对象重建，JSON 明确标记 fresh_objects=true / fresh_process=false。之后增加保存与恢复两个独立 Worker 进程的阶段，使用同一远程持久化目录，才能声称跨进程恢复成功。不要把同一模型对象直接 load 回去的结果当作重建恢复。

RNG 另外验证 Python random、NumPy、Torch CPU 和当前 MUSA 设备：保存后读取各流下一段样本作为 expected；重建会消耗 RNG，load 后读取同样长度样本作为 actual 并对照。MUSA 随机值与 CPU 随机值不互相比对，只比较同一流保存/恢复前后序列。JSON 记录涵盖的 RNG 流和结果，不输出完整 RNG 状态。

## 5. 需要保留的真实失败边界

- DCP 模式的 `Checkpoint.state_dict()` 会使用 MCCL 默认组执行 all_gather_object 保存各 rank RNG，DCP 本身也做元数据 collective。world size 1 仍不能预先假定这些操作为空。若出现 uint8/int64 或对象通信错误，保留实际调用与最后 stage；先做对应真实最小 collective 探针，再讨论是否需要生命周期明确的 Gloo 控制组。
- NO_SHARD 中 get_state_dict 预期取得普通 tensor，但仍必须实测厂商 FSDP get/set 状态、CPU offload 和 optimizer FQN 映射，不能以源码签名代替。
- DCP metadata 中当前 RNG tuple 应保留为 `fsdp_checkpoint.rng` 的序列化叶子；检查真实 metadata，再验证现有 legacy_rng_state 判定。遇到 schema 不符时局部修复，不直接跳过 RNG。
- checkpoint 路径由协调者选择新目录。FileSystemWriter 假定空或不存在的目录，失败不覆盖历史 checkpoint。

记录每个 save/load/继续更新 stage，实际文件清单和格式、source/probe/patch hashes、world size/backend、是否完整保存 model/optimizer/scheduler/RNG，以及逐类数值对照结果。DCP 的 CPU/Gloo 通过和 MUSA/MCCL 通过必须各自标记。

## 6. 已核对的公开 API

固定 PyTorch 源码参照为 `8ac9b20d4b090c213799e81acf48a55ea8d437d6`，本地核对日期 2026-10-05；不是已安装厂商 wheel 的源码认证。

- [DCP save 签名与 Stateful 处理](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/state_dict_saver.py#L37)。
- [DCP load 签名与 Stateful 恢复](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/state_dict_loader.py#L33)。
- [FileSystemWriter 的真实构造接口](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/filesystem.py#L322)。
- [StateDictOptions](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/state_dict.py#L81)、[get_state_dict](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/state_dict.py#L648) 和 [set_state_dict](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/state_dict.py#L837)。
- [Torch 2.2 optimizer 隐式初始化](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/torch/distributed/checkpoint/state_dict.py#L414)。

实施前再读取实机 signature；此前 imports probe 已输出 dcp.save/load，下一轮补充 FileSystemWriter、get_state_dict/set_state_dict 和 StateDictOptions。所有测试与 GPU 阶段由协调者顺序调度。
