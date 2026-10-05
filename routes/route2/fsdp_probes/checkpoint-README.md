# FSDP1 checkpoint 实验入口

本节点基于已验证 strategy 提交 `f9b74d95f97ad4234311580ab32c7b586e1dc7af`，只增加 Torch 2.2 DCP 保存签名兼容和独立 checkpoint 探针。增量保存到 [checkpoint-compat.patch](../patches/checkpoint-compat.patch)，不覆盖已验证的 `fsdp1-experimental.patch`。本机只有静态检查能力，实机由协调者执行。

2026-10-05 实机 CPU/Gloo 两种格式 **2/2 通过**，MUSA/MCCL 的 local_shard 与独立 DCP **均通过**；原始证据见 [checkpoint-cpu-first.txt](../evidence/fsdp1/checkpoint-cpu-first.txt)、[checkpoint-local-first.jsonl](../evidence/fsdp1/checkpoint-local-first.jsonl) 和 [checkpoint-dcp-first.jsonl](../evidence/fsdp1/checkpoint-dcp-first.jsonl)。两种格式恢复与下一步参数/Adam/scheduler 完全一致，四类 RNG state 和 next samples 完全一致；主动裁剪把 norm `12.79 / 19.06` 降至 `0.25`，CPU 参数最大误差 `1.49e-8`。DCP 实际使用默认 MCCL group、Stateful 和 FileSystemWriter/Reader，没有 no_dist 或 Gloo 替代。

先跑 imports 与 CPU/Gloo 回归，然后由协调者单卡顺序跑 local_shard、dcp。每种格式一个命令、一个新目录；探针拒绝复用任何已存在的 checkpoint 目录，不覆盖历史文件。

## 1. 实际导入与签名

把 [checkpoint_probe.py](checkpoint_probe.py) 上传到远程 `route2/fsdp_probes`，增量应用到匹配基线的隔离源码。按现有路径运行：

```bash
/root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/fsdp_probes/checkpoint_probe.py \
  --source /root/autodl-tmp/s4000-research/route2/RLinf-checkpoint \
  --phase imports --enable-torch22
```

该阶段记录真实 DCP save/load、FileSystemWriter、StateDictOptions、get/set_state_dict 签名与 runtime/distribution 版本，不创建 Worker、process group 或 MUSA tensor。模块导入可能按 RLinf 初始化逻辑探测硬件，不能把这一步称为完全没有 GPU runtime 接触。

## 2. CPU/Gloo 回归

新测试文件为 `tests/unit_tests/test_fsdp_checkpoint.py`：

```bash
cd /root/autodl-tmp/s4000-research/route2/RLinf-checkpoint
RLINF_EXPERIMENTAL_FSDP1_TORCH22=1 \
  /root/autodl-tmp/s4000-research/envs/route2/bin/python -m pytest -q \
  tests/unit_tests/test_fsdp_checkpoint.py
```

两种格式使用真实普通 CPU nn.Module、AdamW、StepLR、Gloo 和 `Checkpoint(Stateful)` 经 strategy.save/load_checkpoint 控制流。checkpoint 执行期间显式设置 Worker device 为 CPU、platform 为 None，RNG 只覆盖 Python/NumPy/Torch CPU；只将 base 的 GPU cache housekeeping 替为真实 gc.collect。DCP、对象通信和 checkpoint payload 均不替代，生产 `clear_memory` 不修改。模块导入阶段可能探测硬件，CPU/Gloo 测试不声称验证 GPU FSDP或MCCL。

测试要求新模型对象及新参数、非空 Adam momentum、正确 step counters、完整 param groups、scheduler/LR、完整 RNG state 与下一段样本，以及同 batch 下一步全部状态完全一致。warmup 仅用于空 state、None gradients 的新 optimizer，验证参数无变化、状态数覆盖全部参数、step=0、momentum=0。没有依赖 Torch2.2 DCP 的隐式 optimizer.step 初始化。

## 3. 真实 MUSA local_shard

以下目录名是示例，协调者须选从未存在的新名字：

```bash
RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
  /root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/fsdp_probes/checkpoint_probe.py \
  --source /root/autodl-tmp/s4000-research/route2/RLinf-checkpoint \
  --phase checkpoint --format local_shard --enable-torch22 \
  --checkpoint-dir /root/autodl-tmp/s4000-research/route2/checkpoints/local-shard-first
```

一个真实 MUSA Worker 显式初始化 MCCL，使用真实 RLinf mesh、FSDPStrategy、FP32 NO_SHARD MLP。每次 update 经 before_micro_batch、backward、真实 strategy.clip_grad_norm_、AdamW、StepLR；loss 放大 16 倍、clip threshold=0.25，探针要求 raw norm>threshold、梯度确实缩小且裁剪后 norm≤0.25+3e-6；实测两次 raw norm 都大于 1。第 1、2 步均与同权重 CPU active-clipping 更新比较参数，CPU 容差为 rtol=3e-4 / atol=3e-6。

真实执行为 update1 → save → 未中断 update2 → 新 strategy/FSDP/AdamW/StepLR → warmup → load → 恢复第1步所有状态 → 同 batch update2。恢复与未中断分支的参数、Adam step/exp_avg/exp_avg_sq、全部 optimizer param groups、scheduler state/LR 要求完全相同。Python、NumPy、Torch CPU、MUSA RNG 的原始状态、下一段样本及第2步后状态也须完全相同。

探针保留实际 GPU clear_memory / barrier / load 原路径，不跳过设备 housekeeping，也不在 probe 内手工实现保存。输出明确 fresh_objects=true、fresh_process=false；这里只证明同 Worker 内新对象恢复，不声称跨进程、设备迁移或多 rank 的恢复。local_shard 原始 optimizer checkpoint 保留 MUSA tensor，当前 load 无 map_location，本轮限定同设备/同拓扑。

## 4. 独立 DCP

local_shard 通过后，独立安排 DCP 命令与新目录：

```bash
RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
  /root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/fsdp_probes/checkpoint_probe.py \
  --source /root/autodl-tmp/s4000-research/route2/RLinf-checkpoint \
  --phase checkpoint --format dcp --enable-torch22 \
  --checkpoint-dir /root/autodl-tmp/s4000-research/route2/checkpoints/dcp-first
```

Torch 2.2（包括厂商 prerelease 字符串）使用真正的 `dcp.FileSystemWriter` / `storage_writer=`；新 runtime 保留 `checkpoint_id=`。load 原来就使用真正 `FileSystemReader` / `storage_reader=`。不做异常 fallback，不设置 no_dist，不替换 DCP/RNG 的 MCCL collective。

所有阶段先输出紧凑 started 记录，异常保留实际 traceback，最终只在所有对照通过后输出 pass JSON，包括源文件与 probe hashes、实际文件及大小、恢复范围和主动裁剪数据。协调者保留 stdout/stderr 后摘取 JSON，避免 Ray 日志被当作结果。若 save/load 失败，依据最后 stage 与 traceback 定位；CPU/Gloo 成功不能替代 MUSA/MCCL 成功。

跨进程恢复不在当前脚本中追加，以免改变已验证 probe 的哈希。拆分方案见 [checkpoint-process-plan.md](checkpoint-process-plan.md)：save 与 restore 使用两个独立命令和独立 reference artifact，先 local_shard 再 DCP。

## 5. 本轮边界

静态证据保存为 [checkpoint-static-validation.json](checkpoint-static-validation.json)。本地未安装 Torch 或运行测试，远程 GPU 由协调者执行。该实现暂不涉及 Manager、官方 Actor、PPO、simulator、offload、mixed precision、真实多卡分片、跨进程恢复、world-size 变化或 full_weights 导出。后续跨进程方案仍见 [checkpoint-plan.md](checkpoint-plan.md)。
