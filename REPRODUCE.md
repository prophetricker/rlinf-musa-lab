# 路线二复现说明

本导出仓库包含路线二补丁、探针、小型证据，以及从路线一目录原样复制的共用 MuJoCo runner。恢复源码可以在 Mac 完成；MUSA/GPU 探针在已配好匹配 Torch-MUSA 的 Linux S4000 上执行。新增的便携入口经过本地语法/路径检查，首轮 GPU 结果来自详细报告中的原命令，尚未重新运行便携入口。

## 1. 恢复固定源码

在此仓库根目录执行：

```bash
python3 scripts/restore_sources.py --apply-patches
git -C worktrees/rlinf-route2 rev-parse HEAD
git -C worktrees/rlinf-route2 diff --stat
```

`HEAD` 应为 `c70606f08cdca259b8dec03d4430926b5b8fac9d`；适配作为工作区改动应用。脚本保留已有工作树，不覆盖现有内容。源码树恢复后的预期 tree 为 `b78ed3694bcd1d69afd887e6ce11b780d7528ecf`，对应本地适配提交 `3e7329c6`；补丁重建不改变上游依赖声明。

## 2. 选择已有的匹配环境

首轮解释器是 `/root/autodl-tmp/s4000-research/envs/route2/bin/python`，通过 system-site-packages 与 `.pth` 继承系统 Torch-MUSA 及早先的轻量依赖。这不是独立安装的一套新 Toolkit。

```bash
export ROUTE2_PYTHON=/root/autodl-tmp/s4000-research/envs/route2/bin/python
"$ROUTE2_PYTHON" -c 'import torch, torch_musa, ray, gymnasium, mujoco; print(torch.__version__, torch_musa.__version__, ray.__version__, gymnasium.__version__, mujoco.__version__)'
```

预期 runtime 版本依次为 `2.2.0`、`1.3.0+81caf0a`、`2.59.0`、`1.1.1`、`3.3.7`。Torch distribution metadata 是 `2.2.0a0+git8ac9b20`，与 runtime 字符串不同。

新服务器需要先获得匹配的系统 Torch/Torch-MUSA，再准备隔离环境与缺少的轻量依赖，参考 [环境说明](locks/environment.md)。`locks/route2-environment.txt` 是历史环境快照，含本地 wheel 路径，不能直接作为可移植 requirements 安装；其中 `rlinf==0.3.0` 是继承的旧包，探针实际使用固定源树。也不要直接执行上游全项目 `pip install .`：它的 Torch 版本要求会引入另一套运行栈。

## 3. 顺序复测

下面的入口自动从源码锁读取 commit、配置 PYTHONPATH（包括 Ray child 需要的 unit_tests 与共用 runner 路径），并把新输出写到 `results/`，历史证据保留在 `routes/route2/evidence/`。

```bash
bash scripts/run_probes.sh imports
bash scripts/run_probes.sh cpu
bash scripts/run_probes.sh channel
bash scripts/run_probes.sh worker
bash scripts/run_probes.sh fsdp
bash scripts/run_probes.sh loop
```

`imports` 会记录预期的 FSDP/FlashAttention 阻塞；以输出里的每项状态为准，进程退出 0 不能代表所有模块可用。`worker/fsdp/loop` 分配 MUSA 资源，按顺序运行。Channel 模式验证 Ray CPU 路径，fsdp 模式只验证原生 FSDP1 的单进程 `NO_SHARD`。

loop 模式固定首轮配方：seed 7、4 env、32 horizon、2 iterations、2 epochs、minibatch 64、1 eval episode、同批恢复检查。结果应该是 256 transitions、8 次 optimizer step、130 条 Channel 消息；浮点值与耗时可能随系统变化。resume 对照不包含模拟器状态。可通过 `ROUTE2_SOURCE` 指定另一固定源树，通过 `ROUTE2_RESULTS` 指定输出目录。

这不是 GR00T 或完整官方 RLinf benchmark 的启动脚本。扩大学习预算及周期性评估见 [后续节点](ROADMAP.md)。

## 4. 学习版

详见 [固定预算协议](routes/route2/learning/README.md)。恢复源码后，仓库根目录配置共用 runner 与学习模块路径，再顺序跑三个 seed：

```bash
export PYTHONPATH="$PWD/worktrees/rlinf-route2:$PWD/probes:$PWD/routes/route2/learning"
export RLINF_MUSA_TORCH_DETECTION_FALLBACK=1
for seed in 7 17 27; do
  "$ROUTE2_PYTHON" routes/route2/learning/ppo_learning.py \
    --source-commit c70606f08cdca259b8dec03d4430926b5b8fac9d \
    --seed "$seed" --resume-check --output "results/learning/pendulum-seed$seed.json"
done
"$ROUTE2_PYTHON" routes/route2/learning/summarize_learning.py \
  results/learning/pendulum-seed7.json results/learning/pendulum-seed17.json \
  results/learning/pendulum-seed27.json --output results/learning/summary.json
```

默认是 Pendulum-v1、每 seed 102,400 transitions、固定 5 validation/20 test episodes。修改预算或测试协议需另记配方，不能沿用历史结论。

HalfCheetah 三种子保持同一预算，增加 `--env-id HalfCheetah-v5 --normalize-observations --batch-evaluation`。这是 MuJoCo 训练；Pendulum 本身不是 MuJoCo 环境。批量 helper 的实机 CPU 对照与 MUSA smoke 已验证，不把串行前测的中断结果计入汇总。

也可运行便携固定配方入口：`bash scripts/run_learning.sh pendulum` 或 `bash scripts/run_learning.sh halfcheetah`，顺序执行。便携入口通过路径与 Bash 语法检查，历史 GPU 结果来自上述同等原命令，未重复跑新入口；默认不画图，安装已有匹配 matplotlib 时可对汇总命令添加 `--plot`。

## 5. 可选 FSDP1 增量

基线补丁保持不变。显式恢复另一份源码，避免把后端实验覆盖到学习基线：

```bash
python3 scripts/restore_sources.py --apply-patches --fsdp1-experimental
"$ROUTE2_PYTHON" routes/route2/fsdp_probes/fsdp1_backend_probe.py \
  --source "$PWD/worktrees/rlinf-route2-fsdp1" --phase imports --enable-torch22
RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
  "$ROUTE2_PYTHON" routes/route2/fsdp_probes/fsdp1_backend_probe.py \
  --source "$PWD/worktrees/rlinf-route2-fsdp1" --phase strategy --enable-torch22
```

恢复 tree 为 `1a4eef5dd1f4984652168ebecce1d3cc194ec598`。GPU strategy 顺序运行；单卡 FP32 NO_SHARD RLinf 后端的 MSE 更新、local_shard 与 Torch 2.2 DCP checkpoint 恢复均有独立入口，仍不含 PPO/官方 Actor/多 rank。详细 CPU 测试及失败历史见 [FSDP 说明](routes/route2/fsdp_probes/README.md)。默认不开放旧 Torch，FSDP2 仍拒绝。

## 6. 独立 checkpoint 增量

保持基线源树，另建包含 checkpoint 增量的源树：

```bash
python3 scripts/restore_sources.py --apply-patches --checkpoint-experimental
export ROUTE2_CHECKPOINT_SOURCE="$PWD/worktrees/rlinf-route2-checkpoint"
RLINF_EXPERIMENTAL_FSDP1_TORCH22=1 \
  PYTHONPATH="$ROUTE2_CHECKPOINT_SOURCE" \
  "$ROUTE2_PYTHON" -m pytest -q \
  "$ROUTE2_CHECKPOINT_SOURCE/tests/unit_tests/test_fsdp_checkpoint.py"
for format in local_shard dcp; do
  RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
    "$ROUTE2_PYTHON" routes/route2/fsdp_probes/checkpoint_probe.py \
    --source "$ROUTE2_CHECKPOINT_SOURCE" --phase checkpoint \
    --format "$format" --enable-torch22 \
    --checkpoint-dir "$PWD/results/checkpoint-new/$format"
done
```

每次使用从未存在的 checkpoint 目录。重建 tree 为 `a6c18c65206804ce621bf78472c5ed0b3b37387d`；两种格式分别验收新对象恢复、Adam/scheduler、Python/NumPy/Torch CPU/MUSA RNG 和继续更新一致性。结果限于 world size 1、FP32 NO_SHARD、同进程新对象，详见 [checkpoint 记录](routes/route2/fsdp_probes/checkpoint-README.md)。

## 7. 真实 action Attention / 双层 DiT

固定来源代码按 manifest 逐个下载并验证 SHA256；模型依赖放到独立 `route2-models` 环境，仅 no-deps 安装 Diffusers 0.30.2，继承匹配 Torch-MUSA：

```bash
python3 scripts/restore_model_audit.py
bash scripts/prepare_route2_model_env.sh
/root/autodl-tmp/s4000-research/envs/route2-models/bin/python \
  routes/route2/model_probes/gr00t_action_interface_probe.py \
  --device musa --phase all --output results/action-interface.json
```

该脚本使用随机权重，不下载模型 checkpoint。预期原始 `required_interfaces_pass=false`，失败仅在解析零 bias 的 relative-L2；独立 `semantic_interfaces_pass=true` 还要求这些 bias 的 actual/reference 梯度均 finite 且绝对值在原阈值内。退出码仍按原 strict gate 返回1，不把语义结果伪装成原总门槛通过。所有输出、输入/非零参数 VJP 与 mask 合约沿用原数值门槛，见 [action 接口记录](routes/route2/planning/action-interface-probe.md)。

## 8. 独立进程 checkpoint

同进程探针字节和证据不变；新探针将 save/restore 拆成两个 Python 命令、两个真实 RLinf Worker。两种格式已在 S4000 上通过完整状态、四类 RNG 与下一步 exact 恢复。继续使用 checkpoint 增量源码和既有 route2 环境；reference 是 CPU 断言 oracle，不能替代真实 checkpoint load。

```bash
for format in local_shard dcp; do
  for phase in save restore; do
    RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
      "$ROUTE2_PYTHON" routes/route2/fsdp_probes/checkpoint_process_probe.py \
      --source "$ROUTE2_CHECKPOINT_SOURCE" --phase "$phase" --format "$format" \
      --enable-torch22 --checkpoint-dir "$PWD/results/process-new/$format" \
      --reference-file "$PWD/results/process-new/$format.reference.pt" || exit
  done
done
```

每次使用从未存在的输出路径。save 完全退出后才执行 restore，reference 放在 checkpoint 目录外。完整路径、SHA256、进程身份和范围见 [跨进程说明](routes/route2/fsdp_probes/checkpoint-process-README.md)。本轮同 host/boot、同型号设备/local index、world size 1/FP32/NO_SHARD；不是多卡或设备迁移结果。

## 9. 冻结骨干组件对照

恢复固定 Eagle config（其他模型源文件仅用于审计；运行使用 hash-checked installed Transformers），准备隔离依赖环境后顺序执行：

```bash
python3 scripts/restore_model_audit.py \
  --repo NVIDIA/Isaac-GR00T --path gr00t/model/backbone/eagle2_hg_model/config.json
bash scripts/prepare_route2_backbone_env.sh
/root/autodl-tmp/s4000-research/envs/route2-backbone/bin/python \
  routes/route2/model_probes/gr00t_backbone_interface_probe.py \
  --device cpu --output results/backbone-cpu.json
/root/autodl-tmp/s4000-research/envs/route2-backbone/bin/python \
  routes/route2/model_probes/gr00t_backbone_interface_probe.py \
  --device musa --output results/backbone-musa.json
```

当前v2脚本CPU8/8、CPU/MUSA整组12/12通过；此前v1 Qwen3 BF16 MUSA因诊断 masked_select 不支持BF16退出1，旧脚本/false证据保留。v2仅将被屏蔽概率的只读检查转FP32，并增加绑定及全局注册表identity检查；attention数学、VJP与容差不变。默认需要随包固定的 Spatial config 和 hash核验通过的Eagle config。Torch-MUSA仍来自原系统环境，无Toolkit或driver升级。

范围是随机两层真实Qwen3Model/SiglipVisionModel的冻结前向，以及同CPU-reference feature上的独立synthetic Linear VJP；不是checkpoint的Identity投影参数更新，也不是完整Eagle/GR00T。详细源码路径、dtype门槛、失败与版本记录见 [骨干探针](routes/route2/planning/backbone-interface-probe.md)。

## 10. GR00T/LIBERO 最小 PPO 适配节点

下面两条命令在同一张 S4000、同一隔离环境和同一次开机周期内顺序执行。第一条只审计官方 Actor 的导入门槛，不创建 GPU tensor；它使用独立的 `RLinf-fsdp1` 源树，并需要先在该源树应用 `patches/fsdp1-actor-dtensor.patch`（`patch -p1 --unidiff-zero < ...`）。第二条仍使用基础 `RLinf` 源树，加载完整 Spatial 权重、运行真实 LIBERO-Spatial rollout，并用真实 reward/done 计算 RLinf GAE 后完成一次 PPO loss/backward/AdamW 更新。

```bash
export R=/root/autodl-tmp/s4000-research
export PYTHONPATH=$R/envs/route2-integration/lib/python3.10/site-packages:$R/envs/route2-integration:$R/route2:$R/rlinf-s4000:$R/route2/RLinf:$R/route2/Isaac-GR00T-spatial-eager
PY=$R/envs/route2-integration/bin/python

RLINF_EXPERIMENTAL_FSDP1_TORCH22=1 $PY $R/route2/model_probes/embodied_actor_torch22_probe.py \
  --rlinf-source $R/route2/RLinf-fsdp1 \
  --output $R/route2/results/embodied-actor-torch22-optin-fsdp1-v3.json

RLINF_EXPERIMENTAL_FSDP1_TORCH22=1 $PY \
  $R/route2/model_probes/libero_rlinf_model_episode_probe.py \
  --rlinf-source $R/route2/RLinf \
  --gr00t-source $R/route2/Isaac-GR00T-spatial-eager \
  --model-path $R/route2/weights/Spatial-73f710e \
  --env-config $R/route2/RLinf/examples/embodiment/config/env/libero_spatial.yaml \
  --steps 8 --worker-start-method spawn --rollout-mode train \
  --ppo-multi-step --action-head-fp32 \
  --output $R/route2/results/libero-rlinf-ppo-multi-step-v5.json
```

第二条命令的通过条件是：真实 GAE 的 advantages/returns 全部有限；每步 PPO loss 和梯度有限；至少一个梯度非零；AdamW 后参数样本发生变化。通过只证明计算与适配闭环，不代表 LIBERO 成功率提升；失败时保留 JSON 中的完整错误、reward、done、value 和张量形状记录。

## 10. Spatial 正式宽度与真实 Eagle

使用新的组合依赖环境；旧环境和系统Torch-MUSA保留。以下在成果仓库根目录执行，输出必须是新路径：

```bash
bash scripts/prepare_route2_integration_env.sh
python3 scripts/restore_model_audit.py
python3 scripts/restore_eagle_sources.py --output worktrees/gr00t-eagle-tiny --tiny
python3 scripts/restore_eagle_sources.py --output worktrees/gr00t-spatial-eager --spatial-full
export ROUTE2_INTEGRATION_PYTHON=/root/autodl-tmp/s4000-research/envs/route2-integration/bin/python
for dtype in fp32 bf16; do
  "$ROUTE2_INTEGRATION_PYTHON" routes/route2/model_probes/spatial_action/spatial_action_layout_probe.py \
    --device musa --phase all --dtypes "$dtype" --output "results/spatial-$dtype-new.json"
done
"$ROUTE2_INTEGRATION_PYTHON" routes/route2/model_probes/eagle_integration/eagle_composite_probe.py \
  --source-tree worktrees/gr00t-eagle-tiny --device musa --output results/eagle-tiny-new.json
"$ROUTE2_INTEGRATION_PYTHON" routes/route2/model_probes/eagle_integration/spatial_shape_audit.py \
  --source-tree worktrees/gr00t-spatial-eager --output results/spatial-meta-new.json
```

上面的 Spatial/Eagle混合精度整组原退出码为1，必须查看逐项结果，不能用shell退出码修改或掩盖门槛。FP32 Spatial独立语义gate通过、严格总gate保留解析零方向失败；BF16扩展norm合同失败。tiny Eagle FP32全部通过，BF16有效features allclose失败。完整meta结构899 keys/shapes匹配不代表权重已加载。

需要真实骨干测试时，只下载两个固定公开权重分片；验证header、尺寸和canonical LFS完整SHA，模型文件不进Git：

```bash
"$ROUTE2_INTEGRATION_PYTHON" routes/route2/model_probes/eagle_integration/download_spatial_weights.py \
  --destination results/weights/Spatial-73f710e --transport hf-mirror --evidence results/download-new.json
"$ROUTE2_INTEGRATION_PYTHON" routes/route2/model_probes/eagle_integration/spatial_pretrained_backbone_probe.py \
  --source-tree worktrees/gr00t-spatial-eager --weights results/weights/Spatial-73f710e \
  --device musa --output results/spatial-pretrained-new.json

# 同输入逐层数值诊断；原始数值门槛不变，结果只作为定位证据
"$ROUTE2_INTEGRATION_PYTHON" routes/route2/model_probes/eagle_integration/spatial_pretrained_backbone_probe.py \
  --source-tree worktrees/gr00t-spatial-eager --weights results/weights/Spatial-73f710e \
  --device musa --diagnose-numerics --output results/spatial-pretrained-diagnostics-new.json
```

网络慢时可使用 `scripts/acquire_route2_weights_parallel.py --support-dir routes/route2/model_probes/eagle_integration --destination results/weights/Spatial-73f710e --evidence results/download-parallel-new.json`，先停止其它写同目录的下载进程。只采用支持精确206 Range的公开mirror，保留prefix/分块，完整文件仍校验官方LFS SHA。该helper的纯下载并发不允许同时跑多个GPU实验。

pretrained runner仅加载所有585个backbone张量并转换FP32；全部strict keys和载入值逐项验收，再比较真实CPU source eager / CPU fallback / MUSA fallback。它使用两幅合成raw tensor image和B1/S570，无tokenizer/图像预处理/完整action/环境/优化器。以 [本轮报告](reports/route2-spatial-eagle.md) 的实际完成范围为准。

## 11. 完整 action head 功能影响

复用第10节准备好的隔离环境、完整源码和固定两片 Spatial 权重，在同一 S4000 上顺序执行：

```bash
"$ROUTE2_INTEGRATION_PYTHON" routes/route2/model_probes/eagle_integration/spatial_action_functional_probe.py \
  --source-tree worktrees/gr00t-spatial-eager --weights results/weights/Spatial-73f710e \
  --device musa --with-backbone --embodiment-id 31 \
  --seed 271829 --backbone-seed 271829 --cpu-threads 4 \
  --output results/spatial-functional-new.json
```

采用 RLinf 的 LIBERO 分支31和7维有效动作 mask；固定标准高斯噪声和4步 Euler，并对照未修改的原始 forward/get_action。每组使用完整314张量 head 执行 backward 和 AdamW 单步，直接比较活动分支和所有共享参数的梯度/更新向量。其他机器人分支的梯度必须严格为0。

输出和退出码验收本轮功能与控制合同；action gate、feature gate、逐参数/全局梯度及更新数值结果分开记录。尤其不能把 finite 参数更新或功能探针退出0解释为原 feature gate 已通过、真实任务成功率提升或官方 RLinf PPO 可用。实测历史及空切片检查修订见 [Spatial/Eagle报告](reports/route2-spatial-eagle.md)。

## 12. 官方 Actor 与两卡增量

先按[官方Actor说明](routes/route2/model_probes/official-actor-README.md)恢复独立源码、GR00T eager/lazy-PyTorch3D源码及隔离环境。既有单卡入口与冻结历史证据保持原范围；两卡v15参数更新与范数检查的精确命令见[两卡记录](routes/route2/planning/two-gpu-results.md)。

两卡增量只应用在完整官方Actor基线之后：

```bash
python3 scripts/restore_sources.py --apply-patches --official-actor-experimental
git -C worktrees/rlinf-official-actor-restored apply --check \
  "$PWD/routes/route2/patches/two-rank-fsdp-compat.patch"
git -C worktrees/rlinf-official-actor-restored apply \
  "$PWD/routes/route2/patches/two-rank-fsdp-compat.patch"
```

[增量源码锁](routes/route2/model_probes/two-rank-source-lock.json)记录补丁与生产文件哈希。`torch22_state_dict_backend=sharded_tensor`使用相同一维process group；`actor_state_mode=full_cpu_rank0`由全部Actor rank汇聚CPU完整字典，只有rank0向单个Rollout发送。均需明确配置。初始同步已通过907项完整状态及固定输出精确对照；fresh8步双卡官方PPO与更新后的版本1同步亦已通过。Runner可显式设置 `runner.data_channel_transport=ray`，三条数据通道保留官方collector/dispatcher，完整官方Runner两轮闭环已通过；更长轨迹稳定性仍在验收。

`musa_fsdp_optim_device.apply()`只在实验进程中处理Torch2.2优化器汇聚的设备调用，以及空CPU/MUSA ShardedTensor的device/is_meta/to；不编辑系统安装包。[small DCP结果](routes/route2/evidence/multi-gpu/small-two-rank-dcp-v3.rank0.json)证明小模型两rank新对象保存/恢复/下一步exact；完整GR00T另有[新进程恢复结果](routes/route2/evidence/multi-gpu/fsdp-official-actor-recovery-v1.json)，模型/active Adam/scheduler/RNG/计数/版本及同批下一次更新通过精确对照。`official_runner_probe.py`已通过官方Runner两轮on-policy闭环，支持 `--iterations` / `--steps-per-env`；实际完成范围以两卡报告为准。
