# 官方 Actor / 独立 Rollout 验证

日期：2026-10-06。限定单卡 S4000，Driver 2.7.0、MUSA 3.1.0、Torch 2.2.0、Torch-MUSA 1.3.0；BF16 Eagle + FP32 action/value head，FSDP1 NO_SHARD，AMP 关闭，critic warmup 为 0。宿主与默认环境不改动。

新增两卡记录：Actor卡0/Rollout卡1的真实LIBERO更新与版本0/1同步已通过；隔离Worker设置`MCCL_P2P_DISABLE=1`后通信可用；官方GR00T两rank FULL_SHARD v15已通过默认同步初始化、完整GAE、四微批次与真正参数更新。显式`legacy`范数兼容解决当前梯度范数耗时造成的rank等待；两rank完整DCP保存/新进程恢复及同批下一次更新精确对照亦已通过；分片同步与完整Runner继续独立验收。详细证据和复现命令见 [两卡验证](../planning/two-gpu-results.md)。以下单卡结果仍按原范围解释。

`official_actor_lifecycle_probe.py` 继承真实 `EmbodiedFSDPActor` 构造和 `init_worker`；初始化已通过。`official_actor_rollout_probe.py` 的两个诊断子类继承官方初始化、预测、trajectory receive、GAE、PPO training、Bucket 同步和 checkpoint，只添加 CPU RPC 边界、哈希和固定输入检查。

连续官方PPO v1和fresh-process recovery v1已通过：三轮8步真实LIBERO GAE/PPO，版本0/1/2/3同步的907个状态哈希和固定输出精确一致。Adam steps1/2/3与309/309/310个参数变化，梯度有限非零。新进程的权重、Adam、scheduler、四类RNG、计数与version，以及同批下一步全部精确一致。全部reward=0，未建立学习收益。Actor reserved峰值33.10 GiB，整卡采样峰值35.43 GiB。

初始同步 v2 已通过：907 个完整状态张量精确一致，322 个可训练参数/持久 buffer 通过 CPU Bucket 传输；固定输入 logprob、selected old logprob、value 精确一致，初始 PPO ratio=1。v1 在固定输入重算时遇到旧 Torch-MUSA 的 strided index 除法断言，原失败记录保留。MUSA 上显式 clone 去噪索引到紧密布局后通过；不改变索引值、损失或阈值。

## 固定源码

可用 `python scripts/restore_sources.py --apply-patches --official-actor-experimental` 创建新的 `worktrees/rlinf-official-actor-restored`；已有目录会保留，补丁哈希会核验。本机已从公开上游加补丁重建并核对17个生产文件精确一致。

RLinf 使用公开上游 `c70606f08cdca259b8dec03d4430926b5b8fac9d`。先应用 `current-minimal-compat.patch` 得到既有 Worker 基线（本地 commit `3e7329c629293f1b0f9c331b5bb24eccc3e3f142`），再只应用 `official-actor-combined.patch`。该补丁已包含 FSDP1、checkpoint、DTensor 兼容及新的 loader/同步/runtime/index 改动，**不要再叠加旧实验补丁**。干净源码重建已逐文件核对17个生产源文件精确一致。

GR00T 使用公开 `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`，按 `scripts/restore_eagle_sources.py --spatial-full --output <new-dir>` 恢复已固定的 eager 源码，再应用 `gr00t-lazy-pytorch3d.patch`。LIBERO 路径不使用 PyTorch3D；真实旋转变换调用时仍需要安装实际依赖。Spatial-73f710e 权重来源与哈希见既有下载 manifest。

设 `RLINF_EXT_MODULE=actor_runtime_extension`，通过 RLinf 支持的模型注册扩展接入已有 FP32 action-head fallback。实际 top/vision/language attention 均断言为 eager。旧 Torch 路径须显式设置 `RLINF_EXPERIMENTAL_FSDP1_TORCH22=1`。

## 服务器入口

以下为研究目录内的命令，不含连接或认证信息。输出和 checkpoint 路径必须为新路径；不要覆盖旧记录。

```bash
export PYTHONPATH=/root/autodl-tmp/s4000-research/envs/route2-integration/lib/python3.10/site-packages:/root/autodl-tmp/s4000-research/envs/route2-integration
/root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/model_probes/official_actor_rollout_probe.py \
  --rlinf-source /root/autodl-tmp/s4000-research/route2/RLinf-official-actor \
  --gr00t-source /root/autodl-tmp/s4000-research/route2/Isaac-GR00T-official-actor \
  --model-path /root/autodl-tmp/s4000-research/route2/weights/Spatial-73f710e \
  --phase train --iterations 3 --steps 8 \
  --checkpoint /root/autodl-tmp/s4000-research/route2/checkpoints/official-actor-continuous-v1 \
  --continuation-trajectory /root/autodl-tmp/s4000-research/route2/results/official-actor-continuation-v1.pt \
  --output /root/autodl-tmp/s4000-research/route2/results/official-actor-continuous-v1.json
```

连续三轮真实 LIBERO 轨迹→GAE→官方 PPO→版本递增→独立 Rollout 同步。随后收集另一批真实轨迹，保存完整 DCP 训练 checkpoint，再执行不中断的同批 continuation。另起 driver 用 `--phase recover --checkpoint ... --continuation-trajectory ... --reference <train-json> --output <new-json>` 验证新进程恢复与同批继续训练。

`--phase sync` 仅验证初始化与版本0同步；`--phase eval --episode-ids 0 1` 评估初始策略的同一 Spatial 任务两个初始状态，完整运行至成功或240步截断。可传入 `--checkpoint` 评估保存策略。两局已通过：task0 trial0/trial1分别79/69步成功，平均预测0.799/0.817秒。环境关闭时出现未修复非致命EGL析构异常，完整原始记录保留。两局不是完整10任务 benchmark，微小学习率的三轮更新也不是学习收益实验。

## 边界

`transport=ray_cpu` 默认限制一 Actor、一 Rollout、CPU Bucket 和 Ray Channel；两卡增量显式设置 `actor_state_mode=full_cpu_rank0` 时允许 FSDP1 多 Actor 汇聚完整 CPU 权重供单 Rollout 同步。默认 collective 路径保留。真实 MCCL 多rank FULL_SHARD 更新/恢复已通过，多rank同步独立验收。Runner 已接入可选 weight channel，但本探针使用显式官方 Worker 编排，尚不声称完整 Runner 已通过。

恢复对照覆盖 Actor 权重、Adam、scheduler、Python/NumPy/CPU/MUSA RNG、计数和策略版本；不恢复仿真器、Rollout RNG 或 Runner 进度。runtime metadata 限当前 warmup=0/AMP关闭配置，checkpoint 必须新建目录。大权重与 checkpoint 不进入 Git。

多卡布局、进入顺序与实测资源要求见 [multi-gpu-entry](../planning/multi-gpu-entry.md)。
