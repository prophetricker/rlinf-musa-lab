# GR00T N1.5 / LIBERO-Spatial：从双卡闭环进入评测与学习

日期：2026-10-07。本文件是下一阶段实施计划，**预算、学习率试验与成功条件尚未实测**；当前实验结果以[两卡记录](two-gpu-results.md)为准。先补完整回合与评测覆盖，再用真实学习率开展有独立评估的 PPO，避免把参数更新当作任务学习。

## 1. 现在已经证明什么

两张同节点 48 GiB S4000 已完成完整 GR00T 的两 rank FULL_SHARD PPO 更新、Actor 分片 DCP 保存/新进程恢复/同批下一步对照、分片 Actor 向单 Rollout 的完整 CPU 权重同步。Runner-v4 实际运行官方 `EmbodiedRunner.run()`、collector 和 dispatcher，完成 2 环境 × 4 步 × 2 轮的 16 个 transitions；每个 Actor 非空 Adam step 到 2，版本 0/1 的 907 项完整状态与 Rollout 一致。

Runner-v5 的 3 轮 × 2 环境 × 80 步验证已通过：480个数据槽、454个mask有效动作、每rank3次Adam更新，版本0/1/2完整同步一致。采样出现真实termination并完成post-terminal mask训练；尚未覆盖240步truncation，也不是完整suite评测或学习结果。现有 Runner 使用任务 0 / trial 0、actor/value 学习率 `1e-8`、critic warmup 0；Runner-v4 奖励为 0。先前少量 task 0 / trial 0、1 的预训练策略成功结果也不是完整 LIBERO-Spatial benchmark。

冻结研究条件：Driver 2.7.0、Toolkit 3.1.0、Torch 2.2.0、Torch-MUSA `1.3.0+81caf0a`、MCCL 2.11.4；冻结 BF16 Eagle、FP32 action/value head、AMP 关闭。宿主驱动、`/usr/local/musa`、默认 Python 保持原状，继续使用 `/root/autodl-tmp/s4000-research` 的隔离环境。逐次保留配置、补丁、源码/探针哈希和版本，而非仅写“基于某 commit”。

## 2. 官方配方与两卡实验配方

官方参数来自固定 RLinf commit `c70606f08cdca259b8dec03d4430926b5b8fac9d` 的 [libero_spatial_ppo_gr00t.yaml](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/examples/embodiment/config/libero_spatial_ppo_gr00t.yaml) 和 [model/gr00t.yaml](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/examples/embodiment/config/model/gr00t.yaml)。下表右列是建议起点，不是声称复现官方大规模实验。

| 参数 | 固定上游配置 | 两卡下一阶段建议 |
|---|---|---|
| 训练环境 / rollout_epoch | 64 / 8 | 2 / 1 |
| episode 上限 / 每环境采样 horizon | 240 / 240 simulator steps | 240 / 240，先完整回合验收 |
| `num_action_chunks` | 5 | 下一轮优先验证 5；此前 Runner 只验证了 1 |
| `denoising_steps` / `action_dim` | 4 / 7 | 4 / 7 |
| `micro_batch_size` / `global_batch_size` | 128 / 1024 | 1 / 96（chunks=5、horizon=240），见下文计数口径 |
| `algorithm.update_epoch` | 4 | 先 1；4 是之后单独命名的试验 |
| `actor.optim.lr` | **`5.0e-6`** | **`5.0e-6`，先做短学习率验证** |
| `actor.optim.value_lr` | **`1.0e-4`** | **`1.0e-4`，先做短学习率验证** |
| `actor.optim.critic_warmup_steps` | **0** | **0** |
| Adam betas / eps / weight decay | 0.9、0.95 / `1e-8` / 0.01 | 相同 |
| clip_grad / actor clip / value clip | 1.0 / 0.2 / 0.2 | 相同 |
| gamma / GAE lambda / Huber delta | 0.99 / 0.95 / 10.0 | 相同 |
| advantage normalization / reward filtering | True / False | 相同 |
| 精度 | 模型与 FSDP 混合精度 BF16 | 当前已验证的冻结 BF16 backbone + FP32 head；FSDP mixed-precision dtype 为 null，AMP 关闭 |
| 评测环境 / rollout_epoch | 500 / 1 | 2 / 按实际 init-state 数量计算 |
| 保存周期 | 100 Runner steps | 长实验初拟 25 steps，先验证真实 Runner 保存/恢复 |

当前 `1e-8` 用来验证“更新发生且系统稳定”。官方 actor/value LR 分别是它的 500 / 10,000 倍；直接换成官方 LR 仍需要验收。两环境的小 batch 与官方 64 环境配置的梯度噪声不同，不能因为来自官方 YAML 就推断在当前配置必然稳定。先用 `1e-8` 验证 chunks=5 / horizon=240，保持该配置再切换 LR；逐项记录损失、梯度、KL、clip 和任务表现，再决定是否调整 LR。

不要把正数 critic warmup 当作默认修复。[fsdp_model_manager.py](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py) 的 `build_optimizer` 会在 warmup 期间冻结 actor、只训练 value head；结束时重建 optimizer 和 scheduler。这是另一套运行状态，当前恢复验收没有覆盖它。首轮按官方 **0** 开始；若以后试 warmup，需单独锁定配方并复核恢复。

### 双卡布局与样本计数

继续使用已通过的布局：Actor ranks 0–1 占 GPU 0/1，FULL_SHARD、`use_orig_params=True`；单 Rollout 占 GPU 1，并在训练阶段 offload；Env Worker 放在 GPU 0 的 placement，MuJoCo 仿真主要由 CPU 执行，具体渲染后端另记。先保持 pipeline stage 1、同步 Runner、不重叠 bootstrap。

数据通道继续 `runner.data_channel_transport=ray`，权重同步继续 CPU/Ray Bucket、`actor_state_mode=full_cpu_rank0`；FSDP 参数/梯度通信仍用 MUSA/MCCL。保持当前成功的 `MCCL_P2P_DISABLE=1`、legacy norm、Torch 2.2 sharded-tensor DCP 表示。这不代表默认 scheduler collective、GPU 权重传输或 P2P 性能已经通过。

固定代码中的 EnvWorker 以 `horizon // num_action_chunks` 计算策略调用轮数。**已通过的 Runner 配置是 chunks=1；下一轮优先验证官方 chunks=5。** 后者在 horizon=240 时，每环境是 48 次 chunk 决策，每次最多执行 5 个 simulator actions；两环境每轮安排 **96 个 Actor chunk 样本槽 / 480 个 simulator transition 槽**。每 rank 收到一条 48 个决策、240 个动作槽的完整轨迹，microbatch 1，累计 **48** 个 microbatches 后做一次 optimizer step，global batch 应设 **96**。`update_epoch=1` 时每 Runner 轮次是一次分布式更新；两个 rank 同步参与这一次更新，不能算两次独立更新。

另记录实际 simulator transitions、episode 数、有效 action-slot mask 和 chunk-level `loss_mask`，二者不可混用。`auto_reset=False` 下提前成功之后仍有固定长度数据槽；原始动作 mask 应保留结束动作、排除其后的动作槽，而官方 chunk-level 聚合会保留含结束动作的那个 chunk、排除后续 chunks。Denoising 的 4 个内部步骤不计成 4 个环境 transitions。此前 chunks=1 的 batch=480、每 rank 240 microbatches 只适用于 chunks=1 / horizon=240；切换到 5 后必须重新核对 batch/GAE/mask/同步，不能把 96 个 chunk decisions 写成 96 个 simulator transitions。

## 3. 先做完整 10-task 预训练基线

从原始固定权重重新初始化，不继续用工程探针已经更新的权重。权重锁为 `RLinf/RLinf-Gr00t-SFT-Spatial` 的 revision `73f710e70e7d571f8d828e51e0a428f5a1e0ac22`；元数据见[权重来源](../model_probes/weight_metadata/acquisition.json)。**先通过 chunks=5 的合约验证，主 baseline 优先使用官方 chunks=5**、相同 eager/精度适配，以便之后配对比较。chunks=1 若另测，作为单独对照保留；两者不能混比。即使 chunks=5 相同，精度/attention 适配、并发与 seed 协议仍有差别，成绩标成“本适配配置的完整 Spatial 基线”，不直接声称官方公开成绩复现。

### 固定状态与评测配置

1. 从实际 LIBERO task suite 枚举 10 个 task 的名字、描述和全部 init states，保存 manifest。每个 task 的 trial 数 `N_t = len(get_task_init_states(t))`；global reset ID 为 `sum(N_i, i<t) + trial_id`。通常是 10 × 50 = 500，但以实际 metadata 为准，不盲写 `task_id * 50`。记录 LIBERO、robosuite、MuJoCo 版本、BDDL/init-state 数据指纹及渲染设置。
2. 固定 `is_eval=True`、`use_fixed_reset_state_ids=True`、`specific_reset_id=null`、`auto_reset=True`、`ignore_terminations=True`、`max_episode_steps=240`、`max_steps_per_rollout_epoch=240`、2 环境、group size 1、**chunks=5**、denoising steps=4，使用官方 Rollout 的 `mode="eval"`。
3. 官方 eval 保留回合内的 `success_once`，即使达到成功状态，也运行至 240 步截断再 autoreset 到下一 state。保持这一口径，报告“240 步内曾成功”的 success rate；不要替换成最后一帧是否成功，也不要用累计 reward 代替 success。
4. 单 EnvWorker、2 lanes、上述固定长度设置下，`eval.rollout_epoch = ceil(sum(N_t)/2)`：500 states 对应 **250 epochs**，每 epoch **48 次两 lane 策略调用 / 每 lane 240 个 simulator actions**。先做 **5 epochs / 10 states** 的覆盖 pilot，合计 480 个单环境 chunk decisions / 2,400 个 simulator transition 槽；按官方 interleaved 顺序应覆盖每个 task 的 trial 0。必须用实际 task/trial 核对，不能只看 episode 总数为 10。pilot 成功率只用于检查，不作 benchmark 结论。
5. **不要用 `auto_reset=False` 加多个 eval epochs 来轮转全套状态。** 当前 `EnvWorker.evaluate()` 每 epoch 设置 `is_start=True`，`LiberoEnv.reset()` 因此重置 eval pool、start_idx 和 seen 集，会重复初始状态。`auto_reset=True` 只在第一个 epoch 初始化，后续在截断时轮转；这样才能持续覆盖。也不要仅把官方 `total_num_envs=500` 改成 2 而仍保留 `rollout_epoch=1`。

这一路径可直接沿用官方 EnvWorker / Rollout evaluate 通道；新增评测入口只负责固定配置、episode 审计和结果落盘。上述 epoch 算式是该特定布局的实施起点，**最终验收依据是 manifest 覆盖集合**。遇到奇数 state 数、lane 耗尽 `-1` sentinel、重建 worker 或中途恢复时，需检查缺失/重复；不得仅凭执行完指定 epoch 宣布完整。

### 随机性和结果记录

`mode="eval"` 不等于完全无随机性：[gr00t_action_model.py](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py) 的 `get_rl_action` 仍用 `torch.randn` 初始化 latent actions；eval 改为无额外 denoising 噪声的积分路径。初始/训练后评测必须固定 reset ID、顺序、batch 大小和 policy RNG 协议。

推荐为每个两 lane 的完整回合对定义 `eval_seed = 20261007 + pair_index`，在 Rollout 中于该 pair 第一次预测前设置 Python/NumPy/Torch/MUSA RNG，并记录两个 task/trial 与 seed。两 lanes 都按固定 240 步完成，前后使用同一 pair 顺序和种子。这项 seed hook **尚需在新的评测入口实现**；若第一版仅能固定整次评测 RNG，则明确记录该较弱协议，并保持完全相同的调度，不能宣称独立的 per-episode seed 控制。评测使用独立进程或保存/恢复训练 RNG，不改变训练轨迹。

逐 episode 写 JSONL，至少包括：run/config/source/weight 哈希、checkpoint 与权重版本、task_id/name、trial_id、global_reset_id、eval seed、chunks、denoising steps、`success_once`、首次成功步、结束原因、策略动作数、reset settling 步数、nonfinite/崩溃、耗时。默认关闭大部分视频，仅固定保留少量成功/失败回放；视频不能替代数字结果。

完成后输出各 task 的 `success/total` 和成功率，以及任务等权平均。trial 数不等时另报全部 episode 的 micro 平均，不能混用。用 `(task_id, trial_id)` 去重，完整的一次 500-state 评测应恰有 500 个唯一完成条目、每 task 50 条。超时是任务未成功；基础设施崩溃单独记录并保留原失败，再按相同条件补评，不能删除失败 trial 改小分母。

500 × 240 的上限为 **120,000 个 simulator transition 槽**，chunks=5 时为 **24,000 个单环境 chunk decisions / 12,000 次两 lane 策略调用**，另有 reset settling（当前默认每回合 15 步）等开销。两环境降低并发内存，不降低完整 benchmark 的模拟步预算。先根据 pilot 的纯 rollout 耗时估算总时长；当前包含全状态哈希的工程探针耗时不能用作 benchmark 吞吐。chunks=5 的调用数更少，但端到端加速幅度仍需实测。

## 4. 完整回合的工程验收

Runner-v5 的 chunks=1 / 80 步稳定性通过之后，下一轮优先做 **chunks=5、2 轮 × 2 环境 × 240 simulator steps**、仍用 `lr=1e-8` 的完整 horizon 验证：计划 **192 个 chunk 样本槽 / 960 个 simulator transition 槽**，每 rank 每轮 48 microbatches。准确环境/源码/启动条件参照[已验证入口](two-gpu-results.md)，新增CLI和断言说明见[完整回合入口](../model_probes/embodied-eval-README.md)；这不是学习实验，也不是 chunk5 工具已验证的承诺。

此前v4/v5使用[冻结Runner脚本](../evidence/multi-gpu/versions/official-runner-v4-probe.py)，只支持chunks=1。当前[Runner探针](../model_probes/official_runner_probe.py)已增加`--action-chunks 5`，明确区分`simulator_horizon=240`、`chunk_decisions_per_env=48`、global batch 96，审计`T=48`、`T+1=49`以及包含首个终止动作的mask；此增量GPU验收仍待执行。只传`--steps-per-env 240`仍使用默认chunks=1。dones/terminations/truncations的action维为5，values维度按官方value-head口径核对；原脚本与历史证据按其哈希解释。

验收必须覆盖以下实际边界，缺一项便明确标为未覆盖：

- 官方训练默认 `auto_reset=False / ignore_terminations=False`：至少记录一次 240 步 truncation、下一 Runner 轮次 reset，以及 `T+1` 的 values/dones 与 `T` 个 reward/logprob 的对应关系。
- 真实 task success 的 termination：若当前 train-mode 轨迹没有成功，另外用已经成功的固定 eval 回合检查 Env 成功统计，并准备 train-mode 的 terminal/mask 合约用例；不能把全是 timeout 的轨迹称为成功终止路径已验证。
- 原始 action mask 包含首个终止动作，排除其后的动作槽；chunk-level `loss_mask` 由原始 mask 的 `any` 聚合，保留 terminal chunk、排除其后 chunks。覆盖 chunk 中间成功和末尾截断，并分别记录有效 action 与有效 chunk 数。GAE/advantage normalization 使用官方 mask，不把 reset 后观测当上一回合 terminal observation。分别核对非终止 horizon 尾部、真 termination 和时间上限截断；保留固定上游 bootstrap 约定，任何算法语义改变另立试验。
- 官方 eval 的 autoreset、`final_info`、`success_once`、重复 trial 排除、pool 耗尽和每 task 总数。评测用 autoreset 已通过不代表训练改成 autoreset 也已通过；首个学习配方仍保留 train autoreset=False。
- 每轮 Actor 版本与新收数据版本一致；两 rank 数据互不重复且保持完整时间序列；更新/梯度/非空 Adam 状态有限，完整同步无缺键。跟踪整卡与各 worker 内存、CPU/Ray 队列和实际阶段耗时，排查随轮次增长的积压。

907 项哈希相同证明 Actor 与 Rollout 权重一致，不证明 MUSA 与 CUDA/CPU 的所有算子无误差；完整任务成功率则反映策略功能，不单独证明适配实现正确。两类证据都需保留。若成功率低但上述工程项正常，应先检查动作设置、奖励稀疏性、评测覆盖和学习预算，不能直接归因于 MUSA 数值误差。

## 5. Runner 检查点与最终权重

**已经通过的是 Actor DCP 的恢复及同一 fixture 下一步对照，不是完整 Runner 恢复。** [embodied_runner.py](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/runners/embodied_runner.py) 的 `resume_dir` 解析 Runner global step，再加载 `<resume_dir>/actor`；保存目录是 `checkpoints/global_step_N/actor`。当前没有 simulator state、Rollout RNG、eval pool/seen 或在途队列的端到端恢复等价证据。

长学习前新增一个有真实 `save_interval/resume_dir` 的独立 Runner 入口：

1. 在完整采样/更新结束的边界保存，确保没有在途 batch；记录 Runner global step、Actor optimizer/scheduler/RNG、模型哈希、配置和 reset 调度。
2. 新 driver/Workers 从目录恢复；核对两个 Actor 的非空 optimizer 状态和更新计数、global step，再显式同步 Rollout，核对完整状态及版本。
3. 用新采样运行至少下一轮，核对 collector、GAE/PPO、同步和再次保存。若 Env/Rollout 从新 episode 开始，明确称为“Actor 训练状态恢复后重新采样”，不要求轨迹相同，也不宣称中断前后完全连续。
4. 若要证明不中断等价，另保存/恢复 Env、Rollout 和采样 RNG/状态，再与未中断分支比较；当前尚未实现。优先保证可恢复的完整回合边界，不把精确 simulator 中途恢复作为第一学习实验的隐含能力。

另一个必须处理的边界是**末次更新后的同步**。当前关闭 eval/save 的 Runner 在每轮采样前同步，然后更新；N 轮结束时 Actor 已完成 N 次更新，Rollout 仍持有 N−1 次更新后的策略，版本字段也可能仍是 N−1。正式评测最终模型前，等待所有 Actor 完成训练，按实际 global step/optimizer 状态设置并记录版本，再执行 Actor→Rollout 同步和 manifest 审计。以 checkpoint/权重哈希和 optimizer step 标记最终策略，不能只凭旧版本字段。

当前单个完整 DCP 约 **15.08 GiB**，100 GiB 研究盘上保存新旧两个 checkpoint 就占约 30.16 GiB。保存前检查实际空间，先确认新 checkpoint 可恢复再轮换旧文件；源码、紧凑证据进 Git，权重/轨迹/完整 checkpoint 保留在研究盘。

## 6. 真实学习的阶段预算

下列均为**未验证的建议**，不保证给定预算出现成功率提升。每次从固定预训练权重和全新 optimizer 开始，不把 `1e-8` 工程探针的 optimizer 接到学习实验。

| 阶段 | Runner 配置 | 计划 transitions 槽 / seed | 分布式 optimizer updates / seed | 用途 |
|---|---|---:|---:|---|
| 官方 LR 短验收 | chunks=5，2 env × 240 simulator steps × 3 iterations；update_epoch=1 | 1,440（288 chunk 槽） | 3 | 检查 LR 切换后 nonfinite、梯度、KL、同步和行为崩溃 |
| 首次探索性学习 | chunks=5，2 × 240 × 50；update_epoch=1 | 24,000（4,800 chunk 槽） | 50 | 单训练 seed，观察非零奖励与固定 validation 曲线 |
| 多 seed 学习验证 | chunks=5，2 × 240 × 200；update_epoch=1 | 96,000（19,200 chunk 槽） | 200 | 3 个预先固定训练 seeds：7、17、27；合计 288,000 simulator 槽 |
| 后续 PPO epoch 对照 | 同采样预算，update_epoch=4 | 相同 | 分别 200 / 800 | 只在 epoch=1 配方功能稳定后单独比较 |

上表假设 chunks=5 已通过、每轮每 rank 48 个 chunk 样本、global batch 96，尚未改变 rollout_epoch、分发或数据过滤。有效 action-slot 与有效 chunk 样本按各自 mask 另报，evaluation/reset 等步数另记；重复 PPO epochs 不增加新环境数据。先测真实纯采样、反向、同步、保存和审计时间，再估算 24k/96k 时长。若 microbatch=1 太慢，先在已固定的 global batch 下逐项验证更大 microbatch，不能直接声称扩大并行即可保持数值/内存性质。

24k 只做诊断，96k 也不是大模型具身 PPO 的充分学习预算。若基本数值与行为正常但没有提升，依据非零奖励覆盖、critic 拟合、KL/clip 和学习曲线决定扩预算或改配方；增加预算与更换超参数各自命名，保留原结果，不能把未学习当作适配失败或隐藏它。

每轮至少记录 task/trial、成功 episode 数、非零 reward 覆盖、有效/无效样本数、return/value/advantage 分布、actor/critic loss、clip fraction、实际 LR、裁剪前 grad norm、KL 指标、动作越界/饱和、吞吐和内存。`approx_kl` 的有符号估计可能为负，应保留原定义；如额外记录 `mean(exp(log_ratio)-1-log_ratio)`，标明它的口径。对常数 return，explained variance 的 NaN 是指标未定义，应单独处理，不等于模型 tensor 非有限。

## 7. 配对评测与学习验收

先固定学习协议，再看训练后的成绩。建议在初始 manifest 上用固定 split seed 1234，逐 task 对 trial IDs 做一次可复现划分：约 60% train、20% validation、20% final test。常规 50 trials/task 对应 30/10/10，即共 300/100/100；保存具体列表与哈希。这只能保证本轮 PPO 不采样所划出的 states，不能据此声称预训练/SFT 阶段也未见过它们。

当前上游原生 `task_id_filter` 只按 task 限制，`specific_reset_id` 只指定一个 global ID，**尚无已验证的 trial allowlist 配置**。若采用上述严格划分，需在隔离实验入口补 allowlist 支持并审计每次 reset 的真实 task/trial，确认训练不进入 validation/test。实现完成前，可以按官方从全套状态采样做探索，但报告必须写“同一初始状态分布上的学习”，不得把其中挑出的 eval states 称为训练未见的 held-out 集。完整 500-state 初始 benchmark 与这个学习协议分别记录。

第一轮 50 iterations 用训练 seed 7，每 10 iterations 在固定 validation 集评测，全部采用同一个 policy RNG 协议。长验证固定 7/17/27，各跑同一预算；final test 评测预先声明的最终 checkpoint，不能看见 test 成绩后挑 best checkpoint。若需要 best validation checkpoint，作为另一个预先声明的报告项，注明选择依据。

先对每个最终 test episode 做 pretrained/trained 配对，再按 task 汇总成功率差；报告每个训练 seed 的增益、任务等权均值/中位数，以及失败转成功和成功转失败的数量。评测用与训练独立的 rollout 随机数，同一个 reset state 上多评测 policy seeds 时保持完整覆盖、单独命名，不能靠挑 seed 提高分数。主线训练前后都使用同一已验证的 chunks=5 设置，不和 chunks=1 基线混比。

| 验收层次 | 通过条件 | 允许的结论 |
|---|---|---|
| 长轨迹工程 | 第 4 节边界都有证据；每轮有限、数据版本一致、双 rank 更新与完整同步成立；Runner 边界保存/重启继续有独立记录 | 固定两卡配置可运行并恢复该 workload；不意味着学习 |
| 完整预训练 benchmark | 实际 10-task init-state manifest 全覆盖；唯一完成次数/分母正确；无未补评的基础设施异常；逐 episode/task 数字可复核 | 当前适配策略的完整任务表现；不单凭分数宣称所有数值实现等价 |
| 探索性学习 | 固定 budget/checkpoint、完整配对测试；至少 2/3 训练 seeds 增益为正，中位增益为正；公开全部 seed/task 结果 | 有初步学习证据；3 seeds 仍不支持强统计显著性结论 |
| 可靠收益 | 扩大独立训练 seeds/评测重复后，增益与不确定性仍支持稳定提升；说明训练是否见过 test states | 更可信的学习收益与实际适用范围 |

“2/3 seeds 为正”是本项目的探索性决策规则，不是 RLinf 官方阈值，也不是统计检验。只有一两个 episode 的净改善应继续标为微弱证据；单次 loss 下降、Adam step 增长、task 0 成功或挑出的最好 seed 均不满足学习验收。完整评测本身不设必须达到的成功率门槛：低分也可能是有效结果，应据诊断决定下一步。

## 8. 下一次执行清单

1. 协调者完成 Runner-v5 并冻结证据；若通过，新增CLI执行 chunks=5 / 2 × 240 的完整 horizon 工程轮（每 rank 48 chunk decisions、global batch 96），补 terminal/truncation/mask 证据。
2. 已准备独立[评估探针](../model_probes/official_eval_probe.py)，复用官方evaluate、逐pair记录并核对覆盖，GPU执行待验收。先10-state pilot；完整init-state manifest、独立pair seed hook与全套10-task / 通常500-state基线继续分阶段实施。当前单episode历史结果不能冒充这个入口。
3. 新建可配置的学习 Runner 入口，支持 LR/value LR、horizon、训练 seed、更新 epoch、保存/恢复、最终同步；**当前 `official_runner_probe.py` 硬编码 `lr=1e-8`、save/eval 关闭、resume=null，不能只加 CLI 参数就当正式学习工具。** 保留原探针及历史证据。
4. 验证一次 Runner 保存 → 新进程恢复 → 新轨迹下一轮 → 再保存；按实际能力标明是否只恢复 Actor、是否从新 episode 开始。
5. 确定状态划分/采样协议；若用严格 held-out，先验收 trial allowlist。执行官方 LR 的 3 轮短验收，再按 24k 单 seed、96k × 3 seeds 的预算推进。
6. 每次阶段结束保存配置/来源/紧凑证据，由协调者做 Git milestone。暂不需要继续加卡：先获得实际吞吐与两卡长期内存数据，再判断独立 Rollout 第三卡或更大环境并发的价值。

## 来源锁与代码定位

- [现有 source lock](../model_probes/official-actor-source-lock.json)：RLinf 上游 `c70606f08cdca259b8dec03d4430926b5b8fac9d`，适配基线 `82505b27d4d27f8e9af1d7e52524d524dc0f0d50`；当前增量见[两 rank 源码锁](../model_probes/two-rank-source-lock.json)及逐次结果哈希。
- Isaac-GR00T 上游 [4af2b622892f7dcb5aae5a3fb70bcb02dc217b96](https://github.com/NVIDIA/Isaac-GR00T/tree/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96)，适配基线 `bdd6c7426ff62c1fc07153f4ae41769cac51289b`。
- [LIBERO env config](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/examples/embodiment/config/env/libero_spatial.yaml)：固定 init states、240 步、相对 reward、默认 train autoreset=False、256×256 camera。
- [libero_env.py](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/libero/libero_env.py)：`_compute_total_num_group_envs`、`get_reset_state_ids_all`、`_get_ordered_reset_state_ids`、`reset`、`_record_metrics`、`_handle_eval_auto_reset`。
- [LIBERO utils.py](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/libero/utils.py)：interleaved task/trial 顺序、round-robin 分配、`record_completed_episode_task_stats` 去重。
- [EnvWorker](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/workers/env/env_worker.py)：chunk 步数、bootstrap/reset、`env_interact_step`、`env_evaluate_step`、`evaluate` 固定次数循环。
- [Embodied FSDP Actor](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/workers/actor/embodied_fsdp_actor_worker.py)：`_process_received_rollout_batch` 的 loss_mask、官方 GAE 输入、`run_training` 的 batch/epoch/microbatch/update 顺序。

代码定位来自固定上游与当前隔离适配源码核对；独立评估入口代码已准备但GPU未验收，文中pair seed hook、trial allowlist、正式learning入口仍是待实施项，不能作为已有通过结果。
