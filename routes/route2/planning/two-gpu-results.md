# 两张 S4000：本轮验证记录

日期：2026-10-07。**官方chunk5/240步训练、十任务初始策略pilot及Runner保存/新进程恢复续训全部通过**：训练192动作块、960有效模拟动作槽，超时截断/reset正常；pilot十个task各trial0，原始策略4/10回合曾成功；Runner保存后两rank共22项训练状态精确恢复，新轨迹PPO将累计Adam step增到2，末次版本2的907项完整状态与Rollout一致。此前v5真实termination/post-terminal mask及Actor同批恢复对照保留。LR仍1e-8，没有具身学习收益或完整500-state benchmark结论。

## 环境、源码与证据范围

同节点两张 MTT S4000，保留 Driver 2.7.0、MUSA Toolkit 3.1.0、Torch 2.2.0、Torch-MUSA `1.3.0+81caf0a`、MCCL 2.11.4。宿主驱动、`/usr/local/musa` 与默认 Python 环境均不改动，探针使用研究目录的隔离环境与源码。研究盘已扩至 100 GiB，本轮协调者现场检查约 45 GiB 可用；可用空间会随 checkpoint 和结果增加而变化。

源码基线见 [官方 Actor source lock](../model_probes/official-actor-source-lock.json)：

- [RLinf](https://github.com/RLinf/RLinf)：公开上游 `c70606f08cdca259b8dec03d4430926b5b8fac9d`；既有适配基线 `82505b27d4d27f8e9af1d7e52524d524dc0f0d50`。
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)：公开上游 `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`；既有 eager / lazy PyTorch3D 适配基线 `bdd6c7426ff62c1fc07153f4ae41769cac51289b`。
- GR00T N1.5 Spatial-73f710e 权重；冻结 BF16 Eagle，FP32 action/value head；AMP 关闭，critic warmup 为 0，学习率 `1e-8`。

source lock 是既有基线。当前两卡实验的具体探针哈希、配置与逐文件源码指纹，以各次结果 JSON 为准；不能把基线 commit 当作所有后续实验修改的完整记录。v12 探针已冻结为 [update-v12-probe.py](../evidence/multi-gpu/versions/update-v12-probe.py)，SHA256 为 `4c62cdd91f6e2e73b93e74ea57d27e5a6c2d478c867d4590ff17c83f8464e282`。

## 已通过的七个层次

| 层次 | 实测结果 | 能证明的范围 |
|---|---|---|
| 独立 torchrun / MCCL | 每 rank 6/6：两次 barrier、broadcast、SUM all-reduce、list all-gather、send/recv | 同节点两卡、小 FP32 张量、显式 MCCL 通信 |
| 小模型 FSDP1 FULL_SHARD | 两 rank 均完成一次 AdamW 更新；梯度范数 `0.618055522441864`；rank 0 / 1 分别 1 / 4 个本地参数视图变化 | 真正分片的小 FP32 模型前向、反向、梯度归约与更新 |
| Actor / Rollout 分卡 | Actor 卡 0、Rollout 卡 1，各组 world size 1；真实 8 步 LIBERO → 官方 GAE/PPO，版本 0/1 同步后全部 907 个状态哈希一致 | 两组分开部署、CPU/Ray Bucket 传输、官方单 rank Actor 更新与跨卡同步 |
| GR00T 两 rank FULL_SHARD | v15 两 rank 各用四个独立时间样本完成一次官方 PPO 更新；非空 Adam step `0→1`，梯度范数一致，参数与优化器哈希变化 | 默认同步初始化、完整轨迹 GAE、分片前后向、梯度归约/裁剪及真实 AdamW 更新；输入是既有轨迹 fixture |
| GR00T 分片DCP保存与恢复 | checkpoint-v4两rank保存版本1；recovery-v1新进程恢复和下一次更新每rank各11项对照通过，metrics精确一致，非空Adam step与Actor版本均`1→2` | 固定两rank拓扑、同一fixture下完整GR00T训练状态恢复与续跑等价；不覆盖world size变化 |
| 分片Actor→单Rollout更新与同步 | sync-v1通过版本0；train-sync-v1新采集8步后更新并同步版本1，两个版本各907项状态与固定输出全等，316项完整状态变化均在322项selected中 | 新轨迹、官方GAE/PPO、两rank更新及训练后策略一致；手工time分样本，尚不覆盖Runner collector |
| 官方Runner两轮闭环 | runner-v4官方collector/dispatcher共16 transitions，每rank两次GAE/PPO；Adam step 2；版本0/1各907项完整状态同步一致 | 固定2环境×4步×2轮、CPU/Ray通道、MUSA FULL_SHARD闭环；末次更新未再同步，较长稳定性与学习仍待验证 |

torchrun 数值检查包括 broadcast `[17, 23]`、SUM `[3, 6]`、all-gather `[[0, 10], [1, 11]]`、接收 `[31, 37]`。证据：[rank 0](../evidence/multi-gpu/mccl-multirank-v1.rank0.json)、[rank 1](../evidence/multi-gpu/mccl-multirank-v1.rank1.json)。这两次 JSON 记录的各操作耗时合计为 0.810670 / 0.812276 秒，仅是小张量功能探针。

小 FULL_SHARD 模型为 `Linear(8,16) → Tanh → Linear(16,4)`，使用 `use_orig_params=True`。两 rank 的 loss 均为 `0.13031381368637085`，更新后参数平方和归约均为 `6.591341972351074`。两卡的变化项合计 5 是本地分片视图计数，同一原始 weight 可跨两卡，不能当作 5 个独立原始参数。证据：[rank 0](../evidence/multi-gpu/fsdp-multirank-v2.rank0.json)、[rank 1](../evidence/multi-gpu/fsdp-multirank-v2.rank1.json)。两 rank 使用相同小批次，未在这里建立单卡/两卡数值等价或学习收益。

[Actor / Rollout 分卡 v1](../evidence/multi-gpu/official-actor-disaggregated-v1.json) 中，第一次真实 8 步更新有 312 个参数变化，Adam step `0→1`，梯度范数 `603.681884765625`，训练 ratio `1.0`。随后另收集 8 步轨迹并执行 continuation，310 个参数变化，Adam step `1→2`，梯度范数 `473.5367736816406`，ratio `1.0`。版本 0/1 的固定输入 logprob、old logprob、value 均精确相等，最大差值 0；每次核对覆盖全部 907 个状态张量，Bucket 选择集为 322 项。

最终 Actor 已到版本 2，Rollout 最后一次同步停在版本 1；不能把版本 2 的同步算作通过。v1 保存了版本 1 的训练 checkpoint，但本轮记录没有新的分卡 fresh-process 恢复结果。[分卡 sync v2](../evidence/multi-gpu/official-actor-rollout-disaggregated-v2.json) 另独立复核了初始版本 0 的 907 个状态与固定输出。

分卡 v1 的 Actor allocator peak allocated / reserved 为 27.023738 / 33.103516 GiB，Rollout 为 8.373157 / 8.476563 GiB。训练段记录约 32.18 / 32.75 秒，包含哈希和审计调用，不是纯训练吞吐。两批奖励均为 0，微小学习率用于验证更新；这些结果尚不证明任务学习。此布局仍是 Actor 单 rank NO_SHARD；多rank分片同步由下面的独立探针验证。

## 分片 Actor → 单 Rollout：初始版本0同步

[official-two-rank-sync-v1](../evidence/multi-gpu/official-two-rank-sync-v1.json)为pass，`phase=sync`，布局为两rank FULL_SHARD Actor占卡0/1，单Rollout占卡1并按阶段offload。显式`actor_state_mode=full_cpu_rank0`让两个Actor参与完整状态重建，rank0持有907项CPU参数/buffer manifest并传输；rank1按`rank0_only`返回空manifest，这不是缺少其分片参与。rank0 manifest与Rollout的907项SHA256、shape、dtype及版本0完全一致；322项selected名称在完整状态中全部覆盖。

两个Actor rank与单Rollout分别对同一固定输入重算`logprobs / prev_logprobs / values`，六组比较均`torch.equal=True`、最大差值0。输入来自既有版本3轨迹fixture，只用于固定输入重算；没有在此phase采集fresh轨迹或训练，旧轨迹存储的logprob也不应当作当前版本0的on-policy分数。探针总耗时160.932334秒，含初始化、重建、哈希及输出审计，不是单次权重传输吞吐。训练后版本1由下面的独立train-phase验证。

已执行的[two_rank_rollout_probe.py](../model_probes/two_rank_rollout_probe.py)SHA256为`a7bd554b45bf145c3a9a84b9f3479d3a25c044ce2b59c16ff627f9cb37c1ec93`；四份support文件指纹见结果JSON。该记录的三处生产源码指纹与[冻结v1锁](../evidence/multi-gpu/versions/two-rank-lock-v1.json)完全一致，对应[冻结v1增量补丁](../evidence/multi-gpu/versions/two-rank-compat-v1.patch)SHA256为`19a90ef713a8508facb14bccd02e5527d585d9729e2d19a5edd5e6382199171d`。冻结锁的status是当时状态，不代表最新验收。协调者后续新增了Runner的显式Ray数据通道选项与初始化适配；它们已在下面Runner-v4的固定配置中执行，历史失败和较长验证边界分别记录。

## 新 LIBERO 轨迹 → 双rank PPO → 版本1同步

[official-two-rank-train-sync-v1](../evidence/multi-gpu/official-two-rank-train-sync-v1.json)为pass，`phase=train`，与sync-only使用相同配置、探针、support及源码指纹。版本0同步后，单Rollout用当前权重在LIBERO-Spatial任务0、trial 0新采集8步，并额外计算末端bootstrap value；任务为把盘子与小碗之间的黑碗放到盘子上。8步reward均为0，termination/truncation均为false，数据版本全部为0。新轨迹文件SHA256为`684052bacad36c9741cf779fee2d920fc1f486acc73acac94177f0ec0e774a67`，批数据树SHA256为`eff5a2fdff9f8e559250f47963a931210af5961e4a98d482f95600fe4e509e0c`；文件保存在远端研究盘。

两个Actor各对完整的1环境×8步轨迹执行官方GAE与优势归一化，完整GAE哈希均为`c2a87c89ecd00d16ae387c0d1b161cc10d3ff099f61d6ae38bc269eb941effda`，再手工沿time轴分给rank0的0–3和rank1的4–7；全局恰有8个唯一样本。随后调用继承的官方`run_training()`，每rank四个micro batch、一次AdamW更新。该探针直接编排env/rollout/actor调用，未走官方Runner collector/dispatcher；后述Runner-v4使用2环境×4步、沿环境batch维分发，每rank对自己的完整环境轨迹计算GAE和优势归一化。因此两者的轨迹边界、优势归一化范围与数据分发不同，不能用此结果代替Runner或要求其梯度数值相同。

| 新轨迹更新验收项 | rank 0 | rank 1 |
|---|---|---|
| 七项更新检查 | 7/7通过 | 7/7通过 |
| trainable分片 / optimizer哈希 | 均变化 | 均变化 |
| 非空Adam step / manager更新计数 | `0→1` / `0→1` | `0→1` / `0→1` |
| 裁剪前全局梯度范数 | 1029.08984375 | 1029.08984375 |
| 新数据训练ratio / approx KL | `1.0` / `0.0` | `1.0` / `0.0` |
| 模型、优化器、metrics有限 | 全通过 | 全通过 |

更新后显式设置Actor版本1，再执行完整CPU状态重建与Bucket同步。版本0和1分别核对907项状态的SHA256/shape/dtype与版本一致；每个版本的两个Actor对Rollout各有三项固定输出比较，共12组`torch.equal=True`、最大差值0。版本0→1共有316项完整状态变化，全部属于322项selected集合，没有遗漏已变化状态。此处用于输出对照的fixture仍是既有版本3数据，重新计算当前权重输出；用于PPO训练的是另行新采集的版本0轨迹。

版本1审计结束时记录的内存如下，单位GiB；GPU值为各Worker allocator的累计peak，宿主内存为各进程的VmHWM/VmRSS：

| Worker | GPU peak allocated / reserved | 宿主VmHWM / VmRSS |
|---|---|---|
| Actor rank 0 | 23.461606 / 27.613281 | 16.635368 / 12.609390 |
| Actor rank 1 | 23.454576 / 27.615234 | 12.456734 / 6.495949 |
| Rollout | 8.373157 / 8.476562 | 12.763046 / 11.250744 |

各Worker峰值发生于不同阶段，Rollout训练时offload，不能相加作为物理卡瞬时峰值；Runner双环境推理的实测另见后述v4。完整探针耗时317.541772秒，包括初始化、新轨迹采集、GAE、更新、两次同步与审计，不是纯训练或rollout吞吐。学习率仍为`1e-8`，8步没有任务成功或非零reward；此结果验证新数据更新和策略同步，不证明任务学习、长时间稳定性或benchmark收益。

## 官方 Runner：两轮闭环与失败历史

Runner-v4已通过官方`EmbodiedRunner.run()`、`TrajectoryCollector`和`least_loaded` dispatcher，不手工切time样本。实际配置为2环境×4步、全局batch 8、每rank micro batch 1、update epoch 1、两次Runner迭代；Actor占卡0/1，单Rollout占卡1，Env占卡0。显式`runner.data_channel_transport=ray`将env/rollout/actor数据通道设为CPU/Ray传输，保留collector与dispatcher；默认通道选项仍为collective。FSDP模型的all-gather/梯度归约仍用MUSA/MCCL，Ray隔离作业仍设`MCCL_P2P_DISABLE=1`并保留SHM。成功的固定配置不代表默认scheduler多成员collective、原生GPU权重传输或P2P吞吐已通过。

[official-two-rank-runner-v4](../evidence/multi-gpu/official-two-rank-runner-v4.json)为pass，耗时451.578820秒，包含初始化、官方采样/GAE/PPO、两次完整权重同步与哈希审计，不是纯训练吞吐。每轮两个Actor各收到一条4步完整环境轨迹，policy字段首两维`T=4,B=1`，value与done边界为`T=5,B=1`，保留末端bootstrap；两rank每轮的`forward_inputs + prev_logprobs`哈希不同。数据版本依次为0/1，每rank各执行两次官方GAE/PPO；实际收到的环境transitions是`8+8=16`，不按denoise链或logprob内部维度重复计数。两个LIBERO实例均为task 0、trial 0；不是多任务验证。

| Runner更新验收 | 第0轮，两个rank | 第1轮，两个rank |
|---|---|---|
| 数据接收六项检查 / GAE有限 / PPO七项检查 | 全通过 | 全通过 |
| 新数据版本 / 每rank环境transitions | 0 / 4 | 1 / 4 |
| manager更新计数 / 非空Adam step | `0→1` / `0→1` | `1→2` / `1→2` |
| 裁剪前全局梯度范数 | 1958.786376953125 | 961.2060546875 |
| 训练ratio / approx KL | 0.9999979138374329 / 2.0712614059448242e-6 | 0.9999992251396179 / 7.37607479095459e-7 |
| 本地trainable/optimizer哈希变化，状态有限 | 全通过 | 全通过 |

版本0/1各有9项同步检查通过：两个Actor共同重建、rank0持有907项完整状态，322项selected唯一且覆盖；与Rollout的全状态SHA256/shape/dtype及版本精确一致。每轮7项综合检查及最后7项检查也全部通过。[Actor rank0 events](../evidence/multi-gpu/official-two-rank-runner-v4.actor.rank0.events.jsonl)与[rank1 events](../evidence/multi-gpu/official-two-rank-runner-v4.actor.rank1.events.jsonl)记录receive→GAE→training enter/complete两轮边界。Runner此处验收完整manifest同步，固定输入三项输出精确相等的验收来自前面的train-sync-v1，不混为本轮新增证据。

末态为Runner `global_step=2`，两个Actor `optimizer_steps=2`、非空Adam step为2，Actor和Rollout的版本字段仍为1。Rollout持有第一次PPO更新后的权重，用于采集第1轮；Actor已完成第二次PPO更新。Runner按迭代开始时同步，此次没有追加最终同步，不能把末态版本字段同为1解释成第二次更新后的完整权重仍一致，也不能声称版本2已同步。

| Runner-v4进程 | GPU peak allocated / reserved，GiB | 进程RSS高水位，GiB |
|---|---|---|
| Actor rank 0 | 23.456852 / 26.353516 | 15.989670 |
| Actor rank 1 | 23.453743 / 26.062500 | 12.458157 |
| Rollout | 8.380953 / 8.513672 | 12.461044 |
| Env Worker | 未记GPU值 | 0.582851 |
| Driver | 未记GPU值 | 0.499020 |

GPU值为每Worker allocator峰值，不能相加为物理卡瞬时峰值；宿主RSS不包含LIBERO simulator子进程及Ray共享object storage。Rollout最终已offload，记录allocated为0，reserved约0.001953 GiB。这证明本次双环境4步配置能完成两轮，尚不能推导80步或更大batch的显存/内存余量。全部16个reward为0，dones/terminations/truncations计数为0，loss mask全有效；学习率仍为`1e-8`，未证明任务成功、学习收益或benchmark性能。

[冻结Runner-v4/v5探针](../evidence/multi-gpu/versions/official-runner-v4-probe.py)SHA256为`8361ab1f0b2e15efcf9934e5fd631eabdd7afb99c0d9862635bb6805898f4bcc`。本轮六份生产源码指纹逐项与[当前源码锁](../model_probes/two-rank-source-lock.json)核对一致，增量补丁SHA256仍为`5da7ecf0ce3965cd154bdc1ffcf3e3816352772f85115040fe2a5658294d21f0`；support指纹与配置以结果JSON为准。后续当前CLI已增加chunk选项与新审计，旧结果按冻结脚本解释。

### Runner-v5：三轮80步、真实终止与mask

[v5原始结果](../evidence/multi-gpu/official-two-rank-runner-v5.json)为pass，耗时2709.071960秒，包含加载、采样、更新、完整同步和哈希审计，不是纯训练吞吐。配置仍为动作chunk1、任务0/trial0、2环境、每rank microbatch1、全局batch160、update epoch1、actor/value LR均`1e-8`。两个rank各完成3次真实Adam更新，参数和optimizer哈希变化且状态有限；每轮官方receive、GAE、PPO、版本与完整状态同步检查通过。[独立结果核对](../evidence/multi-gpu/official-two-rank-runner-v5.audit.json)18项通过，关联原始结果、探针和[运行前源码锁快照](../evidence/multi-gpu/versions/two-rank-lock-before-runner-v5.json)的精确SHA256。

| v5实际数据 | 第0轮 | 第1轮 | 第2轮 |
|---|---:|---:|---:|
| 环境数据槽 | 160 | 160 | 160 |
| rank0 / rank1 mask有效动作 | 80 / 80 | 65 / 80 | 80 / 69 |
| rank0 / rank1 reward sum | 0 / 0 | 1 / 0 | 0 / 0 |
| rank0 / rank1 termination true标记数 | 0 / 0 | 7 / 0 | 0 / 6 |
| 两rank共同的裁剪前全局梯度范数 | 52.18792724609375 | 52.485992431640625 | 45.53128433227539 |
| 官方聚合ratio | 1.0 | 0.9062501192092896 | 0.9312501549720764 |

总计480个槽、454个有效动作、26个终止后补齐槽。termination标记7/6可在同一回合后续动作上重复，不是7/6个成功episode。第2轮rank1出现非零正负相对reward但sum为0，不能据reward sum推断任务从未成功；完整评测使用`success_once`。本轮没有truncation，80步也不覆盖240步时间上限；该边界由后续完整horizon另验收。

**上述ratio小于1来自当前官方微批次指标聚合口径，不能直接解释为策略偏离。** 固定`losses.py`对全mask掉的micro batch返回ratio0；Actor先对每rank所有80个micro batch的指标取普通均值，再跨rankAVG。因此第1/2轮的有效微批次比例分别为145/160=0.90625、149/160=0.93125。将原ratio除以该比例，派生有效微批次平均约1.000000132/1.000000166；保留原指标，不替换原始证据，也不把此公式推广到更大microbatch或其他PPO epoch。约`-1e-7`的原有符号approx KL同样保留，不放宽数值gate。

末态Runner step3、Actor各Adam step3，Actor/Rollout版本字段仍为2。Rollout用于第三轮采样，持有第二次PPO更新后的权重；Actor已完成第三次更新。没有额外最终同步，不能声称第三次更新后的策略已与Rollout一致。版本0/1/2每次同步各核对907项完整状态；固定输入输出等价仍引用此前train-sync探针，不混为v5新增验收。

Actor0/1最终GPU allocator peak allocated分别23.456852/23.453743 GiB，reserved分别26.353516/26.062500 GiB；进程RSS高水位16.330467/12.445137 GiB。各进程峰值不能相加为整卡瞬时占用，宿主数值不含simulator子进程与Ray共享object storage。驱动退出后两卡均4 MiB、0%利用率，随后在同次开机周期启动chunk5/240步，十任务pilot在其通过后串行执行；未关机或重启。

新增CLI/评估源码与精确命令见[完整回合和评估入口](../model_probes/embodied-eval-README.md)，后续学习率、全套状态覆盖、Runner恢复与学习收益计划见[学习入口计划](embodied-learning-entry.md)。chunk5和十任务pilot的新增结果见下文，不用v5扩大其历史验收范围。

### chunk5-v1：完整240步与超时截断

[原始结果](../evidence/multi-gpu/official-two-rank-chunk5-v1.json)和[独立核对](../evidence/multi-gpu/official-two-rank-chunk5-v1.audit.json)均为pass，独立核对16项通过；两个rank各14条JSONL事件与最终receive/GAE/training记录逐字段一致。探针字节见[冻结chunk5脚本](../evidence/multi-gpu/versions/official-runner-chunk5-v1-probe.py)，SHA256为`a5b59230ef4bc469d261a3c639a6a643b565506bb3fd45e0859777c2b8adf451`，六份生产文件与[运行前锁快照](../evidence/multi-gpu/versions/two-rank-lock-before-chunk5-v1.json)一致。完整耗时1235.217697秒，包括初始化、采样、训练、同步和哈希审计，不是纯训练吞吐。

固定任务0/trial0，两环境、每环境240模拟步、每次策略决策执行5动作；global batch96、microbatch1、update epoch1、actor/value LR均1e-8。每rank每轮收到48个chunk决策，policy时间维T=48、value/done边界T+1=49；每rank48个microbatches后共同完成一次分布式optimizer更新。两轮合计192个chunk样本、960个模拟动作槽，全部mask有效。不能把192个chunk或4步denoising链当独立环境transition。

| 完整horizon检查 | 第0轮 | 第1轮 |
|---|---:|---:|
| 两rank有效动作槽 / 有效chunk | 480 / 96 | 480 / 96 |
| 各rank truncation true标记 | 1 / 1 | 1 / 1 |
| 各rank termination / reward sum | 0 / 0 | 0 / 0 |
| 非空Adam step / manager计数 | 1 / 1 | 2 / 2 |
| 全局裁剪前梯度范数 | 34.53612518310547 | 33.59235382080078 |
| 官方训练ratio | 0.9999997615814209 | 1.0000001192092896 |

四条真实240步轨迹均以时间上限truncation结束，下一Runner轮次的pre-action done重新为false，bootstrap与独立first-ending-action mask检查通过。没有成功termination；chunk5中间成功的真实路径尚未覆盖，CPU边界case及chunk1-v5的真实termination证据分别保留。两个Actor训练分片/optimizer都变化，metrics、参数、optimizer及GAE有限，版本0/1各907项完整状态与Rollout一致，selected322项完整覆盖。

末态Runner step2、Actor各Adam step2，Actor/Rollout元数据version1；Rollout只有首次更新后的权重，第二次更新后没有追加同步。后续[生命周期入口](../model_probes/runner-lifecycle-README.md)显式验收最终同步，不能将其计划当本轮已通过项。

Actor0/1 allocator peak allocated为23.456853/23.453744GiB，reserved为26.353516/26.062500GiB，进程RSS高水位16.073986/12.477627GiB。峰值为各进程累计值，不相加为整卡瞬时占用。随后同次开机周期串行启动十任务初始策略pilot，没有关机或重启。

### 十任务pilot-v1：执行完成，汇总探针失败

[原始v1结果](../evidence/multi-gpu/official-spatial-ten-task-pilot-v1.json)为fail，原因见[失败说明](../evidence/multi-gpu/official-spatial-ten-task-pilot-v1.failure.txt)。官方eval已完成10个唯一task/trial0和240次双环境预测，官方metrics报告10 trajectories、`success_once≈0.4`、`success_at_end≈0.3`、episode_len240；Env统计成功task为0/5/6/9，其余失败。每task实际有50个init states，此处仅评trial0，不是全500-state基线。耗时409.582088秒包括加载、权重哈希、评估和审计。

失败发生在评估结束后`result["rollout"]["prediction_batches"]`：`rollout.probe_report().wait()`返回单元素列表，v1未解包便按字典读取。它是探针结果契约错误，不能作为MUSA、模型或模拟器执行失败的证据；也不将失败结果改成通过。冻结v1 SHA为`abb80db445de6f1b81f00b84a4bdd42dd320c890b4cd87dbf65e0203ea752569`；[v2冻结脚本](../evidence/multi-gpu/versions/official-eval-pilot-v2-probe.py)只核对返回一个Rollout并取首项，SHA为`72b32a3bee427dd0ab6d7faebec6d4c4414d280869f671122a327048a0fd6206`，已按相同配置独立重跑，结果见下一节。

### 十任务pilot-v2：覆盖、统计和初始策略功能通过

[v2原始结果](../evidence/multi-gpu/official-spatial-ten-task-pilot-v2.json)为pass，六项官方入口检查与[独立核对](../evidence/multi-gpu/official-spatial-ten-task-pilot-v2.audit.json)20项全部通过。实际仅启动Env与Rollout，调用继承的`EmbodiedEvalRunner.init_workers/evaluate`，没有Actor/PPO更新；重新加载固定Spatial权重，两分片完整SHA256/bytes与公开LFS元数据一致。六份锁定生产文件与评估Runner的第七份指纹匹配，显式Ray数据通道已在GPU评估路径执行；补丁仍只改变显式ray选择，默认collective未在此验收。

两lane、chunk5、每lane1200模拟步、auto_reset=True/ignore_terminations=True；共240次双环境策略预测、480个单环境chunk、2400模拟动作槽，十个唯一task/trial0完成，各task分母1。Task0/5/6/9成功，success_once=4/10、success_at_end=3/10；其余六task失败。独立核对逐episode事件、worker seen集合、task统计、global reset ID、language、240步长度与官方聚合，全部相符。v1/v2十条episode的所有非时间字段精确一致，不只总成功数一致。

实际init-state manifest为每task50、合计500，pilot只覆盖2%；`full_suite_coverage=false`。策略seed1234在model初始化前整次设置，未做per-episode reseed。4/10是单seed、小样本初始策略表现，不是全suite benchmark，也没有训练前后增益。`success_once`包含回合内曾成功、最后又偏离目标的情形，不能用最后帧或reward sum代替。

耗时381.840998秒，包含加载、权重哈希、采样和审计；Rollout allocator peak allocated/reserved为8.381166/8.513672GiB。JSONL相邻48次预测批次间隔包含采样与reset开销，粗估完整500-state基线约4–5小时，不是已测完整时长。Runner生命周期验证随后在同次开机周期串行执行。

### Runner保存-v1：末次同步与完整DCP

[保存结果](../evidence/multi-gpu/official-runner-save-v1.json)与[独立核对](../evidence/multi-gpu/official-runner-save-v1.audit.json)均pass，19项核对通过。独立[生命周期冻结脚本](../evidence/multi-gpu/versions/official-training-lifecycle-v1-probe.py)SHA为`7dcd689df311e0b96c9ef9c63b67d3c398debea6b70d35189489beffa8cfe194`，复用已锁定生产代码，显式启用legacy norm与已验证的checkpoint device-handle helper。每环境10模拟步、chunk5、global batch4、每rank两个microbatches，继承官方Runner一次GAE/PPO更新；共4chunk、20有效模拟动作槽，LR仍1e-8，不是完整回合或学习实验。

更新完成后，显式将Actor/Env/Rollout的global step设为1，再调用继承的权重同步；版本0及末尾版本1各907项完整状态与Rollout一致。调用真实`runner._save_checkpoint()`保存`global_step_1/actor`，两rank非空Adam step/manager/version均1，保存前后模型、非空optimizer、scheduler、RNG、计数与版本快照全等。完整DCP共5个文件、15.081598GiB，耗时451.955428秒，包含初始化、采样、同步、保存与哈希审计。

检查点在研究盘`route2/results/official-runner-save-v1.logs/official-runner-save-v1/checkpoints/global_step_1`，完整权重不入Git。此前独立Actor恢复证据不能代替此Runner checkpoint的恢复，新进程结果见下一节。原Runner-v4/v5/chunk5没有追加最终同步，新入口的结果不扩大它们的历史范围。

### Runner恢复-v1：新进程恢复、新采样及末次版本2

[恢复结果](../evidence/multi-gpu/official-runner-resume-v1.json)与[独立核对](../evidence/multi-gpu/official-runner-resume-v1.audit.json)均pass，20项独立核对通过。Runner真实`resume_dir`指向上述global_step_1，官方`init_workers()`恢复global step及Actor checkpoint；模型前向和新采样之前，两rank各11项、合计22项精确匹配：rank/version、model/trainable/buffer/topology、非空optimizer、scheduler、RNG、manager步数和真实非空Adam step。探针、support、完整RLinf/GR00T源码及训练配置与保存参考一致，未用人工重置优化器计数补齐恢复。

两个driver顺序独立启动；[进程观察](../evidence/multi-gpu/official-runner-lifecycle-v1.processes.jsonl)确认Actor0/1从PID201957/201959变为212460/212463。初次记录中的`matching_probe_process_pids`包含继承命令行的fork子进程，不把整列称为driver；按shell父进程筛选的`driver_candidates`及实际Actor日志身份分别保留。

恢复后官方Runner采集新的版本1数据，两环境各10模拟步、chunk5，4个chunk/20个有效模拟动作槽；每rank两个microbatches、一次GAE/PPO，非空Adam step与manager计数1→2，状态/metrics有限、参数和optimizer哈希变化。保存进程与恢复进程的裁剪前全局梯度范数分别2298.517578125/2330.434814453125；两rank相同。两种进程的每rank各5条事件与最终receive/GAE/training记录逐字段一致。

恢复进程在采样前同步版本1，训练后显式set global step并同步版本2，两次各907项完整状态/shape/dtype与Rollout精确相等；末态Runner step2、Actor manager/非空Adam step2、Actor/Rollout version2，Rollout持有第二次更新后的真实权重。耗时439.498548秒，包含初始化、DCP恢复、审计、采样和同步。

此节点证明**Actor训练状态恢复后用新episode继续官方Runner训练**。Env现场、Rollout RNG和队列未恢复，不能称为整个流程与不中断训练逐位等价。十步生命周期探针不代替完整240步回合边界恢复；周期性自动保存、恢复后再次保存、AMP/warmup或改变world size均未在此验证。恢复进程没有再保存checkpoint2；研究盘保留checkpoint1及旧Actor DCP，当前可用15107616768字节（约14.07GiB）。完整DCP保存规则需先确定空间及轮换方案。

实验driver退出后，协调者实测两张S4000各4MiB、0%利用率，`mthreads-gmi`无运行进程。本轮证据和Git同步完成后按用户新指令从平台关机，不继续启动长benchmark。

成功前的v1/v2都在`official_runner_init_workers`阶段失败，结果中的`synchronizations`为空；v3已通过初始化和版本0同步，再在Actor审计代码失败。这三轮均未完成Runner的GAE/PPO训练验收：

| Runner尝试 | 失败节点 | 原始记录 |
|---|---|---|
| v1 | Env初始化调用单成员`Worker.broadcast()`，到`CollectiveGroup.broadcast()`时查询`Worker.torch_platform.is_initialized()`；Torch-MUSA 1.3顶层未导出该函数，报`AttributeError` | [partial](../evidence/multi-gpu/official-two-rank-runner-v1.partial.json)、[failure](../evidence/multi-gpu/official-two-rank-runner-v1.failure.txt) |
| v2 | 增加真实初始化查询alias后越过该节点，单成员broadcast仍创建process group；Torch 2.2的`distributed_c10d`缺少scheduler所导入的`_register_process_group`，报`ImportError` | [partial](../evidence/multi-gpu/official-two-rank-runner-v2.partial.json)、[failure](../evidence/multi-gpu/official-two-rank-runner-v2.failure.txt) |
| v3 | 全部Worker初始化、Runner版本0同步与官方collector/dispatcher到Actor的数据接收已执行；`AuditActor.recv_rollout_trajectories()`错误要求`batch['actions']`，报`KeyError`，属于探针契约错误 | [partial](../evidence/multi-gpu/official-two-rank-runner-v3.partial.json)、[failure](../evidence/multi-gpu/official-two-rank-runner-v3.failure.txt) |

v1–v3探针SHA256均为`d9ed6ebffd951f5f9d892919e8d51289f07ffb4015a87c960d41359d0f9db06d`，配置及源码指纹以各partial JSON为准。Rollout随后被`ray.kill`的日志是失败清理，不能算作另一个显存或模型kernel故障。v1/v2揭示Runner初始化和scheduler对旧版软件接口的兼容问题；v3则是审计探针错误。这些失败没有推翻之前的双rank官方PPO更新、checkpoint恢复与权重同步验收。

v3的Runner初始同步记录包含9项检查，全部为true：两Actor参与、完整状态907项、selected 322项且唯一并覆盖、全状态SHA256/shape/dtype一致、Actor/Rollout版本均为0、非leader空manifest规则及两rank selected名称一致。原始manifest复核也一致。官方`recv_rollout_trajectories()`已返回，随后审计subclass读取`actions`才失败；这支持collector/dispatcher到Actor的数据链路已到达，尚没有完成该批数据的GAE、PPO或第二次Runner迭代。

官方Rollout传给Actor的`PolicyOutput`只含`prev_logprobs / prev_values / forward_inputs / versions`，动作另外发给Env；训练包会由collector加入reward和边界等字段，顶层`actions`不属于本路径的数据契约。新版[official_runner_probe.py](../model_probes/official_runner_probe.py)改为检查实际policy metadata有限性、递归检查`forward_inputs`中的张量，并对`forward_inputs + prev_logprobs`计算policy数据哈希，同时记录顶层keys和`actions_field_present`。修正已在v4通过；新探针与失败v3脚本不同，旧轮次按其原始哈希追溯。

当前`MUSAGPUManager.get_torch_platform()`仅在缺少顶层`is_initialized`时，alias到已有的`torch_musa.core._lazy_init.is_initialized`真实查询函数。[musa-initialized-alias-v1](../evidence/multi-gpu/musa-initialized-alias-v1.json)验证函数身份相同，分配MUSA张量前返回false、分配后返回true，张量值为1；顶层导入的`_initialized`拷贝boolean却始终为false，不能拿它判断设备初始化状态。真实查询函数SHA256为`633e2b62675d5001065e230d400aebd029c636f0aa8c4aec4575ba11f9bac8da`。

当前`Worker.broadcast()`保留已有groups与本worker成员检查，再核对src属于group；对只含本worker的单成员广播直接返回原对象，不创建通信组。`async_op=True`使用原`AsyncFuncWork`封装已完成结果，保留调用方的异步接口。此改动避免本次Env单成员广播无须使用的process-group私有API；尚未证明所有多成员scheduler collective与Torch 2.2兼容。

新增Ray通道后的四文件版本已冻结为[v2补丁](../evidence/multi-gpu/versions/two-rank-compat-v2.patch)与[v2锁](../evidence/multi-gpu/versions/two-rank-lock-v2.json)，补丁SHA256为`7d96b3341800c2b651c0aef5f2338fcef1d521997a2253a449d8dc9c5ab8f47c`。含初始化query alias和单成员broadcast适配的[当前六文件补丁](../patches/two-rank-fsdp-compat.patch)SHA256为`5da7ecf0ce3965cd154bdc1ffcf3e3816352772f85115040fe2a5658294d21f0`；[源码锁](../model_probes/two-rank-source-lock.json)记录六文件指纹，[重建v3](../evidence/multi-gpu/two-rank-patch-reconstruction-v3.json)确认在适配基线`82505b27d4d27f8e9af1d7e52524d524dc0f0d50`上叠加既有combined补丁后6/6文件重建精确一致。重建是CPU源码可复现检查；真实执行覆盖由Runner-v4的固定配置提供，独立API验证范围如下。

[singleton-broadcast-v2](../evidence/multi-gpu/singleton-broadcast-v2.json)与[rank0明细](../evidence/multi-gpu/singleton-broadcast-v2.rank0.json)均为pass，48个case全部通过：同步bool/None/object/CPU张量及容器、默认/显式src、整数rank group、异步wait/async_wait/then，以及非法groups/self/src的sync/async错误检查。真实NodePlacement Worker为单rank、`has_accelerator=True`，可见MUSA卡，但payload只用CPU；MUSA runtime query和default process group前后均为false。单成员case共0次通信组创建请求；两地址control恰好触发1次guarded创建请求，证明多成员仍进入原路径，但guard阻止真正建组，因此不证明多rank或GPU通信。探针总耗时17.358628秒，SHA256为`bab745a8553e813be10188fcca1381ce9b4d7b27ce96abb7d11fe53298a8981e`。

该API探针的[v1失败](../evidence/multi-gpu/singleton-broadcast-v1.failure.txt)是脚本错误要求NodePlacement Worker必须无GPU，在case执行前断言退出；不是框架broadcast失败。[原v1脚本](../evidence/multi-gpu/versions/singleton-broadcast-v1-probe.py)冻结保留；修正后的[singleton_broadcast_probe.py](../model_probes/singleton_broadcast_probe.py)与v2结果哈希一致。

## Ray 隔离设备下的通信问题与已验证配置

默认 RLinf Ray Worker 每 rank 只看一张物理卡：rank 0 的 `MUSA_VISIBLE_DEVICES=0`，rank 1 为 `1`；两者 `device_count=1`、逻辑设备 `musa:0`、`LOCAL_RANK=0`。历史日志物理 bus ID 分别为 `13000` / `16000`，没有把两个 Worker 误绑到同一张卡。实例重启后物理 bus 编号可能变化，不能把历史编号当作固定设备身份。

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

## 暂停前诊断与失败历史

[v14阶段日志和堆栈](../evidence/multi-gpu/fsdp-official-actor-update-v14.diagnostic.txt)及[partial JSON](../evidence/multi-gpu/fsdp-official-actor-update-v14.partial.json)表明：两个rank各完成四次`train_micro_batch`，每次返回后显式`torch.musa.synchronize()`通过；随后进入optimizer。四个微批次约38.09秒，不含初始化和后续梯度裁剪。[rank0 events](../evidence/multi-gpu/fsdp-official-actor-update-v14.rank0.events.jsonl)与[rank1 events](../evidence/multi-gpu/fsdp-official-actor-update-v14.rank1.events.jsonl)保留各边界。[冻结v14脚本](../evidence/multi-gpu/versions/update-v14-probe.py)保留当时诊断与验收逻辑，不能当作已通过的训练实现。

faulthandler显示两rank停留在`get_grad_norm_for_mixed_precision`，最终MCCL报`enqueue.cc:371 Musa failure 'wait operation timed out'`，rank1异常退出。rank0仍在逐梯度本地范数计算，rank1已到范数 all-reduce。不能沿用v12堆栈把首因定为AdamW，也不能据此定为某个归约算子不支持。rank0有227个非空梯度视图，rank1有98个；rank1最大的单个梯度是一维150,994,944元素。完整元数据：[rank0](../evidence/multi-gpu/fsdp-official-actor-update-v14.rank0.gradients.json)、[rank1](../evidence/multi-gpu/fsdp-official-actor-update-v14.rank1.gradients.json)。后续独立范数对照和v15已验证：用数值等价的快速本地范数路径可消除本次更新中的通信超时，详见下一节。

[补充标量通信](../evidence/multi-gpu/worker-collectives-scalars-v1.json)在相同隔离mesh和`MCCL_P2P_DISABLE=1`下每rank10/10通过，包括真正零维FP32 SUM/MAX。此前8项探针只覆盖16元素；补充结果支持标量通信可用，不能推广为所有训练通信顺序都正确。

[独立大向量norm](../evidence/multi-gpu/large-norm-direct-v1.json)覆盖1,048,577至150,994,944元素的FP32常量向量（offset=3），六种长度全通过；与解析值的最大相对误差约6.72e-8。因此“大向量norm本身必然失败”的假设未获支持。它没有复现真实梯度值、分片offset或多rank流依赖。

[分块norm诊断](../evidence/multi-gpu/large-norm-chunked-v1.json)仅是候选实现测试，用户暂停时被终止，顶层状态仍是预设fail/最后一行running；[暂停记录](../evidence/multi-gpu/two-gpu-stop.json)说明终止原因。已完成行可单独读取，但不能把整组算通过或部署进官方训练。末段小通信测试曾与该独立诊断重叠，不使用其时间建立性能结论。暂停时`musa_fsdp_norm.py`未注册到模型扩展；续跑采用的是后续独立验证的`legacy`模式，不能把分块候选视为当前成功实现。

最小Adam探针的[原始v1](../evidence/multi-gpu/adam-shards-v1.json)在空张量有限性断言失败，非空scalar/offset/large检查通过；空张量的optimizer调用本身已返回。当前检查已对空分片使用数学上的空集判定，修正未重跑。v13也因相同的空状态检查问题提前终止，不计为新backend失败。

## 续跑：范数兼容验证与 v15 真实分片更新

[norm-compat-v1](../evidence/multi-gpu/norm-compat-v1.json)的15行数值验证全部通过：9行小规模随机张量对照CPU FP64参考，覆盖`1e-8 / 1 / 1e8`三个尺度；6行大周期张量对照解析L2范数，覆盖1,048,577至150,994,944元素，各比较两次`torch.linalg.vector_norm`和两次`torch.norm`。最大相对误差`8.365470690223045e-8`；探针验收阈值为`2e-5`，用于FP32归约数值兼容检查，不是任务成功率或学习效果阈值。

六种大向量的同步计时中，`torch.linalg.vector_norm`为0.789946–7.360643秒，`torch.norm`为0.000543–0.002591秒。9,437,184元素一行中，两次linalg为4.841947 / 4.143022秒，两次legacy为0.002130 / 0.000572秒。它们在当前Torch-MUSA版本走不同的底层归约实现，因此“数值正确”和“耗时可用于多rank同步”需要分别判断。此记录是固定输入算子对照，不是模型吞吐 benchmark，也没有定位通用归约内核慢的最终原因。

已冻结的[norm-helper-v2.py](../evidence/multi-gpu/versions/norm-helper-v2.py)只在明确启用、MUSA非空梯度且L2范数时，把逐梯度及最终stacked范数改用`torch.norm`；保留FP32输入、空梯度回退、分布式SUM、全局开方和原裁剪逻辑。辅助文件SHA256为`352609cf12d53a712298e603868fd629147b07065c3f310a88192609bdb3d1be`。这是RLinf兼容适配，不需要修改或重建MUSA Toolkit或宿主驱动。

[v15完整结果](../evidence/multi-gpu/fsdp-official-actor-update-v15.json)使用`norm_mode=legacy`、`MCCL_P2P_DISABLE=1`和官方`sync_module_states=True`，并使用与v12相同的8步既有Trajectory fixture。两rank切片前完整GAE张量哈希均为`134481cb57c145664026f02986c1fd07585b47786a1e2d432822ff4a65cb74b4`；样本索引分别为0–3与4–7，全局8个唯一时间样本。两rank各完成四个micro batch，随后更新成功。

| v15验收项 | rank 0 | rank 1 |
|---|---|---|
| 本地trainable元素数 | 536,566,785 | 536,566,784 |
| 模型/trainable/Adam哈希 | 三者均变化 | 三者均变化 |
| 拓扑保持、模型/优化器/metrics有限 | 全通过 | 全通过 |
| 非空Adam step / manager更新计数 | `0→1` / `0→1` | `0→1` / `0→1` |
| 预建空Adam状态数 | 95，step仍为0 | 224，step仍为0 |
| 全局裁剪前梯度范数 | 816.5087890625 | 816.5087890625 |
| optimizer enter→exit同步边界 | 0.204220秒 | 0.157862秒 |
| allocator peak allocated / reserved | 23.461606 / 27.613281 GiB | 23.454576 / 27.615234 GiB |

边界计时见[rank0 events](../evidence/multi-gpu/fsdp-official-actor-update-v15.rank0.events.jsonl)与[rank1 events](../evidence/multi-gpu/fsdp-official-actor-update-v15.rank1.events.jsonl)，不含optimizer前额外的梯度元数据审计。两个rank聚合metrics相同，ratio为`1.0003015995025635`；输入来自单卡先前版本3，与当前新初始化Actor版本并不相同，不能要求ratio严格为1或把它当作fresh on-policy PPO。探针最后显式设置Actor版本为1，尚未将此分片Actor版本同步到Rollout。

[冻结v15脚本](../evidence/multi-gpu/versions/update-v15-probe.py)SHA256为`53fc49d751889937edde338d5bddb691b7ece3044259f2f45753de76799cae35`。完整探针耗时153.432930秒，包含初始化、哈希和审计，不能当作纯训练耗时。这次成功证明固定S4000软件栈可执行该大模型的两rankFULL_SHARD参数更新；没有证明单卡/两卡数值等价、长期稳定性或任务学习。

## GR00T 检查点保存、恢复与续跑

协调者已实际运行两rank`--phase train`保存和新的`--phase recover`恢复入口，checkpoint-v4与recovery-v1均已通过。成功前的失败集中在Torch 2.2的检查点表示和Python设备接口，不能倒推v15的实际参数更新失败。各轮保留原始失败与精确探针版本：

| 保存尝试 | 实际到达位置与失败 | 原始证据与冻结脚本 |
|---|---|---|
| GR00T checkpoint-v1 | `_allgather_orig_param_states`中rank0的`torch.cuda.memory_summary()`报`KeyError`，rank1的`torch.cuda.synchronize()`报未编译CUDA支持 | [partial](../evidence/multi-gpu/fsdp-official-actor-checkpoint-v1.partial.json)、[failure](../evidence/multi-gpu/fsdp-official-actor-checkpoint-v1.failure.txt)、[probe](../evidence/multi-gpu/versions/checkpoint-v1-probe.py) |
| GR00T checkpoint-v2 | 进程内改用device handle后越过优化器汇聚，在DCP的DTensor保存规划器`_local_shard_size_on_dim`失败；待分片首维长度1小于world size 2。尚未定位具体FQN，不能称为Adam step导致 | [partial](../evidence/multi-gpu/fsdp-official-actor-checkpoint-v2.partial.json)、[failure](../evidence/multi-gpu/fsdp-official-actor-checkpoint-v2.failure.txt)、[probe](../evidence/multi-gpu/versions/checkpoint-v2-probe.py) |
| GR00T checkpoint-v3 | 显式`torch22_state_dict_backend=sharded_tensor`，FSDP构造改用原一维mesh的同一个process group；随后日志读取空本地分片的`.device`，旧实现回退`torch.cuda.current_device()`并报`_cuda_getDevice`不存在 | [partial](../evidence/multi-gpu/fsdp-official-actor-checkpoint-v3.partial.json)、[failure](../evidence/multi-gpu/fsdp-official-actor-checkpoint-v3.failure.txt)、[probe](../evidence/multi-gpu/versions/checkpoint-v3-probe.py) |
| small-two-rank-dcp-v1 | 缩小为`Linear(4,8) → Tanh → Linear(8,1)`、torchrun两rank、`cpu:gloo,musa:mccl`复合PG；`get_state_dict`读取空本地ShardedTensor的`.is_meta`，旧实现调用`local_tensor()`后报`Only single local shard is supported` | [failure](../evidence/multi-gpu/small-two-rank-dcp-v1.failure.txt)、[probe](../evidence/multi-gpu/versions/small-dcp-v1-probe.py) |
| small-two-rank-dcp-v2 | 空分片`.device/.is_meta`适配后继续进入CPU offload；空本地ShardedTensor的`.to(cpu)`仍用`torch.cuda.current_device()`推断当前设备，报`_cuda_getDevice`不存在 | [failure](../evidence/multi-gpu/small-two-rank-dcp-v2.failure.txt)；与small-v1使用相同冻结探针，区别是进程内适配helper版本 |

checkpoint-v2和v3冻结探针SHA256分别为`b35b6521da74178477ddae0923dce83d157cf3a1a9166067a7395e87a17f1b50`、`6f7c7cfd4f1041029a1356e3055a6e4f659a05982a5bd96c2d3713d3d69db2a0`，均与对应partial JSON一致。小DCP探针SHA256为`cb89c33b3e4fe3a5b47c8c29422ac82ae32c825367cc670187548a88953ff09f`，后续small-v3保持同一探针，仅使用更新的适配helper。

目前的显式、进程内适配包括：将已审计函数的1处CUDA memory_summary和3处synchronize改用`fsdp_state._device_handle`，让零本地分片的ShardedTensor `.device/.is_meta`依据实际CPU/MUSA storage metadata返回结果，并在其`.to()`中以已解析的`self.device`获取当前设备，保留非空分片及其他设备的原处理。最新冻结实现为[optim-device-v4.py](../evidence/multi-gpu/versions/optim-device-v4.py)，SHA256 `6375d7519bb4757d58678f4a37a812ad9b17653f025df12f107f8e5bf8c80789`，与checkpoint-v4结果一致。FSDP1仅在Torch 2.2实验模式、一维mesh下切换到同PG的标准ShardedTensor检查点表示；manager的mesh、梯度归约组和分片训练逻辑保留，未修改底层DTensor分片算法。

[small-two-rank-dcp-v3 rank0](../evidence/multi-gpu/small-two-rank-dcp-v3.rank0.json)与[rank1](../evidence/multi-gpu/small-two-rank-dcp-v3.rank1.json)均为pass，四项检查`restored_exact / continuation_exact / loss_exact / sharded_tensor`全部通过。覆盖本地模型、非空Adam状态、参数组、新对象恢复后下一次更新和loss。小探针是在同一进程创建新的model/optimizer对象，没有测试新driver/Worker、GR00T、scheduler或RNG恢复。

[完整GR00T checkpoint-v4](../evidence/multi-gpu/fsdp-official-actor-checkpoint-v4.json)为pass：两rank完成第一步更新并保存版本1训练状态，非空Adam step均为1；保存后的未中断分支再次使用同一fixture，完成第二步更新，非空Adam step均为2、Actor版本均为2，模型和优化器状态有限。保存后snapshot含本地参数、buffers、active optimizer、scheduler与各rank RNG，供新进程对照。耗时296.645622秒，包括两次更新、DCP保存、初始化和哈希审计，不是保存吞吐或纯训练耗时。v4沿用已冻结的checkpoint-v3探针，通过三个显式兼容选项：`--norm-mode legacy --optim-device-mode device_handle --state-dict-backend sharded_tensor`。

协调者现场统计此次checkpoint包含两个`.distcp`数据文件，分别8,095,977,568与8,096,422,220字节，`.metadata`为1,340,195字节，两份manager runtime各1,128字节；合计16,193,742,239字节，约15.081598 GiB。这些文件保存在远端研究盘，本地仓库存储探针、配置与紧凑证据。

[完整GR00T recovery-v1](../evidence/multi-gpu/fsdp-official-actor-recovery-v1.json)为pass。协调者启动新的driver/Workers，以checkpoint-v4保存的版本1状态及其未中断续跑记录为参考，执行恢复与下一次更新。两次记录的配置、轨迹、探针、norm/helper哈希及逐文件源码指纹完全一致；恢复JSON没有单独记录checkpoint/reference路径，执行关系由协调者的命令记录与以下状态对照确认。

| 新进程验收项 | rank 0 | rank 1 |
|---|---|---|
| 保存状态与恢复后状态对照 | 11/11精确一致 | 11/11精确一致 |
| 未中断与恢复后下一次更新对照 | 11/11精确一致 | 11/11精确一致 |
| 非空Adam step / manager更新计数 / Actor版本 | `1→2` / `1→2` / `1→2` | `1→2` / `1→2` / `1→2` |
| 模型、优化器状态有限 | 全通过 | 全通过 |

每组11项为本地模型、trainable参数、拓扑、buffers的SHA256，buffer数量，active optimizer、scheduler、各rank RNG的SHA256，manager更新计数，非空Adam step与Actor版本。active optimizer对照包含所有`requires_grad`且本地`numel>0`的参数状态及完整`param_groups`。两个rank各有2个buffer，恢复及续跑哈希一致；下一次更新的training metrics也与未中断分支精确一致。直接复核原始snapshot与metrics确认了JSON中的比较结论。恢复探针耗时218.498649秒，包含初始化、加载、哈希审计与更新，不是加载吞吐。

FSDP标准恢复实际删除了warmup预建的空本地分片Adam状态：

| Adam状态差异 | rank 0 | rank 1 |
|---|---|---|
| 保存前全部状态条目 → 恢复后 | `322→227` | `322→98` |
| 保存前空状态条目 → 恢复后 | `95→0` | `224→0` |
| 保存前原始step集合 → 恢复后 | `[0,1]→[1]` | `[0,1]→[1]` |
| 下一次更新的原始step集合：未中断 / 恢复分支 | `[0,2]` / `[2]` | `[0,2]` / `[2]` |

因此全部optimizer状态的原始哈希和条目数不同；非空Adam状态与参数组的active哈希在恢复及续跑两个边界均完全一致。此差异已记录，没有补造空状态来强求原始哈希相同。

默认Torch包、默认Python环境、`/usr/local/musa`和宿主驱动均未修改。**完整GR00T两rank DCP保存、新driver/Worker恢复，以及下一次更新与未中断分支的等价性已通过**。检查点对照范围限于相同两rank拓扑、既有8步fixture、冻结BF16 Eagle与FP32 action/value head；没有验证world size变化、长期稳定性或任务学习。分片Actor→单Rollout的版本0/1同步、新轨迹官方PPO更新及官方Runner两轮闭环已分别通过；较长稳定性与学习评估仍需继续。

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

GR00T 分片 v12 的历史fixture入口，仅用于复核当时的初始化/GAE和失败位置；当前成功的v15入口在下一段：

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

GR00T分片v15成功入口。复现前部署本仓库的`versions/update-v15-probe.py`及`versions/norm-helper-v2.py`；后者置于`route2/model_probes/musa_fsdp_norm.py`，与结果记录的helper哈希一致：

```bash
env -u MCCL_SHM_DISABLE RLINF_MUSA_FSDP_INDEPENDENT_INIT=0 \
  MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN \
  "$lab_python" "$lab_root/shared/update-v15-probe.py" \
  --rlinf-source "$lab_root/route2/RLinf-official-actor" \
  --gr00t-source "$lab_root/route2/Isaac-GR00T-official-actor" \
  --model-path "$lab_root/route2/weights/Spatial-73f710e" \
  --trajectory "$lab_root/route2/results/official-actor-continuation-v1.pt" \
  --phase update --diagnostic --norm-mode legacy \
  --output "$lab_result_dir/full-shard-v15.json"
```

GR00T两rank checkpoint-v4保存、未中断续跑与recovery-v1新进程恢复入口。部署上文冻结的`versions/checkpoint-v3-probe.py`到`shared/checkpoint-v3-probe.py`、`versions/optim-device-v4.py`到`route2/model_probes/musa_fsdp_optim_device.py`，并保留已验证的norm-helper-v2。源码须包括显式同PG ShardedTensor模式；具体修改与指纹见v4及recovery-v1 JSON。保存前checkpoint路径必须不存在；train结束后以新的driver执行recover，使用同一份保存状态及参考JSON：

```bash
lab_checkpoint_dir="$lab_root/route2/checkpoints/two-gpu-dcp-repro"

env -u MCCL_SHM_DISABLE RLINF_MUSA_FSDP_INDEPENDENT_INIT=0 \
  MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN \
  "$lab_python" "$lab_root/shared/checkpoint-v3-probe.py" \
  --rlinf-source "$lab_root/route2/RLinf-official-actor" \
  --gr00t-source "$lab_root/route2/Isaac-GR00T-official-actor" \
  --model-path "$lab_root/route2/weights/Spatial-73f710e" \
  --trajectory "$lab_root/route2/results/official-actor-continuation-v1.pt" \
  --phase train --diagnostic \
  --checkpoint "$lab_checkpoint_dir" \
  --norm-mode legacy --optim-device-mode device_handle \
  --state-dict-backend sharded_tensor \
  --output "$lab_result_dir/full-shard-checkpoint-train.json"

env -u MCCL_SHM_DISABLE RLINF_MUSA_FSDP_INDEPENDENT_INIT=0 \
  MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN \
  "$lab_python" "$lab_root/shared/checkpoint-v3-probe.py" \
  --rlinf-source "$lab_root/route2/RLinf-official-actor" \
  --gr00t-source "$lab_root/route2/Isaac-GR00T-official-actor" \
  --model-path "$lab_root/route2/weights/Spatial-73f710e" \
  --trajectory "$lab_root/route2/results/official-actor-continuation-v1.pt" \
  --phase recover --diagnostic \
  --checkpoint "$lab_checkpoint_dir" \
  --reference "$lab_result_dir/full-shard-checkpoint-train.json" \
  --norm-mode legacy --optim-device-mode device_handle \
  --state-dict-backend sharded_tensor \
  --output "$lab_result_dir/full-shard-checkpoint-recover.json"
```

分片Actor→单Rollout初始同步入口。源码使用上文冻结v1增量补丁对应的三文件指纹，探针及四份support文件须与sync-v1结果匹配。此命令只复现版本0同步：

```bash
env -u MCCL_SHM_DISABLE RLINF_MUSA_FSDP_INDEPENDENT_INIT=0 \
  MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN \
  "$lab_python" "$lab_root/route2/model_probes/two_rank_rollout_probe.py" \
  --rlinf-source "$lab_root/route2/RLinf-official-actor" \
  --gr00t-source "$lab_root/route2/Isaac-GR00T-official-actor" \
  --model-path "$lab_root/route2/weights/Spatial-73f710e" \
  --fixture "$lab_root/route2/results/official-actor-continuation-v1.pt" \
  --phase sync --diagnostic --norm-mode legacy \
  --output "$lab_result_dir/two-rank-sync.json"
```

同一探针的独立train-phase入口，fixture仅用于固定输入输出对照，训练轨迹由当前版本0策略新采集。源码与support指纹使用train-sync-v1记录；新轨迹与结果路径必须不存在：

```bash
env -u MCCL_SHM_DISABLE RLINF_MUSA_FSDP_INDEPENDENT_INIT=0 \
  MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN \
  "$lab_python" "$lab_root/route2/model_probes/two_rank_rollout_probe.py" \
  --rlinf-source "$lab_root/route2/RLinf-official-actor" \
  --gr00t-source "$lab_root/route2/Isaac-GR00T-official-actor" \
  --model-path "$lab_root/route2/weights/Spatial-73f710e" \
  --fixture "$lab_root/route2/results/official-actor-continuation-v1.pt" \
  --trajectory-output "$lab_result_dir/two-rank-fresh.pt" \
  --phase train --diagnostic --norm-mode legacy \
  --output "$lab_result_dir/two-rank-train-sync.json"
```

官方Runner-v4实际参数对应的复现入口。先将冻结的`versions/official-runner-v4-probe.py`部署到`route2/model_probes/official_runner_probe.py`，保留结果JSON中五份support文件的准确版本，生产源码使用已核对的六文件补丁；输出使用新的repro路径：

```bash
env -u MCCL_SHM_DISABLE RLINF_MUSA_FSDP_INDEPENDENT_INIT=0 \
  MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN \
  "$lab_python" "$lab_root/route2/model_probes/official_runner_probe.py" \
  --rlinf-source "$lab_root/route2/RLinf-official-actor" \
  --gr00t-source "$lab_root/route2/Isaac-GR00T-official-actor" \
  --model-path "$lab_root/route2/weights/Spatial-73f710e" \
  --iterations 2 --steps-per-env 4 --diagnostic \
  --output "$lab_result_dir/two-rank-runner-v4.json"
```

独立单成员broadcast API的CPU payload验证入口，使用singleton-v2记录的脚本哈希；Worker仍处于真实GPU可见NodePlacement环境，命令不进行分布式/GPU通信：

```bash
"$lab_python" "$lab_root/route2/model_probes/singleton_broadcast_probe.py" \
  --source "$lab_root/route2/RLinf-official-actor" \
  --output "$lab_result_dir/singleton-broadcast-v2.json"
```

小探针、分卡单rank更新、大模型分片PPO更新、完整DCP保存与恢复续跑、分片Actor→Rollout版本0/1同步、新轨迹更新、官方Runner三轮80步及chunk5/240步两轮、十任务初始策略pilot、Runner保存/新进程新轨迹续训和末次同步已分别通过。后续重点为官方LR短验收、完整500-state基线及有配对评估的学习实验；本轮暂不继续加卡。进入顺序见[学习计划](embodied-learning-entry.md)。
