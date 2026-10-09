# 路线二成果仓库移交

2026-10-05，远端已设为用户创建的 [prophetricker/rlinf-musa-lab](https://github.com/prophetricker/rlinf-musa-lab)。独立本地成果仓库位于 `artifacts/route2-github/`；研究根仓库保存全部路线，成果仓库只导出路线二与复现所需的共用 runner。GitHub CLI 已完成 `prophetricker` 账号认证；已推送并核对远端 `main` 与本地 commit 一致。首个学习节点发布提交为 [`6a99f21`](https://github.com/prophetricker/rlinf-musa-lab/commit/6a99f2128424c19be22f97b1badab103017395e2)，对应研究提交 `c22ba8b`。

## 固定材料

| 材料 | 来源 |
|---|---|
| 上游 | RLinf `c70606f08cdca259b8dec03d4430926b5b8fac9d` |
| Worker 学习适配基线 | `3e7329c629293f1b0f9c331b5bb24eccc3e3f142`，tree `b78ed3694bcd1d69afd887e6ce11b780d7528ecf` |
| FSDP1 实验增量 | `f9b74d95f97ad4234311580ab32c7b586e1dc7af`，tree `1a4eef5dd1f4984652168ebecce1d3cc194ec598` |
| checkpoint 实验增量 | `c46875bb1515716b487c0cde5e6cf8a2277a483b`，tree `a6c18c65206804ce621bf78472c5ed0b3b37387d` |
| 系统栈 | Driver 2.7.0 / MUSA 3.1.0 / Torch 2.2.0 / Torch-MUSA 1.3.0 |
| 新学习结果 | Pendulum 与 HalfCheetah 各三种子，全部 102,400 transitions 与独立 20 局 test；见 [学习报告](route2-learning.md) |
| 训练后端 | 真实 RLinf FSDP1 单卡更新；local_shard 与 Torch 2.2 DCP 的 MUSA/MCCL 恢复/继续更新通过；独立进程恢复同样通过；尚无官方 Actor、多卡或offload |
| 模型合约 | 真实 Diffusers Attention 与固定双层 GR00T DiT 在 MUSA 完成 12 rows；严格 gate 保留解析零方向失败，semantic gate 全部通过；新增真实Qwen3/Siglip小型冻结骨干FP32/BF16 CPU8/8及CPU/MUSA12/12通过；未加载完整 GR00T |

## 导出与检查

研究根仓库的 `scripts/export_route2.py` 按清单生成新目录，拒绝覆盖已有输出。共用 `routes/route1/mujoco_ppo_worker.py` 原样导出为 `probes/mujoco_ppo_worker.py`。下载的 audit 源码只导出 URL/commit/hash manifest。

导出不包含其他路线结果、上游完整 Git 历史、环境、权重、checkpoint 或认证材料。`EXPORT_MANIFEST.json` 保存各文件 SHA256、字节数及研究来源 commit。既有成果 Git 的 `.git` 保留，刷新时只同步清单中的材料；不强推。

上一节点离线恢复已验证 Worker 基线、FSDP1 与 checkpoint 实验三个源码 tree 完全一致；重复恢复保留既有 worktree。该节点导出验证覆盖 111 个材料文件与 manifest，约7 MB；64 个相对 Markdown 链接、26 个 Python AST、3 个 Bash 文件、共用 runner 逐字对照和凭据模式扫描均通过。新增 checkpoint 证据覆盖 local_shard 与 DCP，两种格式都在 MUSA/MCCL 上精确恢复并继续更新；新增 action-interface 证据覆盖 36 rows，MUSA 12 rows 完成前向/VJP，strict 与 semantic 两种 gate 均原样记录。历史学习语义/评估 CPU 测试为10/10，该节点未重复训练。FSDP1 从基线加实验补丁重建，保持基线与实验边界。

提交使用本地显式身份 `Codex <codex@localhost>`，不修改全局 Git 身份。登录令牌由 GitHub CLI 管理，不写入研究记录。

## 跨进程与骨干组件节点（2026-10-05）

本节点新增四份独立checkpoint save/restore JSONL：local_shard/DCP都在MUSA/MCCL、FP32、world size1、NO_SHARD下通过真实load；两类进程身份不同，save driver已退出。完整模型/Adam/scheduler、四类RNG、下一段随机样本和未中断对照的下一步全部exact，模型与optimizer误差0。保留旧同进程probe/两份生产补丁，新增source/command/run manifests；没有扩大为多rank、offload或官方Actor结果。

骨干组件的v1 CPU8/8通过、整组CPU/MUSA11/12：Qwen3 BF16只读blocked-probability诊断触发Torch-MUSA masked_select类型限制。v2将该诊断输入转FP32并增加binding/registry identity验收，CPU8/8及整组12/12均通过。两个版本及全部原始JSON保存；不改变attention计算/VJP/数值容差，也不重复先前PPO训练。

新增独立Transformers4.51.3环境no-deps安装，系统Torch/Torch-MUSA路径与source hashes核对；原驱动/Toolkit/默认Python不修改。公开Spatial配置与bundled Eagle实际构造路径单独审计，采用Qwen3Model/SiglipVisionModel，而不从仓库名称推断Siglip2实现。synthetic Linear仅为独立梯度基础能力检查；checkpoint projector是Identity，未进行完整Eagle/RADIO、预训练权重、Flow-SDE或LIBERO验证。

最终导出在根Git提交后生成，准确research commit和文件哈希由EXPORT_MANIFEST.json关联，scripts/verify_route2_export.py核验字节、JSON/JSONL、Python/Bash语法与相对链接。远端同步采用已有成果仓库正常commit/push；二进制checkpoint/reference仅留本地忽略目录备份与服务器研究磁盘，不进入GitHub。

## 官方 Actor 与 Rollout 初始同步（2026-10-06）

本地研究提交 `030490e` 已导出并推送为成果仓库 `a8e86f5f19800e12200662d3f51ab21ad2b88223`，远端main已核对。真实官方Actor初始化、322个Adam状态、版本0独立Rollout的907个完整状态哈希与固定输入输出精确一致，初始PPO ratio=1。采用显式单rank CPU Bucket/Ray同步，完整Runner和多rank尚未验证。组合生产补丁、源码恢复入口、17文件重建核验和12个CPU transport guard检查一起交付。

连续PPO/恢复测试正独立推进；本提交不提前声明通过。新源码与精确命令见 `routes/route2/model_probes/official-actor-README.md`，原始初始化、同步、失败修复与CPU检查见 `routes/route2/evidence/official-actor/`。

## 官方持续PPO、恢复与完整episode（2026-10-06）

连续3轮真实LIBERO8步GAE/PPO及版本0–3完整权重/固定输出对照通过；完整DCP与runtime状态保存后，新进程恢复及同批下一步全部精确一致。初始策略task0两个trial完整episode均成功79/69步，平均预测约0.80/0.82秒；不是PPO学习收益或完整suite benchmark。Actor reserved33.1 GiB，整卡采样约35.4 GiB。研究盘50 GiB保存约16GB checkpoint后剩9.4GiB；两卡前应扩盘与验证MCCL/分片。原始证据、命令、精确probe版本与checkpoint校验和都在official-actor evidence目录；大checkpoint仍保存在服务器隔离目录。

最终研究提交 `fed34e0c111a0de6b65a1a6dd2792a52e7683070` 已导出并推送为成果仓库 [`72ce62830fbb676ed79120b3f738cf696208d3b2`](https://github.com/prophetricker/rlinf-musa-lab/commit/72ce62830fbb676ed79120b3f738cf696208d3b2)。导出核验309个文件、65个Python AST、5个Bash语法和150个相对Markdown链接通过；`git ls-remote origin refs/heads/main` 与本地成果仓库HEAD一致，成果仓库干净。该段是推送后的本地移交记录，不修改已发布导出的来源manifest。

推送后SSH只读检查确认实例仍运行，驱动2.7.0，S4000显存4 MiB且无GPU进程；未关机或重启。下一阶段按 `routes/route2/planning/multi-gpu-entry.md` 验收同节点两卡通信、独立Actor/Rollout部署与真正分片；本节点没有验证完整Runner、多rank或PPO学习收益。


## 两卡分片恢复与初始同步节点（2026-10-07）

研究提交 `37941bacbe4a269f86ff4396dc63604b7c549634` 已导出为成果仓库 [`97962fadac536b51b1fe33f18a1645f04b1ff718`](https://github.com/prophetricker/rlinf-musa-lab/commit/97962fadac536b51b1fe33f18a1645f04b1ff718)，正常push后已核对远端main。该发布节点包含两rank完整GR00T FULL_SHARD官方PPO fixture更新、完整DCP保存/新driver与Worker恢复/同批下一步精确一致，以及版本0的907项完整Actor/Rollout状态与固定输出精确一致。导出388个文件、94个Python AST、5个Bash语法、216个相对链接通过。源commit与哈希见EXPORT_MANIFEST。

随后独立fresh8步LIBERO双卡官方PPO及版本0/1同步也已通过；完整Runner仍在后续节点验收。这段后续记录不扩大前述发布提交的实验范围。当前研究盘100GiB，保留约15.08GiB的两rank checkpoint；重启实例后实际PCI总线为12:00.0/13:00.0。宿主驱动、Toolkit及默认Python未修改，实例保持运行。

## 完整官方Runner节点（2026-10-07）

研究提交 `0e467cb1d445b4bf435d57e311df04c6b7d30101` 已导出为成果仓库 [`cb320492f65e6b3f74f9cc84c75522ed3ebe0dc2`](https://github.com/prophetricker/rlinf-musa-lab/commit/cb320492f65e6b3f74f9cc84c75522ed3ebe0dc2)，正常push后核对远端main一致。Runner-v4真正运行官方`EmbodiedRunner.run()`，2环境×4步×2轮共16 transitions；两个FULL_SHARD Actor各完成两次官方GAE/PPO更新，非空Adam step为2；版本0/1的907项完整状态与Rollout一致。末次PPO更新未再同步，奖励均为0，不证明学习收益。导出409个文件、98个Python AST、5个Bash语法、240个相对链接通过。

当前6文件增量补丁及源码锁覆盖Runner Ray数据通道、Torch-MUSA真实初始化查询与单成员广播适配；完整状态同步仍使用CPU Bucket/Ray，两rank训练归约使用MCCL。Runner-v1/v2初始化失败、v3探针误读训练包`actions`失败均保留；v4通过不抹去历史失败。Runner-v5已启动3轮、2环境各80步的稳定性验证，后续结果另行记录，不能扩大此前发布提交的验收范围。

## 三轮80步Runner稳定性节点（2026-10-07）

研究提交`93cebe950e59bf65bd4add8b3ed8edde1165d6cd`已导出并正常push为成果仓库[`9b28edabb614fb7ed93a331f1a7164a9e8a53c15`](https://github.com/prophetricker/rlinf-musa-lab/commit/9b28edabb614fb7ed93a331f1a7164a9e8a53c15)，远端main与本地HEAD已核对。Runner-v5完成3轮×2环境×80步，480数据槽、454 mask有效动作，两个Actor各3次Adam更新，版本0/1/2各907项完整状态同步一致。真实termination与post-terminal mask后训练通过，尚无240步truncation、完整suite评估或学习收益；末次PPO更新未额外同步。原始ratio受全mask微批次的普通均值影响，原值与解释都保留。

该发布节点包含新chunk5与十任务评估代码、5项CPU eval通道/精确补丁重建、10case真实loss-mask CPU检查及后续学习计划；新入口当时仍未验收GPU结果。导出428文件、104 Python AST、6 Bash语法和261相对链接通过。实例保持运行，chunk5/240步Runner和初始十任务pilot已由协调者顺序启动；这段运行中状态不扩大本次发布的通过范围。

## 2026-10-07完整horizon发布节点

研究提交`85629dcf44634814615b448cfe5903c2830c436e`已导出并push为成果仓库[`a5ada2a89151be00ce15eb7ecab4345ecbe8490a`](https://github.com/prophetricker/rlinf-musa-lab/commit/a5ada2a89151be00ce15eb7ecab4345ecbe8490a)，main与本地HEAD核对一致。chunk5/240两轮通过：192动作块、960有效模拟动作槽、每rank2次更新、四条240步truncation与下一轮reset、版本0/1的907项完整同步。该入口仍没有末次更新后的额外同步。独立结果16项通过，两个rank各14条事件与最终JSON逐字段一致。

pilot-v1完成十个任务trial0、官方success_once4/10，但最终读取WorkerGroup列表时探针报错，原失败及冻结字节保留；v2只修单Rollout结果解包并独立重跑。新Runner生命周期入口已静态审查并部署，SHA为`7dcd689df311e0b96c9ef9c63b67d3c398debea6b70d35189489beffa8cfe194`；待pilot通过后串行执行保存一次、新进程恢复/新轨迹训练、最终同步。研究盘当前29.16GiB可用，不删除旧checkpoint，恢复端暂不再保存第二份。

本次导出445文件、108 Python AST、7 Bash与275相对链接全部通过；新生命周期与eval-v2当时还不是GPU已通过项。实例继续运行，无关机或重启。

## 2026-10-07本轮关机前移交

pilot-v2与Runner生命周期现已完成：十个唯一task/trial0、success_once4/10、success_at_end3/10，20项独立核对通过；每task实际50个init states，pilot只覆盖2%，不是全500-state benchmark。原pilot-v1的汇总失败保留。

Runner save-v1调用官方保存接口后保存global_step_1/actor（15.081598GiB、5文件），保存前后状态相等、19项独立核对通过。resume-v1由新driver/Workers恢复，两rank共22项模型/非空optimizer/scheduler/RNG/步数/版本精确匹配，再采新版本1轨迹完成下一次官方GAE/PPO，累计Adam step2，版本1/2各907状态同步一致，20项独立核对通过。冻结入口SHA7dcd689df311e0b96c9ef9c63b67d3c398debea6b70d35189489beffa8cfe194；恢复端step2只同步审计，没有保存第二份。

旧Actor DCP与新Runner checkpoint1都保留，研究盘可用15107616768字节（约14.07GiB）。恢复范围是Actor训练状态后重新采样，新Env/Rollout现场未恢复，十步入口不覆盖周期自动保存或完整回合边界恢复。后续优先官方LR短验收、完整500-state初始基线和配对学习评估，下一次学习从固定预训练权重/全新optimizer开始，不能把本轮1e-8工程更新当学习成果。

本轮GPU全部退出，两卡各4MiB、0%，无运行GPU进程。用户明确追加实验完成后关机；协调者完成最终记录及GitHub同步后从平台关闭唯一两卡S4000实例，不用guest poweroff代替平台操作。实际关机状态另记。

## 2026-10-07最终同步与平台关机确认

研究提交`4237de1e65162d1810732b9e327f016897025d92`已导出并正常push为成果仓库[`0758ebd2745611fac61ff0d63640d6e988319cff`](https://github.com/prophetricker/rlinf-musa-lab/commit/0758ebd2745611fac61ff0d63640d6e988319cff)，`git ls-remote origin refs/heads/main`与本地HEAD精确一致。该节点包含chunk5/240步、十任务pilot-v2和Runner保存/新进程恢复/新轨迹更新/最终同步的原始证据及独立审计。导出461文件、108 Python AST、7 Bash语法和288相对Markdown链接通过。

2026-10-07约02:21（Asia/Shanghai），协调者通过已登录的AutoDL实例列表，对唯一实例`77de47b334-d32e95e0`（MTT S4000×2）执行平台“关机”并确认。页面先显示“关机中”，刷新后明确显示“已关机”、可用“开机”按钮；实例仍保留，没有释放或销毁。关机前已完成服务器`sync`，全部GPU实验已退出。紧凑确认记录见[平台关机证据](../routes/route2/evidence/multi-gpu/platform-shutdown-lifecycle-v1.json)。本段及确认记录是关机后的本地补充，将作为单独收尾提交同步，无需再次开机。

## 2026-10-07正式学习率短验收

实例重新运行后，使用同一套两卡适配、`chunk=5`、每环境240个模拟步、2环境和`update_epoch=1`，从固定Spatial预训练权重与全新optimizer开始，连续3轮切换到官方 actor/value 学习率`5e-6 / 1e-4`。每轮两rank各收到48个policy chunks，共288个chunk决策和1,440个模拟动作槽；每rank完成3次真实GAE/PPO/Adam更新，版本0–3的完整状态同步通过。三轮参数、Adam状态、梯度和loss统计均有限，独立21项审计通过。

第三轮采样出现少量非零奖励（每rank聚合奖励约`0.00217`），但这只是短预算中的行为信号，不能作为学习收益结论。每轮真实更新约6.4分钟，3轮总耗时约31.9分钟；显存峰值仍低于两卡容量。GPU原始结果、两个rank事件和独立审计见[正式LR短验收](../routes/route2/evidence/multi-gpu/official-spatial-lr-check-v1.json)、[rank0事件](../routes/route2/evidence/multi-gpu/official-spatial-lr-check-v1.actor.rank0.events.jsonl)、[rank1事件](../routes/route2/evidence/multi-gpu/official-spatial-lr-check-v1.actor.rank1.events.jsonl)及[审计](../routes/route2/evidence/multi-gpu/official-spatial-lr-check-v1.audit.json)。服务器端收尾命令曾引用不存在的审计脚本路径，未影响训练结果；结果下载后用研究仓库审计脚本重新核对通过。

完整500-state初始策略评测已在同一开机周期串行启动，使用固定预训练权重、10个task、每task实际50个trial、`success_once`口径，不更新模型；最终成功率与覆盖审计待评测完成后补录。

## 2026-10-08完整500-state初始策略基线

官方 `EmbodiedEvalRunner` 评测已完成并通过独立20项审计。固定Spatial预训练权重、2个Env lane、chunks=5、denoising=4、`auto_reset=True`、每回合240步；全程只启动Env/Rollout，不创建Actor、不更新参数。实际完成500个唯一`(task_id, trial_id)`，每个10个task各50个state，120,000个模拟动作槽，12,000次两lane策略调用，覆盖集合和worker seen-trials完全一致。

初始策略结果：`success_once=234/500=46.8%`，`success_at_end=223/500=44.6%`，平均reward`0.0050519`，平均return`0.468`，所有episode长度240。按task分别为`18/50, 37/50, 21/50, 27/50, 1/50, 39/50, 21/50, 33/50, 20/50, 17/50`（task0–9）。评测耗时`16315.48s`，约4.53小时；whole-run policy seed为1234，当前仍是整次评测固定seed，不是per-episode reseed。这个结果是本适配配置的初始策略基线，不直接声称复现官方CUDA成绩，也不代表MUSA适配已产生学习收益。

原始结果和事件见[500-state结果](../routes/route2/evidence/multi-gpu/official-spatial-full-baseline-v1.json)、[Env事件](../routes/route2/evidence/multi-gpu/official-spatial-full-baseline-v1.env.events.jsonl)、[Rollout事件](../routes/route2/evidence/multi-gpu/official-spatial-full-baseline-v1.rollout.events.jsonl)及[独立审计](../routes/route2/evidence/multi-gpu/official-spatial-full-baseline-v1.audit.json)。下一步应从同一固定权重和全新optimizer开始长期PPO，使用相同评测协议比较配对的训练前后成功率；不要把短LR验收中的少量非零奖励当作学习结果。

## 2026-10-08发布核对与用户关机报告

完整基线和正式学习率证据已正常push为成果仓库提交[`81febe5485d204cc80e61b97fec6d8b89affa614`](https://github.com/prophetricker/rlinf-musa-lab/commit/81febe5485d204cc80e61b97fec6d8b89affa614)，收尾时本地成果HEAD与远端main精确一致，对应研究提交`37229195ea28914c60da5d19f4d936412da157ff`。此前协调者的本轮平台关机操作因Mac锁屏及浏览器连接不可用而未完成；不能引用10月7日的旧关机证据作为本轮确认。

10月8日用户明确回复“我已经关了”。本轮据此记为用户报告实例关机，实际操作时间未知，协调者没有独立读取平台状态；记录见[用户关机报告](../routes/route2/evidence/multi-gpu/platform-shutdown-full-baseline-user-report-v1.json)。本次仅整理本地文档和发布，不再连接或启动GPU作业。

下一次学习需先补齐三项准备：短验收的训练配置固定为task0/trial0，需验证训练侧跨任务/初始状态调度；长实验需验证周期保存与完整回合边界的Actor状态恢复后重新采样；评测继续冻结500-state覆盖、batch、顺序和整次policy seed。当前基线没有per-episode reseed，若改成新的随机性协议，需另立配对基线。拟议50轮探索预算约24,000模拟动作槽；按本轮含审计的耗时约32分钟/3轮粗估为约9小时，另加完整评测约4.5小时，实际时长应由新配方重新测量。该预算用于探索，不承诺学习收益。

## 2026-10-08七卡扩展验证

用户启动了 7 张可见 S4000（GPU 0–6）；本轮保留 GPU 6 作为独立 Rollout，GPU 0–5 作为 6 个 FULL_SHARD Actor rank，Env 使用 GPU 0。由于 RLinf 轨迹路由要求训练环境数可被 Actor world size 整除，短验收使用 6 个 LIBERO 环境、`chunk=5`、每环境80步、`global_batch_size=96`，两轮共192个 action chunk、960个模拟动作槽；6个Actor均完成两次GAE/PPO/Adam更新和权重同步，主结果为 `pass`。

随后用同一布局进行完整 horizon 验证：6 个环境、每环境240步、`chunk=5`、`global_batch_size=288`、`update_epoch=1`、两轮。每轮收集 288 个 chunk 样本，每个 Actor 每轮一次全局 PPO/Adam 更新；总计 576 个 chunk、2,880 个模拟动作槽。6 个 rank 均记录两次 trajectory receive、GAE 和 training complete，14项独立审计全部通过；两轮 MCCL 梯度归约后的 `actor/grad_norm` 分别在六个 rank 间精确一致（25.657573699951172、25.989818572998047），参数、优化器和指标保持有限。完整运行耗时约 40.9 分钟，GPU0–5显存约20–21GiB，未发生 OOM 或通信错误。

原始结果、每 rank 事件和独立审计见 `routes/route2/evidence/multi-gpu/seven-gpu-full240/`；短验收见 `seven-gpu-smoke80/`。本节点证明 6-rank Actor + 独立 Rollout 的 RLinf/MUSA 功能闭环和完整 horizon 稳定性，不证明吞吐线性扩展，也不是学习收益或 benchmark 成绩。使用版本仍为 Driver 2.7.0、MUSA Toolkit 3.1.0、Torch 2.2.0、Torch-MUSA 1.3.0、MCCL 2.11.4。实验结果已下载并准备发布，随后清理训练进程并关闭 S4000 实例。

## 2026-10-08七卡发布与关机收尾

研究提交 `a14da50` 已导出并推送为成果仓库提交 [`bdf0be189c8d19f5817e31fe046190f284d824b7`](https://github.com/prophetricker/rlinf-musa-lab/commit/bdf0be189c8d19f5817e31fe046190f284d824b7)，本地成果 HEAD 与远端 main 已再次核对一致。导出487文件、109个Python AST、7个shell语法和301个Markdown相对链接通过。

关机前七卡均为0%利用率、约4MiB显存，无训练、Ray或GPU进程，并已执行 `sync`。平台关机提交后曾显示“关机中”；2026-10-08约15:12（Asia/Shanghai），通过本机Edge的AutoDL实例列表独立确认同一七卡实例显示“已关机”且“开机”按钮可用。实例保留，未释放；实际完成关机的精确时间未知。见[关机确认](../routes/route2/evidence/multi-gpu/platform-shutdown-seven-gpu-v1.json)。

本轮到此收尾。下一阶段先验证训练侧跨任务/初始状态调度，以及七卡布局下周期保存和完整回合边界恢复后重新采样，再从固定预训练权重与全新optimizer进行有预算的PPO学习实验。与冻结的500-state初始策略基线按同一协议比较；若评测batch、顺序或随机性协议发生变化，先建立对应的新基线。

## 2026-10-08七卡跨任务与恢复入口准备

本地新增七卡生命周期入口 [`seven-gpu-next-validation`](../routes/route2/planning/seven-gpu-next-validation.md) 和 [`run_route2_seven_gpu_lifecycle.sh`](../scripts/run_route2_seven_gpu_lifecycle.sh)。它把原两卡 Runner 保存/恢复探针参数化为任意连续 Actor world size，并保留默认两卡 task0 配置不变。七卡配方为 Actor 0–5、独立 Rollout 6、Env 0、6个环境、`chunk=5`、全局 batch 288、每个 Runner step 240个模拟动作；训练 reset 使用全 suite 有序池而非固定 task0，并要求初始环境报告至少覆盖两个 task id。

当前入口先连续训练两轮，通过官方 `save_interval=2` 路径保存 `global_step_2`，再用新进程恢复并在采样前逐 rank 核对模型、Adam、scheduler、RNG、计数和版本，继续一个完整 horizon 后显式同步到版本3。新增审计观察实际 bootstrap/horizon 的 task、trial、描述和每 lane 步数，并对 checkpoint 文件逐一记录大小及 SHA-256，恢复前核对文件集合和内容。本轮正在七卡实例上执行，最终结果待下文补录。它仍是样本路由和生命周期验收，不是学习收益实验。

七卡生命周期结果见[本轮原始证据](../routes/route2/evidence/multi-gpu/seven-gpu-lifecycle-v2/)。两轮保存进程完成 step1/step2 共576个策略chunk和2880个模拟动作槽，task reset 批次从 `[4,8,2,6,7,1]` 推进为 `[7,4,5,3,3,4]`，每轮六个lane覆盖至少四个task且各运行满240步。六rank每轮各收到48个chunk、完成有限的GAE/PPO/Adam更新，rank间梯度范数完全一致；step2 checkpoint约15.09 GiB、13个文件哈希齐全，保存前后Actor模型、optimizer、scheduler、RNG、计数和版本均未变化。保存进程因运行期间磁盘临时峰值不足而由协调者终止，因此原始partial保持`status=fail`，不能声称该driver正常退出；已完成的保存事件由29项独立审计核对通过，见`official-seven-gpu-save-v2.completed-events.audit.json`。

未重跑前两轮，另启恢复driver验证此step2 checkpoint。恢复前全部文件大小和SHA-256通过，六个Actor的66项状态完全一致；从Adam step2/version2继续第三轮、收集48 chunks/rank、做PPO更新至step3并完整同步Rollout成功。32项独立恢复审计通过，见`official-seven-gpu-resume-v3.audit-verified.json`。全程共864个策略chunk、4320个模拟动作槽；冻结策略初始500-state基线仍为46.8% success_once，以上三轮工程验收不能解读成成功率提升或正式训练收益。

七卡保存会同时使用约15.1 GiB最终checkpoint和约16 GiB量级临时空间。实测远端盘峰值曾从35 GiB可用降至约1 GiB；现将多rank保存前门槛提高为40 GiB，并保留现有checkpoint。`official-seven-gpu-resume-v3.json`及所有事件、step2保存partial、冻结保存/恢复探针、source lock和独立审计均随此节点一同归档。

结果、日志和事件下载后，所有训练/Ray worker已退出。协调者在AutoDL列表对`77de47b334-d32e95e0`（S4000×7）确认关机；页面先显示“关机中”，随后显示“已关机”且“开机”按钮可用。实例未释放，step2 checkpoint仍保存在实例持久研究盘。平台状态证据见[本轮关机确认](../routes/route2/evidence/multi-gpu/seven-gpu-lifecycle-v2/platform-shutdown.json)。
## 八卡第一轮验证入口（2026-10-10）

用户现有八张 S4000 后，新增 [`eight-gpu-next-validation`](../routes/route2/planning/eight-gpu-next-validation.md)、[`run_route2_eight_gpu_validation.sh`](../scripts/run_route2_eight_gpu_validation.sh) 和独立审计 [`audit_route2_eight_gpu_result.py`](../scripts/audit_route2_eight_gpu_result.py)。第一轮暂不写 checkpoint：GPU 0–6 为 7 个 FULL_SHARD Actor，GPU 7 为独立 Rollout，Env placement 为 GPU 0；7 个 LIBERO 环境、`chunk=5`、每环境240步、全局 batch 336、两轮共672个 policy chunks和3360个 simulator action slots，训练 reset 要求至少覆盖两个 task id。

入口保留 `MCCL_P2P_DISABLE=1`、隔离 route2 环境和已验证的 RLinf/MUSA 适配变量。独立审计会重新核对八卡 placement、每个 rank 的 trajectory/GAE/PPO 事件、跨轮 reset、有限参数与指标以及七个 rank 的归约梯度范数一致性；这一轮只证明八卡执行闭环，不证明吞吐线性扩展或学习收益。七卡 checkpoint 不复制、不覆盖；八卡保存/恢复待第一轮通过且研究盘至少有40 GiB可用后另行安排。

本地静态验证：既有生命周期审计 `8/8` 通过；新脚本 Bash/Python 编译、`git diff --check` 和非法 batch 参数拒绝通过。开机后在远端研究目录执行 `scripts/run_route2_eight_gpu_validation.sh`，结果写入 `route2/results/official-eight-gpu-validation-v1.*`。

## 八卡第一轮实测收尾（2026-10-10）

用户启动八卡实例后，按上述配方完成两轮真实官方 `EmbodiedRunner.run()`。主结果和事件已下载到 [`eight-gpu-validation`](../routes/route2/evidence/multi-gpu/eight-gpu-validation/)；远端脚本审计和本地重新审计均为 `pass`。7 个 Actor rank 每轮各完成一次 trajectory、GAE 和 PPO/Adam 更新，7-way MCCL 归约后的梯度范数逐轮精确一致；两轮各 336 个全局 policy chunks，合计 672 chunks、3360 simulator action slots。任务批次从 `[4,8,2,6,7,1,7]` 变为 `[4,5,3,3,4,7,0]`，7 条 lane 每轮均完整运行240步，跨任务 reset 和 horizon 边界检查通过。

运行时仍是 Driver 2.7.0、MUSA Toolkit 3.1.0、Torch 2.2.0、Torch-MUSA 1.3.0、MCCL 2.11.4；本轮没有写 checkpoint，研究盘约49 GiB可用。运行约40.7分钟，不能据此宣称八卡吞吐提升；两轮奖励均为零，因此不能作为学习收益或 LIBERO benchmark 结果。所有训练/Ray进程已退出，下一步才考虑在空间门槛满足时单独验证八卡 checkpoint 保存/恢复。
