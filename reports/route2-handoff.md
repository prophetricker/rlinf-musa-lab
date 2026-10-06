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
