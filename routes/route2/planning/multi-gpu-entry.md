# 多卡实验进入条件

日期：2026-10-06。前三个单卡进入条件已通过：官方Actor初始化；独立Rollout版本0–3完整权重与固定输入对照；真实LIBERO三轮GAE/PPO及fresh-process完整Actor状态恢复、同批下一步精确一致。完整episode基线另行记录。

用户已提供两张同节点48 GiB S4000，研究盘已扩至100 GiB，核验时约45 GiB可用。已通过torchrun双rank MCCL、小模型FULL_SHARD更新，以及Actor卡0/Rollout卡1的真实LIBERO更新与版本0→1同步。RLinf隔离Worker的默认P2P/IPC通信失败；作业设置`MCCL_P2P_DISABLE=1`后，8项小通信每rank全部通过，实际使用shared-memory transport。官方GR00T默认同步初始化和GAE也已通过，完整分片更新仍在排查。证据和准确边界见 [两卡记录](two-gpu-results.md)。

## 是否需要加卡

现在一张48 GiB S4000足够接口与小规模连续PPO验证。Actor peak allocated为27.02 GiB、peak reserved为33.10 GiB；每5秒整卡采样峰值35.43 GiB。外部采样可能漏掉短峰值，进程allocator统计不包含其他进程和驱动占用，两类数字不能直接相减当作精确上限。

继续放大batch、增加并行环境或开展分片训练时，建议先增加到**两张同型号、同节点S4000**。先核对两卡可见性、PCIe拓扑、MCCL版本和通信，再决定是否需要更多卡。当前的单rank MCCL不证明两卡已通过。

研究盘原为50 GiB，其中单个完整GR00T+Adam checkpoint约16 GB。现已扩至100 GiB；仍需设置checkpoint保留数量，并在保存前检查空间。大模型、数据和checkpoint无需推入Git。

## 两卡按什么顺序验收

| 阶段 | 布置 | 要确认的事实 |
|---|---|---|
| 设备与MCCL | 两个小型rank，各占一卡 | SUM/AVG all-reduce、broadcast、barrier与非空张量收发数值正确，有限超时；记录实际backend和设备绑定 |
| Actor/Rollout分开 | Actor卡0，Rollout卡1；两组各1rank | 907个完整状态哈希与固定输入一致；版本至少0/1/2；真实GAE/PPO；可先保留现有CPU/Ray Bucket传输，不声称加速 |
| FSDP1分片 | Actor两个rank，先小MLP，再GR00T | 真正FULL_SHARD，固定全局batch与单卡/CPU对照；梯度归约、clip norm、Adam变化、各卡显存、DCP保存恢复及同批下一步 |
| 多rank同步 | 分片Actor→Rollout | 完整参数重建和选择键覆盖；版本、dtype/shape、哈希与输出一致；同步时不出现死锁、OOM或失配 |
| 完整Runner | 官方EnvWorker/collector/Actor/Rollout编排 | 跨episode边界、auto-reset/终局bootstrap、队列积压、训练进度和checkpoint正确 |
| 学习收益 | 固定完整任务预算和多seed | 初始与训练后独立任务成功率、动作/奖励/超时、吞吐与显存；与硬件/环境变化分开记录 |

其中现有`ray_cpu`明确只允许Actor world size=1且Rollout world size=1。它可以用于两组分别放在两卡的第一阶段；不能用于两个Actor rank或两个Rollout rank。多rank时须先解决旧Torch2.2缺少RLinf通信helper所需的`_register_process_group`等具体接口，并验证真实MCCL。增加卡数不会自动解决这些API差异。

两卡跑两个Actor rank时，Rollout可暂时与一张Actor卡分阶段共用，通过CPU offload控制显存；更宽裕的独立布局是两卡Actor加一卡Rollout。应由实测分片显存决定是否增至三卡。冻结BF16 Eagle与FP32 action/value head需继续分别包装，不能把整个FSDP模型统一转BF16来省显存。

## 当前性能数字的用途

8步rollout加一次最终bootstrap约13.4–13.7秒，包含官方预测、Ray RPC、CPU环境step与offload边界。每轮记录的训练段约32秒含一次完整状态快照哈希，整轮还含其他哈希/sync审计；不能把它写成纯optimizer或最终训练吞吐。完整episode的新基线单独保留逐步预测耗时。

三轮都为零reward，学习率1e-8仅用于验证真实参数更新。进入长期训练前，需采用合理学习率、充分rollout预算、独立评估和多seed；当前loss下降不证明任务学会了。S5000或Lambda-Sim到位后，重复相同进入条件并另记硬件、驱动、Toolkit和环境合约。
