# 七卡下一阶段验收

七卡 `6 Actor + 1 Rollout` 的完整 horizon 已通过，但此前六个训练环境都固定在 `task_id=0, trial_id=0`。因此下一轮先验证训练样本调度和 checkpoint 生命周期，再开始有预算的 PPO 学习。

## 实验入口

[`scripts/run_route2_seven_gpu_lifecycle.sh`](../../../scripts/run_route2_seven_gpu_lifecycle.sh) 使用已经通过的隔离环境和源码，固定以下配置：

- Actor GPU `0-5`，独立 Rollout GPU `6`，Env placement GPU `0`；
- 6 个 LIBERO 环境、`chunk=5`、每环境 240 步、全局 batch `288`；
- `specific_reset_id=null`，使用有序的全 suite reset pool，初始环境至少出现 2 个不同 task id；
- 第一个进程完成 2 个完整 Runner step，并通过官方 `save_interval=2` 路径保存 `global_step_2` checkpoint；
- 新进程从 `global_step_2` 恢复，在任何采样前核对模型、Adam、scheduler、RNG、计数和版本，再采集第三个完整 horizon 并显式同步到版本 3。

默认脚本会写入新的 `official-seven-gpu-save-v1` 和 `official-seven-gpu-resume-v1` 结果；已有路径会拒绝覆盖。`ROUTE2_SEVEN_STEPS` 可用于先做更短的 10 步 smoke，但正式验收应保持 240。

## 通过条件

1. 初始环境报告包含多个 task id，且 Actor 每个 rank 收到一条完整的 48 chunk 轨迹；
2. 6 个 Actor 的 GAE、PPO/Adam、MCCL 梯度归约和版本 0/1/2 同步通过；
3. 保存后新进程的 6 个 Actor 状态逐 rank 精确恢复，恢复后完整 horizon 能继续更新并同步到版本 3；
4. 结果记录 checkpoint 文件大小、显存、每个 task/trial 身份、有效 action mask 和 truncation/reset 边界。

实际采样的 bootstrap 和完成事件必须交替，每轮每 lane 运行240步，任务描述与环境身份匹配；第1、2轮 reset 批次应不同。独立审计使用 [`audit_route2_runner_result.py`](../../../scripts/audit_route2_runner_result.py)，七卡调用时附加 `--min-distinct-task-ids 2`；本地回归入口为 [`test_route2_runner_audit.py`](../../../scripts/test_route2_runner_audit.py)。

本配方的周期保存恰好与第一个进程的结束步重合，验证官方 `_maybe_eval_and_checkpoint`/`_save_checkpoint` 路径，不覆盖“非末轮保存后继续同一进程训练”。新 Env/Rollout 进程重启 reset 池，未恢复模拟器现场或其 RNG，不能声称不中断训练等价，也不能声称恢复后必然使用之前未见的 state。两轮同进程内的 reset 推进和新进程的 Actor 精确恢复分别检查。为保留现有 checkpoint，本轮只新增一个完整保存点。

## 实测记录（2026-10-08）

完整原始文件及事件见 [`seven-gpu-lifecycle-v2`](../evidence/multi-gpu/seven-gpu-lifecycle-v2/)。第一driver完成两轮step1/step2训练和save_interval调用，实际六条lane的task批次不同：第一轮task `[4,8,2,6,7,1]`，第二轮task `[7,4,5,3,3,4]`；不同trial保留。每轮每条lane恰好240模拟步，六个Actor每轮接收288个全局policy chunks并完成PPO更新，MCCL归约后的六rank梯度范数一致。

Step2 checkpoint共13个文件、16,202,414,772字节（15.09 GiB）。Actor保存前后的模型/Adam/scheduler/RNG/计数/版本摘要一致。保存事件独立审计29项通过。协调者在保存阶段磁盘暂时仅余约1 GiB时终止driver；最终文件树及摘要完整，但原driver未走完末尾审计与退出，因此partial仍诚实标为fail。用该点另启恢复driver，保存文件在加载前重哈希通过，6 rank × 11项（模型、参数、buffers、拓扑、optimizer、scheduler、RNG、计数和版本）逐项一致。恢复后从step2/version2重新启动Env/Rollout、采样一个240步horizon、Actor Adam到step3、MCCL同步到version3。独立恢复审计32项全部通过。

实测峰值证实全rank checkpoint约15.1 GiB，保存时还需要大块临时空间。后续多rank保存要求至少40 GiB空闲并保留既有checkpoint。重置池只在各driver内部推进；恢复后新Env从自己的有序池起点抽样，可能与保存进程的第一批身份重复。此次没有恢复模拟器现场/RNG，也没有证明被中断driver的末尾清理行为。三次小步PPO是适配/续训执行证据，不是学习收益；正式收益仍需固定的500-state配对评测。

恢复汇总：[`official-seven-gpu-resume-v3.json`](../evidence/multi-gpu/seven-gpu-lifecycle-v2/official-seven-gpu-resume-v3.json)，独立审计 [`official-seven-gpu-resume-v3.audit-verified.json`](../evidence/multi-gpu/seven-gpu-lifecycle-v2/official-seven-gpu-resume-v3.audit-verified.json)。保存driver原始partial和单独审计也在同目录，不能把partial的fail状态覆盖为pass。

这一步仍然是适配和生命周期验收，不作为学习收益结论。通过后才使用官方学习率和固定 500-state 评测协议进行配对训练前后比较。
