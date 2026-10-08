# 七卡下一阶段验收

七卡 `6 Actor + 1 Rollout` 的完整 horizon 已通过，但此前六个训练环境都固定在 `task_id=0, trial_id=0`。因此下一轮先验证训练样本调度和 checkpoint 生命周期，再开始有预算的 PPO 学习。

## 实验入口

[`scripts/run_route2_seven_gpu_lifecycle.sh`](../../../scripts/run_route2_seven_gpu_lifecycle.sh) 使用已经通过的隔离环境和源码，固定以下配置：

- Actor GPU `0-5`，独立 Rollout GPU `6`，Env placement GPU `0`；
- 6 个 LIBERO 环境、`chunk=5`、每环境 240 步、全局 batch `288`；
- `specific_reset_id=null`，使用有序的全 suite reset pool，初始环境至少出现 2 个不同 task id；
- 第一个进程完成 1 个完整 Runner step 后保存官方 Runner checkpoint；
- 新进程从 `global_step_1` 恢复，在任何采样前核对模型、Adam、scheduler、RNG、计数和版本，再采集第二个完整 horizon 并显式同步到版本 2。

默认脚本会写入新的 `official-seven-gpu-save-v1` 和 `official-seven-gpu-resume-v1` 结果；已有路径会拒绝覆盖。`ROUTE2_SEVEN_STEPS` 可用于先做更短的 10 步 smoke，但正式验收应保持 240。

## 通过条件

1. 初始环境报告包含多个 task id，且 Actor 每个 rank 收到一条完整的 48 chunk 轨迹；
2. 6 个 Actor 的 GAE、PPO/Adam、MCCL 梯度归约和版本 0/1 同步通过；
3. 保存后新进程的 6 个 Actor 状态逐 rank 精确恢复，恢复后完整 horizon 能继续更新并同步到版本 2；
4. 结果记录 checkpoint 文件大小、显存、每个 task/trial 身份、有效 action mask 和 truncation/reset 边界。

这一步仍然是适配和生命周期验收，不作为学习收益结论。通过后才使用官方学习率和固定 500-state 评测协议进行配对训练前后比较。
