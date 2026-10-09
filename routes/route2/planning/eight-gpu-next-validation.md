# 八卡第一轮验证

八张 S4000 的第一轮只验证 RLinf 官方 Runner 在 MUSA 上的八卡拓扑和完整 horizon，不写新的 checkpoint。这样可以把八卡通信、轨迹路由、跨任务 reset 和 PPO 更新与已有七卡保存磁盘风险分开。

## 配方

- Actor GPU `0-6`：7 个 `FULL_SHARD` rank；Rollout GPU `7`；Env placement GPU `0`。
- 7 个 LIBERO 训练环境，每个 Actor rank 对应一个环境。
- `chunk=5`，每环境 240 个模拟步，48 个 policy decisions。
- 每轮全局 batch `336`，两轮共 `672` 个 action chunks、`3360` 个 simulator action slots。
- `specific_reset_id=null`，使用有序 suite reset pool，初始和每轮 reset 至少覆盖 2 个 task id。
- 保存参数保持 `save_interval=-1`、`save_final=false`；已有七卡 checkpoint 不覆盖，也不复制。

入口是 [`run_route2_eight_gpu_validation.sh`](../../../scripts/run_route2_eight_gpu_validation.sh)，远端研究目录默认仍为 `/root/autodl-tmp/s4000-research`。结果和事件保存到 `route2/results/official-eight-gpu-validation-v1.*`，随后由 [`audit_route2_eight_gpu_result.py`](../../../scripts/audit_route2_eight_gpu_result.py) 独立审计。

## 通过条件

独立审计必须同时确认：8 卡 placement、7 个 Actor rank、每轮每 rank 两次 trajectory/GAE/PPO、每轮 336 个全局 chunk、每环境 240 步、task/trial 在 horizon 内稳定且轮次间 reset 推进、MCCL 归约后的 7 个 grad norm 每轮一致、参数/Adam/loss 全部有限，并且没有请求 checkpoint。

这一轮仍然是适配和执行闭环证据，不是吞吐线性扩展、500-state benchmark 或学习收益。通过后再单独规划八卡保存/恢复；多 rank checkpoint 只有在研究盘至少 40 GiB 可用时才允许尝试。

## 实测结果（2026-10-10）

八卡实例实际运行通过。主结果、7 个 Actor 事件、Env 事件、日志和远端审计见 [`eight-gpu-validation`](../evidence/multi-gpu/eight-gpu-validation/)，主运行状态为 `pass`，独立审计为 `pass`，本地重新审计也为 `pass`。

- 7 个 Actor 每轮各收到 48 个 policy chunks；两轮全局分别为 336、336，共 672 个 chunks 和 3360 个 simulator action slots。
- 两轮每个 rank 都完成 trajectory receive、GAE 和 PPO/Adam，最终 Runner step 为2；7 个 rank 的归约 `actor/grad_norm` 逐轮精确一致：`12.937379837036133`、`16.034832000732422`。
- 两轮 bootstrap 的 task id 分别为 `[4,8,2,6,7,1,7]`、`[4,5,3,3,4,7,0]`，每条 lane 都完成 240 个模拟步，task/trial 在 horizon 内保持一致。
- 运行时为 Torch `2.2.0`、Torch-MUSA `1.3.0+81caf0a`；没有请求或写入 checkpoint。完整运行约 40.7 分钟，不能由此推出吞吐线性扩展。

这次训练奖励两轮均为0，因此结果只证明八卡拓扑、MCCL/FSDP 同步、跨任务 reset、轨迹路由和 PPO 执行没有失败，不是学习效果或 LIBERO benchmark 成绩。下一步可以在保持该拓扑的前提下单独评估八卡 checkpoint 保存/恢复；开始前需要再次确认至少 40 GiB 可用空间。
