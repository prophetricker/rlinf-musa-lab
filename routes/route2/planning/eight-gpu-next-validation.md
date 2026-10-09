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
