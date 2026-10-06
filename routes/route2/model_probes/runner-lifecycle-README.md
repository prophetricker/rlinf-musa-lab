# 官方 Runner 保存、恢复与最终权重同步

2026-10-07：保存-v1与新进程恢复-v1已通过，独立核对分别19/20项，恢复前22项状态精确一致；末次版本2完整同步也通过。此前Actor DCP新进程恢复已经通过，但不能用它代替此处实际Runner入口的结果。chunk5/240步两轮与初始策略十任务pilot-v2已通过，同次开机周期完成本验证。

[official_training_probe.py](official_training_probe.py)复用当前官方Runner、collector、dispatcher、GAE/PPO和已锁定的DCP代码，新增可配置LR/seed、Runner恢复、最终同步和保存。原[Runner探针](official_runner_probe.py)及冻结脚本不变。此新入口尚不是正式学习收益评估工具：训练仍固定task0/trial0、epoch1/microbatch1，每轮一个global batch，eval关闭。

最小验证分成两个独立driver进程：

1. 两环境各10模拟步、chunk5，执行一轮继承的Runner更新。将Actor/Env/Rollout global step设为1，完整同步907项状态，调用官方`runner._save_checkpoint()`保存`global_step_1/actor`。
2. 新进程设置真实`resume_dir`，官方`init_workers()`从目录恢复Runner global step及Actor训练状态；在任何采样或forward之前逐rank比较模型、trainable、buffers、拓扑、非空分片optimizer、scheduler、RNG、计数及版本共11项。再采集新轨迹、完成下一轮更新，显式同步到版本2。

研究盘当前约29GiB可用、单个完整DCP约15GiB。首版保留一份新的Runner checkpoint，恢复进程不再保存第二份；现有Actor checkpoint保留。`--save-final`要求至少18GiB可用。恢复后的再次保存、完整240步回合边界恢复、长学习周期自动保存仍需后续验证。

```bash
cd /root/autodl-tmp/s4000-research
bash shared/run_route2_runner_lifecycle.sh
```

结果默认写`route2/results/official-runner-save-v1.json`和`official-runner-resume-v1.json`及各自事件/log。脚本拒绝覆盖已有结果，第一进程失败时不启动恢复。checkpoint路径由实际成功的保存结果读取。CLI的`--iterations`表示本进程新增轮数，恢复起点N时上限设为N+iterations。

末次同步是显式调用，在所有Actor训练完成之后执行；Rollout的`set_global_step()`仅设置调度计数，真实权重version由继承的sync apply更新。仅给`save_checkpoint(..., step=N)`传入N不能修正Actor版本，保存前必须先设置并同步。原Runner-v4/v5/chunk5入口没有额外末尾同步，不扩大其历史通过范围。

恢复只包含Actor训练状态：Env/Rollout重新初始化并从新episode采样，未保存模拟器现场、Rollout RNG或队列。恢复后新轨迹不同是预期行为，不能声称整个流程与不中断训练逐位等价。AMP关闭、critic warmup0、epoch1与单batch的范围保留；版本N与累计Adam step=N只适用于本配置。

该入口可显式设置`--lr 5e-6 --value-lr 1e-4`，但首次生命周期验收保留1e-8。官方LR、完整评估协议、trial划分和学习收益另按[学习计划](../planning/embodied-learning-entry.md)推进。

实际证据：[保存结果](../evidence/multi-gpu/official-runner-save-v1.json)、[恢复结果](../evidence/multi-gpu/official-runner-resume-v1.json)、[恢复独立核对](../evidence/multi-gpu/official-runner-resume-v1.audit.json)。每rank累计Adam step达到2，版本0/1（保存进程）和1/2（恢复进程）各907项完整状态与Rollout一致。实际checkpoint为global_step_1，约15.081598GiB；恢复后的step2仅同步与审计，未再次保存。
