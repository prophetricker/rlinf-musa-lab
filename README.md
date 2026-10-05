# RLinf on MTT S4000：路线二适配研究

在单卡 MTT S4000、Driver 2.7.0、MUSA 3.1.0、Torch 2.2.0、Torch-MUSA 1.3.0 上，探索固定版本的当前 RLinf 主线适配。首轮已验证 MuJoCo HalfCheetah-v5 → RLinf Worker/Ray CPU Channel → MUSA PPO 参数更新与同批次 checkpoint 恢复。

**当前是功能验证成果，尚未证明策略学会任务。** 这套闭环使用自定义普通 `nn.Module` Worker 扩展；官方 EmbodiedRunner、EmbodiedFSDPActor、GR00T N1.5、LIBERO 和 Lambda-Sim 尚未通过全链路验证。上游要求 `torch>=2.5.0`，本项目没有建立官方 Torch 2.2 支持。

## 首轮结果（2026-10-04）

| 验证 | 结果与边界 |
|---|---|
| 真正的 RLinf Cluster / Worker | 识别 1 张 MTT S4000、15 CPUs |
| 上游 MLPPolicy / GAE / PPO loss | MUSA 前向、反向、AdamW 更新；CPU/MUSA GAE 对照通过 |
| HalfCheetah-v5 小闭环 | 4 env × 32 horizon × 2 iterations，256 transitions、8 次 optimizer step、130 条 Channel 消息 |
| checkpoint 同批恢复后更新 | 参数与 AdamW 状态最大绝对误差均为 0；未保存 MuJoCo simulator state |
| Ray CPU Channel 回归 | 8/8；涵盖 CPU Tensor、sync/async/nowait/batch、重连与 consumer dispatcher |
| 原生 Torch FSDP1 / MCCL | 单进程 `NO_SHARD` 更新通过；未使用 RLinf FSDP 包，未验证多卡分片、offload 或 FSDP checkpoint |
| 学习收益 | 未建立；唯一 held-out episode 回报 `0.177940 → 0.154520` |

rollout 与 update 共驻 ActorWorker，独立 Actor/Rollout 权重同步仍待验证。显式 `transport="ray"` 传 CPU 对象，不构成默认 collective/MCCL 多进程张量通信通过的证据。

## 代码与证据

- [详细首轮报告](routes/route2/README.md)、[小闭环指标](routes/route2/evidence/mujoco-loop.json)、[Channel 证据](routes/route2/evidence/channel-tests.json)。
- [适配补丁](routes/route2/patches/current-minimal-compat.patch)：legacy DTensor 导入回退、显式 Torch-MUSA 设备探测回退、Ray CPU Channel 公共 API、回归测试及中英文文档。
- [源码锁](locks/sources.json)、[实测环境](locks/environment.md)、[包清单](locks/route2-environment.txt)。
- [复现说明](REPRODUCE.md)、[后续节点](ROADMAP.md)、[来源说明](PROVENANCE.md)。

上游固定为 [`c70606f08cdca259b8dec03d4430926b5b8fac9d`](https://github.com/RLinf/RLinf/tree/c70606f08cdca259b8dec03d4430926b5b8fac9d)，本地适配提交为 `3e7329c629293f1b0f9c331b5bb24eccc3e3f142`。这里保存补丁与探针；本地提交尚未上传，GitHub 上游没有这个适配提交。恢复脚本从固定上游应用补丁，生成同样的源码内容，不承诺重建相同 commit ID。

研究主线为路线二；路线一保留回归参考，路线四保留 backend 移植备选。后续目标是 `RLinf + GR00T N1.5 + LIBERO-Spatial + PPO`，取得 Lambda-Sim SDK 后并行验证环境接口。
