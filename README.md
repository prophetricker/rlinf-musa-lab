# RLinf on MTT S4000：路线二适配研究

在单卡 MTT S4000、Driver 2.7.0、MUSA 3.1.0、Torch 2.2.0、Torch-MUSA 1.3.0 上，逐步适配固定版本的当前 RLinf 主线。Mac 管理开发与记录，GPU 实验在隔离 Linux 环境执行。

2026-10-05 已完成 Pendulum 三种子 PPO 学习基线：每种子 102,400 transitions，独立 20 局 test 平均回报全部提升。模型是普通 Gaussian MLP，使用真实 RLinf Worker/Cluster/Ray CPU Channel、GAE 和 PPO loss，rollout/update 共驻。完整官方 EmbodiedRunner/Actor、GR00T N1.5、LIBERO 和 Lambda-Sim 尚未验证。

| 训练种子 | 初始 test 回报 | 训练后 test 回报 | 提升 |
|---|---:|---:|---:|
| 7 | -1191.12 | -286.31 | +904.81 |
| 17 | -1180.24 | -435.37 | +744.87 |
| 27 | -1186.04 | -606.41 | +579.63 |

满足本项目预先声明的初步工程目标，三个种子不足以建立统计显著性。详见 [学习报告](reports/route2-learning.md)、[原始结果](routes/route2/evidence/learning/pendulum-summary.json) 与 [配方](routes/route2/learning/README.md)。

![学习曲线与独立测试](routes/route2/evidence/learning/pendulum-curves.png)

## 其他验证

- MuJoCo HalfCheetah-v5：三个训练种子各 102,400 transitions，独立 20 局 test 平均回报从约 -0.33 提升到 694/885/678；同批 model/Adam/normalizer/输出/RNG 恢复通过。[结果与曲线](reports/route2-learning.md) 保留全部种子及统计边界。
- 真实 RLinf FSDP1 strategy：显式旧 Torch opt-in、单卡 FP32 NO_SHARD 包装/前反向/梯度范数/AdamW 更新通过；local_shard 与 Torch 2.2 DCP 的 MUSA/MCCL checkpoint 恢复和继续更新均通过，CPU 参数最大误差 1.49e-8、继续更新模型/optimizer/scheduler 最大误差为0。仍未验证跨进程、多卡分片、offload 与官方 Actor，见 [说明](routes/route2/fsdp_probes/README.md)。
- 真实 Diffusers Attention 与固定双层 GR00T DiT：MUSA 12/12 rows 完成 forward、输入 VJP 和参数 VJP；除解析零方向外的严格门槛通过，独立 semantic gate 全部通过。严格总 gate 保留 false，不能替代完整 GR00T 权重加载或训练结论，见 [action 接口探针](routes/route2/planning/action-interface-probe.md)。
- GR00T attention 合约：FP32 eager 基础探针 32/32 rows 通过；SDPA 广播 mask 报错，展开 mask 的 BF16 通过，FP32 梯度超原严格门槛。实际模型尚未加载，见 [分析](routes/route2/planning/attention-results-analysis.md)。
- 首轮 Ray CPU Channel 8/8 回归、256 transitions 的 MuJoCo 小闭环及原生 Torch FSDP1 证据保留在 [历史报告](routes/route2/README.md)。

## 固定来源与复现

上游为 [RLinf c70606f0](https://github.com/RLinf/RLinf/tree/c70606f08cdca259b8dec03d4430926b5b8fac9d)，已验证 Worker 基线适配提交 `3e7329c629293f1b0f9c331b5bb24eccc3e3f142`；FSDP1 增量提交 `f9b74d95f97ad4234311580ab32c7b586e1dc7af`。这里保存可移植补丁，恢复不要求 GitHub 上存在本地适配 commit。

checkpoint 增量提交 `c46875bb1515716b487c0cde5e6cf8a2277a483b`，对应 tree `a6c18c65206804ce621bf78472c5ed0b3b37387d`。恢复脚本支持独立 checkpoint 源树；模型来源通过审计 manifest 下载并核对，保留前两版 action 探针的准确哈希与失败证据。

上游声明 `torch>=2.5.0`，本项目提供限定实验路径，不构成官方 Torch 2.2 支持。系统驱动、Toolkit 与默认 Python 保持原状；不直接安装上游全项目依赖，也不覆盖 Torch-MUSA。

[复现说明](REPRODUCE.md)、[源码锁](locks/sources.json)、[环境](locks/environment.md)、[后续节点](ROADMAP.md)、[来源与许可证](PROVENANCE.md)。后续具身目标为 `RLinf + GR00T N1.5 + LIBERO-Spatial + PPO`；Lambda-Sim SDK 到位后验证其环境接口。
