# 路线二后续节点

记录日期：2026-10-05。路线二作为主开发路线，固定上游 commit 后递增适配补丁；路线一只承担回归对照，路线四在具体 backend 缺口阻塞主线时再投入。已完成部分标明，剩余项按独立验收继续推进。

| 节点 | 工作内容 | 验收材料 |
|---|---|---|
| R2-0 已完成 | 当前主线的 MUSA MLP/PPO、MuJoCo/Worker/Channel 小闭环、同批恢复与原生 FSDP1 探针 | 首轮 JSON/JSONL、固定源码与补丁、环境记录 |
| R2-1 基线完成 | Pendulum 与 HalfCheetah-v5 各三种子、每种子 102,400 transitions，独立 test 均提升 | 原始回报、学习曲线、超参数、各 seed 的配对结果与离散程度；更大预算另作实验 |
| R2-2 部分完成 | 分离导入与真实单卡 NO_SHARD strategy 更新、local_shard 和 Torch 2.2 DCP checkpoint 同进程与独立进程恢复通过，训练状态/四类 RNG/下一步 exact；下一步 Manager/Actor/Rollout、多卡同步与 offload | 真正 RLinf FSDP1 actor 的更新/恢复证据；策略版本与同步前后输出对照；多 rank 与 offload 单独验收 |
| R2-3 部分完成 | synthetic eager attention、外围算子通过，SDPA 失败已定位；真实 Diffusers processor 与固定双层 GR00T DiT 的 MUSA 前向/VJP 已完成，严格 gate 保留零方向失败并有独立 semantic gate；真实Qwen3/Siglip小型冻结骨干FP32/BF16已通过CPU/MUSA对照 | 真实 checkpoint 权重、更多模型布局、实测显存和训练闭环；不能以随机 DiT 代替完整模型可用 |
| R2-4 模型与环境 | 算子通过后加载 Spatial checkpoint；独立验证 LIBERO headless 渲染、观测/动作和完整 episode，再组合 rollout | 模型单步推理、双视角图像与 proprioception 合约、动作块语义、episode 结果 |
| R2-5 极小 VLA PPO | 初始冻结视觉/语言骨干，1～2 env、microbatch 1；验证策略更新、恢复和 Actor/Rollout 同步 | 有限 loss/梯度、参数变化、显存峰值、恢复对照；再以固定预算与多 seed 成功率验收学习 |
| R2-L Lambda-Sim | SDK 到位即做独立 reset/step、headless、图像/动作、reward/terminated/truncated 验证，再接 RLinf 环境接口 | 最小可运行例子、接口映射、seed/自动 reset/最终观测语义和成功条件 |

本轮新增 R2-3 的Spatial正式action宽度FP32前向/VJP、小型真实Eagle FP32组合、完整GR00T899 keys/shapes匹配和真实骨干585权重strict load。完整FP32骨干已执行S4000前向，结构/有限性合同通过，features/logits逐元素数值失败已做逐层诊断；下一步隔离视觉传播与语言网络/最终norm差异，再连接完整action、真实预处理和LIBERO，推进官方Actor/Rollout/PPO。BF16失败保留；R2-2 的Manager/Actor/多rank仍独立待验收。一张S4000上的GPU作业顺序执行。详见 [本轮报告](reports/route2-spatial-eagle.md)。Pendulum与HalfCheetah是学习与接口基线，10万transitions是起步预算。

R2-1 已实现周期性 validation、独立 test、多 seed 汇总、完整 normalizer/RNG 恢复对照及批量评估。Pendulum 是 Gymnasium 控制任务，HalfCheetah 才覆盖 MuJoCo 长训练；两者都不替代最终具身目标。

R2-2 保留显式 Torch 2.2 实验开关，默认门槛与 FSDP2 拒绝保持。R2-3 保持原数值阈值与失败证据，不用放宽容差代替验证。路线四只在具体 backend 缺口阻塞主线时重启，完整编译另排期。

Lambda-Sim 的独立接口验证不必等待 GR00T 训练完成。S5000 到位后重复相同配方，另记其驱动/Toolkit/Torch-MUSA 环境，不把 S4000 结果自动推广到新硬件。

后续每次实验记录：日期、目标、上游/适配 commit、环境、完整命令、训练与评估 seed、预算、通过范围、失败原因、原始证据和下一步。性能对照需固定 warmup、采样区间和环境配置。
