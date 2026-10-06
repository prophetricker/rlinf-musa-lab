# 路线二后续节点

记录日期：2026-10-05。路线二作为主开发路线，固定上游 commit 后递增适配补丁；路线一只承担回归对照，路线四在具体 backend 缺口阻塞主线时再投入。已完成部分标明，剩余项按独立验收继续推进。

2026-10-06续跑：基础通信、小模型FULL_SHARD更新、Actor/Rollout分卡真实轨迹更新和版本0/1同步通过；GR00T两rank FULL_SHARD官方PPO更新v15通过，范数兼容15行数值检查通过。2026-10-07两rank完整DCP新进程恢复与同批下一步精确对照通过；fresh8步双卡官方PPO与版本0/1精确同步也已通过；完整官方Runner两轮闭环也已通过；三轮80步稳定性亦已通过（480槽、454有效动作），chunk5/240步两轮亦已通过（192动作块、960有效动作槽、truncation/reset）；当前执行十任务pilot，Runner保存/恢复及最终同步入口已准备，见 [两卡证据](routes/route2/planning/two-gpu-results.md)。

| 节点 | 工作内容 | 验收材料 |
|---|---|---|
| R2-0 已完成 | 当前主线的 MUSA MLP/PPO、MuJoCo/Worker/Channel 小闭环、同批恢复与原生 FSDP1 探针 | 首轮 JSON/JSONL、固定源码与补丁、环境记录 |
| R2-1 基线完成 | Pendulum 与 HalfCheetah-v5 各三种子、每种子 102,400 transitions，独立 test 均提升 | 原始回报、学习曲线、超参数、各 seed 的配对结果与离散程度；更大预算另作实验 |
| R2-2 部分完成 | 分离导入与真实单卡 NO_SHARD strategy 更新、local_shard 和 Torch 2.2 DCP checkpoint 同进程与独立进程恢复通过，训练状态/四类 RNG/下一步 exact；官方Actor初始化、连续三轮真实LIBERO GAE/PPO、版本0–3同步与fresh-process同批恢复已通过；两rank FULL_SHARD更新、分片DCP新进程恢复和版本0/1同步亦通过；Actor offload另行验收 | 真正 RLinf FSDP1 actor 的更新/恢复证据；策略版本与同步前后输出对照；两rank证据见两卡报告；offload单独验收 |
| R2-3 部分完成 | synthetic eager attention、外围算子通过；真实 Diffusers processor/双层 GR00T DiT 前向与 VJP 完成；真实完整 Spatial 骨干585张量和完整 action head 314张量严格加载，已开展固定噪声动作、loss、梯度和 AdamW 更新功能对照 | 原骨干逐元素 gate 失败保留；完整功能影响结果见 Spatial/Eagle 报告，不能代替真实 episode 或官方 PPO |
| R2-4 部分完成 | 完整 Spatial checkpoint 已严格加载并完成固定合成输入下 backbone→action、backward和单步更新；真实预处理、headless渲染、8步观测/动作与单模型GAE/PPO已通过；官方三轮持续PPO/恢复通过，初始模型两个完整episode成功；完整双卡Runner两轮闭环也已通过；接下来更长轨迹与跨任务学习收益 | 真实双视角图像/proprioception合约、动作块反归一化与episode结果；合成输入不替代真实任务 |
| R2-5 单/双卡工程闭环通过 | 官方Actor/独立Rollout，真实LIBERO三轮8步GAE/PPO，版本0–3同步，完整Actor状态fresh-process同批恢复；初始模型两个完整episode成功；新增两rank完整恢复与官方Runner两轮on-policy闭环 | 原始907张量哈希/固定输出、Adam步数/实际变化、恢复metrics精确对照、显存与预测耗时；长轨迹、完整suite评估、Runner恢复和学习收益另行验收 |
| R2-L Lambda-Sim | SDK 到位即做独立 reset/step、headless、图像/动作、reward/terminated/truncated 验证，再接 RLinf 环境接口 | 最小可运行例子、接口映射、seed/自动 reset/最终观测语义和成功条件 |

本轮新增完整 backbone→action head 的功能影响验证：分开比较 CPU/MUSA action head 的设备差异与 backbone feature 的误差传播，固定噪声、时间步和权重，检查动作、loss、全部梯度与更新方向。使用 RLinf 的 LIBERO 分支31、8维 state 和7维有效 action，保留25维 padding；合成输入尚不代表真实预处理或 episode。正式功能与动作门槛通过；端到端7维动作relative-L2为2.82e-6，梯度为2.25e-5，更新为1.52e-3。原严格feature失败保留，暂时停止深入算子，优先验证真实预处理、LIBERO和官方Actor/Rollout/PPO。BF16失败保留；该合成组件阶段未验收Manager/Actor/多rank，后续独立官方Actor和两卡结果见上方最新节点。GPU作业顺序执行。详见 [本轮报告](reports/route2-spatial-eagle.md)。

R2-1 已实现周期性 validation、独立 test、多 seed 汇总、完整 normalizer/RNG 恢复对照及批量评估。Pendulum 是 Gymnasium 控制任务，HalfCheetah 才覆盖 MuJoCo 长训练；两者都不替代最终具身目标。

R2-2 保留显式 Torch 2.2 实验开关，默认门槛与 FSDP2 拒绝保持。R2-3 保持原数值阈值与失败证据，不用放宽容差代替验证。路线四只在具体 backend 缺口阻塞主线时重启，完整编译另排期。

Lambda-Sim 的独立接口验证不必等待 GR00T 训练完成。S5000 到位后重复相同配方，另记其驱动/Toolkit/Torch-MUSA 环境，不把 S4000 结果自动推广到新硬件。

后续每次实验记录：日期、目标、上游/适配 commit、环境、完整命令、训练与评估 seed、预算、通过范围、失败原因、原始证据和下一步。性能对照需固定 warmup、采样区间和环境配置。
