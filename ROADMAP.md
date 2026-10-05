# 路线二后续节点

记录日期：2026-10-04。路线二作为主开发路线，固定上游 commit 后递增适配补丁；路线一只承担回归对照，路线四在具体 backend 缺口阻塞主线时再投入。以下都是待完成工作，不是已通过的结果。

| 节点 | 工作内容 | 验收材料 |
|---|---|---|
| R2-0 已完成 | 当前主线的 MUSA MLP/PPO、MuJoCo/Worker/Channel 小闭环、同批恢复与原生 FSDP1 探针 | 首轮 JSON/JSONL、固定源码与补丁、环境记录 |
| R2-1 学习基线 | Pendulum-v1 调试算法，再用 HalfCheetah-v5；先约 10 万 transitions，至少 3 个训练种子、固定多 episode 评估，视结果扩大预算 | 原始回报、学习曲线、超参数、各 seed 的初始/训练后配对结果与离散程度；出现可重复收益才通过 |
| R2-2 训练后端 | 分离 RLinf FSDP1/FSDP2 导入；先单卡 NO_SHARD，再接 checkpoint、独立 Actor/Rollout 与权重同步 | 真正 RLinf FSDP1 actor 的更新/恢复证据；策略版本与同步前后输出对照；offload 单独验收 |
| R2-3 GR00T 算子 | 核查 N1.5 的 attention、视觉、BF16；为 SDPA/eager 路径做 CPU/MUSA 前向与梯度对照，测峰值内存 | 误差、失败算子、正确 mask/layout、实测显存；不能以导入成功代替模型可用 |
| R2-4 模型与环境 | 算子通过后加载 Spatial checkpoint；独立验证 LIBERO headless 渲染、观测/动作和完整 episode，再组合 rollout | 模型单步推理、双视角图像与 proprioception 合约、动作块语义、episode 结果 |
| R2-5 极小 VLA PPO | 初始冻结视觉/语言骨干，1～2 env、microbatch 1；验证策略更新、恢复和 Actor/Rollout 同步 | 有限 loss/梯度、参数变化、显存峰值、恢复对照；再以固定预算与多 seed 成功率验收学习 |
| R2-L Lambda-Sim | SDK 到位即做独立 reset/step、headless、图像/动作、reward/terminated/truncated 验证，再接 RLinf 环境接口 | 最小可运行例子、接口映射、seed/自动 reset/最终观测语义和成功条件 |

下一轮先推进 R2-1，同时准备 R2-2 与 R2-3 的源码和小型 CPU 检查；一张 S4000 上的 GPU 作业顺序执行。Pendulum 和 HalfCheetah 是学习与接口基线，不替代最终具身目标。10 万 transitions 是起步预算，不是收益承诺。

R2-1 先确认现有 runner 对 Pendulum 的动作缩放、截断 bootstrap 和评估语义；目前只实测过 HalfCheetah。现有 runner 只提供终点配对评估，长训练的周期性评估与多 seed 汇总需要补充，尚未实现。评估使用独立种子、明确冻结的 normalizer；CPU 与 MUSA 比较保持算法和预算一致。

R2-2 与 R2-3 不预先删除版本门槛或替换不存在的 API。保留失败日志，依据具体 API/算子证据决定局部移植；仅当主线受阻才启动路线四的单个翻译单元检查，完整 backend 编译另排期。

Lambda-Sim 的独立接口验证不必等待 GR00T 训练完成。S5000 到位后重复相同配方，另记其驱动/Toolkit/Torch-MUSA 环境，不把 S4000 结果自动推广到新硬件。

后续每次实验记录：日期、目标、上游/适配 commit、环境、完整命令、训练与评估 seed、预算、通过范围、失败原因、原始证据和下一步。性能对照需固定 warmup、采样区间和环境配置。
