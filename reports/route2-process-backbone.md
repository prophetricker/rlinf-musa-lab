# 路线二：跨进程恢复与 GR00T 骨干组件验证

记录日期：2026-10-05。使用现有单卡 S4000，Driver 2.7.0 / MUSA 3.1.0 / Torch 2.2.0 / Torch-MUSA 1.3.0；宿主驱动、系统 Toolkit 和默认 Python 未修改。新增 Transformers 依赖放到独立 `route2-backbone` 环境，继承原匹配的 Torch-MUSA。

本轮完成了两个工程节点：训练状态能在结束旧进程后恢复并准确继续；真实 Qwen3/SigLIP 组件在 S4000 上能用受限 eager attention 路径完成冻结前向。它们补齐后端与模型组件的证据，但完整 `RLinf + GR00T N1.5 + LIBERO-Spatial + PPO` 尚未跑通。

| 验证项 | 实际结果 | 说明 |
|---|---|---|
| local_shard checkpoint | 独立 save / restore 均退出0 | 新 Python driver、新 RLinf Worker、真实 MCCL 与 FSDPStrategy |
| Torch 2.2 DCP checkpoint | 独立 save / restore 均退出0 | 同样使用真实 checkpoint load，未用 reference 复制参数 |
| 恢复后继续训练 | 两格式模型/Adam 最大误差0 | scheduler、loss/norm/LR 与四类 RNG 也 exact 一致 |
| 骨干 CPU v2 | 8/8 rows 通过 | Qwen3/SigLIP × FP32/BF16 × source/fallback |
| 骨干 CPU/MUSA v2 | 12/12 rows 通过 | 包含8个CPU对照和4个真正MUSA病例，并非12个GPU病例 |
| 骨干 v1 | CPU8/8，CPU/MUSA11/12 | Qwen3 BF16 的只读概率检查触发 masked_select 类型限制；原失败与脚本保留 |

## 恢复验证为什么有用

checkpoint 不只是模型参数，还要保存 Adam 的动量和步数、学习率进度以及随机数状态。缺少其中一项，重启后的训练就可能走上另一条轨迹。

这次在保存进程中完成第1步，调用真正的 RLinf 保存接口，随后继续第2步作为未中断对照。保存命令完全退出后，另一命令创建全新的 FSDP、AdamW 和 StepLR，并用真正的 RLinf load 恢复第1步，再运行相同第2步。独立 CPU reference 仅提供断言数据，不能替代真实 load；没有在 load 后重新设 seed。

两种格式都逐项比较完整训练状态、Python/NumPy/Torch CPU/MUSA 的原始随机状态和下一段样本，结果 exact 一致。PID、boot id、process start ticks 证实 driver/Worker 身份不同；显式检查保存 driver 已结束，但没有单独强制检查旧 Worker 已结束。

范围限定同 Linux host/boot、相同 runtime 和设备型号/local index，world size1、FP32、NO_SHARD tiny MLP。尚未验证多卡分片、设备迁移、offload、混合精度训练、官方 Actor 或 PPO checkpoint。原同进程探针和生产补丁保持原字节，见 [跨进程完整记录](../routes/route2/fsdp_probes/checkpoint-process-README.md)。

## 骨干验证说明了什么

GR00T 的视觉和语言骨干负责把图像、文本变成特征；RLinf 的固定 GR00T builder 默认冻结这部分，因此先验收冻结前向符合后续起步方向。

本轮使用真实 Transformers 4.51.3 的两层随机 Qwen3Model 与 SiglipVisionModel，保留 Qwen3 的 RMSNorm、RoPE、GQA、因果/右侧padding和视觉attention的D72。模型整体宽度、层数、图像和词表缩小；权重未预训练。被屏蔽概率必须为零，padding/未来token扰动不应改变有效前缀，冻结参数必须无梯度；这些都参与验收。

v1 的 Qwen3 BF16 病例在只读诊断操作 `masked_select` 上失败。v2 仅把诊断输入无损转FP32，并增强绑定/全局注册表检查；没有改变 attention 数学、VJP 或数值门槛。复测CPU8/8、整组CPU/MUSA12/12通过。独立 synthetic Linear 用同一CPU参考特征验证 forward 与输入/参数 VJP；实际 Spatial checkpoint 的 `eagle_linear` 是 Identity，因此这不代表其投影参数训练。

checkpoint 名称里含 Siglip2，固定源码实际从 bundled Eagle config 构造 SiglipVisionModel，不能仅凭名称换类。完整 Eagle/RADIO 的导入、强制 FlashAttention 构造路径、连接器/图像token替换、预处理与真实权重都未验证，见 [配置审计及版本证据](../routes/route2/planning/backbone-interface-probe.md)。

## 下一步收窄方向

1. 将 action 组件测试对齐 Spatial 实际布局：H32/D48/cross2048，另验收 VL self-attention H32/D64；旧toy H4/D64/cross1536不覆盖这组布局。
2. 在隔离固定源树中处理完整 Eagle 构造与连接器接口；明确加载真实 Spatial 权重时的 missing/unexpected keys，再验收单步动作。下载和显存预算单独记录。
3. 独立验证 LIBERO headless 渲染、双视角观测、动作块语义和完整 episode；通过后接真实 RLinf Actor/Rollout/Manager，再进行极小 PPO。

目前继续用单卡即可。多卡不是已验证组件的前提；等完整模型测得显存、rollout与update需求后再决定是否加卡。之前的 Pendulum/HalfCheetah 三seed学习结果保留，本轮没有重跑或声称新增学习结果。
