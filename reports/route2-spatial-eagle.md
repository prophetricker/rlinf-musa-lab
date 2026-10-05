# Spatial action 与真实 Eagle 集成验证

日期：2026-10-05。实机是单卡 MTT S4000 48 GiB，Driver 2.7.0 / MUSA 3.1.0 / Torch 2.2.0 / Torch-MUSA 1.3.0。系统运行栈保持原状；新增 `route2-integration` 隔离环境组合 Transformers 4.51.3、Diffusers 0.30.2、safetensors 0.8.0、dm-tree 0.1.8，均以 `--no-deps` 安装。

本轮完成真实 Spatial 骨干585张量的严格加载和完整前向；CPU参考与CPU fallback完全一致，S4000前向有限且结构合同成立，但最终features/logits未通过原逐元素数值门槛。正式宽度action的FP32前向和全部VJP已完成，小型真实Eagle组合FP32数值对照通过。FP32/BF16失败均保留原门槛与证据，尚不能推出完整GR00T action推理、LIBERO rollout或PPO已可用。

## 1. 正式 Spatial action 几何

固定 Spatial revision `73f710e70e7d571f8d828e51e0a428f5a1e0ac22`，Isaac-GR00T commit `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`。Action H32/D48、inner1536、cross2048、Q49/K570；VL Attention H32/D64、width2048、Q/K570。实际 DiT 层数由16缩至2；测试使用随机权重、合成嵌入和 eval/no-dropout，未执行完整 action head。

| 独立运行 | 完成行数 | 实际 MUSA 行数 | 原严格总门槛 | 独立语义门槛 | 真实退出码 |
|---|---:|---:|---|---|---:|
| CPU FP32 | 20 | 0 | false | true | 1 |
| CPU BF16 | 20 | 0 | false | false | 1 |
| CPU/MUSA FP32 | 30 | 10 | false | true | 1 |
| CPU/MUSA BF16 | 30 | 10 | false | false | 1 |

总计100行，包括重复建立的CPU参考，实际GPU执行20行。没有运行异常或缺失梯度。FP32 的输出、输入 VJP 和非解析零参数 VJP 都通过原严格阈值；解析零 bias 的 relative-L2 仍导致严格总门槛失败。独立语义门槛同时要求这些解析零方向的 actual/reference 梯度 finite 且绝对值在原阈值内，不能把它改称原严格门槛通过。

BF16 的额外失败在扩展 `action_norms_4d_rightpad_additive` 合同：CPU fallback 的 `norm_k.weight` VJP relative-L2 为0.06792，大于0.04；MUSA 相对 native CPU 为0.15340，另有 `norm_q.bias` 为0.04316。`norm_k.weight` 和 `norm_q.bias` 是正常梯度方向，不能按解析零 bias 排除。该扩展 per-head norm 配置不同于正式 DiT 的无 q/k norm 配置；同一失败已经在CPU复现，也不能笼统归因于S4000。

独立 Attention 覆盖真实右侧 padding、bool/additive、4D mask、屏蔽输入扰动、可见输入扰动、移除 mask 正控制及全部 VJP。固定 DiT 的源码却忽略传入 `encoder_attention_mask`，VL SelfAttentionTransformer 也没有 mask 参数。本轮按原源码执行 no-mask baseline，并单独验证“传入 mask 与 None 完全相同”；这不能作为完整 DiT 屏蔽 padding 成功的证据。

证据见 [四次运行汇总](../routes/route2/evidence/spatial-action/run-validation.json)、[实测 FP32](../routes/route2/evidence/spatial-action/spatial-action-musa-all-fp32-v2.json)、[实测 BF16](../routes/route2/evidence/spatial-action/spatial-action-musa-all-bf16-v2.json) 和 [合约说明](../routes/route2/planning/spatial-action-layout.md)。

## 2. 小型真实 Eagle 组合

直接运行原 `EagleBackbone.__init__/forward`，真实 SigLIP → 原 connector → image token substitution → Qwen3 → 原 hidden-state selection → Identity。补丁只使选中分支的依赖延迟导入，并让 SigLIP 遵循显式 eager 配置；七个关键源码方法的 AST 均不变。没有伪造 PEFT/RADIO、FlashAttention availability 或手写 multimodal forward。

临时 AutoConfig/AutoModel 注册的是实际类，并在结束恢复；attention registry、源类 forward 和 FlashAttention availability 前后不变。构造后 top/vision/language 的实际 attention implementation 都是 eager。tiny fixture 使用两层 Qwen3、两层 SigLIP 和随机权重，不能替代正式宽度/深度。

| 组合 | FP32 | BF16 |
|---|---|---|
| native CPU source eager | 通过 | 通过 |
| CPU FP32 attention fallback | 通过 | 有效 feature allclose 失败 |
| MUSA FP32 attention fallback | 通过 | 有效 feature allclose 失败 |

FP32 MUSA 有效 feature 最大绝对误差6.3181e-6、relative-L2 1.2339e-6，低于原2e-4阈值。BF16 CPU/MUSA feature 最大绝对误差分别0.046875/0.0546875；relative-L2虽然低于0.04，混合 allclose 仍失败。logits、connector 以及MUSA相对CPU fallback的 BF16 feature 比较通过；原组合总 gate 仍为false。

六行的冻结状态、梯度为空、state hash、image slot 与原 connector 精确对应、Identity、causal/padding、被过滤图像负控制和可见图像正控制全部通过。CPU fallback 也出现 BF16 feature 失败，下一步可用不改变门槛的分层诊断定位舍入误差；本轮不以放宽容差推进。

证据见 [完整 CPU/MUSA JSON](../routes/route2/evidence/eagle-composite/eagle-composite-musa-v2.json)、[源码方法复核](../routes/route2/evidence/eagle-composite/source-method-review.json) 和 [集成设计](../routes/route2/planning/eagle-integration.md)。

## 3. 完整模型结构与权重来源

真实完整 `GR00T_N1_5` 在 meta 上构造后，全部899 state keys/shapes与固定 checkpoint headers 匹配：missing=0、unexpected=0、shape mismatch=0。包括585个 BF16 backbone 张量和314个 FP32 action 张量。meta 不执行前向或加载数据；临时关闭的 Beta 参数校验和默认dtype都已恢复。

完整 Eagle 保留27层视觉网络、12层语言网络、1152→2048连接器及Identity。权重约7.59 GB，两个分片大小分别4,999,367,032和2,586,705,312 bytes。canonical HF API 的固定 revision 提供完整 LFS SHA256：

- 分片1：`e8e38e607765f47658ba2ccac8c7b311e57bafc58dd8490a143332c01e929556`。
- 分片2：`ce44c6277867741d906c3b25d281ea645edf78b0c5b8454d3c69e1a4ed6458eb`。

下载仅涉及这两个公开 safetensors 文件，不包含 optimizer/scheduler。远端 canonical 连接超时；mirror 支持实际206 Range。串行传输停止后保留原 prefix，采用32 MiB bounded ranges并发续传，完成文件仍必须通过canonical LFS完整 SHA256、长度、原header SHA与全部tensor layout检查。signed redirects和认证信息不记录。

两个完整分片均已下载并通过canonical LFS完整SHA及header/layout校验。下载结果见 [验证JSON](../routes/route2/evidence/spatial-weights/spatial-download-parallel-v1.json)。证据还包括 [完整899项结构审计](../routes/route2/evidence/spatial-weights/spatial-shape-audit-v1.json)、[权重来源](../routes/route2/model_probes/weight_metadata/lfs-metadata.json)。

真实骨干调用 `load_state_dict(strict=True)`：全部585 keys，missing/unexpected都是0；载入后逐tensor内容与checkpoint BF16→FP32转换精确相同。source里共享的embedding/lm_head两份checkpoint值也精确相同，未静默丢弃或覆盖不一致张量。未加载/执行314个action张量。

完整fixture为B1/S570、两幅224×224合成raw tensor image、512个image slots、有效token550。真实wrapper保留12层Qwen3和27层SigLIP，共39个实例attention绑定；image substitution、Identity、mask、无cache、冻结/no-grad、全部feature/logits/connector/pooler有限及state不变均通过。

| 完整预训练骨干 v1 | 结构/有限性合同 | 原数值门槛 |
|---|---|---|
| native CPU source eager | 通过 | 通过 |
| CPU fallback | 通过 | 通过，features/logits/connector与source精确相同 |
| MUSA fallback | 通过 | features与logits失败，connector通过 |

MUSA有效features最大绝对误差0.01650047、relative-L2 2.82969e-5；logits最大绝对误差0.00706100、relative-L2 2.19074e-5。两者整体relative-L2低于2e-4，但逐元素 `atol=rtol=2e-4` 的混合allclose不通过。不能只按整体误差改写通过结果。原进程退出1、整体pass=false、没有执行异常，见 [v1完整证据](../routes/route2/evidence/spatial-weights/spatial-pretrained-backbone-musa-v1.json)。

MUSA该进程测得 peak allocated 8,119,447,040 bytes（约7.56 GiB）、peak reserved 8,443,133,952 bytes（约7.86 GiB），统计在clone转移后重置并包含真实forward/后续state核对；不包含训练/optimizer/多worker。当前单卡足够继续模型诊断，这一数字不能推导完整PPO的显存需求。

v2仅增加只读逐层hooks和失败元素计数，kernel、权重、输入及原数值门槛不变。复跑结果与v1完全相同，退出1；有效features有2,960/1,126,400个元素超混合阈值（约0.263%），logits有246,774/83,424,000个（约0.296%）。connector没有超阈值元素。最差feature的actual/reference为0.13700652/0.13451497，绝对误差0.00249155，而对应混合阈值0.00022690；不能简单解释成“全部是接近零的无意义项”。

逐层诊断观察到：视觉第24～26层出现少量逐元素失败，后续vision norm/connector通过；语言第3层开始有单个失败，第7层后数量增长，最终language norm后有效feature失败2,960个。每层整体relative-L2仍很小。这定位了误差增长发生的位置，尚未证明是某一个kernel、累积规则或设备算子的单一根因。见 [v2逐层证据](../routes/route2/evidence/spatial-weights/spatial-pretrained-backbone-musa-v2.json)。下一次应在相同fixture下分开“CPU视觉特征送入GPU语言网络”和“同一hidden输入的选定层/最终norm对照”，再用更高精度CPU参考核对候选算子；不改写原失败结果。

## 4. 修订历史与复现边界

Spatial v1 的CPU两个dtype各留下18行partial JSON，在DiT effective_config含torch.dtype时报告序列化失败。Eagle v1 的FP32 stdout两行通过，但最终报告序列化失败、退出1。v2只修复JSON dtype表示与证据保护等诊断逻辑；没有改数值数学或门槛。原脚本、partial JSON、stdout和命令退出码保留。

上游完整169个归档文件的Git blob与固定commit tree匹配。最小补丁源树提交 `62c7d52e625a8172f3c42453453706c329b71d7b`；tiny fixture提交 `e441b8f7c17d4f25c49cfa1f20731e6085e7a962`；full eager/FP32 fixture提交 `c9bd9a2403d14d492be88a9af9ae3989d4b711fc`。恢复脚本只新建目录，不覆盖既有源码；源码锁、原始命令及依赖列表随包保存。

下一步需要把正式 pretrained backbone 与完整 action head 连起来，再完成真实 tokenizer/image processor、LIBERO 双视角和 proprioception 输入、动作块解码和 episode。之后才能接入官方 Actor/Rollout/PPO；import、随机前向、结构匹配、strict load与学习结果分别验收。
