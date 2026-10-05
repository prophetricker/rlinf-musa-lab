# GR00T Libero-Spatial backbone 配置审计与真实组件探针

v2 实测 CPU 8/8、MUSA runner 12/12 rows pass，其中实际 MUSA 比较为4 rows。它检验冻结的真实 Qwen3/SigLIP 随机小模型 forward 接口与独立 Linear VJP；没有加载权重、导入完整 Eagle/RADIO 或运行训练。此前 action interface 的脚本与数值证据保持独立。

## 固定 checkpoint 与实现选择

RLinf 上游 `c70606f08cdca259b8dec03d4430926b5b8fac9d` 的 `examples/embodiment/config/libero_spatial_ppo_gr00t.yaml:96,109`，actor 与 rollout 均指向 `RLinf/RLinf-Gr00t-SFT-Spatial`。本地文件 SHA256 `c7923bfaa7cb28609aeb0bc0a3ce84e61501fa7cb07665ca79f4e68a232903a8`。

官方 Hugging Face API 读取时间为 2026-10-05 09:33:19 UTC，返回 revision `73f710e70e7d571f8d828e51e0a428f5a1e0ac22`。仅下载公开 `config.json`（1706 bytes），SHA256 `6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526`，固定 URL：

`https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial/resolve/73f710e70e7d571f8d828e51e0a428f5a1e0ac22/config.json`

checkpoint 的 `eagle_path` 名称含 Qwen3 和 Siglip2，但该 NVEagle 仓库公开 API 返回 HTTP 401；本轮未使用认证、未获得内部 config。因此名称不足以确认内部类或预训练 vision 权重结构。

固定 Isaac-GR00T `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` 的 `EagleBackbone` 接受 `eagle_path` 参数，却从 `DEFAULT_EAGLE_PATH` 读取 bundled `eagle2_hg_model/config.json`，没有使用该参数。此 config 明确指定 Qwen3 和 `siglip_vision_model`；`modeling_eagle2_5_vl.py` 实际导入、构造 `Qwen3ForCausalLM` 与 `SiglipVisionModel`。因此新探针采用 Qwen3Model/SiglipVisionModel，而不凭名称改成 Siglip2VisionModel。RADIO 仍在 Eagle 文件顶层导入，vision constructor 仍强制 `flash_attention_2`；本探针并未解决这些完整构造问题，也没有验证服务器安装的 GR00T 源码与此固定源码相同。

| 组件 | 固定 config | 本轮 tiny probe |
|---|---|---|
| Qwen3 | hidden2048，H16/KV8/D128，28层，source backbone 截断至12层 | hidden512，H4/KV2/D128，2层，vocab256，B2/S13 |
| Qwen3 norm/RoPE | RMS eps1e-6，theta1e6，no scaling/no sliding window/cacheFalse | 相同 eps/theta/行为；真实 q_norm/k_norm 与 source RoPE |
| Vision | SiglipVisionModel，hidden1152，H16/D72，27层，image224/patch14 | hidden288，H4/D72，2层，image56/patch14，16 tokens |
| Vision norm/MLP | LayerNorm eps1e-6，gelu_pytorch_tanh，intermediate4304 | 相同 eps/activation，intermediate1076（宽度按1/4缩小） |
| Eagle connector | 一层 Linear1152→2048，pixel_shuffleFalse | 不集成 multimodal connector/token substitution |
| Eagle feature projector | checkpoint `project_to_dim=null` → Identity | 独立 synthetic Linear(feature_width→64) 仅作 VJP 检查 |
| Action cross attention | H32/D48/cross_dim2048 | 本轮不运行 action；此前 action toy H4/D64/cross1536 保持其原范围 |
| VL self attention | H32/D64 | 本轮不运行 action/VL self attention |

checkpoint `tune_visual=true`，但 RLinf builder `gr00t_n1d5/__init__.py:73-85` 明确传入 `tune_visual=False,tune_llm=False`。固定 GR00T 的冻结方法会同时冻结 language、vision 与 `mlp1`；由于此 checkpoint 的 `eagle_linear` 为 Identity，不能声称存在训练中的 backbone Linear。新的 synthetic Linear 明确独立于 checkpoint 参数。

公开配置、源码 pins、URL/hash/HTTP 状态与范围的机器可读记录在 `model_probes/backbone_sources/backbone-config-audit.json`；源码原文位于既有不导出的 `audit_sources`，所有新增 Transformers 文件已经加入 `audit_sources/manifest.json`。

## 新脚本与 gate

当前 v2 `model_probes/gr00t_backbone_interface_probe.py` SHA256：
`eb69db022e7be0c9b0252af5106ec53b0c1e8200950ba7aae66a4ae074ee97ec`

当前 v2 `model_probes/fp32_backbone_attention.py` SHA256：
`db580ebeea14b7f7b174927e2d589ddb720f4f31a1991f4ddecc512d12be7cf1`

脚本强制 Transformers 4.51.3，并验证完整已安装 modeling_qwen3.py、modeling_siglip.py 和两份固定 config 的 SHA256。对应 Transformers commit 为 `5f4ecf2d9f867a1255131d2461d75793c0cf1db2`。模型随机初始化后先在 CPU 量化到目标 dtype，再复制到目标设备，记录每个 state/input/config/hash，检查设备转移后 state 一致。

每个真实 Attention 实例的 source `forward` 用相同 code object 和 private globals 字典绑定；唯一被替换的 lookup 是 `eager_attention_forward`。类方法与全局注册表不改动，config 明确保持 `eager`。Q/K/V 投影、Qwen3 q_norm/k_norm、RoPE、GQA 的 source repeat_kv、实际 scale、输出投影与 BSHD kernel 返回均沿用 source。FP32 fallback 只把 QK、softmax、PV 累积提升至 FP32，然后转回目标 dtype。CPU baseline 直接调用 pinned source eager kernel；BF16 baseline 自身的 QK/PV 仍是 BF16，不能等同 CPU FP64 真值。

Qwen3Model 用 source 自动生成的 4D additive causal + right-padding mask，batch 有效长度9/11；不在 kernel 重复叠加因果 mask。有效 token 的输出必须满足 gate，padding 与未来 token 的独立扰动必须使有效 token／prefix5 输出差精确为零，blocked probability 必须精确为零；cache 返回必须为 None。attention traces 记录真实 QKV shape/stride/dtype、mask shape/stride、GQA 与 scale。SigLIP 保留真实 patch convolution、位置 embedding、encoder、LayerNorm 和 MLP，但设置 `vision_use_head=False` 排除供 Eagle feature 使用时被忽略的 pooler 计算；图像预处理、池化头、动态切片均不在范围内。

整个 backbone 在 eval/no_grad 下运行，全部参数 requires_gradFalse、gradNone、输出不含图。独立 Linear 用同一 CPU source 有效 feature 创建 detached leaf，所有设备使用相同 quantized feature/state/upstream；检查 forward、feature leaf VJP、weight VJP、bias VJP。它验证 trainable projection 基础能力，没有把随机投影冒充 checkpoint 的 Identity 或完整 action projector。没有 optimizer update。

严格 gate 与此前数值证据一致：FP32 atol/rtol/relative-L2 都为2e-4；BF16 atol0.025、rtol0.04、relative-L2 0.04。源模型完整输出保留为 diagnostic，有效 token forward 与全部 projection VJP 为 required gate。MUSA 还比较同 dtype 的 CPU fallback forward。

v2 `no_global_patch` 记录并要求 class forward 未改变、global attention registry 的 keys 与逐项 entry identity 前后相同及真实 FA availability 前后相同。每个实例绑定必须满足 source code object 相同、private globals 与 instance-only，且加入 `scope_pass`。没有全局修改 availability。

## 依赖与协调者 runner

独立 `envs/route2-backbone` 使用 system-site Torch/Torch-MUSA，不覆盖默认 Python。Transformers 固定 dependency table 要求 `huggingface-hub>=0.30.0,<1.0`、`tokenizers>=0.21,<0.22`、`safetensors>=0.4.3`、`numpy>=1.17`、`packaging>=20`、`pyyaml>=5.1`、`tqdm>=4.27`，以及 filelock、requests、regex。协调者已使用 no-deps 安装 `transformers4.51.3/hub0.30.2/tokenizers0.21.4/regex2024.11.6`，其余来自现有环境；此前 Diffusers action 环境不变。

协调者确认真实 Qwen3Attention/SiglipVisionModel import 通过；这个 import 仅为依赖检查。torchvision.io image extension 警告没有验证或否定本探针 raw tensor convolution，但图像预处理能力尚未证明。

```sh
envs/route2-backbone/bin/python routes/route2/model_probes/gr00t_backbone_interface_probe.py --device cpu --output routes/route2/evidence/backbone-interface/backbone-interface-cpu.json
envs/route2-backbone/bin/python routes/route2/model_probes/gr00t_backbone_interface_probe.py --device musa --output routes/route2/evidence/backbone-interface/backbone-interface-musa.json
```

CPU 默认2 families×2 dtypes×2 kinds＝8 rows。显式 MUSA runner 再加 fallback_musa，共12 rows；GPU仅由协调者安排。本地 macOS 无 Torch，两版均已完成 AST 与 --help 检查。

## v1 实测与 v2 修复

v1 两个脚本的准确字节一起保存于 `model_probes/versions/backbone-interface-v1/`，分别为 runner `e136cc33b556d1e20bb09051d4126add048cfc68ca3bec6bdc97bf7d950bb514` 与 kernel `ee6caf6745a92cf9b339a70c71067fb562730f4fe14fea80cfed2450d6ceef01`。历史 runner 导入该目录同名 sibling kernel；复现需显式传入未复制的两份 config 路径，命令在版本目录 README 中。旧证据保持原字节。

v1 CPU `artifacts/run-20261005-process-backbone/backbone-cpu.json` SHA256 `a96bc0febc985116fd9e02ea05003aa11f34e09de771ff8f0c3c8549b69ebde2`，8/8 rows pass。

v1 MUSA `artifacts/run-20261005-process-backbone/backbone-musa.json` SHA256 `10092a9d2caf55fc9a10fcd711e8d42a0c234f324ea67f880afcd17df63d40be`，11/12 rows pass；Qwen3 BF16 fallback_musa 在 `weights.masked_select` 诊断读取处报错：Torch-MUSA 仅支持该算子的 Float32/Int32/Int64 输入，当前输入 BFloat16。该行没有取得最终 feature/VJP 比较，不能推断该行已通过数值 gate。其他三条 MUSA 行有完整比较。

v2 唯一数值代码变化是诊断用 `weights.float().masked_select`，不改 attention output/weights/VJP，不放宽阈值。runner 同时增加前述绑定不变量与 registry entry identity gate。协调者已重新执行 CPU 与 MUSA；全部19条已成功的 v1 rows，其 feature output、projection output 和全部 projection gradient hash 与对应 v2 rows 完全相同（包括两份 runner 中重复的 CPU cases）。

## v2 结果与结论边界

四份正式证据保存于 `routes/route2/evidence/backbone-interface/`，旧 v1 false 原样保留；`run-validation.json` 关联文件 hash、脚本版本、真实执行命令、退出码、source pins、误差与范围。v2 CPU JSON SHA256 `1cb64a269306df934e6a219e08a8ff4316f0c13545869257bb13ebc01b383e6e`；v2 MUSA JSON SHA256 `806bdddc06813874928fc9393b8fc6112880e75c62b537fc42995c767588616d`。

| v2 MUSA component | 有效 feature max abs | 相对L2 vs CPU source eager | 相对L2 vs CPU fallback | 结果 |
|---|---:|---:|---:|---|
| Qwen3 FP32 | 5.126e-6 | 1.181e-6 | 1.181e-6 | pass |
| Qwen3 BF16 | 0.046875 | 0.0082044 | 0.0060291 | pass |
| SigLIP FP32 | 5.603e-6 | 1.048e-6 | 1.048e-6 | pass |
| SigLIP BF16 | 0.046875 | 0.0069139 | 0.0052989 | pass |

BF16 max abs 高于 atol0.025仍可通过混合 allclose 条件 `abs(error) <= atol + rtol*abs(reference)`，并且相对L2须≤0.04；这里没有把 atol 冒充独立 max-abs 上限，也没有改变 gate。独立 projection 的4个 MUSA cases forward 与 feature leaf/weight/bias VJP 全通过；FP32 最大 VJP max abs5.960e-8／relativeL2 9.811e-8，BF16 最大 VJP max abs2.980e-8／relativeL2 5.114e-8。BF16 projection forward 最大 max abs0.001953125／relativeL2 7.404e-5。

两份 v2 runner 的全部 required rows 均有限、backbone/projection state hash 匹配 CPU template、冻 backbone gradNone/输出 no_grad、instance binding invariants 成立。Qwen3 右 padding 与未来 token 扰动的有效 token／prefix5 输出差均精确为零，blocked probabilities 精确为零。registry 三个 entry identity、source class forward、真实 FA availability 前后相同；FA availability 始终 false。环境为 Torch `2.2.0a0+git8ac9b20`、Torch-MUSA distribution `1.3.0`（此版本字符串不额外证明 backend commit）、Transformers4.51.3/hub0.30.2/tokenizers0.21.4/regex2024.11.6/safetensors0.8.0；Torch 来自保留的系统环境。

本轮证据支持真实源码封装后的随机小尺寸冻结 Qwen3/SigLIP feature forward 与独立 projection VJP 在 MUSA 上运行并通过所声明 gate。backbone 自身没有反向传播；projection 用 CPU reference detached leaf，未检查完整 multimodal connector、动态图像、pretrained weights、Eagle constructor、full-depth full-width、570-token end-to-end feature、optimizer update、性能或学习。

## 具体下一步建议

1. 新建正式 Spatial action-layout probe，保留此前 toy 证据。首轮使用真实 Diffusers Attention 与固定 GR00T DiT 两层，保持 H32/D48（inner1536）、cross_dim2048、input_embedding1536、output_dim1024；VL self attention 另测 H32/D64（inner2048）。query 顺序保留 source 的 state→future→action，长度为 `Tstate+32+16`，state长度1时才是49 tokens。RLinf `gr00t_action_model.py:605,613` 将 input_ids 与 attention_mask 右 padding 至配置 `padding_value570`，故先覆盖 key length570与真实右pad mask，再核对实际 batch 的未padding长度。CPU source baseline → CPU FP32 fallback → MUSA，继续全部输入／参数 VJP、真实 norm/4D mask、analytic-zero bias diagnostics；前向 dropout 设为 eval0，之后另设训练 dropout 合同，不把 eval gate推广到dropout训练。
2. 对隔离环境实际安装的 Isaac-GR00T 文件做 commit/hash核对，确认 `eagle_path` 忽略、bundled config 与 checkpoint weights 的形状关系。准备可审查的 source patch，使未选 RADIO 分支不顶层加载 RADIO 依赖，且显式 eager 配置能在构造前覆盖硬编码 FA2；不要伪造 FA availability。先不载权重，用缩小的 composite Eagle 构造测试连通 SigLIP→mlp1→image token substitution→Qwen3→Identity，核对有效 mask、select_layer、vision head、connector 冻结与 layout。这一步独立于本轮组件 pass。
3. composite 构造通过后，再明确 checkpoint 下载范围，固定同一 HF revision 的权重清单／索引／hash及实际 config。核对 `load_state_dict` missing/unexpected keys 和所有张量 shape，先单样本真实预处理与冻结 feature，再接真实 Spatial action。完整宽度、12层 Qwen3 与27层 vision、570 feature token 的内存／时间应单独测量后决定 full rollout/PPO 集成；上述 gate均不足以推出可训练、成功率或学习结论。
