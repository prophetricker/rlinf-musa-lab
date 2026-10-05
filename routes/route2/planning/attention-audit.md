# R2-3：GR00T N1.5 attention 与 BF16 检查

这份审计确定当前 RLinf 的 GR00T N1.5 在旧 MUSA 栈上必须保留的 attention 行为，并提供不加载权重的小型数值、梯度和内存探针。当前结论支持先检查 SDPA/eager 替代接口；没有验证完整 GR00T 模型能加载或训练。本轮仅读取固定源码、下载小型公开源码文件及做本地静态检查，GPU 执行交给协调者串行安排。

## 固定来源与尚未确定的配置

| 来源 | 固定版本或 commit | 用途 |
|---|---|---|
| RLinf current worktree | `c70606f08cdca259b8dec03d4430926b5b8fac9d` | `worktrees/rlinf-route2` 中 N1.5 builder、RL action head、MUSA patch、安装依赖与 YAML |
| NVIDIA/Isaac-GR00T `n1.5-release` | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` | 本轮观察到的该分支 commit；仅读取所需模型源码 |
| Transformers 4.51.3 | `5f4ecf2d9f867a1255131d2461d75793c0cf1db2` | RLinf N1.5 固定依赖；Qwen2/Qwen3、Flash、SDPA、mask 实现 |
| Diffusers 0.30.2 | `f63c12633f154c2a1d79c17f4238fb073133652c` | RLinf N1.5 固定依赖；action DiT 的 attention processor |

公开文件在本地缓存于 `routes/route2/model_probes/audit_sources/`，版本控制保存其中的 `manifest.json`，记录原始 URL、完整 commit、SHA256 和本地 RLinf 文件 hash；下载源码缓存不进入成果包。下载传输使用 `gh-proxy.com`，来源仍为固定 GitHub raw URL；未执行下载源码，也未安装任何依赖或下载权重。RLinf installer 指定的是可移动的 `n1.5-release` 分支，因此这次外部快照不能代替未来安装的 dependency lock。

当前尚未读取用户最终选择 checkpoint 的 `config.json`。外部源码目录自带 Eagle 配置为 Qwen3（16 query heads、8 KV heads、head_dim 128）和 SigLIP vision（16 heads、1152 width，对应 head_dim 72）；该文件不能证明任务 checkpoint 使用相同 vision/config。RADIO 实现仍须审计，因为 Eagle 模型模块在顶层导入 RADIO，而且 RLinf 的 MUSA patch 专门处理 RADIO 的 Flash/capability 路径。

## 从 builder 到需要执行的算子

[RLinf N1.5 builder](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/gr00t/gr00t_n1d5/__init__.py#L85) 传入 `tune_visual=False`、`tune_llm=False`，模型 dtype 默认 BF16，之后再次 `model.to(torch_dtype)`。它还按配置替换 Dropout 为 Identity；默认配置的 `disable_dropout=True` 是 rollout/actor log-prob 一致性所需。SDPA 自身的 `dropout_p` 不会因 Module.eval 自动变为零，替换时必须显式使用 0。

[GR00T 的 trainable 设置](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/gr00t/model/backbone/eagle_backbone.py#L65) 先把 backbone 全部参数设为 trainable，再冻结 language_model、vision_model 和视觉 connector `mlp1`。**`eagle_linear`（默认 2048→1536）没有因此冻结。** 因此不能把“冻结视觉/LLM”简化为 backbone 全部无梯度；须在最终实例上导出 trainable 参数清单。Action projector、DiT 默认为可训练，action head 的 `vlln`、可选 `vl_self_attention` 也在 `process_backbone_output` 中执行。

| 路径 | 必需前向 | 当前训练所需反向 | 主要接口 |
|---|---|---|---|
| Frozen vision / Eagle LLM | 是，每次 `default_forward` 都重新调用 backbone | 冻结参数且普通输入不 requires_grad 时不需要这两部分参数梯度 | vision self-attention；LLM causal attention、GQA、RMSNorm、RoPE、GELU/SiLU、Linear |
| `eagle_linear` 投影 | 是 | 当前默认仍需要 | BF16 Linear 的输入与权重梯度 |
| Action `vl_self_attention` | checkpoint 启用该配置时需要 | 是，默认 trainable | 非 causal、无 mask 的 self-attention |
| Action DiT self/cross attention | 是，取决于 `interleave_self_attention` 层配置 | 是 | 非 causal self/cross-attention，Linear、LayerNorm/AdaLayerNorm、GELU、残差、布局变换 |
| State/action projector 与 decoder | 是 | 是 | embodiment 索引选择权重 + `bmm`、embedding、sin/cos、SiLU |
| RL value / Flow-SDE log-prob | 是 | 是 | value MLP、reduction、normal/noise、sqrt/log/div/where；这不因 attention 探针通过而自动受验证 |

冻结减少 optimizer 状态、梯度和 frozen transformer 的反向 activation 保存，但不省略权重、图像/text 预处理、attention 前向与其临时矩阵，也不等于缓存视觉特征。`output_hidden_states=True` 和完整序列的前向仍影响峰值内存。本轮不覆盖解冻视觉/LLM训练、KV cache、long-context、dropout训练或完整 FSDP。

## 三种 attention 的 mask、dtype 和布局

**Action DiT 当前没有使用 padding mask。** [RLinf `sample_mean_var_val`](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py#L162) 只传 `hidden_states=sa_embs`、`encoder_hidden_states=vl_embs`、timestep。外部 [DiT.forward](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/gr00t/model/action_head/cross_attention_dit.py#L282) 也明确给各层传 `attention_mask=None`、`encoder_attention_mask=None`；action transformer 的 block 不转发 encoder mask。虽然外部 SFT action forward 取得 `backbone_attention_mask`，这个 DiT 实现仍忽略它。给替代 action processor 增加 padding mask 会改变现有输出，不能作为无行为变化的兼容修复。

Diffusers 默认 [AttnProcessor2_0](https://github.com/huggingface/diffusers/blob/f63c12633f154c2a1d79c17f4238fb073133652c/src/diffusers/models/attention_processor.py#L2287) 对 Q/K/V 做 `[B,S,H*D] → [B,H,S,D]` 的 view/transpose，没有在该边界显式 `.contiguous()`，调用 `F.scaled_dot_product_attention(..., dropout_p=0.0, is_causal=False)`。输出再转回 `[B,S,H*D]`。默认 eager processor 使用 score softmax 与 `bmm`；`upcast_attention`/`upcast_softmax` 会改变累积 dtype。BF16 改为 FP32 score/softmax/value 累积是需要数值对照的决定，不能只比较输出 dtype。

**Eagle LLM 需要 causal 与 padding mask，且可能使用 GQA。** Transformers [Qwen3Attention](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3/modeling_qwen3.py#L170) 的 Q/K/V 为 BHSD，Q/K 做每 head RMSNorm 与 RoPE，KV heads 经 `repeat_kv` 按各 head 成组重复。SDPA integration 先使 Q/K/V contiguous，再用 scale 参数和 causal/mask。Flash interface 则转为 BSHD，并在 padding 路径 unpad/pad。SDPA bool mask **True 表示允许**；浮点 additive mask 的允许位置为 0，阻止位置为负无穷或 source 指定的最小 dtype 值，不能把整数 0/1 mask 直接当 additive mask。显式 composite causal+padding mask 时必须 `is_causal=False`，避免同时传两个 causal 来源。

RLinf rollout 把 `eagle_input_ids`/`eagle_attention_mask` 右侧补零到默认 `padding_value=570`。右 padding 是当前主要测试；左 padding 下，前几个 query 可能没有可见 key。Transformers 的 `_unmask_unattended` 设备条件包含 CUDA/XPU，不能假定 MUSA 也自动执行相同处理。探针另报告 fully-masked row 边界，默认不把这些不参与有效 token 输出的行当成当前右 padding 路径必需通过项。迁移 wrapper 必须明确 padding query 输出和选择有效 token 的规则，不能只用 `nan_to_num` 处理 loss。

**RADIO vision 使用 packed QKV，非 causal、无 padding mask。** [RADIO `_flash_attn`](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/gr00t/model/backbone/eagle2_hg_model/radio_model.py#L142) 从 `[B,N,3,H,D]` 拆分/置换 Q/K/V，可能先执行 q_norm/k_norm，然后输出 BSHD。其 Flash wrapper 使用 varlen packed API，并 assert `qkv.dtype in [fp16,bf16]` 与 `qkv.is_cuda`，外层 forward 又要求 BF16。N1.5 Eagle SigLIP 路径还硬设 vision `_attn_implementation="flash_attention_2"`；Qwen2 分支 assert Flash2，而当前目录的 Qwen3 分支没有该 assert。后续替换需处理各实际子模型，不是仅改顶层一个字符串。

## 当前 MUSA patch 不提供 SDPA fallback

[RLinf MUSA patch](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/gr00t/gr00t_n1d5/musa_patches.py#L35) 直接导入 vendor `flash_attn_func`、`flash_attn_varlen_func` 与 padding helper，给 Transformers 绑定 FA2，并把 availability 函数设为 True。该绑定在构建后持续生效，只有 CUDA capability sentinel 被 restore。当前旧环境缺少 vendor package，所以在模型构建阶段就会失败，不能把 availability 返回 True 当 kernel 支持。

RLinf 有通用 `resolve_attn_implementation` fallback utility，但本轮 N1.5 builder/MUSA patch 没有调用它。完整替代需要显式选择实现、处理 RADIO 的 import/forward，以及保持 Eagle 子模型和 action processor 的各自契约。当前 probe 记录 MUSA tensor `is_cuda`/`is_musa` 属性，但没有修改这些属性；RADIO 的 `is_cuda` assert 在现有运行栈是否直接阻断须以实测属性为准。

## 小型探针的执行与判定

`routes/route2/model_probes/attention_probe.py` 仅依赖已匹配的 Torch；`--device musa` 才导入 Torch-MUSA 并运行设备操作。没有 Transformers/Diffusers/GR00T 导入、权重加载或全局 patch。它验证候选数学算子，不验证实际 processor/module 的导入和所有参数。

每个数值 case 使用 CPU 生成相同输入，先量化到被测 FP32/BF16，再将这份量化输入转成 CPU FP64 参考。因此 error 分离 kernel 误差与输入量化误差。反向使用同一个确定性上游向量做 vector-Jacobian product，分别比较 Q/K/V 梯度；在 `--device musa` 下，还运行 CPU 同 dtype、同实现 peer 对照。算子周边的小 block 同时比较 Linear、category-selected bmm、RMSNorm、LayerNorm、GELU、SiLU 与手写 RoPE 的输出和参数梯度。

| 数值 case | 默认行为 | 对应边界 |
|---|---|---|
| action cross / action self / VL self | 无 mask、非 causal、前向 + Q/K/V 反向 | 当前 action head 必需 |
| Eagle GQA causal + right padding | bool composite mask，KV 成组重复；仅前向 | 冻结 LLM 必需 |
| vision packed QKV | head_dim 72，packed base 转到目标设备后再切 Q/K/V；仅前向 | 冻结 vision 的布局与 >64 head_dim |
| bool / additive key mask | 前向 + 反向 | fallback wrapper 的 mask 契约，不能据此声称 action 当前使用 mask |
| causal left padding / empty rows | 单独标为 edge，输出有限性与零输出参考 | 非当前主要路径，失败单独呈现 |

实现候选包括 SDPA contiguous、SDPA source layout、dtype 原生 eager、Qwen-like FP32 softmax eager 和全 FP32 累积 eager。`--sdpa-backend math` 通过 shared ATen SDP context 禁用 flash/efficient，只请求 math 路径；源码 v1.3 的 backend selector读取这些 context 标志，但 probe 未用 profiler确认最终 kernel，不能把 SDPA 成功称为 vendor FlashAttention 成功。

预设门槛：FP32 `atol=rtol=relative_L2_limit=2e-4`；BF16 `atol=0.025`、`rtol=0.04`、relative L2 ≤0.04，前向和每个梯度都须有限、满足 allclose 与相对 L2。门槛是在运行前声明的算子 gate，属于小随机输入范围，不代表模型任务容忍误差。required case 任一失败脚本非零退出；edge case 仍保存失败，不掩盖 NaN。

协调者先上传脚本到 route-owned 目录，再以隔离解释器依次运行：

```bash
ROUTE2_PYTHON=/root/autodl-tmp/s4000-research/envs/route2/bin/python
ROUTE2_MODELS=/root/autodl-tmp/s4000-research/route2/model_probes

# 先 CPU，只有该阶段输出可信，才进入设备对照。
$ROUTE2_PYTHON "$ROUTE2_MODELS/attention_probe.py" \
  --device cpu --phase all --profile small \
  --output "$ROUTE2_MODELS/cpu-attention.json"

# 由协调者按单卡序列执行。先自动 dispatch，再隔离 math 请求。
$ROUTE2_PYTHON "$ROUTE2_MODELS/attention_probe.py" \
  --device musa --phase numerics --profile small \
  --output "$ROUTE2_MODELS/musa-attention-auto.json"
$ROUTE2_PYTHON "$ROUTE2_MODELS/attention_probe.py" \
  --device musa --phase numerics --profile small --sdpa-backend math \
  --kinds sdpa,sdpa_layout,eager_fp32 \
  --output "$ROUTE2_MODELS/musa-attention-math.json"
$ROUTE2_PYTHON "$ROUTE2_MODELS/attention_probe.py" \
  --device musa --phase primitives \
  --output "$ROUTE2_MODELS/musa-bf16-primitives.json"

# 数值 gate 通过后，再测较大单层 geometry 的实际 allocator 峰值。
$ROUTE2_PYTHON "$ROUTE2_MODELS/attention_probe.py" \
  --device musa --phase memory --profile feature-570 \
  --kinds sdpa,eager_fp32 \
  --cases action_cross_unmasked,vl_self_unmasked,eagle_gqa_causal_rightpad,vision_packed_unmasked \
  --output "$ROUTE2_MODELS/musa-attention-memory.json"
```

`feature-570` 是显式假设的单层形状：B=1，action Q=49（假设 state 1 + future 32 + action 16），KV=570，action H=8/D=64（源码构造默认值），LLM H=16/KV=8/D=128，vision S=256/H=16/D=72。它们不是最终 checkpoint 的配置，不应用来报告 GR00T 全模型显存。较大或实际 checkpoint 几何应在读取配置后调整并记录，脚本默认拒绝单个 FP32 score 矩阵超过 128 MiB。

内存结果报告目标 MUSA allocator 的 baseline、peak allocated、peak reserved 及增量，包含本 kernel 输入、输出和可选反向；不混淆 reserved 与 live allocated。CPU 或缺失 peak API 时只报告理论 score 矩阵字节，明确 `theoretical_only`，不把进程 RSS 高水位伪装成 kernel 内存。eager FP32 至少显式生成 `[B,H,Q,K]` score/probability，层数、batch、图像 tiles 与 sequence 增大时呈平方或交叉乘积增长；完整模型仍需另测。

## 本轮结果与下一步

本轮静态结果：全部下载 Python 源码和 probe AST 解析通过、文件 SHA256 校验通过，`attention_probe.py --help` 可运行。Mac 无 Torch，未在本轮子任务运行 CPU 数值或 GPU；数值与 allocator 结果由协调者执行后保存 JSON，不能预填“通过”。

若仅 eager FP32 通过而 SDPA 失败，先保留失败 case 的 dtype、mask、布局、错误与可用 fallback；若 BF16 前向通过但 action 梯度失败，不能采用 BF16 action 训练路径。只有 required kernel gate 与单层内存检查通过，才继续安装 **no-deps** 的固定外部源码、以随机小配置实例化实际 Diffusers processor 与 Eagle/RADIO wrapper，验证真实 mask/preprocessing/参数梯度。最终 checkpoint 配置、BF16 normal/noise/Flow-SDE log-prob、processor integration、模型加载与 trainable 参数清单仍是独立 gate。冻结视觉/LLM的结论不得外推到解冻训练或 GR00T 任务学习效果。
