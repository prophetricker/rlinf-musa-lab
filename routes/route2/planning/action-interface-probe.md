# R2-3b：真实 GR00T action 接口的最小验证

本节点把已检查的 FP32 attention 数学路径接到真实 Diffusers 0.30.2 `Attention` 和固定 GR00T 的双层 `DiT`，用随机权重验证输出、输入 VJP 与全部 trainable 参数 VJP。它不加载 checkpoint、不构造 Eagle、不运行 optimizer，也不声称完成训练或学习。实现和本地静态检查由 route2 完成；隔离依赖、CPU 与单卡 MUSA 执行由协调者安排。

## 文件、固定来源与依赖

| 文件 | SHA256 |
|---|---|
| 当前 `model_probes/fp32_attention_processor.py` | `0edcb656696bcd3a47e014c3d549cf029a268c4a2f3f9694bea9993bd9d53a6f` |
| 当前 `model_probes/gr00t_action_interface_probe.py` | `c1a0e36336cbefc5499ffa8bb7c1f38bfe2384f64505c55f7a976d8dad3645f4` |
| 未修改 `model_probes/attention_probe.py`，只复用门槛/同步 helper | `704b68ef1e7bcde4b0578519f6d9c2c0a51298fd8a5e27527754718ce693b775` |
| Diffusers 0.30.2 `attention_processor.py` | `afe47ff6d33c865af745e28eed4bd8048b66fd52de7c2b3bad37867e826776c9` |
| 固定 GR00T `cross_attention_dit.py` | `1366e3f80c0e076953965eeb43d800cb354fdcdf681ab54f0c2ad8f1d6c9fe62` |

Diffusers 来源 commit `f63c12633f154c2a1d79c17f4238fb073133652c`，GR00T 来源 commit `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`，完整原始 URL 见 `audit_sources/manifest.json`。runner 对已安装 Diffusers 的版本和 `inspect.getsourcefile(Attention)` 所指文件执行严格 hash 检查；DiT 阶段对直接导入的源码也严格检查。源码不匹配或缺依赖时保存 `setup_error` 并非零退出，不会静默改用替代 module。

最小新增模型依赖是 **Diffusers 0.30.2**；Torch 与 Torch-MUSA 必须继续使用已匹配的旧栈。NumPy 用于 tensor 原始字节 hash，Diffusers 自身 import 所需的 Hub、packaging、Pillow、safetensors、requests、filelock、regex、importlib-metadata 等复用隔离环境可见的既有依赖，runner 记录这些 distribution 的实际版本。协调者已报告 `/root/autodl-tmp/s4000-research/envs/route2-models/bin/python` 中以 no-deps 安装的 Diffusers 0.30.2 能导入真实 Attention、FeedForward、Timesteps、TimestepEmbedding 与 positional embedding，且上述 source hash 匹配；**现有 Hub 1.33.0 的实际 import 已通过，不需要为了本节点降级**。本报告不将这个 import 结果扩大为未来 Transformers 的依赖兼容结论。

固定 `cross_attention_dit.py` 只有 Torch、Diffusers、typing 导入，没有 GR00T 相对导入，因此用 `importlib.util.spec_from_file_location` 按固定文件直接加载即可。此节点不需要安装整个 Isaac-GR00T、Transformers、timm、flash_attn、FA vendor package 或额外图像预处理包。两个 offline 环境变量只在 runner 自身进程设默认值；从不调用 `from_pretrained` 或 Hub 下载 API。

## processor 的实际语义

`FP32AttentionProcessor` 是普通 Python callable，经 **每个实例** 的 `set_processor` 绑定，不修改 Diffusers 类、全局 SDPA、`is_cuda` 属性或 vendor availability。它沿用 `AttnProcessor2_0` 的次序：spatial norm → 可选4D展平 → source prepare mask → group norm → Q projection → source encoder cross norm → K/V projection → BHSD transpose → q/k norm → attention → BSHD reshape → output projection/dropout → 恢复4D → residual → rescale。

只有 kernel 使用 FP32 QK、softmax、PV 累积，随后 cast 回 query 的 FP32/BF16 dtype。外层 projection/norm/parameter 保持原 dtype。Q/K/V 没有被整体提前改为 contiguous；trace 保存实际 shape、stride 和 dtype，便于确认真实 processor 的 transpose 边界。kernel 与 source 相同使用默认 `1/sqrt(D)`、non-causal、dropout=0；output dropout module 保留，本探针 config 将 dropout 设0且 eval，避免随机差异。

范围限定为普通 MHA、default scale、常规 output projection 和 per-head Q/K norm。GQA/MQA、added KV、scale_qk=False、pre_only/context_pre_only、across-head normalization 都会明确拒绝。这里没有把 action processor 用作 Eagle causal/GQA wrapper。

mask API 输入使用 source 接受的 `[B,1,K]`，source `prepare_attention_mask` repeat heads 后 reshape 为 `[B,H,1,K]`；FP32 eager 按 query 轴 broadcast，bool True 表示允许，additive 为0/-inf。没有新增 mask 到 DiT baseline。仅对 mask **明确** 全false或全-inf的行定义 kernel zero context；外层 output bias/residual 仍保留，因此不能把 kernel zero context 等同于整个 module 输出为零。required masks 每行非空，empty-row的实际 module 行为不是本节点 gate。Q/K/V、未mask score 的非有限值及 NaN/+inf mask 会直接失败，不以 empty-row逻辑隐藏坏值；这些诊断增加同步，当前 runner 不用于性能测量。

## cases 与比较口径

| case | 实际 module/config | 输入与额外检查 |
|---|---|---|
| `attention_self` | Attention，query_dim256，H4/D64，QKV/out bias=True | B2/Q17；hidden与全部参数VJP |
| `attention_cross` | 同上，cross_attention_dim1536 | B2/Q17/K31；hidden/encoder与全部参数VJP |
| `processor_mask_bool` | 同一个 cross config | `[B,1,K]` bool；扰动被屏蔽encoder，不影响输出/所有VJP；被屏蔽encoder梯度严格0 |
| `processor_mask_additive` | 同一个 cross config与相同seed | 等值0/-inf mask；相同独立契约 |
| `processor_norms_4d_residual` | cross；spatial_dim32、group_norm32组、cross LN、per-head q/k LN、residual=True、rescale1.5 | hidden `[2,256,1,17]`，temb `[2,32,1,5]`，encoder `[2,31,1536]`；包括所有norm/Conv参数和temb VJP |
| `gr00t_dit_two_layers` | 直接导入固定源码；H4/D64、2层、cross_dim1536、interleave_self_attention=True、ada_norm、gelu-approximate、dropout0/final_dropout=False | B2/Q17/K31、timestep=[3,11]；第0层cross、第1层self、output_dim source默认26；encoder_mask=None |

width1536 是 source action feature 默认值，随机 config 不代替最终 checkpoint。DiT source 的所有其余 constructor defaults 保存为 `effective_config`，包括 timestep与sinusoidal计算。source 模块、AdaLayerNorm、timestep encoder、GELU feed-forward、residual、output projection 均直接执行；没有复制一个“类 DiT”网络。

每个 case/dtype 先在 CPU 初始化 FP32 权重并量化为 FP32/BF16，从这一个 template deepcopy，所有执行共享同一份初态、输入与量化后的上游向量。seed_offset固定于 case，过滤 phase/cases 不改变结果；bool/additive刻意使用与cross相同seed。每个初态参数和 buffer、输入、上游、mask、timestep都保存 shape/dtype/hash；每次运行检查 state hash仍一致，记录全部参数名称、requires_grad、dtype、device和缺梯度。

对照依次为 CPU 原 `AttnProcessor2_0`、CPU 新 processor，以及显式 `--device musa` 时的 MUSA 新 processor。baseline所有输出/梯度必须有限且不存在缺梯度；CPUfallback/MUSAfallback分别比较native CPU的输出、每个可微输入与**每个** trainable参数VJP；MUSA还比较CPUfallback。此节点的reference是 **native CPU实际module**，不是CPU FP64，不与前一轮算子probe的FP64参考混淆。

数值 gate复用未改的 `error_metrics`：FP32 atol=rtol=relative_L2_limit=`2e-4`；BF16 atol=`0.025`、rtol=`0.04`、relative L2 limit=`0.04`。每个 tensor都需要finite、allclose与相对L2同时通过，未设置新的全模块容忍误差。relative L2保留原 `expected.norm().clamp_min(1e-12)`，并额外保存actual/expected L2，便于识别近零参考。汇总遇到任何运行异常都会明确 `pass=False`，保存异常、全部已得到的错误项和非零exit。

**运行前声明的零梯度解释：** 无K norm的Attention共享 `to_k.bias` 对所有可见key的logit产生相同平移，softmax数学上不变，因此该参数的解析梯度应为0。norm case中per-head `norm_k.bias` 在qk dot之前作共享平移，也有相同解析零梯度。有限精度取消噪声可能令两份近零梯度的relative L2很大，不能只凭该字段归为普通参数反向偏差。runner预先单列 `analytic_zero_parameter_diagnostics` 的名称、原因、max absolute与L2，但这些参数**仍参与原逐参数 gate**，没有排除、抹零、放宽或代替总pass。若因此失败，必须同时报告具体名称、绝对误差和其它输入/参数的原门槛结果。

## 协调者执行

上传三个本地Python文件到同一model_probes目录，DiT源码可保留默认缓存相对目录，或通过`--gr00t-source`指定真实文件。以下只运行随机小配置：

```bash
ROUTE2_MODEL_PYTHON=/root/autodl-tmp/s4000-research/envs/route2-models/bin/python
ROUTE2_MODELS=/root/autodl-tmp/s4000-research/route2/model_probes
ROUTE2_DIT_SOURCE="$ROUTE2_MODELS/audit_sources/Isaac-GR00T/gr00t/model/action_head/cross_attention_dit.py"

$ROUTE2_MODEL_PYTHON "$ROUTE2_MODELS/gr00t_action_interface_probe.py" \
  --device cpu --phase attention --output results/action-interface-cpu-attention.json
$ROUTE2_MODEL_PYTHON "$ROUTE2_MODELS/gr00t_action_interface_probe.py" \
  --device cpu --phase dit --gr00t-source "$ROUTE2_DIT_SOURCE" \
  --output results/action-interface-cpu-dit.json

$ROUTE2_MODEL_PYTHON "$ROUTE2_MODELS/gr00t_action_interface_probe.py" \
  --device musa --phase attention --output results/action-interface-musa-attention.json
$ROUTE2_MODEL_PYTHON "$ROUTE2_MODELS/gr00t_action_interface_probe.py" \
  --device musa --phase dit --gr00t-source "$ROUTE2_DIT_SOURCE" \
  --output results/action-interface-musa-dit.json
```

`--phase attention`默认只测前5 cases，CPU20 rows、MUSA30 rows；`--phase dit`为CPU4 rows、MUSA6 rows；`--phase all`可合并。每个MUSA运行仍带完整CPU baseline/fallback，不需要把不同进程的CPU输出拼成比较。`--cases`可选comma-separated子集，仅用于定位并保留该次完整arguments；不得把过滤后的通过代替全范围gate。

本地Mac没有Torch，仅完成两份源码AST和runner `--help`。另有route1代理只读审查，已核对norm顺序、mask入口、全部parameter/temb VJP、非有限值处理和异常汇总；没有本地数值通过结论，也没有安装、远程、GPU或Git操作。协调者的首次实测与随后修订记录如下，失败原样保存。

## 首次实测：CPU算完，原gate仍失败；MUSA被guard算子阻断

| 证据（`evidence/action-interface/`） | SHA256 |
|---|---|
| `action-interface-cpu-first.json` | `385bcfdeb5d969f46481765cc471e8b36cb2eab00311e22399d8a9b7e6a6f134` |
| `action-interface-musa-attention-first.json` | `250c509c09cc53827ca833fd7e2551cafa4962b3b1870283e36520a01ac9e6ea` |

这两份JSON使用首版processor `324a31fc3924c1df6f5c7d10d204990bb91e73f1d0a457b661764299f6c200b3`、首版runner `8477a503…`。首版两个文件及未改的helper已逐字保存在 `model_probes/history/action-interface-v1/`。当前 processor 修订只把正/负inf检查换成相等/不等标量比较；runner 随后增加了零梯度 semantic diagnostics，当前 hash 单独列在表内，未改变模型数学、原始严格 gate 或 final evidence 的已执行 source hash。final JSON 的 `source_sha256` 记录当时 runner 为 `46ab074d…`，与当前 runner 的后续 diagnostics-only 更新区分保存。

CPU `--phase all` 的24 rows都完成计算、有限、没有缺梯度，初态/运行后state hash全匹配。12个nativeCPU baseline rows通过；12个CPUfallback rows严格gate全失败。每个fallback输出和每个输入VJP都通过原门槛；参数失败**只**来自运行前标出的解析零方向：普通case的 `to_k.bias`、normcase的 `norm_k.bias`、DiT的两层 `attn1.to_k.bias`。其余参数共166项比较全部通过原逐tensor门槛（含重复mask cases，不是166个独立模型），不能据此重写总gate为pass。

| CPUfallback对nativeCPU | FP32输出relative L2 / 最大非解析零参数relative L2 | BF16输出relative L2 / 最大非解析零参数relative L2 | 失败参数 |
|---|---|---|---|
| self | 1.391862e-7 / 2.240566e-7 | 2.705147e-3 / 3.665406e-3 | `to_k.bias` |
| cross | 1.255153e-7 / 2.163284e-7 | 2.692880e-3 / 3.572778e-3 | `to_k.bias` |
| bool/additive mask（同结果） | 1.234354e-7 / 1.924045e-7 | 2.834712e-3 / 4.981558e-3 | `to_k.bias` |
| 4D norms/residual | 4.185463e-8 / 1.186271e-6 | 1.495831e-3 / 1.648805e-2 | `norm_k.bias` |
| 双层DiT | 3.626374e-7 / 6.453843e-7 | 4.155918e-3 / 8.367473e-3 | 两个 `attn1.to_k.bias` |

FP32解析零参数的delta max absolute范围为 `4.074536e-10–7.654307e-9`，relative L2约1.05–1.22；nativeCPU参考L2本身仅 `1.741781e-9–3.303410e-8`。BF16该方向delta max absolute为 `1.013279e-5–1.525879e-4`，参考L2为 `4.364976e-5–4.974055e-4`，relative L2约1.02–1.14。所有absolute allclose都通过，只在原relative-L2门槛失败；完整actual/expected L2保留在JSON。例如DiT两层BF16的delta max absolute分别为 `3.182888e-5`、`1.113415e-4`，normcase BF16为 `1.525879e-4`。这些是解析零方向上的有限精度结果，不能无条件声称误差无害或训练等价，下一步解释必须保留该差异而不是抹零。

独立mask契约在CPU native/fallback、FP32/BF16都通过：被屏蔽encoder的梯度严格0，扰动后输出、所有输入及参数VJP的最大absolute差都严格0，bool与additive结果一致。该mask仍未接入DiT baseline。

MUSA首次attention run共有30 rows：10个CPU baseline pass、10个CPUfallback解析零参数gate失败、**10个MUSAfallback运行错误**，均为 `NotImplementedError: aten::isposinf.out` 没有MUSA实现。因此这份first证据没有MUSA输出或参数VJP，不能把CPU结果外推为MUSA成功。当前修订改用 `(x == float("inf"))` 检查+inf、`(mask != float("-inf"))`检查允许位置，仍单独检查NaN、QKV与未mask scores finite；对FP32/BF16同等数学判定。首轮没有走到 `isneginf`，它是否支持未被实测，但新版本也移除了该依赖。协调者将按新processor hash重测，runner与门槛不变。

## 修订版 MUSA attention 实测

协调者随后以 processor hash `0edcb656…` 与 runner hash `c1a0e363…` 完成 `action-interface-musa-semantic.json`（SHA256 `7aa6d33cbe8d99dfd5a1d51531d4e82210968aad9c3cb14e0cb54bd899c0bf39`）。它包含 36 rows：6 cases × 2 dtypes × native CPU、CPU fallback、MUSA fallback；没有 setup error 或运行异常。MUSA 的所有 12 rows（5 Attention cases × 2 dtype 加双层 DiT × 2 dtype）均完成 forward 与 VJP，state hash 匹配，所有输出、输入梯度和参数梯度 finite。

原始逐参数 gate 仍为 0/12，因为同一组解析零方向参数的 relative-L2 不稳定：普通 self/cross/mask cases 是 `to_k.bias`，norm case 是 `norm_k.bias`，双层 DiT 是两层 `attn1.to_k.bias`。除这些零方向外，MUSA 对 native CPU 和 CPU fallback 的每个输出、输入 VJP、其它参数 VJP 都通过原 FP32/BF16门槛；对应 `native_cpu_semantic_comparison.semantic_pass` 与 `cpu_fallback_semantic_comparison.semantic_pass` 全部为 true。当前 runner 另外把每个零方向的 actual/reference 梯度都要求 finite 且 `max_abs <= atol`（FP32 `2e-4`、BF16 `0.025`），并在顶层记录 `semantic_pass` 与 `semantic_interfaces_pass=true`。该 semantic 字段是诊断视图，未替换或放宽原始 strict gate。

| MUSA case | dtype | MUSA/native output relative L2 | MUSA/CPU-fallback output relative L2 | 原始失败项 | semantic pass |
|---|---|---:|---:|---|---|
| self | FP32 | 1.717917e-7 | 1.760782e-7 | `to_k.bias` | true |
| self | BF16 | 2.713150e-3 | 3.010336e-4 | `to_k.bias` | true |
| cross | FP32 | 5.212361e-7 | 5.210031e-7 | `to_k.bias` | true |
| cross | BF16 | 2.707085e-3 | 6.969887e-4 | `to_k.bias` | true |
| bool/additive mask | FP32 | 5.375920e-7 | 5.375269e-7 | `to_k.bias` | true |
| bool/additive mask | BF16 | 2.832277e-3 | 7.639566e-4 | `to_k.bias` | true |
| 4D norms/residual | FP32 | 2.084581e-7 | 2.090671e-7 | `norm_k.bias` | true |
| 4D norms/residual | BF16 | 1.495728e-3 | 3.178989e-4 | `norm_k.bias` | true |
| two-layer GR00T DiT | FP32 | 4.838800e-7 | 4.834609e-7 | two `attn1.to_k.bias` | true |
| two-layer GR00T DiT | BF16 | 4.817118e-3 | 4.568863e-3 | two `attn1.to_k.bias` | true |

The BF16 output comparisons remain below the unchanged `0.04` relative-L2 gate. The actual MUSA result is therefore “the real module path executed and passed all meaningful nonzero VJPs, with strict aggregate status false only for separately diagnosed zero-gradient directions”; it is not a blanket claim that the original all-parameter gate passed. The first MUSA file remains historical evidence of the missing `aten::isposinf.out` guard and is not merged with this final result.

`action-interface-musa-final.json` 与 `action-interface-musa-second.json` 使用的 `46ab074d…` runner 已逐字归档为 `model_probes/history/action-interface-v2/`；当前 `c1a0e363…` 仅新增零方向 finite/绝对值 gate 与明确汇总，并对完整范围重新实测。后续可单独做 bias 扰动不变性、真实 checkpoint 和训练更新；这些不在本节点的通过范围内。
