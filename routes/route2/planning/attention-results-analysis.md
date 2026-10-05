# R2-3 attention 数值结果分析（2026-10-05）

现有证据支持把 `eager_fp32` 作为后续实际 GR00T module 集成的候选：小形状的 CPU/MUSA FP32、BF16 前向及所需 Q/K/V 梯度全部通过原门槛，额外 fully-masked-row 前向也通过。**当前 MUSA SDPA 不能按原 gate 宣告完整通过**：`B×1×1×K` key mask 在 muDNN 前向直接报错；三个无 mask 训练形状的 FP32 梯度相对 L2 超过 `2e-4`。追加 auto 与 math 诊断均确认显式展开 query 轴后可运行，屏蔽契约成立、BF16 全部通过，但 FP32 梯度仍超过原门槛。这两类失败必须分开，不能把梯度误差描述成算子不可运行，也不能把 mask 展开后可运行描述成数值 gate 已通过。

本分析仅读取协调者已保存的 JSON 和固定源码，并追加独立的 mask 表示诊断脚本。没有修改原探针、放宽门槛、安装软件、运行 GPU 或操作 Git。完整模型导入、processor、checkpoint、Flow-SDE 与训练仍须各自验证。

## 证据与版本边界

运行 JSON 位于 `routes/route2/evidence/attention/`，均记录 Torch runtime `2.2.0`、Linux 与 MUSA device。环境锁区分 Torch distribution `2.2.0a0+git8ac9b20`、Torch-MUSA distribution `1.3.0` 与 runtime `1.3.0+81caf0a`。公开 Torch-MUSA v1.3.0 源码 commit `73c9f5ba21dc6a0dc0ed07ef027c804dc53644a9` 未被证明与已安装 `+81caf0a` 是同一个构建；以下源码定位只能提供可检验假设。

| 文件 | SHA256 |
|---|---|
| `auto.json` | `76e633346a672255c41b47e7ef9f48082e51222ed5d382f667a45b088e69ed23` |
| `math.json` | `46bc1f5059766b2e603e17991689defc14083b1499a3db07495810e4b95754ac` |
| `fallback.json` | `6aeca103cc97ba864e502a28df5c6c847ff38b57f6900bd4730e7991445e83b1` |
| `primitives.json` | `c2a39446204252fedefd93c3f6f07e704409759b369f692af9339b9115966227` |
| `memory.json` | `f8035db7fc8fb0a9d44dfeec3938010fb5f0a10d9f77d6bcf3ca5a286a5e872b` |
| `mask-expand-auto.json` | `507f043e0487be7a882f57e77153a327f1ea9b040b8d788d28baa81ba496bb7d` |
| `mask-expand-math.json` | `7b3153ebf039a8953729466a89d324fc5855693f9d7b9d6dd5add986e5aef823` |
| 未修改 `model_probes/attention_probe.py` | `704b68ef1e7bcde4b0578519f6d9c2c0a51298fd8a5e27527754718ce693b775` |
| `model_probes/mask_contract_probe.py` | `403e34af884d5e65c69e526ca7ab4b24ad19355dea9b39d678b83a4b0e55762b` |

GR00T 契约所依据的固定版本沿用 `attention-audit.md` 与 `model_probes/audit_sources/manifest.json`：RLinf upstream `c70606f08cdca259b8dec03d4430926b5b8fac9d`，route2 本地兼容 commit `3e7329c629293f1b0f9c331b5bb24eccc3e3f142`，Isaac-GR00T `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`，Transformers 4.51.3 `5f4ecf2d9f867a1255131d2461d75793c0cf1db2`，Diffusers 0.30.2 `f63c12633f154c2a1d79c17f4238fb073133652c`。外部 GR00T 分支快照与最终 checkpoint 配置不能混为一谈。

## gate 的统计口径

输入先量化为被测 FP32/BF16，再使用同一份输入和确定性上游向量计算 CPU FP64 参考；因此比较的是量化之后的 kernel 差异。每个反向 case 比较 Q、K、V 三份 VJP。原 FP32 gate 是 `atol=rtol=relative_L2_limit=2e-4`；BF16 是 `atol=0.025`、`rtol=0.04`、relative L2 ≤0.04。finite、allclose、相对 L2 均须通过，本分析保持原门槛。

| 运行 | 总 rows | required rows | required 通过 | required 运行报错 | required 数值失败 | excluded edge rows |
|---|---:|---:|---:|---:|---:|---:|
| auto：5 个实现，CPU + MUSA | 160 | 140 | 126 | 8 | 6 | 20 |
| math：SDPA、SDPA layout、eager FP32，CPU + MUSA | 96 | 84 | 70 | 8 | 6 | 12 |
| fallback：仅 eager FP32，CPU + MUSA | 32 | 28 | 28 | 0 | 0 | 4 |
| primitives：CPU/MUSA × FP32/BF16 | 4 | 4 | 4 | 0 | 0 | 0 |
| mask expand auto：SDPA/eager FP32，CPU + MUSA | 64 | 64 | 54 | 4 | 6 | 0 |
| mask expand math：SDPA/eager FP32，CPU + MUSA | 64 | 64 | 54 | 4 | 6 | 0 |

原始 auto/math 中所有 required 失败均来自 MUSA 的两个 SDPA 变体；追加 mask expand 也只有 MUSA SDPA rows 失败。CPU 所有 required rows 都通过。上表同时计入不同实现对相同 geometry 的重复测试，不是独立任务数量。memory 的 8 rows 没有数值比较及 `pass` 字段，`required_numerics_pass=null`；它们不能计为 8 个 numerical pass。

在 auto 下，每个 MUSA SDPA 变体有 16 rows：7 通过、4 个 key-mask 前向错误、3 个 FP32 梯度 gate 失败、2 个 excluded left-padding edge 失败。CPU SDPA 每个变体有 14 required pass 和 2 edge fail。CPU/MUSA 的 eager native、eager Qwen-like 各通过所有 14 required rows，只在 2 个 fully-masked edge rows 出现非有限值；eager FP32 的 16 rows 全部通过。math 请求保持同样的 required 分类。

## FP32 前向通过，反向严格门槛未通过

MUSA 的 SDPA contiguous 和 source-layout 结果相同；强制 math 的这三个 FP32 case 也与 auto 相同。所有输出和梯度有限，没有 RuntimeError。前向最大绝对误差低于 `9.35e-8`，relative L2 约 `1.06e-7–1.28e-7`，均通过。

| MUSA SDPA FP32 case | Q 梯度 relative L2 | K 梯度 relative L2 | V 梯度 relative L2 | 三份梯度最大 absolute error |
|---|---:|---:|---:|---:|
| action cross，无 mask | 4.098853e-4 | 4.124586e-4 | 2.975063e-4 | 5.799969e-6 |
| action self，无 mask | 4.150185e-4 | 4.106296e-4 | 2.791569e-4 | 9.054073e-6 |
| VL self，无 mask | 4.157848e-4 | 4.239384e-4 | 2.971838e-4 | 6.643921e-6 |

失败原因是相对 L2 大于 `2e-4`，不是 absolute allclose 超限。整体梯度最大绝对误差是 **`9.054073160427029e-6`**，不能概括为“全部小于 6e-6”。误差在当前随机单次输入下很小，但既定训练算子 gate 未通过；没有多 seed、真实参数梯度或学习曲线证据可据以放宽。

**`cpu_peer_gradient_errors` 是 MUSA-vs-CPU，不是 CPU-vs-FP64。** 原脚本在 MUSA row 中用 `actual` 与此前 CPU 的同 dtype、同实现 peer 比较。CPU 的自身参考误差应读取 `device="cpu"` row 的 `gradient_errors`。例如 auto 的 action cross FP32 SDPA，CPU Q/K/V relative L2 分别为 `2.297114e-7`、`2.234429e-7`、`1.607507e-7`，最大绝对误差分别为 `2.190197e-9`、`2.386154e-9`、`3.872290e-9`，全部通过。MUSA-vs-CPU 误差接近 MUSA-vs-FP64 正是因为 CPU 很接近参考。已有数据不支持“导入 Torch-MUSA 使 CPU peer 产生同样 3e-4 误差”的说法，也没有做导入前后因果实验。

公开 v1.3 [SDP.cpp](https://github.com/MooreThreads/torch_musa/blob/73c9f5ba21dc6a0dc0ed07ef027c804dc53644a9/torch_musa/csrc/aten/ops/attention/mudnn/SDP.cpp#L118) 的 math forward 明确调用 `SetComputeMode(GetComputeModeFromCtx(dtype))`，math backward 在 L159–237 创建新的 muDNN attention 对象后只设置 embed dim、heads、training，再调用 `RunMathBwd`，没有对应 `SetComputeMode`。`aten/utils/Context.cpp` L24–29 按 allow-TF32 或 Half/BF16 选择 TENSOR，否则 SCALAR；`Context.h` L35–36 的默认 allow-TF32 为 false。这是前后向精度配置可能不同的定位线索，**尚未证明是当前安装栈梯度误差的根因**。本轮没有改变任何 TF32 或 compute-mode 默认值。

## key-mask 形状与 muDNN 运行错误

bool 与 additive key mask 的原形状均为 `[2,1,1,31]`；Q/K/V 为 B=2、H=4、Q=17、K=31、D=64。bool 的 True 表示可见，additive 的可见位置为 0、不可见位置为 -inf；每个 query 都至少有一个可见 key。

CPU 两种 dtype、两种 SDPA layout 都通过。MUSA 两种 dtype、两种 layout 均失败，错误为：

- FP32 auto/math：`RuntimeError: MuDNNMathSDPAFwd MUDNN failed in: Run SDPA`。
- BF16 auto：`RuntimeError: MuDNNFlashSDPAFwd MUDNN failed in: Run SDPA`。
- BF16 math：`RuntimeError: MuDNNMathSDPAFwd MUDNN failed in: Run SDPA`。

强制 math 将 BF16 错误名从 Flash 改为 Math，说明该请求确实改变了这条失败路径；仍没有 profiler 证据可描述全部成功 case 的内部 dispatch。这里的 muDNN Flash 路径也不等于缺失的独立 vendor `flash_attn` Python package。

Eagle causal+right-padding 使用完整 Q 轴的 `[B,1,Q,K]` composite mask，在同一运行中通过 FP32/BF16 前向。结合 key-mask 失败，不能写成“MUSA 不支持任何 mask”。公开 [SDP.cpp](https://github.com/MooreThreads/torch_musa/blob/73c9f5ba21dc6a0dc0ed07ef027c804dc53644a9/torch_musa/csrc/aten/ops/attention/mudnn/SDP.cpp#L80) math L80–85、flash L284–289 在送入 muDNN 前只做 `.contiguous()`，没有显式把 mask broadcast 为 Q/K geometry。L25–27 的 `is_pad_mask` 还只识别二维 mask，且第二维与 query 的 Q 长度比较。四维 `[B,1,1,K]` 不走该 special pad-mask 模式。追加实验已将失败定位到这个被测 key-mask 的 query 广播表示：相同内容只展开 Q 轴即消除运行错误。muDNN 的内部实现和安装栈源码一致性仍未核对，因此不能把它外推为所有形状的根因或完整 SDPA 修复。

新增 `model_probes/mask_contract_probe.py` 与原探针同目录，复用未修改的输入、参考与误差函数，默认 CPU。它保持 B/H/Q/K/D、seed、dtype 与 mask 内容不变，只在目标设备上构造四种表示：

| 变体 | 形状 | 目的 |
|---|---|---|
| `B11K` | `[2,1,1,31]` | 保留原广播形状与原失败 |
| `B1QK_contiguous` | `[2,1,17,31]` | 展开 query 轴，仍广播 heads |
| `BHQK_expand_view` | `[2,4,17,31]`，expand view | 全形状但保留零 stride；在目标设备上展开，避免 transfer 预先物化 |
| `BHQK_contiguous` | `[2,4,17,31]`，dense contiguous | 全形状并显式物化 |

各表示比较 CPU FP64 前向、Q/K/V VJP、CPU 同实现 peer，并以 eager FP32 作控制；把被屏蔽 K/V 改为大幅随机值后再次检查输出/VJP不变，并要求被屏蔽 K/V 的梯度为数学上的零。原形若成功，还保存 expanded-vs-original 数值对照。失败 row 保留 mask shape/stride、dtype 与异常。`all_variants_pass`、`expanded_variants_pass` 分别汇总，但不会把原失败隐藏；任一 row 失败则非零退出，全部 JSON 仍写入。

本地已完成 AST 与 `--help` 检查，Mac 无 Torch，未做本地数值执行。协调者已完成 auto 与 math 并保存证据。本探针执行命令如下（脚本依赖同目录的原 `attention_probe.py`）：

```bash
ROUTE2_PYTHON=/root/autodl-tmp/s4000-research/envs/route2/bin/python
ROUTE2_MODELS=/root/autodl-tmp/s4000-research/route2/model_probes

$ROUTE2_PYTHON "$ROUTE2_MODELS/mask_contract_probe.py" \
  --device musa --sdpa-backend auto \
  --output results/attention/mask-expand-auto.json
$ROUTE2_PYTHON "$ROUTE2_MODELS/mask_contract_probe.py" \
  --device musa --sdpa-backend math \
  --output results/attention/mask-expand-math.json
```

若展开后前向可执行但仍有 FP32 梯度失败，应分别记录“广播形状障碍消除”和“原数值 gate 未通过”。当前 Action DiT 基线本来不使用 padding mask；任何兼容 wrapper 都不应因此把 mask 新增到 action 路径。

## mask 展开 auto 与 math 的实测结论

`mask-expand-auto.json` 与 `mask-expand-math.json` 均记录 Torch-MUSA runtime `1.3.0+81caf0a`，并保存上表中的两份脚本 hash，确认调用的是未修改的原输入与门槛。每次 64 rows 中 54 通过、4 个原 `B11K` muDNN 前向错误、6 个 expanded FP32 数值 gate 失败。每次 CPU 32/32 通过，MUSA eager FP32 16/16 通过；MUSA SDPA BF16 为 6 pass + 2 原形错误，FP32 为 6 数值失败 + 2 原形错误。两次的 `all_variants_pass=false` 与 `expanded_variants_pass=false` 保留这些失败。

bool 与 additive 得到完全相同的 expanded 数值 metrics；每种 dtype 内，三种 expanded 表示也得到相同结果。`B1QK_contiguous` 的 stride 为 `[527,527,31,1]`，已足以消除原 muDNN 错误，无须显式展开 heads。`BHQK_expand_view` 的 stride 是 `[31,0,0,1]`，目标设备上仍是非 contiguous view；`BHQK_contiguous` 为 `[2108,527,31,1]`。view 能运行只表明 API 接受这个 view，源码边界仍可能内部 `.contiguous()` 物化，不能据此宣称零拷贝或相同内存。

| MUSA expanded SDPA，对 CPU FP64 | forward relative L2 / max absolute | Q/K/V gradient relative L2 | Q/K/V gradient max absolute | 原门槛 |
|---|---|---|---|---|
| FP32（bool/additive、三种表示一致） | 1.259547e-7 / 8.058893e-8 | 4.126222e-4 / 4.171972e-4 / 2.919192e-4 | 2.706874e-6 / 2.457983e-6 / 6.824823e-6 | 前向通过，三份梯度失败 |
| BF16 auto（bool/additive、三种表示一致） | 2.270278e-3 / 1.388147e-3 | 2.307223e-3 / 2.400157e-3 / 2.340733e-3 | 1.691552e-5 / 1.935279e-5 / 5.630476e-5 | 全部通过 |
| BF16 math（bool/additive、三种表示一致） | 2.409260e-3 / 1.117367e-3 | 3.340371e-3 / 3.334347e-3 / 2.392396e-3 | 2.540614e-5 / 2.173571e-5 / 6.417439e-5 | 全部通过 |

FP32 每个 expanded row 的失败具体是 `numerics.gradient_errors`、`perturbed_numerics.gradient_errors`、`cpu_peer.gradient_errors` 各自三份 Q/K/V 的 relative L2 超 `2e-4`。这三个比较中的 forward 全部通过；梯度 absolute allclose 仍通过，所有值有限。屏蔽扰动前后的 numerical metrics 完全相同，不是扰动使误差增加。CPU peer 的 relative L2 为 `4.126212e-4 / 4.171970e-4 / 2.919236e-4`，其语义仍是 MUSA-vs-CPU。

**屏蔽契约没有失败。** 每次两种 dtype 的全部 12 个 MUSA expanded SDPA rows，`blocked_key_invariance` 的输出及 Q/K/V 梯度最大 absolute error 均为严格的 0，`blocked_key_gradients` 中被屏蔽 K/V 的梯度也严格为 0。这说明在当前输入上 mask 确实阻断 K/V 影响，而不是“能跑但忽略 mask”。原 `B11K` 在 MUSA 无输出，因此没有 MUSA expanded-vs-original 对照，不能补写该比较通过；CPU 及 eager 成功的 original-representation 对照均通过。

BF16 auto 对 CPU peer 的 forward relative L2 是 `3.269689e-3`、max absolute `1.953125e-3`，Q/K/V gradient relative L2 是 `4.659143e-3 / 4.783164e-3 / 2.419481e-3`；math 为 forward relative L2 `2.520298e-3`、相同 max absolute，梯度 `3.824272e-3 / 3.833947e-3 / 2.392932e-3`。这些比较也全部通过。原 `B11K` FP32 两次均报 `MuDNNMathSDPAFwd`；BF16 auto 报 `MuDNNFlashSDPAFwd`、math 改为 `MuDNNMathSDPAFwd`，原失败保持。

逐项比较两份 JSON，所有 CPU rows、MUSA eager FP32 rows，以及 MUSA FP32 SDPA rows 的 pass/error/数值与屏蔽 metrics 都完全相同；差异只有 MUSA BF16 SDPA 的原形错误名与展开后 numerical/CPU-peer metrics。BF16 math 的梯度 relative L2 高于 auto，仍远低于 `0.04`，不能以单输入结果推断性能或任务效果。auto 与 math 一致支持显式 Q 展开作为这个 key-mask 表示的兼容措施，也一致证明它不消除 FP32 SDPA 反向数值问题。

## BF16、fully-masked rows 与周边 primitive

BF16 action cross/self/VL self 的 SDPA 前向和 Q/K/V 反向通过原 BF16 gate，auto 与 math 均如此。Eagle GQA causal/right-pad、packed vision D=72 的 FP32/BF16 SDPA 前向也通过；这两条 frozen 路径未测反向。primitive 4 rows 全部通过，覆盖 Linear/matmul、category-selected bmm、RMSNorm FP32 reduction、LayerNorm、GELU tanh、SiLU、手写 RoPE 与参数梯度；它不包含完整 DiT/Eagle module、normal/noise 或 Flow-SDE/log-prob。

Fully-masked left-padding edge 被原设计排除在 required 之外，仍保存失败。CPU SDPA/native/Qwen-like 在两种 dtype 下产生非有限值；MUSA FP32 也如此。MUSA BF16 auto SDPA 输出有限，但零输出 oracle 最大绝对误差 `0.279296875`、relative L2 `0.1352471284`，因此失败；math BF16 为非有限值。eager FP32 在这些边界前向通过零输出契约，**edge case 未测反向**，不能据此声称 fully-masked 行训练梯度已验证。真实 wrapper 仍需明确 padding query 的处理与有效 token 选择。

## 单层内存可用，完整模型与性能尚无证据

`memory.json` 仅测试 `eager_fp32`；没有 SDPA 的对照峰值。8 rows 都有限且没有异常，allocator baseline allocated/reserved 均为 0。下表是设备 allocator 的 live allocated peak 和 reserved peak（MiB），包含单个 kernel 的输入、输出及指定 VJP；不是进程 RSS、整卡占用或完整模型峰值。

| geometry | FP32 allocated / reserved MiB | BF16 allocated / reserved MiB | 是否测 VJP |
|---|---:|---:|---|
| action cross，Q49/K570/H8/D64 | 7.395 / 24 | 8.460 / 26 | 是 |
| VL self，Q570/K570/H8/D64 | 48.996 / 62 | 48.608 / 66 | 是 |
| Eagle GQA，S570/H16/KV8/D128 | 107.542 / 122 | 105.316 / 122 | 否 |
| packed vision，S256/H16/D72 | 21.504 / 44 | 22.629 / 44 | 否 |

这些 B=1 的 `feature-570` 形状是源码默认值/序列假设，最终 checkpoint 未核对。大形状没有 CPU FP64 numerical check，结果顶层 `required_numerics_pass=null`。BF16 eager FP32 仍构造 FP32 score/probability/value 累积，因此不能期待内存减半。首个 action FP32 耗时 2.286 s、随后 BF16 为 0.00349 s；初始化与 warmup 未控制，不能用于 FP32/BF16 性能比较，也不能宣称 fallback 训练吞吐可接受。

下一步可沿已通过的 eager FP32 数学路径验证真实 Diffusers processor、Eagle/RADIO wrapper 与参数梯度，同时保留原 SDPA 失败。现有 RLinf N1.5 MUSA patch 仍直接导入 vendor FlashAttention 并宣告 availability=True；本环境缺失该包，算子探针不会修复其导入。RADIO source 的 `qkv.is_cuda` assert 与实测 MUSA tensor `is_cuda=False`、`is_musa=True` 不兼容；SigLIP 强制 FA2、Qwen2 assert FA2、实际 checkpoint 子模型、预处理与 GQA/RMSNorm/RoPE 整体集成仍待检查。冻结视觉/LLM 仍每次执行 backbone 前向，`eagle_linear` 默认可训练；完整权重、optimizer/FSDP、Flow-SDE 和任务学习必须独立建立证据。

## 下一步最小实际 module 适配范围（不加载权重）

首个实现与 GPU gate 建议只覆盖 **GR00T action 的真实 Diffusers attention processor 与一个双层 DiT**，先把数学 fallback 接到模型真正的 projection、归一化与输出布局；不要立刻实例化完整 Eagle 或 GR00T builder。最小范围如下，保持依赖使用上述固定源码、隔离目录及协调者串行 GPU：

1. 在 route2 增加一个独立 `AttnProcessor2_0` 兼容 processor，通过 `attn.set_processor(...)` 绑定到指定 action attention 实例。复用 source 的 spatial/group/cross norm、`to_q/to_k/to_v`、可选 q/k norm、`[B,S,H*D]→[B,H,S,D]` transpose、`to_out[0/1]`、residual/rescale；只将 kernel 改为 FP32 QK/softmax/PV 累积，输出 cast 回原 dtype 后按 source 转回 BSHD。保留 source 非 contiguous Q/K/V 输入语义、默认 scale `1/sqrt(D)`、`dropout_p=0`、`is_causal=False`，不得绕过 output projection 或把所有 module `.float()`。当前 action 调用明确 mask=None；wrapper 的 mask 分支单独验证，不能把新增 key mask 接入 DiT 基线。
2. 首先用 Diffusers 0.30.2 的真实 `Attention` 做 self 与 cross 两个实例：B2、H4、D64、query width256、Q17；cross encoder K31、width1536（Eagle action feature 的 source 默认，记录为配置假设）。共享同一份量化初始参数，比较 CPU 原 `AttnProcessor2_0`、CPU 新 processor、MUSA 新 processor 的输出，以及 hidden/encoder 输入和 `to_q/to_k/to_v/to_out` 参数 VJP，保存 parameter 名称、dtype、hash、grad 有限性与原 FP32/BF16 gate。独立 masked contract 只作为 processor API 检查，不改变 action 模型调用。
3. 再直接从固定 `cross_attention_dit.py` 构造 `DiT(num_attention_heads=4, attention_head_dim=64, num_layers=2, cross_attention_dim=1536, interleave_self_attention=True, dropout=0, final_dropout=False, norm_type="ada_norm", activation_fn="gelu-approximate")`，其他 source 默认保留。第0层 cross、第1层 self，Q17/K31、固定 timestep，随机权重即可；绑定两个 `block.attn1` processor。这比复制“类 DiT 小网络”多覆盖真实 AdaLayerNorm、timestep projection、sinusoidal position、GELU feed-forward、residual 与 output projection。检查 CPU/MUSA 同权重输出和完整 trainable 参数 VJP、encoder 与 hidden 输入梯度；将 source 中固定 FP32 的时间/位置计算与量化来源记清，不能误称整条 reference 已经全 FP64。只有这一步通过后再考虑 `SelfAttentionTransformer` 与 RL action head 接入。

冻结 backbone 的下一批范围也应是三个独立实例，不与首个 action gate 混在一起：

- **Qwen3Attention + 实际 RoPE/RMSNorm**：以 Transformers 4.51.3 的真实类构造 hidden width512、H4/KV2/D128 的随机 config，禁用 cache/sliding-window/dropout；保留每 KV head 成组重复。用显式 composite causal/right-padding `[B,1,Q,K]` additive mask，必要 key-only mask 先展开 Q，再返回 source 期望的 BSHD/None tuple并保留 scaling。当前 frozen 目标先查前向与有效 token；不要把 action 非 causal processor直接套给 Eagle。
- **SigLIP vision attention**：构造小 SigLIPVisionModel 随机 config，仅减少层数、token数和head数，保持 D72；给这个实例选择已验证 kernel，保留独立 q/k/v/out projection、non-causal/no-padding、dropout=0。先证实 source constructor 不再硬选 FA2；最终 checkpoint 若确为 SigLIP，就优先这条实际路径。
- **RADIO/timm packed attention**：其 `_flash_attn` 是 `qkv Linear→[B,S,3,H,D]→可选q_norm/k_norm→inner_attn→BSHD→proj/drop`。仅适配实例的 `inner_attn` 为 packed FP32 fallback，返回 `(context, None)`，先限制 `key_padding_mask=None`、`causal=False`，保持外层 BF16 与 packed 非 dense布局。RADIO 文件在 import 时导入 FA并全局替换 timm Attention，故需在隔离进程中明确移除这两个 source 阻碍；不能通过伪造 `is_cuda` 或 availability 进入未知 kernel。若实际 checkpoint 不用 RADIO，它仍有顶层 import blocker，但不因此增加 RADIO 全模型训练范围。

这里所有 config 只用于无权重接口验证，不代替最终 checkpoint 的 head/dtype/feature size。首个 action module gate 已足以判断 fallback 能否穿过真实 projection 与参数梯度；完成后再按 checkpoint 实际选择扩展 frozen 路径、导出 trainable 清单、检查 `eagle_linear` 与 Flow-SDE，而后才开始完整权重加载。
