# 正式 Spatial action-layout 接口探针

本节点把已验证的 toy Attention/DiT 接口推进到 `RLinf/RLinf-Gr00t-SFT-Spatial` 固定配置的正式宽度。新实现位于 [spatial_action_layout_probe.py](../model_probes/spatial_action/spatial_action_layout_probe.py)，依赖清单见 [source-manifest.json](../model_probes/spatial_action/source-manifest.json)。当前仅完成本地静态验证，CPU/MUSA 数值结果由协调者执行后单独保存；既有 action/backbone probe、生产补丁、锁文件均保持原字节。

四条完整 argv、导出文件与预期行数另保存为 [run-plan.json](../model_probes/spatial_action/run-plan.json)。解释器沿用协调者新建的 `envs/route2-integration`；旧 action/backbone 环境保留。默认batch1、CPU线程1，每case结束释放临时模型与snapshots，JSON只保留metrics/hash；尚未实测RSS或吞吐。

固定 checkpoint revision 为 `73f710e70e7d571f8d828e51e0a428f5a1e0ac22`，config SHA256 为 `6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526`。真实 GR00T `cross_attention_dit.py` 来自 commit `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`，SHA256 `1366e3f80c0e076953965eeb43d800cb354fdcdf681ab54f0c2ad8f1d6c9fe62`。Diffusers 必须为 `0.30.2`，实际 `Attention` 源文件 SHA256 必须为 `afe47ff6d33c865af745e28eed4bd8048b66fd52de7c2b3bad37867e826776c9`；版本或字节不符均保存异常并失败。

## 正式几何与减小范围

| 组件 | 本轮几何 | 与正式配置的差别 |
|---|---|---|
| Action self/cross Attention | H32/D48、inner/query width1536、cross width2048；Q49/K570 | 使用真实独立 Diffusers Attention、随机权重 |
| DiT | 同上、output1024、真实 AdaLayerNorm/time encoder/FFN/output projection | 正式16层减至2层；第0层cross、第1层self；未加载checkpoint |
| VL self Attention | H32/D64、inner/query width2048；Q=K570 | 独立 Attention，不构造正式4层 SelfAttentionTransformer 或 vlln |
| Norm/4D合同 | Action H32/D48；真实 spatial/group/cross LN、per-head q/k LN、residual/rescale | 扩展 processor API 合同，不声称这些 norm 都是 Spatial DiT 的正式参数 |

正式 config 的 `dropout=0.2`、`final_dropout=true` 在本节点明确改为 `dropout=0`、`final_dropout=false`，所有模型均 eval。DiT 的 `positional_embeddings=null`、interleave self attention、norm 类型与完整宽度保持配置值，其余真实 constructor defaults保存为 `effective_config`。默认 FP32，可显式另测 BF16；两者都是共享量化权重/输入的真实 CPU baseline，均不称为完整 FP64 参考。此结果不能推出训练 dropout、完整深度、pretrained模型、optimizer更新或学习已经通过。

Action hidden 输入先按 state1 → future32 → action16 顺序拼为 `[B,49,1536]`，再作为一个可微 leaf；这些是合成的已嵌入 token，未调用 state/action encoder 或 future embedding。Q49 因本节点明确设置 `Tstate=1` 而成立。encoder 为 `[B,570,2048]`。默认 batch1、代表性未补齐长度513；`--lengths 513,497` 可显式构造 batch2。未补齐长度是声明的测试fixture，尚未观测真实 LIBERO batch。

## 真实源码的 mask 边界

固定 DiT 的 `forward` 虽接受 `encoder_attention_mask`，但给所有真实 block 传的 `attention_mask` 与 `encoder_attention_mask` 都是 `None`。BasicTransformerBlock 实际只向 Attention 传 `attention_mask`。action head 中即使传入 `vl_attn_mask`，这份固定 DiT 也忽略它；VL SelfAttentionTransformer 同样没有 mask 参数。因此，本轮保留真实 DiT/VL Attention 的 no-mask baseline，不修改或包装其源码来注入 padding mask。

DiT 另做同权重、同输入的 `encoder_attention_mask` 参数实验：传入代表性右pad mask 与传入 None 的输出、所有输入/参数 VJP 须 exact。这一项明确叫 `dit_mask_argument_ignored`，只验证源码忽略参数，不能作为 DiT 屏蔽 padding 的成功证据。独立 Attention 的 mask cases才承担以下合同：

1. 从未补齐的全1 token mask 使用真实 `torch.nn.functional.pad(..., value=False)` 右补至570，compact入口为 `[B,1,570]`，bool True表示允许，additive为0/-inf。
2. 在 source 接受的4D入口 `[B*H,1,Q,570]` 使用真实 `prepare_attention_mask`，其真实 view 结果为 `[B,H,Q,570]`。source 对原始 `[B,H,Q,K]` 会按dim0再repeat heads，不能把该形状直接当成它的合法入口。本节点保留prepare和view，没有跳过源码mask准备。
3. 4D mask在右pad基础上，每个head/query额外屏蔽一个可见key以检查轴语义；token0始终可见。它是API合同fixture，未称为真实LIBERO padding。核对所有行非空、所有右pad key被屏蔽、head和query轴均确实不同；过短 `--lengths` 无法形成变动时明确失败。fallback trace须记录最终实际4D mask shape且与source准备结果一致。
4. compact bool/additive和4D bool/additive分别共享seed、权重、inputs、upstream，输出与所有输入/参数 VJP要求 exact；等价项单独保存为 `bool_additive_equivalence_exact`。

## 前向、反向与 mask 扰动门槛

每个case/dtype先构造同一份CPU量化template，记录全部state/input/upstream/mask/timestep的shape、dtype、SHA256；再依次执行 native CPU `AttnProcessor2_0`、CPU `FP32AttentionProcessor`、显式选MUSA时的同processor。既有processor与比较helper通过明确目录加载、逐文件hashguard，不依赖cwd或未固定的import搜索结果。processor保持真实norm、QKV/out projection、4D还原、dropout模块、residual/rescale，只把QK/softmax/PV累积放在FP32。实例 `set_processor` 绑定，不做全局patch。

原始逐tensorstrict gate复用已验证helper：FP32 `atol=rtol=relative_L2_limit=2e-4`，BF16 `atol=0.025, rtol=relative_L2_limit=0.04`，finite、allclose、relative L2均须通过。检查output、每个可微输入与每个trainable参数VJP，没有动态丢弃小reference。每次state hash须匹配初态，所有输入/参数均须有有限梯度；MUSA还同时比较native CPU与CPUfallback。CPU peer缺失时相应比较明确失败。

analytic-zero方向仍按固定helper从module结构提前枚举：无K norm时的 `to_k.bias`，per-head K LN的 `norm_k.bias`。strict门槛仍包含它们；独立semantic视图要求其它项通过原门槛，并对这些解析零方向同时要求actual/reference有限且max_abs不超过各dtype atol。`strict_interfaces_pass` 与 `semantic_interfaces_pass` 分开，**exit0只认strict与所有合同通过**。如果仅零方向relative L2失败，原始status/exit仍保持fail/1，不能凭semantic字段改写strict通过。

每个masked case与每种CPU/device实现另执行完整扰动VJP：

- 对所有右pad token使用非恒定feature扰动；输出与全部输入/参数VJP须exact，屏蔽位置输入梯度严格为0。非恒定方向避免cross LayerNorm把常量扰动消掉。
- 移除mask须改变可见输出，作为mask有效性的正控制。
- 扰动保证可见的token0的交错feature须改变输出，作为模型确实读取输入的正控制。
- VL self合同只比较有效query输出；padding-query upstream置0，保留实际padding-query输出，不凭空定义整个module的padding输出为0。检查全部input/parameter VJP和padding位置hidden梯度仍严格0。

## 协调者独立运行

保留以下相对布局，或通过三个明确路径参数指定相同固定字节；新脚本和manifest需放在同一个 `spatial_action/` 目录。所需外部文件共5份：未改的 `attention_probe.py`、`fp32_attention_processor.py`、`gr00t_action_interface_probe.py`，固定 `backbone_sources/checkpoint-config.json` 与固定GR00T DiT源码。runner自身也按source-manifest检查哈希。无需下载checkpoint或安装完整GR00T。

以下每条独立命令均会自己重建nativeCPU/CPUfallback基线。输出文件必须全新；GPU运行只由协调者排期。

```bash
/root/autodl-tmp/s4000-research/envs/route2-integration/bin/python \
  /root/autodl-tmp/s4000-research/route2/model_probes/spatial_action/spatial_action_layout_probe.py \
  --device cpu --phase attention --dtypes fp32 --lengths 513 \
  --support-dir /root/autodl-tmp/s4000-research/route2/model_probes \
  --spatial-config /root/autodl-tmp/s4000-research/route2/model_probes/backbone_sources/checkpoint-config.json \
  --gr00t-source /root/autodl-tmp/s4000-research/route2/model_probes/audit_sources/Isaac-GR00T/gr00t/model/action_head/cross_attention_dit.py \
  --output /root/autodl-tmp/s4000-research/route2/results/spatial-action-cpu-attention-first.json

/root/autodl-tmp/s4000-research/envs/route2-integration/bin/python \
  /root/autodl-tmp/s4000-research/route2/model_probes/spatial_action/spatial_action_layout_probe.py \
  --device cpu --phase dit --dtypes fp32 --lengths 513 \
  --support-dir /root/autodl-tmp/s4000-research/route2/model_probes \
  --spatial-config /root/autodl-tmp/s4000-research/route2/model_probes/backbone_sources/checkpoint-config.json \
  --gr00t-source /root/autodl-tmp/s4000-research/route2/model_probes/audit_sources/Isaac-GR00T/gr00t/model/action_head/cross_attention_dit.py \
  --output /root/autodl-tmp/s4000-research/route2/results/spatial-action-cpu-dit-first.json

/root/autodl-tmp/s4000-research/envs/route2-integration/bin/python \
  /root/autodl-tmp/s4000-research/route2/model_probes/spatial_action/spatial_action_layout_probe.py \
  --device musa --phase attention --dtypes fp32 --lengths 513 \
  --support-dir /root/autodl-tmp/s4000-research/route2/model_probes \
  --spatial-config /root/autodl-tmp/s4000-research/route2/model_probes/backbone_sources/checkpoint-config.json \
  --gr00t-source /root/autodl-tmp/s4000-research/route2/model_probes/audit_sources/Isaac-GR00T/gr00t/model/action_head/cross_attention_dit.py \
  --output /root/autodl-tmp/s4000-research/route2/results/spatial-action-musa-attention-first.json

/root/autodl-tmp/s4000-research/envs/route2-integration/bin/python \
  /root/autodl-tmp/s4000-research/route2/model_probes/spatial_action/spatial_action_layout_probe.py \
  --device musa --phase dit --dtypes fp32 --lengths 513 \
  --support-dir /root/autodl-tmp/s4000-research/route2/model_probes \
  --spatial-config /root/autodl-tmp/s4000-research/route2/model_probes/backbone_sources/checkpoint-config.json \
  --gr00t-source /root/autodl-tmp/s4000-research/route2/model_probes/audit_sources/Isaac-GR00T/gr00t/model/action_head/cross_attention_dit.py \
  --output /root/autodl-tmp/s4000-research/route2/results/spatial-action-musa-dit-first.json
```

`--phase attention`有9 cases，每dtype CPU18 rows / MUSA27 rows；`--phase dit`有1 case，每dtype CPU2 rows / MUSA3 rows；`--phase all`有10 cases，每dtype CPU20 rows / MUSA30 rows。BF16可独立使用全新output重跑 `--dtypes bf16`，或用 `--dtypes fp32,bf16`，不能把一个dtype的通过替代另一个。`--cases`用于定位，保存完整筛选arguments，不把子集通过代替全范围。

clone/transfer、forward、全部VJP、snapshot、mask扰动、DiT ignored-mask均输出逐阶段紧凑JSON；每行保存实际exception、traceback、last_stage。每行完成后刷新输出JSON，最后写expected rows、两个独立gate与exit_code；协调者还需独立保留命令真实退出码和完整stdout/stderr。Elapsed包含hash/同步/CPU基线/扰动，不是性能benchmark。

本地静态记录见 [static-validation.json](../model_probes/spatial_action/static-validation.json)。Mac无Torch/Diffusers/MUSA运行，本节点不包含数值通过结论。[route1只读审查](../model_probes/spatial_action/review.json)已完成，无阻断项，核对了所有来源hash、真实mask路由、扰动/VJP门槛、DiT no-mask边界、strict/semantic及异常处理。当前v2 runner冻结SHA256为 `353fbb6380edd20f5bdb6cea14189dd23ea2c97b0d0b4224ad795bc86fb3a97d`，manifest为 `44b94fa5f3fb5673948b7de0a9dfef805f71adac17b77e5767f916ff3589b197`；实机结果仍待协调者调度，边界不扩展到官方完整action head、冻结Eagle、真实预处理、570-token端到端、训练dropout、pretrained权重、optimizer、PPO或LIBERO学习。

## 首次CPU作业与序列化修复

协调者报告首次CPU作业在DiT阶段写入证据时遇到 `Object of type dtype is not JSON serializable`，原因是实际DiT的 `effective_config`包含 `torch.dtype`。此前partial JSON保留前9 Attention cases的18 rows，命令exit1；没有完整all范围通过结论。runner、manifest、run-plan与原静态/审查记录逐字保存到 [v1快照](../model_probes/spatial_action/history/layout-v1-json-dtype/README.md)，实机原始partial JSON和日志由协调者保存。

v2仅增加证据树的递归 `json_safe`：dict/list/tuple中真实torch.dtype转str，所有其它类型与数值原样交给JSON。AST比较确认全部数值函数、模型/输入构造、VJP、strict/semantic gate与main除persist之外语句未变；不是模型数值修复。v2静态检查通过，真实torch.dtype小回归与全CPU/MUSA数值由协调者重跑，不能把序列化失败改称某个数值gate通过或失败。


## 协调者 GPU 结果补记

v2 MUSA runner每dtype完整30行，包含20个CPU参考行和10个实际GPU行。FP32 semantic=true、strict=false；BF16 semantic=false、strict=false；真实退出码均1，无运行异常。BF16扩展norm4D合同在MUSA上有normal `norm_k.weight`（relativeL2 0.15340）和`norm_q.bias`（0.04316）超过0.04，不能按analytic-zero排除。正式无q/k norm DiT的meaningful项两dtype都通过；扩展合同失败不推广为整个BF16action不可执行。所有原失败/门槛保留，详见 [四次独立运行验证](../evidence/spatial-action/run-validation.json) 和 [协调报告](../../../reports/route2-spatial-eagle.md)。
