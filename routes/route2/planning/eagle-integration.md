# 缩小真实 Eagle composite 集成探针

本轮直接构造固定 Isaac-GR00T 的真实 `EagleBackbone`，通过真实 `Eagle2_5_VLForConditionalGeneration` 跑通 SigLIP→`extract_feature`→`mlp1`→image token substitution→Qwen3→source hidden-state selection→Identity。此前 action/backbone 的源码、证据和审计 manifest 不改；新文件限于 `model_probes/eagle_integration/` 与本报告。

## 固定 source 与依赖

Isaac-GR00T 固定 commit `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`。协调者取得官方 archive（46,491,700 bytes），SHA256 `f98e0507e96281ffa289da9468e328b30de3733a00472f249f432e14050d1763`，并验证既有七份源文件 hash。完整 source tree 由协调者管理，路线代理没有 Git、安装、远程/GPU操作。

新 audit cache 只读副本与获取记录位于 `eagle_integration/audit_sources/manifest.json`。关键新增小文件：`configuration_eagle2_5_vl.py` SHA256 `d1578a2d5b7654da9a5e4cb18acd5d9d92b04a227ddab9365c2e3265d6cf2f93`，root LICENSE `58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd`，model package init `279c47a1bcae3a98ac661206e04d0b1e9a15c73ce8262536431ba433589c3854`。Canonical raw URL 都固定同一 commit；cache 由协调者导出时排除，不把完整第三方源码复制进精简成果。

选定 no-LoRA/no-RADIO backbone 不需要 PEFT、accelerate、einops、timm 或完整 policy/data 依赖。保留既有系统 Torch/Torch-MUSA，独立 integration environment 使用 no-deps 安装已选 Transformers4.51.3/hub0.30.2/tokenizers0.21.4/regex2024.11.6/safetensors 等；完整 pyproject 的 Torch/CUDA optional dependencies 不安装。若将来真的选 LoRA/RADIO，仍须安装它们的真实依赖，不能把本轮 lazy-import 结果推广到未选分支。

## 可审查 source patch 与完整树构造

`prepare_eagle_sources.py` 只读 supplied fixed source tree，核对七个原 hash，生成统一 diff、tiny config 和 source-preparation manifest；它不会修改 source tree。补丁 SHA256 `99c8522a350617dd85e8534382be446a652190ec086e80dee179ff5ae37719ab`，实际修改三个 source 文件：

- Eagle config 的 RADIOConfig import 移到真实 radio config 分支。
- Eagle modeling 的 RADIOModel import 移到真实 radio model 分支；PEFT import 移到两处真实 wrap_*_lora 方法；SigLIP 构造遵循显式 top-level attention implementation，默认 FA2 保留其原配置含义。
- `gr00t/model/__init__.py` 通过 PEP562 `__getattr__` 延迟导入原有 GR00T_N1_5/Gr00tPolicy 公开导出；访问这些导出时仍导入真实模块。

没有 fake modules、FlashAttention availability shim 或重新实现的 multimodal forward。`eagle_backbone.py` 原字节、Eagle `forward`、`extract_feature`、connector、token replacement 均不改。完整 source 而非模块替身用于 import，确保未选依赖分支的真实文件仍存在。

协调者已建立 eager/lazy 基线 `worktrees/gr00t-eagle-integration` commit `62c7d52e625a8172f3c42453453706c329b71d7b`／tree `0b0d6c6481683238ccb0f0262b6d3f8efe840154`，以及独立 tiny fixture `worktrees/gr00t-eagle-tiny` commit `e441b8f7c17d4f25c49cfa1f20731e6085e7a962`／tree `66e0fadc90837a2212f373431bef5ed06d804c6e`，七个 prepared hash 匹配；commit/tree 属于协调者报告的源锁，不是路线代理自行执行 Git 取得的结果。

tiny config 从 fixed bundled config 生成，移除 `auto_map`，临时通过公开 AutoConfig/AutoModel.register 注册真实 Eagle config/model classes。原 EagleBackbone 的 DEFAULT_EAGLE_PATH、AutoConfig.from_pretrained、AutoModel.from_config 调用保留；模型 registry 在运行结束完整恢复。这里存在明确的临时 model registry 修改，不能笼统声称“无任何全局修改”。attention registry/class forward/FA availability 必须前后完全相同。去掉 tiny `auto_map` 是避免 HF dynamic-code loader 静态递归扫描未选 RADIO dependency；没有替换真实 constructor。

## 缩小配置与接口合同

| 合同 | 本轮 tiny composite |
|---|---|
| Qwen3 | H4/KV2/D128，hidden512，RMS eps1e-6，RoPE theta1e6，无cache/window |
| 层数与选择 | 构造3层，原 wrapper `select_layer=2` 裁至2层；取 source hidden_states[2]，共3份 hidden state |
| SigLIP | H4/D72，hidden288，2层，image56/patch14，16 patch tokens，LayerNorm eps1e-6/gelu_pytorch_tanh |
| Vision pooler | 默认真实 head 保留并执行；Eagle 丢弃其输出，只检查 finite/layout，不要求 pooler 数值 parity |
| Connector | 原真实 mlp1 Linear288→512，one layer，pixel_shuffleFalse |
| 输入 | B2/S24，各16个 image token，有效长度22/23；3幅 raw tensor image，flags[1,1,0] |
| image mapping | 正好32个 selected tokens与2×16 vision tokens匹配；额外图仍经过vision/connector后被过滤 |
| 输出 | 真实 source EagleBackbone BatchFeature，mask不变，Identity等于selected hidden state |
| 冻结 | 原 set_trainable_parameters(False,False)，vision/connector/language全部 requires_gradFalse/gradNone |

HF Spatial config 原 `tune_visual=true` 被 RLinf builder 显式 False 覆盖；project_to_dim=null 是 Identity。本轮冻结方向与 builder 一致，不声称训练独立 backbone projection。tiny sequence与宽度/depth不能代表真实570-token/full-width成本。

真实类的 forward hooks 仅读取 connector output、fused inputs_embeds、vision pooler/hidden、Eagle logits/selected hidden state。image位置必须精确等于实际 connector 输出，非image位置必须精确等于真实 token embedding lookup；source except 的截断 warning必须没有触发。负控制分别扰动 padding text、future text、被flags过滤但已计算的额外图，相关有效输出须差精确为零。正控制扰动两幅选中图，必须改变有效输出，同时图像之前的prefix3须因果不变。

CPU baseline 用固定 Transformers source eager；CPU/MUSA fallback 复用已通过组件 gate 的 kernel copy，SHA256 `db580ebeea14b7f7b174927e2d589ddb720f4f31a1991f4ddecc512d12be7cf1`，只提升 attention QK/softmax/PV 到FP32。source q_norm/k_norm/RoPE/GQA/scale/output布局保持，pooling head仍走真实 Torch MultiheadAttention。FP32 gate atol/rtol/relativeL2 2e-4；BF16 0.025/0.04/0.04，allclose 为混合 abs+relative 条件。比较有效features、有效logits、connector outputs；MUSA另比较CPUfallback features。全部 frozen state hash在转移前后和5次forward后必须一致。

## 公共 metadata 与权重形状

HF revision 保持 `73f710e70e7d571f8d828e51e0a428f5a1e0ac22`。本轮仅另取 index（104,606 bytes，SHA256 `bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746`）与 experiment metadata（12,968 bytes，SHA256 `0b8fffeb48a0bde4b12fb8cb2244136ddfebce60cd8b69c2233fdb4d55b85b24`），没有下载 full weights。index包含899 tensors、两个shards，总 tensor bytes7,585,939,328；key清单有Qwen layers0..11、vision layers0..26、真实 vision pooling-head参数、mlp1.0参数；没有eagle_linear参数，与12层裁剪、27层vision、one-layer connector、Identity相符。

协调者已用公开 safetensors headers 确认关键 shape：patch conv[1152,3,14,14]、mlp1[2048,1152]、embedding[151680,2048]。这些 shape与 bundled config/source相符，不意味着全部 pretrained weights 已读入或可推理；headers详细证据由协调者维护。

## Runner 与待验证项

```sh
/root/autodl-tmp/s4000-research/envs/route2-integration/bin/python /root/autodl-tmp/s4000-research/route2/model_probes/eagle_integration/eagle_composite_probe.py --source-tree /root/autodl-tmp/s4000-research/route2/Isaac-GR00T-eagle-tiny --device cpu --output /root/autodl-tmp/s4000-research/route2/results/eagle-composite-cpu.json
```

显式 --device musa 会在CPU source/CPUfallback之后增加MUSAfallback；两种dtype默认共CPU4 rows／MUSA runner6 rows。GPU排期仅协调者负责。代码本地 AST/help通过，runtime结果待协调者；构造时须先确认 AutoConfig/AutoModel 没把 eager自动改成sdpa，再允许称CPU source eager baseline。未检验 pretrained load、图像预处理/动态tile、full width/depth、GR00T action连接、optimizer、rollout/PPO、性能或学习。


## 协调者实机结果补记（2026-10-05）

tiny v2构造的实际top/vision/language均为eager。CPU四行3/4 strict通过，MUSA runner六行4/6；FP32全部通过，BF16仅fallback有效feature混合allclose失败，所有结构/冻结/状态/正负控制通过。未调整门槛。原v1序列化失败的stdout和准确源码保留。

随后 full eager/FP32 fixture commit `c9bd9a2403d14d492be88a9af9ae3989d4b711fc` 构造真实GR00T meta，全部899 keys/shapes匹配。两个Spatial权重分片完整LFS SHA通过，585个backbone权重strict加载后与BF16→FP32值精确相同；完整B1/S570、两224px图像的真实骨干前向已经在CPU与S4000执行，CPU两实现完全一致，MUSA features/logits逐元素门槛失败，结构/有限性/state合同通过。完整action、真实processor、LIBERO和PPO仍待验证。v2追加逐层误差和失败元素诊断，不改kernel/门槛。最终证据与范围以 [协调报告](../../../reports/route2-spatial-eagle.md) 为准。
