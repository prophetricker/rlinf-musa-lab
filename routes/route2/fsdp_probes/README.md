# 路线 2：真实 RLinf FSDP1 最小实验

2026-10-05 已在 S4000 实机完成真实 RLinf FSDP1 单卡 FP32 NO_SHARD 更新，CPU 回归为 9/9 通过。实机证据见 [strategy-third.jsonl](../evidence/fsdp1/strategy-third.jsonl) 和 [imports-regression-third.txt](../evidence/fsdp1/imports-regression-third.txt)。该节点只证明 strategy 包装、forward/backward、梯度范数与 AdamW 更新；尚未执行 checkpoint、Manager、官方 Actor、PPO、offload 或仿真器。

本机 Python 3.9.6 未安装 Torch、pytest、Ray、OmegaConf 或 Accelerate，本地只做源码和静态检查；实机测试由协调者执行。第一版已通过 imports 和 6 个导入用例，GPU 首次遇到一维 mesh 切片限制；第二版 CPU 结果为 6 passed / 2 failed，原因是厂商保留 list 维名。当前第三版归一化维名并覆盖 list/tuple，9 个 CPU 用例和 GPU strategy 均通过；历史补丁与静态记录保留在 [history](history)。

源码基线为 RLinf 上游 `c70606f08cdca259b8dec03d4430926b5b8fac9d` 加已验证适配提交 `3e7329c629293f1b0f9c331b5bb24eccc3e3f142`。本轮增量单独保存为 [fsdp1-experimental.patch](../patches/fsdp1-experimental.patch)，必须应用到该适配基线；没有覆盖 `current-minimal-compat.patch`，也没有改变 Torch、Toolkit、驱动或依赖声明。

## 补丁范围

- 只在显式设置 `RLINF_EXPERIMENTAL_FSDP1_TORCH22=1` 时开放 Torch 2.2 的真实 FSDP1 类型，包含该小版本的厂商预发布字符串。未设置时仍拒绝旧版本；该开关不会放行 2.1 或 2.3。
- 共用 base、utils、checkpoint 使用真实 DTensor fallback 和延迟 FSDP2 类型注解。选择 FSDP2 或调用其 helper 时明确失败，没有假类或功能替代。
- Transformers 的 auto-wrap helper 在真正使用时再导入；探针关闭 auto-wrap，仍然创建真实 FSDP1 包装。
- 添加导入开关、版本范围及 FSDP2 拒绝行为的子进程回归测试，以及中英文 MUSA 说明。测试中的版本字符串模拟只验证分支选择，实际 Torch-MUSA runtime 必须另跑下面的探针。
- `gradient_reduction_group()` 将维名归一化为 tuple，在明确命名为 `fsdp` 的一维 mesh 上直接调用真实 `get_group()`，兼容厂商保留的 list 维名并避免 Torch 2.2 不支持的一维切片；二维 mesh 保留原先按 `fsdp` 维切片的逻辑。一维 `ddp` 或未命名 mesh 仍明确拒绝，避免误选复制维。

此补丁没有修复 DCP 保存：当前代码调用 `checkpoint_id=`，Torch 2.2 需要 `storage_writer=`。不能据导入成功声称 checkpoint 已适配。Manager、官方 Actor、weight syncer、offload、多进程、混合精度同样未覆盖。

## 按顺序运行

协调者把本目录探针上传到远程对应目录，并将增量应用到远程隔离源码。以下路径沿用已经验证的路线 2 环境；服务器中的源码与环境若更换，先修正 `--source` 与解释器。不要对默认栈使用 `pip install -U torch`。

### 1. 导入与实际 API（不启动 Cluster / process group）

```bash
/root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/fsdp_probes/fsdp1_backend_probe.py \
  --source /root/autodl-tmp/s4000-research/route2/RLinf-fsdp1 \
  --phase imports --enable-torch22
```

通过条件：真实 RLinf FSDP package、utils、checkpoint、base 和 FSDPStrategy 全部导入；factory 创建真实 FSDP1 strategy；旧 runtime 的 FSDP2 请求明确被拒绝。JSON 记录实际 runtime/distribution 版本、模块路径、FSDP/MixedPrecision/DCP 签名与源码哈希。该阶段仅检查 API，不执行 checkpoint，也不创建 MUSA Tensor。

### 2. CPU 导入分支回归测试

在已具备 pytest 的隔离环境运行；本轮没有为测试安装或替换任何依赖：

```bash
cd /root/autodl-tmp/s4000-research/route2/RLinf-fsdp1
RLINF_EXPERIMENTAL_FSDP1_TORCH22=1 \
  /root/autodl-tmp/s4000-research/envs/route2/bin/python -m pytest -q \
  tests/unit_tests/test_fsdp.py \
  -k 'legacy_fsdp1 or one_dimensional_sharding_group or gradients_reduce_over_the_sharding_dimension'
```

该命令选择 6 个参数化导入用例和 3 个真实 CPU/Gloo DeviceMesh 用例，不使用 GPU。Torch 2.2 环境需要父进程开关以完成测试模块收集，负例会在新子进程中明确清除它。新增一维用例分别保留 list/tuple 维名，并临时令切片报错以复现旧 API 限制，仍使用真实 mesh / process group 并执行 CPU all_reduce；另一用例保留二维 `fsdp` 维选择，拒绝一维 `ddp` 和未命名 mesh。失败时保留第一条具体错误，不补假类型或假 process group。

### 3. 单卡 strategy 更新（必须由协调者安排 GPU 时段）

```bash
RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
  /root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/fsdp_probes/fsdp1_backend_probe.py \
  --source /root/autodl-tmp/s4000-research/route2/RLinf-fsdp1 \
  --phase strategy --enable-torch22
```

真实路径为一个 MUSA Worker → 显式 MCCL 默认组（使用 `Cluster.get_collective_timeout()`）→ RLinf `create_device_mesh(1)` 与 `gradient_reduction_group` → strategy factory → `wrap_model` → `before_micro_batch` → backward → **strategy 本身的 `clip_grad_norm_`** → AdamW.step。

探针模型为 FP32 `Linear(17,32)→Tanh→Linear(32,6)`，batch size 8，固定 CPU seed 42 和同一初始权重。策略是 `NO_SHARD`、`use_orig_params=true`、不启用 offload、auto-wrap、梯度累积。FSDP 包装保持原代码的整数 `device_id`、`sync_module_states=true` 和真实 mesh，没有修改包装行为来绕过失败。

通过条件包括有限 loss/梯度/norm、非零参数变化，及 CPU 对照的 loss、裁剪前后梯度、范数和更新后所有参数。对照容差显式为 `rtol=3e-4, atol=3e-6`；证据记录实际误差。裁剪后 L2 还须不超过 `1.0 + atol`。

此阶段主动建立 MCCL 默认组，因此不代表 RLinf 未指定 backend 的默认初始化入口已适配。world size 1 / NO_SHARD 也不证明真正跨卡分片或 collective 已通过。导入、包装、裁剪和更新前均输出 `started` 记录，失败时从最后阶段定位；日志可能夹杂 Ray 输出，保留完整 stdout/stderr，再由协调者摘取 JSON。

当前实机结果为 loss `0.8063545227`（与 CPU 对照相同）、梯度最大绝对误差 `2.235e-8`、更新后参数最大误差 `1.490e-8`、最大参数变化 `0.0010024309`。真实 `strategy.clip_grad_norm_` 已执行；本批次梯度 norm 为 `0.86544746`，低于阈值 1，因此没有触发实际缩放，主动裁剪的数值验证仍未覆盖。

## 本地验证与后续节点

静态记录见 [static-validation.json](static-validation.json)。`ruff` 解析检查与格式检查、探针 `py_compile` / `--help`、`git diff --check` 和补丁应用检查均属于源码验证；manifest 单独关联协调者保存的实机证据，并核对 GPU 记录中的 5 个 FSDP 源文件与 probe 哈希。首版补丁 SHA256 为 `59f970f17df26a516937f8d46bb688441a14915748685abcef222487df880e72`，对应此前的导入通过和第一次 strategy 阻塞，不能与更新后补丁的结果混用。

若真实 strategy 通过，再按 [FSDP1 审计](../planning/fsdp1-audit.md) 推进：先 local_shard checkpoint 恢复，单独修复并验证 DCP，再接 Manager、官方 Actor 与独立 Rollout。MSE strategy 更新成功只能证明训练后端这一层，PPO 和 MuJoCo 学习实验仍须分别验收。

最小 checkpoint 的具体后续方案见 [checkpoint-plan.md](checkpoint-plan.md)。当前补丁仍未实现 DCP 的 `storage_writer` 兼容分支，也没有新增 checkpoint 执行阶段。
