# 真实 HalfCheetah 串行与批量评估核验

2026-10-05，在匹配 route 2 的 Python 环境运行
`evaluation_cpu_probe.py`，完整退出码 0。真实 Gymnasium 1.1.1 / MuJoCo 3.3.7，
NumPy 1.26.4，Torch package 2.2.0a0+git8ac9b20。仅 CPU 模拟和 NumPy 固定线性
策略，未调用 MUSA/GPU API，不运行训练，不抽样动作。

策略权重固定为 `0.004*cos(index+1)`，bias 为 `0.02*sin(action_index+1)`，
动作裁到实际环境 bounds。相同 last-axis reduction 顺序避免单样本/多样本 BLAS
算法差异干扰 MuJoCo 轨迹对照。三组 held-out seeds 为 30000–30002，另用
seed 40000 的 20 步 warm-up 排除首次加载影响；warm-up 不计入比较 episode。

紧凑 JSON 证据如下（完整 JSON 留在服务器
`/root/autodl-tmp/s4000-research/route2/results/learning/halfc-evaluator-cpu.json`）：

```json
{"status":"passed","env_id":"HalfCheetah-v5","seeds":[30000,30001,30002],"serial_returns":[0.7689235698143466,0.9105956022118127,-1.122929348564491],"batched_returns":[0.7689235698143466,0.9105956022118127,-1.122929348564491],"lengths":[1000,1000,1000],"return_max_abs_error":0.0,"lengths_equal":true,"rng_unchanged":{"setup":{"python":true,"numpy":true,"torch_cpu":true},"serial":{"python":true,"numpy":true,"torch_cpu":true},"batched":{"python":true,"numpy":true,"torch_cpu":true}},"policy_calls":{"serial":3000,"batched":1000},"timing_seconds":{"serial":0.2191942110657692,"batched":0.21077685058116913},"timing_serial_over_batched":1.0399349381176877}
```

回报逐项精确一致；脚本使用 `rtol=1e-10, atol=1e-8` 验收，同时要求 episode
种子和长度一致。Python、NumPy、Torch CPU 的全局 RNG 在 setup、串行评估和
批量评估三个阶段分别保持不变。原始 RNG 状态不写入报告。

批量版本将 policy 调用从 3000 次减到 1000 次。本次 NumPy CPU 线性推理成本低，
单次 warmed timing 只有约 1.04× 比值；计时包括 env 创建、reset 和 close。
该结果验证真实环境的评估语义，不代表 MUSA 推理加速或学习 benchmark。

在服务器上复跑：

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/root/autodl-tmp/s4000-research/route2/learning \
/root/autodl-tmp/s4000-research/envs/route2/bin/python \
  /root/autodl-tmp/s4000-research/route2/learning/evaluation_cpu_probe.py \
  --episodes 3 --seed-start 30000 \
  --output /root/autodl-tmp/s4000-research/route2/results/learning/halfc-evaluator-cpu.json
```

本轮仅新增验证脚本和本说明；原学习 runner、批量 helper、共享脚本、上游源码
和 Git 状态没有操作。Ruff 0.14.3 检查及格式化通过。
