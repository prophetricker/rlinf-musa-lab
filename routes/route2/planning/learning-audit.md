# 学习基线审计：历史 route 1 PPO runner

本次只读审计 `routes/route1/mujoco_ppo_worker.py`，对照 v0.3 RLinf GAE 和
PPO 损失实现。未运行远程、GPU 或安装，未修改历史 runner。已有 256 transitions
结果证明训练闭环与同批次恢复成立，但 held-out return 仅 `0.177940 → 0.154520`，
不能证明学习。这份建议用于独立的新学习 runner，不追改历史证据。

## 主要结论

| 设计 | 目前判断 | 新 runner 首轮处理 |
| --- | --- | --- |
| 每个 horizon 只将第一个 observation batch 加入 normalizer | 高优先级学习风险；样本稀疏且 reset 初期方差很窄 | Pendulum 关闭；HalfCheetah 全轨迹 raw observations 累积，采样期固定快照 |
| pre-clip Gaussian action 与环境端剪裁/缩放 | 有效的 latent-action PPO；不是 log-prob 算错 | 保留以减少变量，记录动作饱和、mean、std |
| GAE 真终止与 TimeLimit 截断 | 目前逻辑正确，已有语义测试 | 保留，补多 horizon 边界测试 |
| value clip 0.2 + Huber 10 + reward scale 0.1 | 固定价值单位的 clip 会阻碍远离初值的 critic，Pendulum 尤其明显 | 首轮禁用 value clipping，保留明确的 Huber 与固定 scale |
| 最终 normalizer 下初始权重 vs 最终权重配对 | 可隔离“相同 final normalization 下的权重变化”，不是完整 time-0 policy baseline | 保存/eval完整初始与训练 policy bundle，同时保留该诊断对照 |
| 只有最终一次确定性评估 | 无学习曲线，也容易误读单个初始状态 | 定期固定验证集、预先固定 final test 集、3 个训练种子 |
| 当前 PPO KL / value-clip 指标 | 现有记录不足以诊断信任域与 critic 饱和 | 加可靠 KL、actual value delta、clip fraction、explained variance |

## 1. Normalizer 是最应先调整的学习设计

历史 `collect()` 在第一个消息处仅调用一次 `self.moments.update(...)`（第 305 行），
随后一个 horizon 的所有观测都用这个统计快照；该快照冻结是正确的。但收集全部
raw observations 与更新统计并不冲突：统计可在 horizon 后更新，下一 horizon 使用
新快照，当前优化继续使用保存好的 normalized observations。

在 `num_envs=8, horizon=128, iterations=100` 配方下，训练产生 102,400 transitions，
历史 normalizer 仅收到 800 observations，相当于只覆盖约 1/128。更严重的是第一个
快照只有 8 个 reset states：HalfCheetah 的 reset 速度/关节状态分布很窄，方差很小；
进入实际运动后，归一化值可能频繁被裁到 ±10，后续 Tanh 隐层进一步饱和。`count=1e-4`
只是初始数值稳定项，不构成实质性 warm-up。周期采样还可能对 episode phase 产生偏置。

建议新 runner：

- Pendulum-v1 首轮关闭 observation normalization。3 维状态本身尺度可控，减少
  一个会漂移的状态，使算法与设备验证更易解释。
- HalfCheetah-v5 收集整条 raw `s_t` 轨迹给累计 moments；训练/采样用冻结快照。
  可以先做 2 个 batch（2,048 transitions）随机策略 warm-up，统计量稳定后开始 PPO；
  这些 transitions 纳入 102,400 总预算并明确不产生优化更新。
- 每个统计更新只计算状态一次，避免 horizon 末尾 `s_T` 又作为下轮 `s_0` 被重复计入。
  Reset observations 可按实际采样状态纳入；仅用于 truncation bootstrap 的 terminal
  observations 不应人为重复加权。
- Warm-up 后仍记录每维 raw/normalized mean、variance，以及 normalization clip fraction。
  快照必须随权重一同保存。Evaluation 永远不更新 moments。
- 第一轮用持续累计统计；在 horizon 后更新统计导致下一轮 policy representation 变化，
  这是正常的 running-normalization 设计，但不受 PPO ratio 限制。因此记录快照漂移。
  若这个因素导致明显不稳定，再预先决定固定 warm-up normalizer 的对照实验。

关键约束是同一个 PPO batch 的观测、old logprobs、values、terminal bootstrap 与
new logprobs 采用同样的表示。不能在更新时重新用新的 moments 处理保存 raw obs，
而直接复用旧 logprobs。固定 normalized rollout tensor 最简单。

## 2. 动作剪裁与缩放本身不使 PPO 失效

模型输出 `u ~ Normal(mean, std)`，环境得到
`a = low + (clip(u, -1, 1)+1)*(high-low)/2`（第 282 行），batch 保存的是 u，
new/old logprob 都是 u 的 Gaussian 密度（第 316、390 行）。这可视为在 latent
action space 上定义 policy，再经过确定性环境变换，PPO 重要性比率一致。不能改成
用执行后的 clipped action 计算 Gaussian logprob；端点对应概率质量与内部密度混合，
简单密度值不能代表正确的执行动作分布。

会影响学习的实际问题是动作饱和和冗余。初始 `log_std=0` 即 std=1，mean 接近 0，
每维约 31.7% 的 latent action 越过 ±1。HalfCheetah 6 维动作中至少一个被裁剪的
概率约为 `1-0.6827^6 = 89.9%`（独立、mean=0 时）；Pendulum 单维执行 torque
范围 [-2,2]，std=1 的 latent 对应未裁剪 torque std=2。

许多不同 u 得到同一个极值 torque，增加梯度估计噪声。当前 entropy 是 latent
Gaussian entropy，不是执行动作 entropy；增大 latent std 不一定增加有效探索。
`log_std` 无边界也可能导致 action saturation 长期升高。

首轮保留 clipped Gaussian 与 `log_std_init=0`，作为常见连续控制 baseline；先记录
每维和任一维 saturation fraction、mean/std 范围与 entropy。若 std 无界增长再做
预先固定的 log_std bounds 或 `log_std_init=-0.5` 对照，避免看到失败后不断调参。
若后续改 tanh-squashed Gaussian，则存 pre-tanh u 或使用正确 Jacobian 修正，必须
标明算法变更；这不是首轮建立学习证据的必要条件。现有原始 Gaussian density
无需加入 env 线性缩放常数，因为同一变换下 log-ratio 常数抵消。

## 3. GAE 截断语义目前成立

EnvWorker 在 reset 前保存 final observation，collector 对 `truncated & ~terminated`
将 `gamma * V(final_obs)` 加到 scaled reward；随后将 `terminated | truncated` 设为
trace boundary。RLinf GAE 使用 `dones[t+1]` 同时阻断 next-value 和 GAE trace。

因此：

1. 真终止不 bootstrap，即使同时 truncated 也不补偿。
2. TimeLimit 只 bootstrap final state 一次，不使用 reset state，trace 不跨 episode。
3. 非终止 horizon 尾部用 `V(s_T)` bootstrap；不用把 collector horizon 人为标为 done。
4. 下一 horizon 的 `dones[0]=False` 不影响 RLinf 函数，因为其读的是 `dones[t+1]`。
5. `V` 在 scaled reward 单位中拟合；reward 已乘 scale，而 V 不再乘一次 scale，当前正确。

Pendulum 的 200 steps 和 HalfCheetah 的 1000 steps TimeLimit 采用 continuing-task
bootstrap，这是常见 Gym PPO 约定；evaluation 仍报告有限 episode 的 raw undiscounted
return。如果研究目标改为严格有限 horizon 的任务，需要把剩余时间加入状态并重新定义
terminal，不应悄悄沿用上述约定。

当前已有 terminal-vs-truncation/reset-value=100 语义测试。新 runner 需要额外检查
“horizon 尾部刚好 truncated”“horizon 尾部没 done”“多个 episode in one rollout”。
不建议因 RLinf docstring 写不支持 auto-reset 就删掉正确的补偿层；这里是外部显式
reset 加 TimeLimit correction，使输入满足 GAE 所需语义。

## 4. Critic clipping、Huber 与 scale 必须一起看

历史 runner 固定 `value_clip=0.2, huber_delta=10, value_coef=0.5`（第 398–409 行）。
RLinf 返回 `max(Huber(R-V), Huber(R-V_clipped))`，clipped value 以 old V 为中心。
当当前 V 已超过 clip 边界且朝 target 方向移动，clipped branch 的误差可能更大，
max 选择的是对 V 梯度为 0 的 branch。这是标准 value clipping 的作用，不是代码 bug；
但目标较远时会明显限速。

以一维例子说明：old V=0，R=-50，new V=-1，clip=0.2；clipped V=-0.2。两者
都处于 Huber 线性区，clipped error 49.8 比 original 49 更大，目标对 new V 梯度为 0。
更大 epoch 数无法简单解决该 batch 上的冻结。严格来说它不强制全局每次更新小于 0.2，
因为网络参数耦合/其他样本仍可改变预测，但会压抑该样本的正确学习方向。

Pendulum reward 常为负，每步约 0 到 -16，scale=0.1 后 continuing discounted V
可远离 0；HalfCheetah 初期/学习后的目标规模也不同。因此相同 0.2 threshold 在两个
环境与不同 scale 下不是相同约束。Normalization 的批间漂移还会改变未更新权重的
状态到 value 映射，增加跨 horizon 的拟合负担；但当前 batch 内统计冻结，old/new
value 仍在同一 normalized representation 上比较，这部分并无混用快照错误。

首轮新 runner 关闭 value clipping（明确 unbounded/disabled，不设置 clip=0），
保留 RLinf `huber_delta=10`、`value_coef=0.5` 与 `reward_scale=0.1`。若要复用该 API，
可传递 `float('inf')` 让 clipped value 等于当前 V，并按日志/配置清楚标注 disabled。
这仍然调用 RLinf critic objective，不需要换框架。是否更换普通 MSE 应作为后续单独
对照。Huber 的 quadratic 区是 `0.5 * error²`，再乘 value_coef=0.5 实际是
`0.25 * error²`；不同库常见的系数约定不能直接对照“0.5”这个配置数字。

Advantage normalization 让 actor 对正 scale 的整体幅值更不敏感，但 critic targets、
Huber 区间、value clipping、全局 grad clipping 都仍受 scale 影响。不能把 reward scale
改动说成完全等价。Evaluation/episode logs 必须保留 raw reward；当前 EnvWorker 和
evaluate() 已如此做。

固定的当前主线 `losses.py:426` 已用
`abs(values-prev_values)>value_clip` 记录真实裁剪前超界比例，不能把更早代码的
clip 后比较问题归到本轮版本。新 runner 显式禁用 clip，ratio 为 0 是预期结果；
若重新开启 clipping，可沿用该指标并补充实际 unclipped delta。
保留 RLinf explained variance，但 constant returns 时 NaN 是未定义指标，不能据此
判定训练 tensor 非有限。当前只记录 actor loss、critic loss、grad_norm、approx_kl，
对价值学习很不够。

## 5. Evaluation 配对需要两种明确口径

历史代码用 final moments 分别评估 initial_weights 与 final_weights，在同一 held-out
seed 和 deterministic mean action 下比较；这确实排除了“两个权重使用不同 final
统计”对该对照的影响。但 initial_weights+final moments 不是实际 time-0 policy，
也不是把所有 normalization 改善都从实际 policy 性能中排除掉。完整 policy 是
weights+normalizer bundle，学习曲线需要按每个 checkpoint 的 bundle 评估。

新 runner 建议同时记录：

- 主曲线：initial/trained checkpoint 各自保存的完整 policy bundle，在固定 validation
  seeds 上评估，冻结自身 moments；先从规定 warm-up/preprocessing 后记录 t=0。
- 辅助诊断：initial weights + final normalizer vs final weights + final normalizer，
  明确称为 final-normalizer 下的权重对照。Pendulum no-norm 时两种口径相同。
- Untrained deterministic initial policy 与独立 seeded random-action policy 都可做 baseline；
  不能只拿随机 baseline，忽略初始近零动作本来就能得到的回报。
- 固定训练 seeds [7,17,27]，共用 validation seeds 20000–20019，final test seeds
  30000–30049。训练 seed 不应偏移 evaluation 集，以减少不同 seeds 之间任务差异。
- 验证 checkpoint 不更新 normalizer、不探索、不改变训练随机数状态；如增加 stochastic
  evaluation，明确保存/恢复 Torch/MUSA RNG 或使用独立 evaluator，不污染训练轨迹。
- Validation 可用于诊断，最终成绩用预先固定 budget 的最后 checkpoint，在独立 final
  test 集评估。若报告 best checkpoint，注明选择用了哪一组 seeds，不能混为 final。

当前只有最后一次 evaluation，`eval_episodes=1` 的结果尤其不能推断可靠学习。
建议每 10 batches（10,240 transitions）评估一次，每次 20 episodes；raw returns
记录 mean/median/IQR 与逐 episode 值。最终 test 50 episodes，跨三个训练 seeds
先计算每 seed 的均值，再等权汇总，避免把 150 个 episode 当成 150 个独立训练实验。

## 6. 首套约 100k transitions 配方

下表是新 runner 的起始配方，不是对历史脚本可直接执行的 CLI 承诺，也不是已验证
达到分数的标准超参数。优先先跑 Pendulum，再验证 MuJoCo HalfCheetah。

| 参数 | Pendulum-v1 | HalfCheetah-v5 |
| --- | --- | --- |
| 训练 seeds | 7,17,27 | 7,17,27 |
| 总 transitions 每 seed | 102,400 | 102,400 |
| n_envs / horizon / batch | 8 / 128 / 1024 | 8 / 128 / 1024 |
| Warm-up | 0；不做 obs normalization | 2048 random-policy transitions，计入总预算 |
| 优化轮数 | 100 batches | 98 batches，前 2 batches 只 warm-up |
| MLP | actor/value 各 64×64 Tanh | 相同 |
| Sampling | Gaussian pre-clip u 保存，线性缩放 env bounds | 相同 |
| 初始 log_std / entropy coef | 0 / 0 | 0 / 0 |
| PPO epochs / minibatch | 10 / 256 | 10 / 256 |
| Actor clip | 0.2 | 0.2 |
| gamma / lambda | 0.99 / 0.95 | 0.99 / 0.95 |
| Adam lr / epsilon | 3e-4 / 1e-5，首轮 constant lr | 相同 |
| value_coef / Huber delta | 0.5 / 10 | 相同 |
| value clipping | disabled | disabled |
| reward scale | 0.1，eval raw | 相同 |
| Advantage normalization | 全 batch，population std | 相同 |
| global grad L2 clip | 0.5 | 0.5 |
| Observation normalization | disabled | 全 raw-state累计，rollout快照冻结 |
| Evaluation | 每 10 batches，20 validation episodes，最终 50 test | 相同 |

每 batch 40 minibatch optimizer steps；Pendulum 共 4000 optimizer steps，HalfCheetah
共 3920，不计同 batch resume regression 的校验重放。首次兼容 smoke 与 timing 单独
记录，不冒充学习预算。三个种子总采样 307,200 transitions/环境；evaluation steps
另记，并不混入训练 transitions。

关键 diagnostics：unscaled episodic return/length、scaled reward/value/return quantiles、
advantage mean/std、actor clip fraction、log-ratio range、可靠 KL、latent std/entropy、
执行动作 saturation、normalization clipping、critic explained variance、unclipped value
delta、preclip grad norm、每次 update 的 wall time 与真实 transitions/sec。

RLinf 现有 `approx_kl=-mean(log_ratio)` 可因采样变成负数，不能直接当作非负约束。
建议额外记录 `mean(exp(log_ratio)-1-log_ratio)`，以及预先规定 target KL=0.02 的
epoch early stop（同一 old batch 下评估，超过 1.5×target 则停止余下 epochs）。
若第一版暂不做 early stop，也必须记录这个 KL 并将严重超界单独标为需要诊断，
不能把绝对值套在有符号近似上当精确 KL。

## 7. 验收与失败如何解释

下列分数是本项目预先声明的工程验收目标，不能标成 RLinf 官方 benchmark 标准。

1. **实现正确性**：旧终止/截断测试保留；新增多 horizon 与正常 nonterminal bootstrap
   测试；一个完整 batch 中 normalization snapshot 不变；stored old logprob 与更新前
   policy 重算一致；3 seeds 训练 tensor/gradients 有限；same-batch resume 回归成立。
2. **Pendulum 学习证据**：每 seed 在 final test 上算 trained minus initial bundle 的
   paired mean return gain；至少 2/3 seeds 为正，3 个 gain 的中位数 ≥300。另记录
   final mean return 中位数达到 -500 作为更强的实用目标。改善但未到 -500 可说
   有学习证据，不能说已解决 Pendulum。
3. **HalfCheetah 100k 探索性学习证据**：同样至少 2/3 paired gains 为正，gain 中位数
   ≥300；final mean return 中位数 ≥500 作为较强目标。100k 是早期诊断预算，未达到
   固定分数不等于 S4000 或 RLinf 不兼容；需要看 normalization、value fit、KL 与同
   算法 CPU 参考。不能仅因 loss 下降或 parameter delta 非零就宣布学习。
4. **统计表达**：报告全部 3 seeds 与每个 test episode；以 train seed 为外层单元，
   3 seeds 的区间估计非常不稳定，不给“统计显著”强结论。若缺一个 seed、最终性能
   仅优于 random 而不优于 initial policy、或只挑 best checkpoint，标明限制。
5. **异常诊断顺序**：先排查 nonfinite/错误 bootstrap/normalizer clip/action saturation；
   再检查 critic explained variance 与 target量级、KL/actorclip；最后才改变 epochs/lr/
   reward scale。每个更改独立命名并重新固定配方，保留失败结果与原预算。

若 Pendulum 有清楚提升而 HalfCheetah 100k 没有，首要结论是小连续控制学习链路成立，
MuJoCo workload 还需算法/采样预算诊断。后续可以按预先计划扩到 300k–1M transitions
或做 CPU 同 runner 对照；这些是下一阶段，而非用来抹掉本轮结果。

## 审计来源定位

- 历史 runner：`routes/route1/mujoco_ppo_worker.py`，RunningMoments 第 42–65 行，
  GaussianMLP 第 69–105 行，action transform 第 282–283 行，collector 第 302–357 行，
  PPO update 第 383–420 行，evaluation 第 451–485 行，final comparison 第 696–705 行。
- RLinf v0.3：commit `0505431899574619da86f551bad70b71e0ea2177`；
  `rlinf/algorithms/advantages.py` 的 GAE 使用 dones[t+1]；
  `rlinf/algorithms/losses.py` 第 347–385 行的 clipped Huber/value clip metric。
- 历史证据：`reports/route1-loop.json`，256 transitions，8 optimizer steps，
  paired held-out seed 10007；`routes/route1/test_semantics.py` 的时间截断测试。
