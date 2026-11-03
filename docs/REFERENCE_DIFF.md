# pqn_v5.py 与官方 PQN-MinAtar 实现的逐项对照（第 2 轮）

- 官方：<https://github.com/mttga/purejaxql>，提交 `47af6d7b35c89ddfe633aaf7341bdb8964cb7cce`（`reference to render script`），
  只读克隆在 `third_party/purejaxql`（已 .gitignore）。文件：`purejaxql/pqn_minatar.py`（下称 **官方 L**）与
  `purejaxql/config/alg/pqn_minatar.yaml`（下称 **yaml L**）。
- 我方：`pqn_v5.py`（下称 **v5 L**）。立场：审稿人，默认怀疑；每一行都由 `tests/test_reference.py`（26 项，全部通过）或
  `test_v5.py` 提供数值证据，或在“证据”列说明为何只能靠阅读。
- 状态含义：**一致** = 逐行/逐值相同；**有意** = 计划 v5.1 明确要求的差异（括号内为对应条目）；**无意** = 非计划内的差异（视同 bug，必须报告）。

## 一、对照表

| # | 项目 | 官方（文件行） | pqn_v5.py（行） | 状态 | 证据 |
|---|---|---|---|---|---|
| 1 | 并行环境数 | `NUM_ENVS: 128`（yaml L5） | `--num_envs` 默认 128（v5 L30） | 一致 | `test_default_hyperparameters_equal_official_yaml` |
| 2 | rollout 长度 | `NUM_STEPS: 32`（yaml L6） | `--num_steps` 默认 32（v5 L31） | 一致 | 同上 |
| 3 | epoch / minibatch | `NUM_EPOCHS: 2`、`NUM_MINIBATCHES: 32`（yaml L10-11） | `--epochs` 默认 2、`--minibatches` 默认 32（v5 L32-33） | 一致（E 作为实验变量可改） | 同上 |
| 4 | 总步数与更新次数 | `NUM_UPDATES = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS`（官方 L91-93）= 2441，总梯度步 156 224 | `NUM_UPDATES = int(steps // (num_envs*num_steps))`（v5 L64-66）= 2441，`GRAD_STEPS = 156224` | 一致 | `test_lr_schedule_values_equal_official` 断言 `GRAD_STEPS == 156224` |
| 5 | 学习率 | `LR: 0.0005`（yaml L13） | `--lr` 默认 5e-4（v5 L35） | 一致 | 同 1 |
| 6 | 学习率调度 | `optax.linear_schedule(LR → 1e-20, NUM_UPDATES_DECAY × NUM_MINIBATCHES × NUM_EPOCHS)`，`LR_LINEAR_DECAY: True`（官方 L140-147、yaml L15）；时钟 = 梯度步（optax 内部计数） | `optax.linear_schedule(lr, 1e-20, GRAD_STEPS)`（v5 L87），接在 `scale_by_learning_rate`（v5 L98）；时钟 = 梯度步 | 一致 | 第 0、G/2、G−1 步取值相等（rel 1e-6）；并且在这三个梯度步上用相同状态、相同梯度比较两条优化器链的**更新量**，相等（`test_optimizer_applies_the_scheduled_lr_at_given_gradient_step`） |
| 7 | ε 调度 | `linear_schedule(1.0 → 0.05, EPS_DECAY × NUM_UPDATES_DECAY)`，**浮点**步数 = 244.1，按 `n_updates` 取值（官方 L134-138、L195；yaml L7-9） | `linear_schedule(1.0 → 0.05, max(1, int(0.1 × NUM_UPDATES)))` = **244 步**（v5 L88） | **无意（量级极小，非 bug 级）** | 第 0、50 %、末次更新取值完全相等；仅在 0<u<245 的区间内差 ≤ 3.9×10⁻⁴（u=244 处官方 0.050389，我方 0.05）。见下文“发现 1” |
| 8 | 梯度裁剪 | `clip_by_global_norm(MAX_GRAD_NORM=10)`（官方 L160、yaml L14） | `optax.clip_by_global_norm(10.0)`（v5 L95） | 一致 | `test_v5.py` 第 4 项（`tx` 与 `optax.chain(clip, radam)` 逐步参数差 0.0） |
| 9 | 优化器 | `optax.radam(learning_rate=lr)`，默认 β1=.9、β2=.999、阈值 5（官方 L161） | `scale_by_radam()` + `scale_by_learning_rate(lr_sched)`（v5 L95-99；wd 臂在中间追加 `add_decayed_weights`） | 一致（`--norm_ctrl none`） | `test_v5.py` 第 4 项；wd/radial/np 为有意新增（计划 3.5/4） |
| 10 | 网络结构 | `BatchNorm`（哑，输出未使用）→ `x/255` → Conv(16, 3×3, VALID, He) → LN → ReLU → flatten → Dense(128, He) → LN → ReLU → Dense(|A|)（官方 L24-69） | Conv(16, 3×3, VALID, He) → LN → ReLU → flatten → Dense(128, He) → LN → ReLU → Dense(|A|)；`x/255` 仅当 `--input_scale255`（v5 L71-83） | 一致（去掉 `/255` 是**有意**，对应计划命题 2 / C1；哑 BatchNorm 无输出影响，见“说明”） | `test_param_tree_shapes_equal_official`（逐层形状）；`test_network_function_equals_official`：把官方网络参数映射到我方，两网络输出相差 < 1e-4（`--input_scale255` 与官方一致；去 255 时等于官方网络喂 x/255） |
| 11 | 初始化 | Conv/Dense(128) He-normal；输出层默认（lecun normal）；LN scale=1、bias=0（官方 L38-48、L68） | 同（v5 L80-83） | 一致 | `test_he_init_dense_norm_squared`：20 次抽样 E‖W₀,dense‖² 与 256 的相对误差 < 5 %；conv 与 32 同 |
| 12 | LayerNorm 位置 | conv 之后、ReLU 之前；Dense(128) 之后、ReLU 之前；flax 默认只对通道轴归一化（官方 L45、L49） | 同（v5 L81-82） | 一致 | 上一行的函数等价测试（LN 的 scale/bias 取非平凡值） |
| 13 | γ、λ | `GAMMA: 0.99`、`LAMBDA: 0.65`（yaml L16-17） | `--gamma 0.99`、`--lam 0.65`（v5 L36-37） | 一致 | 同 1 |
| 14 | 输入预处理 | `x = x / 255.0`（官方 L66，对**二值**输入） | 默认**不除**；`--input_scale255` 复现（v5 L77-78） | 有意（C1、命题 2、推论 1） | 同 10 |
| 15 | Q(λ) 目标递推 | `_get_target`（官方 L237-250）：`bootstrap = r + γ(1−d)·next_q`，`λ-ret = bootstrap + γλ(λ-ret − next_q)`，终止时 `= r`；初值 `last_q·(1−done[−1])`（L252）与 `r[−1] + γ·last_q`（L253），**`next_q` 的初值是 V(s_T) 而非 V(s_{T−1})**（命题 1 的 off-by-one） | `targets_public`（v5 L123-137）：`--buggy_target` 时初值 `nq0 = last_q·(1−done[−1])`（逐行同官方）；默认 `nq0 = max Q(s_{T−1})`（修正） | `--buggy_target`：一致；默认：有意（C1、命题 1） | `test_buggy_mode_equals_official_transcription`：把官方 L227-260 用 `lax.scan` 原样转录（done 取 bool，与 gymnax 一致），5 组随机输入（含中途终止与最后一步终止）最大误差 < 1e-5；`test_default_mode_differs...`：修正后只在 t ≤ T−2 处改变目标 |
| 16 | 目标的一般形式 | （命题 1 给出）误差 = γ(1−λ)(γλ)^{T−2−t}[(1−d_{T−1})V(s_T) − V(s_{T−1})]·∏(1−d_j) | — | — | `test_proposition_1_closed_form`：3 组 (γ,λ) × {无终止, 有终止（含 T−1 终止）}，最大误差 < 1e-5；T−1 步误差为 0 |
| 17 | 标准 Q(λ)（含截断） | 官方无截断处理 | `targets_fixed`（v5 L140-152）：`G_t = r_t + γ[(1−d_t)((1−λ)V_{t+1} + λG_{t+1}) + trunc_t·V(final_t)]` | 有意（Pardo 2018；仅 P0 的 `fixed` 臂与 P1/P2 打开 `--bootstrap_timeout`） | `test_fixed_targets_match_brute_force_with_truncation`：对 4 组 (γ,λ)（含 λ=0、λ=1）用 float64 逐步暴力实现比较，误差 < 1e-4（float32 递推 32 步） |
| 18 | 终止/超时处理 | gymnax `done` = 终止 **或** 达到 `max_steps_in_episode`（1000），统一当终止，不 bootstrap | 默认同官方；`trunc = done ∧ (time ≥ MAXT)`（v5 L160-170）；`--bootstrap_timeout` 才 bootstrap | 一致（默认）/有意（`fixed` 臂） | `tests/test_protocol.py`（第 3 轮）核对截断标记 |
| 19 | minibatch 构造 | 把 (T, n) 展平成 T·n，用**同一个 key** 对 transitions 与 targets 做相同置换，再切 32 份（官方 L299-315） | `perm = permutation(kp, N)`，`x[perm].reshape(M, −1, …)`，O/A/tgt 共用同一 `perm`（v5 L220-222） | 一致（置换随机数流不同，分布相同） | 阅读；逐样本配对性由共用同一置换保证 |
| 20 | 损失 | `0.5·mean((Q(s,a) − target)²)`，target 在损失外计算（官方 L285） | 同（v5 L214-216） | 一致 | 阅读 |
| 21 | 动作选择 | 每步用 rollout 开始时的网络；ε-greedy，所有环境同一 ε（官方 L193-196） | 同（v5 L186-191）；另加 sticky（见 22） | 一致 | 阅读 |
| 22 | sticky actions | 无（gymnax 去掉了 MinAtar 的 sticky） | `--sticky p`：以概率 p 重复上一步**执行**的动作，智能体存储自己**选择**的动作（v5 L194） | 有意（C0；主协议 p=0.1） | `tests/test_protocol.py`（第 3 轮） |
| 23 | 环境 auto-reset | gymnax `step`：done 时用 `reset_env` 的状态/观测替换（含 `lax.select`） | `env_step`：同样的 `select`，另外返回 reset 前观测与截断标志（v5 L160-170） | 一致（附加输出为有意） | 阅读；`test_v5.py`、第 3 轮协议测试 |
| 24 | rollout 末尾 bootstrap | `last_q = max Q(next_obs[−1])`（官方 L227-235），再乘 `(1−done[−1])`（L252） | `last_q = max Q(obs)`，`obs` 为最后一步的 post-step（含 reset）观测（v5 L203） | 一致 | 15 行测试 |
| 25 | 评估方式 | 训练中每 `int(NUM_UPDATES·0.05)` = 122 次更新评估一次；128 个环境各跑 `max_steps_in_episode`（1000）步，ε=0（贪心），对**所有结束的局**（含同一环境内的多局）取 `nanmean`（官方 L340-350、L371-413；yaml L24-27） | 每 `round(5e5/4096)` = 122 次更新评估一次（v5 L67）；**双协议**：公开（无 sticky、上限 1000）与稳健（sticky 0.1、上限 10000），各 64 个并行局，**每个环境只取第一局**，局结束即停（v5 L254-274、L327-334） | 有意（C0：公开协议在 gymnax 下饱和；64 局为计划指定；评估点节奏与官方一致） | 公开协议的数字与官方评估口径不完全同构（官方多局平均、128 环境）。因 gymnax SpaceInvaders 动力学确定、贪心策略确定，两种口径给出同一条轨迹的回报，所以对主结论无影响；停止规则 1 比较时需在报告中注明口径差异 |
| 26 | 评估点数量 | 20（每 5 %） | 10M 步时 20 个（余数并入最后一块，CHANGELOG 第 19 条） | 一致 | `test_v5` 之外见第 1 轮冒烟的评估点间隔检查 |
| 27 | 随机数流与 seed | `PRNGKey(SEED)` 再 `split(NUM_SEEDS)` | 训练 seed i 用 `PRNGKey(seed0+i)`（v5 L284-285） | 有意（CHANGELOG：与进程内 seed 数无关） | `tests/test_protocol.py`（第 3 轮） |
| 28 | 版本 | `jax>=0.4.16,<=0.4.38`、`gymnax==0.0.6`（pyproject） | 同：`jax==0.4.38`、`gymnax==0.0.6` | 一致 | `docs/ENV.md` |

## 二、发现

### 发现 1（无意差异，非 bug 级，建议不改）：ε 衰减长度取整
官方 `transition_steps = EPS_DECAY × NUM_UPDATES_DECAY = 0.1 × 2441 = 244.1`（浮点，官方 L137），我方 `int(0.1 × 2441) = 244`（v5 L88）。
- 效果：ε 在第 244 次更新到 0.05（官方在第 244.1 次）。仅在 u=244 附近差 3.9×10⁻⁴，其余更新取值完全相同（测试输出 `eps diffs {0: 0.0, 244: 0.000389, 1220: 0.0, 2440: 0.0}`）。
- 其他步数（如 `--steps 1e6`）下量级同为 ≤ 1 个更新的 ε 变化。
- 建议：**不改**。改动会使 P0–P2 的随机流/数值与已通过测试的版本不同，而偏差远小于任何评估噪声（按 CLAUDE.md 规则 7，若要改须加开关、默认保持、记入 CHANGELOG）。如你们希望与官方逐位一致，可新增 `--eps_float_decay` 开关；本轮未改 pqn_v5.py。

### 说明（不算差异）
- **哑 BatchNorm**：官方 `QNetwork` 在 `norm_input=False` 时仍建一个 `nn.BatchNorm`（官方 L65），其输出 `x_dummy` 从不使用，所以对网络函数和梯度没有影响（但会在参数树中多出全零梯度的 `BatchNorm_0`，全局梯度范数不变）。我方不含这层。函数等价测试已涵盖。
- **参数命名**：官方 `CNN_0/Conv_0/...`，我方 `conv/dense/ln0/ln1/head`；测试中显式映射。
- 官方 `rng` 的分割次数与我方不同，所以相同 seed 下两份代码的轨迹不会逐位一致；本轮的比较都是在**同一份给定输入**上做函数级比较。

## 三、结论
- 公开模式（`--input_scale255 --buggy_target`，sticky 0，上限 1000）与官方一致：目标递推逐行一致（< 1e-5），网络函数一致（< 1e-4），优化器与调度一致。
- 默认模式只差计划所述的两处修正（命题 1、2）以及 `--bootstrap_timeout`（仅 `fixed` 臂）。
- 唯一的“无意差异”是 ε 衰减步数取整（发现 1），量级可忽略，已报告，等待你们决定是否处理。

复现：`python -m pytest -q tests/test_reference.py`；`python test_v5.py`。
