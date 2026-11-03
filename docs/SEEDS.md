# Seed 身份与评估配对（第 3 轮结论）

所有结论由 `tests/test_protocol.py`（23 项，CPU，约 1 分钟）自动检验；下面是每条结论对应的测试与现状。

## 1. 训练 seed 的身份
- 训练 seed `i` 使用 `PRNGKey(seed0 + i)`（`pqn_v5.py`：`SEED_IDS = np.arange(a.seed0, a.seed0 + a.seeds)`，
  `jax.vmap(jax.random.PRNGKey)(SEED_IDS)`）。
- **与同进程 seed 数无关**：`--seeds 2 --seed0 5` 的第 2 个 seed（id 6）与 `--seeds 1 --seed0 6` 的唯一 seed 在
  `hist`（train_return、范数、转角累计、dormant 等）与 `evals`（两种协议的回报、局长、触顶比例）上**逐比特相同**
  （`test_seed_stream_does_not_depend_on_how_many_seeds_share_the_process`，CPU、JAX 0.4.38）。
  注意：这是 CPU 上的结论；GPU 上 vmap 宽度可能改变归约顺序，因此 P0–P3 仍按计划“同一阶段内所有臂使用相同的 seed 个数与进程分组”运行。
- **同一命令跑两次，JSON 中所有数值完全相同**（除 `wall_seconds`、`hist.chunk_seconds`）
  （`test_same_command_twice_gives_identical_numbers`）。
- 每个输出 JSON 记录 `seed_ids` 与 `versions`（jax、jaxlib、flax、optax、gymnax、`jax_threefry_partitionable`、backend）
  （`test_json_records_versions_and_actual_eval_points`）。随机流随 JAX 版本变化，所以**一个阶段内不得更换运行环境**。

## 2. 各阶段的 seed 范围
由 `sweep.py <stage> --dry` 解析出每个臂的最终生效参数（`tools/effective_args.py`）：

| 阶段 | seeds | seed0 | seed id | 说明 |
|---|---|---|---|---|
| P0 | 10 | 0 | 0–9 | 5 个臂完全相同 |
| P1 | 3 | 100 | 100–102 | 19 个臂完全相同（开发 seed） |
| P2 | 10 | 1000 | 1000–1009 | 4 个臂完全相同（确认 seed） |
| P3 | 5 | 2000 | 2000–2004 | 12 个配置（3 个游戏 × 4 臂）完全相同 |

- 四个阶段的 seed 范围两两不相交（开发与确认分离）；每个进程 `--seeds ≤ 10`（CLAUDE.md 规则 8）。
  （`test_all_arms_of_a_stage_use_the_same_seeds`、`test_stage_seed_ranges_are_disjoint_and_match_the_plan`）

## 3. 评估环境的配对
- 评估环境的随机键 `ek = fold_in(PRNGKey(10_000 + 评估点的更新数), seed_id)` **只依赖（评估点，seed id）**，不依赖任何超参数、
  训练状态或训练随机数（`test_evaluation_keys_depend_only_on_eval_point_and_seed_id` 直接对 `pqn_v5.py` 中的这一行求值：
  seed 6 无论与谁同进程，键相同；不同 seed、不同评估点键不同）。因此同一评估点上，不同配置、同一 seed id 的评估环境键相同，
  配对差（analyze.py）成立。
- 评估环境本身（gymnax SpaceInvaders）的动力学与 reset 都是确定的（`step_env` 不使用随机键），评估中唯一的随机性是 sticky
  替换所用的均匀随机数；因此评估键只在稳健协议下有意义，公开协议（sticky 0）与键无关（命题 6a）。

## 4. 协议（命题 6）的测试结论
- **6a 确定性**：sticky 0、上限 1000 时，8 局回报完全相同，且与评估键无关；sticky 0.1 时 8 局回报不全相同
  （手写的反应式策略；随机初始化的 PQN 网络几乎都是常数动作策略，在 sticky 下也几乎不变，所以不能用来演示稳健协议的非退化性）。
- **6b 截断**：把训练/稳健上限设为 10000 时，公开协议的局长 = 1000（`PUBLIC_CAP` 取 gymnax 默认），稳健协议的局长可达 10000
  （用“冻结的游戏”——外星人不动不射——这样只有上限能结束一局）。
- **回报掩码**：`greedy_eval` 的每局回报只累计第一次 done 之前的奖励；用独立的 Python 循环（同样的 key 序列）逐局比较，
  n=1 时 4 个不同评估键逐位相同，n=4 时均值、局长、触顶比例相同，且“不加掩码”的均值确实更大（掩码有作用）。
- **截断标记**：cap=50 时，`trunc` 只在 `time ≥ cap` 导致的 done 上为 1，真正的游戏结束（局长 < cap）为 0，非 done 步为 0；
  用不带上限的游戏重放每一局的动作序列，独立确认哪些是真实游戏结束。
  已知的边界情形：游戏恰好在第 cap 步自然结束时 `trunc=1`（代码以 `time ≥ cap` 判定）。cap=10000 时概率约 1/10000，
  且只有打开 `--bootstrap_timeout` 的臂受影响，量级可忽略。

## 5. sticky actions 的语义（测试结论与一处提示词修正）
- 取代规则：以概率 p 重复**上一步执行的动作**，否则执行选择的动作；回合结束后“上一步动作”重置为 0（无操作）；
  **写入学习数据的是智能体“选择的动作”**，不是执行的动作（`update()` 中 scan 输出的第 2 项是 `chosen`；AST 检查
  `test_update_stores_the_chosen_action_and_remembers_the_executed_one` 固定住这一点）。
- 统计检验：对固定选择序列 0,1,0,1,…，在“选择动作 ≠ 上一步执行动作”的步上，执行动作等于上一步执行动作的频率服从 Binomial(·, p)
  （p=0.1 与 0.3，N=40000，α=0.001，通过）。
- **提示词与实际的不符（已按实际写测试）**：第 3 轮提示词写“执行动作等于上一步执行动作的频率 ≈ p”。对交替序列 0,1,0,1,… 这个**无条件**频率
  不是 p，而是 2p/(1+p)（p=0.1 时 0.1818）：未触发 sticky 的步上，若选择动作恰好等于上一步执行动作，同样算“重复”。
  所以对固定序列检验的是上述**条件**频率；另外加了一个自适应序列 `chosen_t=(prev_t+1) mod 4`（保证 chosen ≠ prev），
  其无条件重复频率恰为 p，通过二项检验。这不是代码问题，只是测试陈述的修正。
- 为了能对真实代码做上述测试，`pqn_v5.py` 抽出了两个纯函数 `sticky_select(u, p, prev, chosen)` 与 `next_prev(done, act)`
  （表达式原样搬出，update() 与 greedy_eval() 调用它们）。抽取前后的冒烟运行（2 seeds、20 480 步、sticky 0.1）的 JSON（除耗时外）
  与保存的参数**逐比特相同**；已记入 CHANGELOG.md。
