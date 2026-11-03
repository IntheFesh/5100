---
title: "Claude Code 多轮实现提示词（DRL 期末项目 v5.1）"
subtitle: "More Updates, More Learning? —— PQN 的评估协议、实现修正与范数自衰减"
author: "Yueyi Li（603437） · Cheng Qian（607522）"
date: "2026 年 11 月 3 日"
---

# 怎么用这份提示词

**起点。**这些提示词基于随计划提供的代码包 `drl_v5_code.zip`（已通过单元测试，并经过一位未参与编写的审查者逐项复核；审查发现的 13 个问题已全部修正，见包内 CHANGELOG.md）和研究计划 `DRL_研究计划v5.1_Li_Qian.pdf`。不要让 Claude Code 从零重写：它的任务是**复核、补测试、跑分析、写论文**，而不是重新发明已经验证过的代码。

**分工。**Claude Code 在你们自己的电脑上工作（读写代码、跑小规模测试、分析结果、写论文）；GPU 训练在 Colab 上由你们手动运行 notebook。流程是：

| 轮次 | 谁来做 | 内容 | 预计时间 |
|---|---|---|---|
| 0 | 你 | 建仓、放入 CLAUDE.md | 10 分钟 |
| 1 | Claude Code | 环境与基线核对 | 30 分钟 |
| 2 | Claude Code | 对照 purejaxql 的独立复核 | 1 小时 |
| 3 | Claude Code | 协议、评估与 seed 的测试 | 1 小时 |
| 4 | Claude Code | 机制分析脚本（预测 1--3 的检验） | 1 小时 |
| 5 | Claude Code + 你 | Colab 打包、计时、算力预算 | 1 小时 |
| — | 你（Colab） | 跑 P0 | 依计时而定 |
| 6 | Claude Code | P0 分析与停止规则判定 | 30 分钟 |
| — | 你（Colab） | 跑 P1 | |
| 7 | Claude Code | P1 筛选、机制检验、冻结 P2 配置 | 1 小时 |
| — | 你（Colab） | 跑 P2（及 P3） | |
| 8 | Claude Code | 确认分析、全部图表、数字宏 | 2 小时 |
| 9 | Claude Code | 展示视频 | 1 小时 |
| 10 | Claude Code | 论文写作（NeurIPS 格式） | 多次会话 |
| 11 | Claude Code | 投稿前自审与修改 | 1 小时 |

**每一轮的操作规范。**

1. 每轮开一个新会话（或在上一轮结束后输入 `/clear`），这样上下文干净，CLAUDE.md 会自动加载。
2. 第 2、3、4、8、10 轮会改代码或产出大量内容：先按 `Shift+Tab` 进入 plan mode，让它给出计划，你确认后再执行。
3. 每轮结束时它会按固定格式汇报（见每个提示词末尾）。**逐条核对“验收标准”**，没过就不要进入下一轮，把失败信息贴回去让它修。
4. 每轮通过后让它 `git commit`（提示词里已经要求），出问题可以回退。
5. 中途它提出要改训练行为（`pqn_v5.py` 的默认值、协议、配置），而该阶段已经开始跑了：**拒绝**，除非你们确认这是 bug，并把改动记入 `CHANGELOG.md`。

**三条不可让步的纪律**（已写进 CLAUDE.md，Claude Code 每轮都会读到）：

- 不挑 seed、不挑检查点、不挑局；P2 配置在看确认 seed 之前冻结。
- 论文里的每个数字都由脚本从结果 JSON 生成，不许手抄。
- 协议效应（C0）、实现修正（C1）、方法（C3）三者分开报告，前者不算任何方法的提升。

\newpage

# 第 0 轮：建仓与 CLAUDE.md（你手动完成）

在终端里：

````bash
mkdir drl-pqn && cd drl-pqn
unzip /path/to/drl_v5_code.zip -d .
mkdir -p docs results paper tests
cp /path/to/DRL_研究计划v5.1_Li_Qian.pdf docs/plan_v5.1.pdf
git init
````

然后在仓库根目录新建 `CLAUDE.md`，内容如下（整段复制）：

````markdown
# 项目：More Updates, More Learning?（DRL Fall 2026 期末项目，Yueyi Li & Cheng Qian）

## 一句话
在 PQN（无回放并行 Q 学习，Gallici et al. ICLR 2025；MinAtar via gymnax）上：
(C0) 修正评估协议——公开设定（无 sticky actions、1000 步上限）下贪心评估饱和且完全确定；
(C1) 两个精确的实现偏差（Q(λ) 边界 off-by-one；二值输入除以 255）；
(C2) 诊断：提高每批更新次数 E 时，LayerNorm 前的权重范数增长造成有效学习率“自衰减” Λ_t=||W_t||/||W_0||；
(C3) 检验范数控制（NP / 径向投影 / 解耦权重衰减）能否把更高的 E 转化为样本效率；
(C4) 游戏展示。主游戏 SpaceInvaders-MinAtar；P3 稳健性用 Breakout/Asterix/Freeway。
交付：7–10 页 NeurIPS 格式论文 + 3 分钟展示视频 + 可复现代码。截止约 12 月中旬。

## 权威文档
docs/plan_v5.1.pdf（研究计划 v5.1：理论、先导结果、P0–P3 设计、停止规则、预注册选择规则）。
本文件与计划冲突时以计划为准，并且必须告诉我冲突在哪里。

## 文件
- pqn_v5.py：训练。默认=修正 PQN；--input_scale255 --buggy_target 复现公开代码；--norm_ctrl none|np|radial|wd；
  --sticky / --max_steps_in_episode / --eval_cap / --eval_sticky 控制协议；--save_params / --ckpt_at 保存参数。
  输出 JSON：seed_ids、versions（JAX 等版本）、ckpts（快照的实际步数）、
  hist{env_steps, train_return, wn_dense, rad_cum, quad_cum, ang_cum, trunc_frac, ...}，
  evals[{env_steps, greedy_return(公开协议), capped_frac, public_len, robust_return(稳健协议,主指标), robust_capped_frac, robust_len}]。
  训练 seed i 用 PRNGKey(seed0+i)，与同进程 seed 数无关。
- test_v5.py（最后打印 ALL PASS）、tests/test_stats.py：单元核验，必须一直全部通过。
- sweep.py：P0–P3 配置，可断点续跑，拒绝“同名不同参数”的重复运行，每次运行有 .log；
  analyze.py：AUC、Student-t 区间、配对差、精确符号翻转检验 + Holm、--filter（按游戏）、--iqm、--json_out；
  render_demo.py：2–3 栏展示视频（--seed_indices 每栏指定训练 seed）；check_protocol.py：任一保存模型在两种协议下复测。
- CHANGELOG.md：所有改动训练/评估行为的修改记录。
- DRL_v5_colab.ipynb：Colab 流程。pilot_results/：CPU 先导的原始日志与 JSON（只读）。
- results/：从 Google Drive 下载的正式结果（只读，不许改）。paper/：论文。tests/：新增测试。

## 硬性规则
1. 主协议：训练与评估都用 sticky actions p=0.1、每局上限 10000 步；只有 P0 的 public_gymnax 臂用公开设定。
2. 已开始运行的阶段，其配置不许再改；P2 配置在 P1 结束后写入 frozen_P2.json 并打 git tag，之后不许改。
3. 不挑 seed、检查点或局。展示视频：训练 seed 取该配置稳健 AUC 的中位数 seed；局取 9 局完整得分的中位数局。
4. 论文/报告里的每个数字都由脚本从 results/ 的 JSON 生成（paper/numbers.tex 宏），禁止手抄或“估计”。
5. C0（协议）、C1（修正）、C3（方法）的效应分开报告；协议效应不算任何方法的提升。
6. 少于 5 个 seed 不报置信区间和 p 值，只报逐 seed 数值；5 个 seed 时精确检验最小 p=0.0625，P3 只作描述。
   置信区间用 Student-t，p 值用精确符号翻转检验（不要换回百分位 bootstrap：n≤10 时偏乐观）。
7. 改 pqn_v5.py 的训练行为：必须加开关、默认保持现状、跑 test_v5.py 与冒烟测试、记入 CHANGELOG.md。
8. 单进程 vmap 的 seed 数 ≤ 10。一个阶段内不得更换 JAX 运行环境（随机流随版本变化，JSON 中的 versions 必须一致）。
9. 不确定就停下来问我，不要猜；发现计划本身有问题，先报告，不要自行“修正”计划。

## 常用命令
- 环境：source .venv/bin/activate
- 测试：python test_v5.py && python tests/test_stats.py && python -m pytest -q tests/
- 冒烟：python pqn_v5.py --steps 16384 --num_envs 32 --minibatches 8 --seeds 2 --eval_episodes 2 --eval_cap 300 --sticky 0.1 --max_steps_in_episode 10000 --bootstrap_timeout --out /tmp/smoke.json
- 配置清单：python sweep.py LIST --outdir /tmp/x ；某阶段命令预览：python sweep.py P0 --outdir /tmp/x --dry
- 分析：python analyze.py results/P0 --metric robust_return --ref public

## 术语
n=128 并行环境，T=32 步/rollout，E=epoch 数，M=32 minibatch；U=EM/(nT) 每条转移的梯度步数；K=EM。
AUC = 每个 seed 在所有评估点上的平均回报（按环境步等间隔）。稳健协议=sticky 0.1、上限 10000；公开协议=无 sticky、上限 1000。
Λ_t=||W_t||/||W_0||（隐藏全连接层，含偏置）；s=d||W||²/dΣη_t²；A=每环境步累计转角。

## 汇报格式（每轮结束必须用）
1. 做了什么（文件与函数级）；2. 验收标准逐条：通过/失败 + 证据（命令与输出摘录）；
3. 发现的问题与风险（含计划层面的）；4. 需要我决定的事；5. git commit 的哈希。
````

最后：

````bash
printf '*.pkl\n*.mp4\n.venv/\n__pycache__/\nthird_party/\n' > .gitignore
git add -A && git commit -m "init: v5.1 code package, plan, CLAUDE.md"
````

\newpage

# 第 1 轮：环境与基线核对

````text
这是一个强化学习课程项目的代码仓库。请先完整阅读 CLAUDE.md，再读 README.md，然后浏览 docs/plan_v5.1.pdf 的第 5、6、7 节（先导结果、正式实验设计、指标）。

本轮目标：在本机建立可复现的 CPU 环境，确认代码包原样可用，并记录基线。不要修改任何已有的 .py 文件。

任务：
1. 用 Python 3.10–3.12 建 .venv，安装：jax==0.4.38 jaxlib==0.4.38 flax==0.10.2 optax==0.2.4 chex==0.1.88 gymnax==0.0.6 six numpy matplotlib imageio imageio-ffmpeg pillow pytest。若某个版本在本机平台装不上，停下来告诉我具体报错，不要自行换版本。把最终 `pip freeze` 存到 docs/ENV.md，并写明操作系统、CPU、Python 版本。
2. 运行 python test_v5.py 与 python tests/test_stats.py，把全部结果贴出来。
3. 运行 CLAUDE.md 里的冒烟命令；读取输出 JSON，确认顶层有 seed_ids、versions、ckpts，evals 里同时有 greedy_return、capped_frac、public_len、robust_return、robust_capped_frac、robust_len，hist 里有 ang_cum、trunc_frac；评估点的 env_steps 没有重复或间隔异常。
4. 运行 python sweep.py LIST --outdir /tmp/x 和 python sweep.py P0 --outdir /tmp/x --dry，把 5 个 P0 臂最终生效的参数逐个列成表（注意 argparse 后者覆盖前者：public_gymnax 必须最终是 --sticky 0 --max_steps_in_episode 1000）。
5. 用 pilot_results/ 里的 JSON 重新生成先导图：在 pilot_results 目录下运行 plot_pilot2.py（需要 proto_summary.json），确认生成 fig/pilot_v51.pdf，并与 docs/plan_v5.1.pdf 的图 2 目测一致。
6. 用 check_protocol.py 复测一个先导模型，确认能复现计划表 3 的量级。如果 pilot_results 里没有 params pkl，跳过并说明。

验收标准：
- test_v5.py 打印 ALL PASS（含 3c：与 purejaxql 原文逐行转录一致）；tests/test_stats.py 打印 ALL PASS。
- 冒烟 JSON 含上述全部字段；P0 参数表中 public_gymnax 为 sticky 0 / cap 1000，其余四臂为 sticky 0.1 / cap 10000；只有 fixed 含 --bootstrap_timeout（两个单项修正臂各只比 public 多一处改动）。
- 先导图重新生成成功。

完成后：git add -A && git commit -m "round1: env + baseline checks"，然后按 CLAUDE.md 的汇报格式汇报。
````

\newpage

# 第 2 轮：对照 purejaxql 的独立复核

> 建议先进入 plan mode。

````text
本轮目标：独立确认 pqn_v5.py 在“公开模式”（--input_scale255 --buggy_target，sticky 0，cap 1000）下与 PQN 官方实现逐项一致，在默认模式下只差计划所述的修正。你的立场是审稿人：默认怀疑，用数值证据说话。

任务：
1. 把 https://github.com/mttga/purejaxql 克隆到 third_party/purejaxql，checkout 提交 47af6d7（若该提交不存在，用离 2025 年 1 月最近的提交并告诉我）。只读，不改。
2. 阅读 purejaxql/pqn_minatar.py 及其 MinAtar 配置文件，做一张对照表（写入 docs/REFERENCE_DIFF.md），逐行列出：并行环境数、rollout 长度、epoch、minibatch、学习率与调度（按什么计数衰减到多少）、ε 调度、梯度裁剪、优化器、网络结构与初始化、LayerNorm 位置、γ、λ、输入预处理、Q(λ) 目标递推（贴出官方代码行号）、终止/超时处理、评估方式。每行标注：一致 / 有意差异（对应计划哪条修正）/ 无意差异（=bug，必须报告）。
3. 新建 tests/test_reference.py（pytest）：
   a. 不要看 test_v5.py 的第 3c 项，自己独立地把官方的 Q(λ) 目标计算（_get_target 及其前两行，含 last_q 乘以 1-done[-1] 的掩码）原样转录进测试（注释写明出处文件和行号），在随机输入（含中途终止、含最后一步终止）上与 pqn_v5.targets_public（--buggy_target 时）比较，要求最大误差 < 1e-5。
   b. 自己按定义写一个逐步的暴力 Q(λ) 实现：G_{T-1}=r+γ(1-d)V(s_T)；G_t=r_t+γ[(1-d_t)((1-λ)V(s_{t+1})+λG_{t+1})+trunc_t·V(final_t)]，与 targets_fixed 比较（含截断）。
   c. 验证计划命题 1 的一般形式：buggy − fixed = γ(1−λ)(γλ)^{T−2−t}[(1−d_{T−1})V(s_T)−V(s_{T−1})]·Π_{j=t}^{T−2}(1−d_j)，在有/无终止两种输入上误差 < 1e-5。
   d. 学习率调度在第 0、G/2、G−1 个梯度步的值；ε 调度在第 0、10%、50% 次更新的值，与官方一致。
   e. 网络参数形状与官方一致；He 初始化下 dense kernel 的范数平方期望 ≈ 256（抽 20 次取均值，相对误差 < 5%）。
   注意：pqn_v5.py 在 import 时会解析命令行参数并运行训练。测试里请用 test_v5.py 相同的办法（exec 文件中 "eval_fn = " 之前的部分，并伪造 sys.argv），不要为了可测试性重构 pqn_v5.py。
4. 运行 pytest -q tests/test_reference.py 和 python test_v5.py。

约束：不修改 pqn_v5.py。发现任何“无意差异”时，只报告（附证据和建议修法），等我决定。

验收标准：对照表完整、无“无意差异”（或已全部报告）；新测试全部通过；test_v5.py 仍通过。
完成后 commit（"round2: reference audit"）并按汇报格式汇报。
````

\newpage

# 第 3 轮：协议、评估与 seed 的测试

> 建议先进入 plan mode。

````text
本轮目标：用测试把计划第 3.7 节（命题 6）和第 6 节开头的协议定义钉死，防止之后任何改动悄悄破坏评估。

新建 tests/test_protocol.py（pytest），覆盖：
1. 确定性（命题 6a）：用一个随机初始化或 pilot 模型，在 sticky 0、cap 1000 下做 8 局贪心评估，8 局回报完全相同；sticky 0.1 下 8 局回报不全相同。
2. sticky 语义：写一个只测动作替换逻辑的小函数测试——给定固定的“选择动作”序列（如 0,1,0,1,...），执行动作等于上一步执行动作的频率 ≈ p（二项检验，p=0.1，N≥20000，α=0.001）；回合结束后 prev 重置为 0；存入学习数据的是“选择动作”而不是“执行动作”（读 update() 的代码并在测试注释里说明依据）。如果为了测试必须抽出一个小的纯函数，可以在 pqn_v5.py 里抽取，但行为必须逐比特不变，并在 CHANGELOG.md 记录。
3. 公开协议上限：训练上限设为 10000 时，公开协议评估的局长度 ≤ 1000（PUBLIC_CAP 取 gymnax 默认值），稳健协议的局长度可以超过 1000。
4. 回报掩码：评估中每局回报只累计第一次 done 之前的奖励。写一个独立的 Python 循环参考实现（同样的 key 序列），与 greedy_eval 在 4 局上逐局比较。
5. 截断标记：在 cap 很小（如 50）的设置下跑几步 rollout，trunc 只在 time>=cap 导致的 done 上为 1，真正游戏结束时为 0。
6. seed 身份：(a) 同样的命令跑两次，JSON 中所有数值完全相同；(b) --seeds 2 --seed0 5 的第 2 个 seed 与 --seeds 1 --seed0 6 逐比特相同（训练 seed i 用 PRNGKey(seed0+i)）；(c) 同一评估点上两个不同配置的评估环境 key 相同（配对评估）；(d) JSON 的 versions 字段存在。把结论写进 docs/SEEDS.md，并检查 sweep.py 中同一阶段所有臂的 seeds/seed0 相同。
7. Freeway：P3 中 Freeway 的训练与评估上限都是 2500（用 sweep.py --dry 的输出验证）。

所有测试在 CPU 上总耗时控制在 10 分钟内（用小 num_envs、小 cap、少局数）。

验收标准：pytest -q tests/ 全部通过；test_v5.py 通过；docs/SEEDS.md 写清结论。
完成后 commit（"round3: protocol tests"）并汇报。
````

\newpage

# 第 4 轮：机制分析脚本（预测 1–3 的检验）

> 建议先进入 plan mode。

````text
本轮目标：写 mechanism.py，把计划第 3.4–3.6、3.8 节的预测 1–3 变成可以对正式结果直接运行的检验，判定标准运行前写死。

背景（详见 docs/plan_v5.1.pdf 第 3 节）：
- 学习率按总梯度步 G 线性衰减：η_t = η0(1 − t/G)，η0=5e-4；在环境步比例 φ 处 Σ_{s<φG} η_s² ≈ η0² G[1−(1−φ)³]/3。
- 预测 1（范数律）：未约束训练中 ||W_t||²−||W_0||² 与 Ση_t² 近似成正比，斜率 s 在后半程稳定；按环境步计 Λ 随 E 超线性增大。
- 预测 2（角位移）：同一 E 下 A_none/A_NP（每环境步累计转角之比，来自 hist.ang_cum 的差分）随训练下降并跟踪 1/Λ_none；E=8 时训练后半段该比值 < 0.4。
- 预测 3（权重衰减平衡）：λ=0.1、0.3 时 Λ_t 趋向 Λ_eq(η_t)=sqrt(s·η_t/(2λ·||W_0||²))，s 取同一 E 未约束臂拟合出的斜率；实测与预测相差超过 2 倍即判为不成立。
- ||W_0||：用 He 初始化期望 256（dense 层 1024→128），并用各运行第一个记录点做 sanity check。

mechanism.py 的接口：python mechanism.py <stage_dir> --out results/mechanism_<stage>/
输出：
1. Λ_t 曲线（按 E、按控制方式分面），s 的拟合表（每个配置、每个 seed，后半程拟合，附 R²）。
2. A_none/A_NP 与 1/Λ_none 的对照曲线（同一 E 配对），以及后半段比值的逐 seed 数值。
3. 权重衰减臂的实测 Λ_t 与预测 Λ_eq(η_t) 曲线、比值。
4. verdicts.json：对每条预测给出 支持 / 不支持 / 无法判断（数据不足），以及所用数字。判定规则只用上面写死的标准，不许事后调整。
5. 所有图用 matplotlib 存 PDF（论文用）与 PNG。配色：公开 PQN 深灰虚线 #55554f，E=2 蓝 #2a78d6，E=8 橙 #eb6834，NP 青 #1baf7a，其他控制依次用 #eda100、#e87ba4；线宽 2，图例不遮挡曲线，单图宽 3.3 英寸、双栏 6.8 英寸，字号 8。

测试：
- 用 pilot_results/ 的 JSON 跑一遍（先导 A–D 没有 ang_cum，脚本要能优雅地跳过并在 verdicts 里写“无法判断”）。
- 再用冒烟级别的小运行（E=2 与 E=8，none 与 np，各 1 seed、约 6 万步）生成带 ang_cum 的 JSON，确认第 2 项能画出来。
- 在 tests/test_mechanism.py 里对 Ση² 闭式与逐步求和的一致性、s 的拟合（用合成数据：已知斜率 + 噪声，恢复误差 < 5%）写单元测试。

验收标准：pilot 与小运行上都能跑通；单元测试通过；verdicts.json 结构清楚。
完成后 commit（"round4: mechanism analysis"）并汇报。
````

\newpage

# 第 5 轮：Colab 打包、计时与算力预算

````text
本轮目标：生成上传到 Colab 的代码包，检查 notebook，并在我提供 Colab 计时数字后给出算力预算。

任务：
1. 写 make_zip.sh：把运行所需文件打成扁平的 drl_v5_code.zip（zip 根目录直接是 pqn_v5.py 等，不要多一层文件夹——notebook 会 unzip 到 /content/code 后在该目录运行）。包含：pqn_v5.py test_v5.py tests/test_stats.py sweep.py analyze.py render_demo.py check_protocol.py mechanism.py README.md CHANGELOG.md DRL_v5_colab.ipynb。不要包含 .pkl/.mp4/third_party/.venv。
2. 逐个单元检查 DRL_v5_colab.ipynb：每条命令里的文件名、参数名、glob 模式是否与 sweep.py 实际产生的文件名一致（例如 P2 的 ours_E8_np_lr2.5e-4__<hash>_params.pkl 与 ..._ckpt1M_params.pkl；展示单元的 g1() 必须只匹配最终参数并断言唯一）。用 sweep.py P2 --steps 8192 --protocol "...--eval_episodes 2" 在本机生成一套假 P2 输出，实际执行一遍 notebook 第 7 节的 Python 部分来验证。在 P0/P1/P2 分析单元后加上 mechanism.py 的调用。只改 notebook，不改训练代码。
3. 在 notebook 第 2 节计时单元之后加一个 markdown 单元，提示我把两条计时命令的完整输出（每个评估点的 [xxx.xs] 时间戳）复制回来给你。
4. 等我把计时输出贴给你之后：按 P0（5 配置×10 seed）、P1（19 配置×3 seed，其中 E=4、E=8 的梯度步分别是 E=2 的 2、4 倍）、P2（4 配置×10 seed）、P3（12 配置×5 seed）估算总 GPU 小时，列表给出。注意：一次进程 vmap 全部 seed，所以耗时主要由配置数和 E 决定，seed 数影响较小，但要用我给的 10-seed 计时为准。
5. 若总时长超过 20 GPU 小时，按以下预注册顺序提出削减方案（不许改动 P0 与 P2 的设计）：先砍 P3 的 Freeway → 再砍 P3 全部 → P1 去掉 E=4 的 5 个配置 → P1 的 wd 只保留 0.1。每一步写明省下多少时间。

验收标准：zip 解压后目录扁平且包含上述文件；notebook 所有路径/名称与代码一致（列出核对表）；拿到计时后给出预算表。
完成后 commit（"round5: colab package"）并汇报。
````

**之后你在 Colab 上：**上传 zip 到 `MyDrive/drl_v5/`，按 notebook 跑第 1–3 节（安装、计时、P0）。P0 跑完后把 Drive 里的 `drl_v5/P0/` 整个下载到仓库 `results/P0/`。

\newpage

# 第 6 轮：P0 分析与停止规则判定

````text
P0 的结果已下载到 results/P0/（只读）。本轮目标：按计划第 6.1 节与第 10 节的停止规则，对 P0 做预注册分析并给出是否进入 P1 的判定。

任务：
1. 完整性检查：5 个配置各有 10 个 seed；每个 JSON 的 args 与 sweep.py 预期一致（列表核对 sticky、max_steps_in_episode、eval_cap、input_scale255、buggy_target、bootstrap_timeout）；评估点数量一致。
2. 运行 analyze.py，三个指标（robust_return 主指标、greedy_return、train_return），--ref public，并用 --json_out 存下全部数字；另对 --ref public_gymnax 跑一次 robust_return，用来量化协议效应（C0）。确认所有 JSON 的 versions 一致。
3. 停止规则 1：public_gymnax 的公开协议曲线是否与公开 PQN 报告的 SpaceInvaders-MinAtar 量级相符。请到 PQN 论文（arXiv:2407.04811）或 purejaxql 仓库中查找 MinAtar SpaceInvaders 的报告数值或曲线，写明出处；查不到就说明查不到，并与 pilot_results 中 A_orig_E2 的公开协议数值比较。
4. 停止规则 4：主指标下每个配置的 10-seed 95% CI（analyze.py 的 Student-t 区间）半宽 / 均值；若任一超过 25%，加 --iqm 同时报告 IQM，并提示我 P2 需增至 15 seeds。
5. 超时比例：报告每个配置的 trunc_frac，确认只有 fixed 打开的超时自举在 10000 步上限下几乎不起作用（给出数字），因此阶梯各级的比较不受它干扰。
6. 公开协议下的触顶比例 capped_frac 随训练的变化——这是 C0 的核心证据，画一张图（results/P0_report/）。
7. 运行 mechanism.py results/P0。
8. 写 results/P0_report.md：表格 + 图 + 每条停止规则的判定（满足/不满足 + 数字）+ 对 C0、C1 的初步结论（措辞遵守 CLAUDE.md：<5 seed 不报 p；协议效应不算方法提升）。

约束：不要修改任何 P1 配置，即使数据“暗示”应该改。若你认为计划需要调整，写在报告的“需要我决定的事”里。

验收标准：报告完整，每个判定都有数字支撑；所有数字可由脚本重现（报告末尾列出生成命令）。
完成后 commit（"round6: P0 analysis"）并汇报。
````

\newpage

# 第 7 轮：P1 筛选、机制检验与冻结

````text
P1 结果已在 results/P1/（只读，开发 seed）。本轮目标：严格按预注册规则选出 P2 的配置并冻结。

预注册选择规则（计划第 6.2 节，原文照办，不许改）：取开发 seed 平均稳健 AUC 最高的配置；若它与修正 PQN（E2_none）的差小于开发 seed 间标准差，则 P2 只确认修正 PQN 本身，不推进方法主张。

任务：
1. 完整性检查：19 个配置各 3 个 seed，args 正确。
2. analyze.py：robust_return 与 train_return，--ref E2_none；列出所有配置的 AUC（3 seed 不报 CI/p，只报逐 seed 值与均值、标准差）。
3. 应用选择规则，写出每一步的数字。说明“开发 seed 间标准差”你用的是哪一个（建议：E2_none 的 3 个 seed AUC 的样本标准差；若计划有不同定义以计划为准，并指出）。
4. 停止规则 2、3：任一控制在 E=8 是否优于 E2_none；最佳控制是否优于“只调学习率”的两个对照臂（E8_none_lr2.5e-4、E8_none_lr1.25e-4）。逐条给数字和判定。
5. 运行 mechanism.py results/P1，报告预测 1–3 的判定（verdicts.json）。
6. 写 frozen_P2.json：{"best_E":…, "best_ctrl":"…"（含 --lr 若有）, "selection_numbers":{…}, "rule":"…原文…", "date":…}；计算其 sha256 写入 docs/FREEZE.md；然后 git commit 并打 tag freeze-P2。
7. 生成我在 Colab 上要运行的 P2 命令（只填 BEST_E 与 BEST_CTRL 两个变量），以及 P3 命令。注意 sweep.py 会把控制写进配置名（如 ours_E8_np_lr2.5e-4），并拒绝同名不同参数的重复运行。

约束：P2 的确认 seed（1000–1009）在本轮之前和本轮中都没有任何数据，不要尝试生成或查看。

验收标准：选择过程每一步可复核；frozen_P2.json 与 tag 存在；给出可直接粘贴的 Colab 命令。
完成后按汇报格式汇报（commit 已在任务 6 中完成）。
````

**之后你在 Colab 上：**把 notebook 第 5 节的 `BEST_E`、`BEST_CTRL` 改成第 7 轮给出的值，跑 P2（时间允许再跑 P3）。下载 `P2/`（含 `*_params.pkl`）和 `P3/` 到 `results/`。

\newpage

# 第 8 轮：确认分析、全部图表与数字宏

> 建议先进入 plan mode。

````text
P2（及可能的 P3）结果已在 results/（只读）。先确认 git tag freeze-P2 存在且 frozen_P2.json 的 sha256 与 docs/FREEZE.md 一致，并核对 P2 的 JSON args 与 frozen_P2.json 一致；不一致就停下来报告。

本轮目标：产出论文需要的全部统计结果、图和数字宏，全部由一个脚本 make_paper_assets.py 从 results/ 生成。

任务：
1. 主结果（P2，10 seed，预注册）：robust_return AUC 与最终值，配对差 + 95% Student-t 区间 + 精确符号翻转检验 + Holm 校正（直接用 analyze.py --json_out 的输出，不要自己另写统计），比较族为：fixed_E2 vs public（C1）、ours vs fixed_E2（C3）、ours vs fixed_E{E*}（控制本身的作用）。按计划第 2 节“较大提升”的定义（相对 AUC 提升 ≥25% 且 CI 不含 0）逐条判定能否使用“较大提升”的表述。同时报告 train_return 与公开协议 greedy_return（次指标），以及若 P0 触发停止规则 4 则加 IQM。
2. 图（paper/figs/，PDF，NeurIPS 单栏 3.3 英寸 / 通栏 6.8 英寸，字号 8，配色同 mechanism.py）：
   F1 协议：公开协议下 capped_frac 随训练上升到 1、同模型两协议对比（来自 P0）。
   F2 修正阶梯：P0 五个臂的稳健 AUC（点 + CI，按 seed 配对的差）。
   F3 机制：Λ_t（E=2/4/8，none vs 控制）与 A_none/A_NP vs 1/Λ（来自 P1 或 P2）。
   F4 主结果：P2 四臂稳健回报学习曲线（均值 ± 95% CI 带）。
   F5 权重衰减平衡：实测 vs 预测 Λ_eq（来自 P1）。
   T1 主结果表；T2 P3 稳健性表（5 seed，报均值与逐 seed 值，标明不做显著性检验：n=5 时精确检验最小 p=0.0625）。
3. paper/numbers.tex：所有将在正文出现的数字定义成宏（如 \newcommand{\aucFixedVsPublic}{+12.3\%}），宏名含义清楚；附 paper/numbers.json 便于核对。
4. results/CLAIMS.md：对照计划表 1 的 C0–C4，逐条写：证据、数字、是否满足“必需证据”、按“不满足时”一栏应如何降级表述。
5. 所有图做一次肉眼检查：把每张 PDF 转成 PNG 看一遍，确认无文字重叠、图例不遮挡、坐标轴单位清楚。

验收标准：python make_paper_assets.py 一条命令重建全部资产；CLAIMS.md 每条都有数字；图无重叠。
完成后 commit（"round8: paper assets"）并汇报，汇报中明确告诉我：哪些主张能写成“显著/较大提升”，哪些只能如实报告效应大小。
````

\newpage

# 第 9 轮：展示视频

````text
本轮目标：用 P2 的模型做 3 分钟展示视频的素材，严格遵守预注册的选取规则。

选取规则（不许改）：
- 训练 seed：对每个配置，取 10 个确认 seed 中稳健 AUC 为中位数的那个 seed（10 个时取第 5 小与第 6 小中 AUC 更接近均值者）。notebook 第 7 节的 median_seed() 已实现，请复核它与此规则一致。render_demo.py 用 --seed_indices 为每栏分别指定（它是 vmap 轴上的位置，与 JSON 的 seed_ids 顺序一致，请核对）。
- 快照：JSON 的 ckpts 字段记录了每个快照的实际步数，视频标签必须用实际步数。
- 局：render_demo.py 已实现“同 9 个环境 seed、按完整得分取中位数局”，sticky 0.1，游戏跑满、只截视频（--max_steps 2400 = 100 秒）。

任务：
1. 学习进程视频：最终方法（ours）的 1M、3M、10M 快照三栏（--panel 三次）。
2. 对比视频 A：public vs ours（public 需 --scale255 0）。对比视频 B：fixed_E2 vs ours。若第 8 轮判定 B 组差异不显著，在视频底部加一行字幕 "difference not statistically significant (10 seeds)"（可给 render_demo.py 加一个 --footer 参数，默认不变）。
3. 标题卡与拼接：用 Pillow 生成 3 张 1920×1080 标题卡（问题一句话；Λ_t 曲线图 F3；结论一句话），用 ffmpeg 把标题卡（各 4 秒）与三段视频拼成 demo_final.mp4，总长 ≤ 3 分 30 秒（必要时把对比视频 --max_steps 调小，但两侧必须相同）。
4. 每段视频导出一帧 PNG 放进 paper/figs/，供论文附录使用。
5. 在 results/DEMO.md 记录：每段用的参数文件、seed_index 及其计算、9 局完整得分列表、展示的中位数局得分与长度。

验收标准：demo_final.mp4 可播放、时长合规；DEMO.md 中的选取过程可复核；没有使用任何“最好的一局”。
完成后 commit（视频本身不入库，只提交脚本与 DEMO.md）并汇报。
````

\newpage

# 第 10 轮：论文写作（可分多次会话）

> 建议先进入 plan mode，先出大纲，确认后再逐节写。

````text
本轮目标：写 7–10 页 NeurIPS 格式论文（英文），所有数字来自 paper/numbers.tex，所有图来自 paper/figs/。

准备：
- 下载 NeurIPS 2025 LaTeX 模板（neurips_2025.sty），用 \usepackage[preprint]{neurips_2025}（课程项目，不匿名，作者 Yueyi Li、Cheng Qian，WashU）。
- 读 docs/plan_v5.1.pdf 第 2–3 节（主张与理论）、results/CLAIMS.md、results/P0_report.md、results/mechanism_*/verdicts.json。

结构（页数为正文目标）：
1. Introduction（1 页）：问题——无回放 Q 学习中“多更新几次”为什么不带来更多学习；三个发现（评估协议饱和、两个实现偏差、范数自衰减）；贡献列表与 C0–C3 一一对应，措辞按 CLAIMS.md 的降级结论。
2. Background（0.5 页）：PQN、MinAtar/gymnax、sticky actions（Machado et al. 2018）、尺度不变层与有效学习率。
3. Evaluation protocol（1 页）：命题 6 + P0 证据（F1）。
4. Implementation biases（1 页）：命题 1、2（含证明梗概，完整证明放附录）+ 修正阶梯（F2）。
5. Norm self-attenuation（1.5–2 页）：引理 3（注明是 AdamP Lemma 2.2，Heo et al. 2021）、命题 3、命题 5（三种“退火时钟”）、推论 2（λ*）、预测与检验（F3、F5）。每个结论标注证据等级 exact / known / approximate / model，与计划一致。
6. Experiments（1.5–2 页）：P1 设计与预注册选择规则、P2 主结果（F4、T1）、P3（T2）、算力记账（回报—墙钟）。
7. Related work（0.5 页）：回放比与可塑性（D'Oro 2023；Nikishin 2022；Lyle 2024 NaP；Juliani & Ash 2024；Mayor et al. 2025）、范数与优化（AdamP；Kosson 2024；Wan 2021；Li & Arora 2020）、评估方法（Agarwal 2021；Pardo 2018；Machado 2018）。
8. Limitations（0.3 页）：单主游戏、MinAtar 规模、LayerNorm ε、seed 数、协议偏离原 PQN 设定。
9. Conclusion（0.2 页）。
附录：完整证明、v4 阴性结果（计划附录 B 精简版）、全部超参数表、复现命令、展示视频帧。

写作规则：
- 任何数字只能用 numbers.tex 的宏；若需要新数字，先在 make_paper_assets.py 里加宏再引用。
- 每个经验性主张后面紧跟它的证据（图/表/宏）；CLAIMS.md 判为“不满足”的主张不许出现在摘要和引言里。
- 理论：我们原创的（命题 1、2 对公开配置的刻画；命题 5 的“退火时钟”对照与 λ* 取法；命题 6 的实测）和已知的（AdamP 引理、旋转平衡、NaP）必须区分清楚并引用。
- 参考文献用 BibTeX，从 DBLP 或 arXiv 取条目，不要凭记忆编造；每条引用你都要能给出 URL。
- 每写完一节就编译一次（latexmk -pdf），修掉所有 warning 中的 undefined reference。

本轮可分多次：第一次只交大纲 + 摘要 + 引言；之后每次 1–2 节。每次结束 commit 并汇报当前页数。
````

\newpage

# 第 11 轮：投稿前自审与修改

````text
本轮目标：以严格审稿人的身份审一遍 paper/，并修复问题。若你的环境里有 submission-self-review 或 paper-preflight-audit 技能，先加载并按它执行；否则按下面的清单。

清单：
1. 数字可追溯：grep 正文中所有阿拉伯数字，凡不是来自 numbers.tex 宏、超参数或引用年份的，列出来逐个处理。重新运行 make_paper_assets.py，确认 PDF 不变（或变化可解释）。
2. 主张—证据对齐：逐条比对摘要、引言贡献列表与 results/CLAIMS.md；任何超出证据的措辞（"significantly", "large", "solves", "always"）改掉。
3. 归属：AdamP Lemma 2.2、旋转平衡（Kosson 2024）、NaP（Lyle 2024）、sticky actions（Machado 2018）、time limits（Pardo 2018）都已在对应位置引用；我们的贡献没有把已知结果说成新的。
4. 统计：所有 CI/p 值都是 ≥5 seed；Holm 校正的比较族在正文写清；预注册与探索性分析在文中分开标注。
5. 可复现：附录有完整超参数表、协议定义、seed 范围、每个阶段的确切命令、代码链接、算力。
6. 格式：正文 ≤ 10 页（参考文献与附录不计）；图字号 ≥ 7pt；所有图表在正文被引用；无 overfull hbox；参考文献格式统一。
7. 逻辑链：按“测量（C0）→ 偏差（C1）→ 机制（C2）→ 方法（C3）→ 展示（C4）”读一遍，每一环的结论是否只依赖前面已给出的证据。
8. 三位独立审稿人视角（方法 / 证据 / 写作）各列出最可能的 3 条批评，并给出你在文中的回应位置；无法回应的写进 Limitations。

输出：先给问题清单（按 致命 / 应改 / 可选 分级），我确认后再改；改完重新编译并汇报页数与剩余问题。
````

\newpage

# 附：常见问题处理

| 现象 | 处理 |
|---|---|
| Colab 上 `pip install jax[cuda12]==0.4.38` 失败或 `jax.devices()` 没有 GPU | 重启运行时，改用 notebook 中的 Option B（Colab 自带 JAX + gymnax 1.0.0）；在第 1 轮让 Claude Code 在 CPU 上用同一组合跑 test_v5.py 确认兼容 |
| Colab 断线 | 重新运行同一个单元即可：sweep.py 跳过已完成的配置。正在运行的那个配置会从头重跑 |
| 显存不足（OOM） | 不要减少 seed；先把 `--eval_episodes` 从 64 降到 32（P0 之前决定，所有阶段统一），并记入 CHANGELOG.md |
| 某个配置结果明显异常（NaN、回报为 0） | 不要删除或重跑“直到正常”；保留原始 JSON，让 Claude Code 诊断原因，确认是 bug 才修复并重跑整个阶段的该配置，并在论文中说明 |
| Claude Code 想“顺手优化”训练代码 | 拒绝。已开始的阶段不改训练行为；确需修改按 CLAUDE.md 规则 7 处理 |
| 结果不支持“较大提升” | 按计划停止规则把论文转为分析型：C0 + C1 + C2（含已检验的预测）仍是完整贡献链；不要在 P2 之后追加配置去“找”提升 |
