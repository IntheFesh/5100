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

## 新增脚本（第 1–9 轮加入；详见 docs/STATUS.md、docs/OPEN_DECISIONS.md）
- mechanism.py（预测 1–3 的预注册检验）、make_paper_assets.py（图表/numbers.tex/CLAIMS.md，先校验 P2 冻结）、make_zip.sh（Colab 扁平包）。
- tools/：effective_args.py（sweep 各配置最终生效参数）、p0_report.py（第 6 轮）、select_p2.py（第 7 轮，默认 dry-run；`--write` 只对 results/ 内目录）、
  budget.py（第 5 轮预算与预注册削减阶梯）、check_notebook.py、make_demo_video.py + demo_select.py（第 9 轮）。
- tests/：test_reference / test_protocol / test_mechanism / test_notebook / test_budget / test_sweep / test_analyze / test_p0_report / test_select_p2 / test_paper_assets / test_demo；
  tests/synth*.py 是**合成**测试夹具（只在临时目录，永远不是结果）。慢测试：`RUN_SLOW=1 python -m pytest -q tests/`。
- sweep.py 新增 `--only`；analyze.py 新增 `--pairs`（单一 Holm 比较族）；render_demo.py 新增 `--footer`/`--json_out`；以上均为加法，默认行为不变（有回归测试）。
