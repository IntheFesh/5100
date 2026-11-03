# 状态总表：提示词 v5.1 的各轮 vs 已完成的代码任务

核心原则（CLAUDE.md）：GPU 结果（P0/P1/P2/P3）还不存在。凡是依赖真实结果的步骤，这里只交付**脚本 + 在合成数据上的测试**，没有生成任何“结果”、没有冻结任何配置、没有写任何论文数字。合成数据只存在于临时目录（`tests/synth*.py`），从不写入 `results/`。

| 轮 | 内容 | 状态 | 交付物 | 验证 |
|---|---|---|---|---|
| 0 | 建仓、CLAUDE.md | 完成 | `CLAUDE.md`、`.gitignore`、`docs/plan_v5.1.pdf` | — |
| 1 | 环境与基线 | 完成 | `.venv`（Python 3.12、固定版本）、`docs/ENV.md`、`tools/effective_args.py` | `test_v5.py` ALL PASS；`tests/test_stats.py` ALL PASS；冒烟 JSON 字段齐全；P0 五臂参数表；先导图复现 |
| 2 | 对照 purejaxql | 完成 | `docs/REFERENCE_DIFF.md`、`tests/test_reference.py` | 26 项通过；唯一“无意差异”：ε 衰减步数取整（≤3.9e-4，建议不改） |
| 3 | 协议/评估/seed 测试 | 完成 | `tests/test_protocol.py`、`docs/SEEDS.md`；`pqn_v5.py` 抽出 `sticky_select`/`next_prev` | 23 项通过；人为植入 6 个 bug 均被检出；抽取前后 JSON 与参数逐比特相同 |
| 4 | 机制分析 | 完成 | `mechanism.py`、`tests/test_mechanism.py`、`tests/synth.py`、`tools/make_mech_smoke.sh` | 先导数据复现计划斜率 2.26e5/1.54e5/2.32e5 与 Λ²−1 比 5.1；合成数据 + 反例；`RUN_SLOW=1` 真实冒烟训练通过 |
| 5 | Colab 打包/计时/预算 | **脚本完成，预算待你贴计时** | `make_zip.sh`、修补后的 `DRL_v5_colab.ipynb`、`tools/check_notebook.py`、`tools/budget.py` | notebook 静态核对 25 条命令；第 5/7 节在真实微型 P2 上经 IPython 执行通过；P0/P1/P3 分析单元在合成树上通过 |
| 6 | P0 分析与停止规则 | **脚本完成，等 `results/P0`** | `tools/p0_report.py` | 合成 P0：完整性检查能抓到错参数/缺 seed/混用运行时；规则 1/4 触发逻辑 |
| 7 | P1 筛选与冻结 | **脚本完成，等 `results/P1`** | `tools/select_p2.py`；`sweep.py --only` | 规则逐步、停止规则 2/3、各模式（含 E*=2、无控制）；`--write` 只允许对 `results/` 内的目录且不覆盖 |
| 8 | 确认分析与全部资产 | **脚本完成，等 `results/P2`+冻结** | `make_paper_assets.py`、`analyze.py --pairs` | 与独立重算比对；Holm 单一比较族；“较大提升”规则阈值测试；冻结校验（sha256、args、seed、tag） |
| 9 | 展示视频 | **脚本完成，等 P2 模型** | `tools/make_demo_video.py`、`tools/demo_select.py`、`render_demo.py --footer/--json_out` | 中位数 seed 规则与 notebook 一致；字幕规则；时长预算；卡片+拼接；`RUN_SLOW=1` 在小型真实模型上端到端 |
| 10 | 论文写作 | **未做**（非代码任务，且依赖 `numbers.tex`） | — | — |
| 11 | 投稿前自审 | **未做**（依赖论文稿） | — | — |

## 拿到真实结果后的命令序列

```bash
source .venv/bin/activate
python test_v5.py && python tests/test_stats.py && python -m pytest -q tests/        # 一直全部通过
# 第 5 轮：把 Colab 计时输出存成 t2.txt / t8.txt
python tools/budget.py --e2 t2.txt --e8 t8.txt --md docs/BUDGET.md
./make_zip.sh                                                                         # -> drl_v5_code.zip，上传到 MyDrive/drl_v5/
# 第 6 轮（results/P0 已下载）
python tools/p0_report.py results/P0 --out results/P0_report --md results/P0_report.md   # 停止规则 1 需加 --ref_value/--ref_step/--ref_source（见 OPEN_DECISIONS）
# 第 7 轮（results/P1 已下载）
python tools/select_p2.py results/P1                 # 先 dry run，核对 results/P1_selection/P1_selection.md
python tools/select_p2.py results/P1 --write         # 写 frozen_P2.json + docs/FREEZE.md；再按屏幕提示 git commit + git tag freeze-P2
# 第 8 轮（results/P2、results/P3 已下载）
python make_paper_assets.py --require_tag
# 第 9 轮
python tools/make_demo_video.py
```
