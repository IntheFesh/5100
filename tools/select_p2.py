"""P1 screening -> P2 configuration (round 7; plan sections 6.2 and 10).  Applies the PRE-REGISTERED rule, nothing else.

Usage:  python tools/select_p2.py results/P1 [--out results/P1_selection] [--write]
Default is a dry run: prints the decision and writes the selection report/JSON into --out. `--write` additionally writes frozen_P2.json and
docs/FREEZE.md (sha256) - allowed ONLY for a stage directory inside results/ (synthetic or test data can never be frozen) and never over an
existing frozen_P2.json. The git commit and the tag `freeze-P2` are printed, not executed.

Rule (plan 6.2, verbatim from the round-7 prompt, not to be changed):
    "取开发 seed 平均稳健 AUC 最高的配置；若它与修正 PQN（E2_none）的差小于开发 seed 间标准差，则 P2 只确认修正 PQN 本身，不推进方法主张。"
"Development-seed standard deviation" is not defined further in the plan; we use the sample standard deviation (ddof = 1) of E2_none's three
per-seed robust AUCs (the round-7 prompt's suggestion). Alternatives (pooled within-config SD over all configs, the largest per-config SD) are
reported next to it, together with whether the decision would change.
Stopping rules (plan 10), evaluated on mean robust AUC (3 development seeds: no CI/p-values, per-seed values are listed):
    2  no control at E=8 (np, radial, wd0.1, wd0.3, np_lr2.5e-4) has a higher mean AUC than E2_none        -> no 'high update ratio' claim
    3  the best E=8 control does not beat the better of the two 'just lower the lr' arms (E8_none_lr2.5e-4, E8_none_lr1.25e-4) -> report comparison only
Modes of the frozen P2:   with_control (public, fixed_E2, [fixed_E*], ours)   |   high_E_no_control (public, fixed_E2, fixed_E*)   |
                          confirm_fixed_only (public, fixed_E2).   P2 seeds 1000-1009 are never read or generated here.
"""
import argparse, datetime, glob, hashlib, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from effective_args import effective_rows

RULE = "取开发 seed 平均稳健 AUC 最高的配置；若它与修正 PQN（E2_none）的差小于开发 seed 间标准差，则 P2 只确认修正 PQN 本身，不推进方法主张。"
E8_CONTROLS = ["E8_np", "E8_radial", "E8_wd0.1", "E8_wd0.3", "E8_np_lr2.5e-4"]
LR_ONLY = ["E8_none_lr2.5e-4", "E8_none_lr1.25e-4"]
GAMES = ("Breakout", "Asterix", "Freeway")
SKIP_KEYS = {"out"}


def lr_str(lr):
    m, e = f"{lr:e}".split("e")
    return f"{float(m):g}e{int(e)}"


def ctrl_string(args):
    """The --best_ctrl string of a P1 config (control + its learning rate), in the notation of sweep.py."""
    parts = [f"--norm_ctrl {args['norm_ctrl']}"]
    if args["norm_ctrl"] == "wd":
        parts.append(f"--wd {args['wd']:g}")
    if abs(args["lr"] - 5e-4) > 1e-12:
        parts.append(f"--lr {lr_str(args['lr'])}")
    return " ".join(parts)


def load_p1(d):
    runs = {}
    for fn in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(fn))
        if "evals" in j:
            runs.setdefault(os.path.basename(fn).split("__")[0], []).append((fn, j))
    return runs


def integrity(runs):
    problems = []
    exp = {n: a for n, _, a in effective_rows("P1")}
    for n in exp:
        if n not in runs:
            problems.append(f"missing config {n}")
        elif len(runs[n]) != 1:
            problems.append(f"config {n}: {len(runs[n])} JSON files")
    problems += [f"unexpected config {n}" for n in runs if n not in exp]
    versions = set()
    for n, a in exp.items():
        if n in runs and len(runs[n]) == 1:
            j = runs[n][0][1]
            diff = {k: (j["args"].get(k, "<absent>"), v) for k, v in a.items() if k not in SKIP_KEYS and j["args"].get(k, "<absent>") != v}
            if diff:
                problems.append(f"config {n}: args differ from sweep.py (json, expected): {diff}")
            if j.get("seed_ids") != [100, 101, 102]:
                problems.append(f"config {n}: seed_ids = {j.get('seed_ids')} (expected 100-102)")
            versions.add(json.dumps(j.get("versions"), sort_keys=True))
    if len(versions) > 1:
        problems.append("JAX runtimes differ between runs")
    return problems


def analyze(stage_dir, metric, out):
    p = os.path.join(out, f"analysis_{metric}.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "analyze.py"), stage_dir, "--metric", metric, "--ref", "E2_none", "--json_out", p], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("analyze.py failed:\n" + r.stderr[-1500:])
    return json.load(open(p))["configs"]


def decide(auc):
    """auc: {config: per-seed robust AUC list}. Returns the full decision record (pure function of the numbers)."""
    mean = {n: float(np.mean(v)) for n, v in auc.items()}
    sd = {n: float(np.std(v, ddof=1)) for n, v in auc.items()}
    ref_mean, ref_sd = mean["E2_none"], sd["E2_none"]
    best = max(mean, key=lambda n: mean[n])
    margin = mean[best] - ref_mean
    alt = {"sd_E2_none (used)": ref_sd, "pooled_within_config_sd": float(np.sqrt(np.mean([s ** 2 for s in sd.values()]))), "max_config_sd": float(max(sd.values()))}
    ctrl_means = {n: mean[n] for n in E8_CONTROLS}
    best_ctrl = max(ctrl_means, key=ctrl_means.get)
    lr_best = max(LR_ONLY, key=lambda n: mean[n])
    rule2_trig = not any(m > ref_mean for m in ctrl_means.values())
    rule3_trig = not (ctrl_means[best_ctrl] > mean[lr_best])
    return dict(mean_auc=mean, sd_auc=sd, best=best, best_mean=mean[best], e2_none_mean=ref_mean, margin=margin, sd_used=ref_sd,
                confirm_fixed_only=bool(margin < ref_sd), decision_under_alternatives={k: bool(margin < v) for k, v in alt.items()}, alternatives_sd=alt,
                rule2=dict(triggered=bool(rule2_trig), control_means=ctrl_means, controls_above_E2_none=[n for n, m in ctrl_means.items() if m > ref_mean],
                           controls_above_E2_none_by_more_than_sd=[n for n, m in ctrl_means.items() if m - ref_mean >= ref_sd]),
                rule3=dict(triggered=bool(rule3_trig), best_control=best_ctrl, best_control_mean=ctrl_means[best_ctrl], lr_only_best=lr_best, lr_only_means={n: mean[n] for n in LR_ONLY}))


def plan(decision, args_by_config):
    """Frozen P2 configuration from the decision (mode, best_E, best_ctrl, --only lists)."""
    if decision["confirm_fixed_only"]:
        mode, E, ctrl = "confirm_fixed_only", 2, "--norm_ctrl none"
        p2_only = ["public", "fixed_E2"]
    else:
        a = args_by_config[decision["best"]]
        E, ctrl = int(a["epochs"]), ctrl_string(a)
        plain = a["norm_ctrl"] == "none" and abs(a["lr"] - 5e-4) < 1e-12
        if plain:                                              # unconstrained high E: no control to name -> only the fixed_E* arm exists
            mode, p2_only = "high_E_no_control", ["public", "fixed_E2", f"fixed_E{E}"]
        else:
            mode, p2_only = "with_control", None
    return dict(mode=mode, best_E=E, best_ctrl=ctrl, p2_only=p2_only)


def p2_arm_names(P):
    rows = effective_rows("P2", P["best_E"], P["best_ctrl"])
    names = [n for n, _, _ in rows]
    return [n for n in names if P["p2_only"] is None or n in P["p2_only"]]


def p3_only(P):
    rows = effective_rows("P3", P["best_E"], P["best_ctrl"])
    names = [n for n, _, _ in rows]
    if P["p2_only"] is None:
        return None
    keep = set(P["p2_only"])
    return [n for n in names if n.split("_", 1)[1] in keep]


def colab_commands(P):
    only = f" --only {','.join(P['p2_only'])}" if P["p2_only"] else ""
    lines = ["BEST_E = %d" % P["best_E"], f"BEST_CTRL = '{P['best_ctrl']}'",
             f'!python sweep.py P2 --outdir $OUT --best_E $BEST_E --best_ctrl "$BEST_CTRL"{only}']
    p3 = p3_only(P)
    p3only = f" --only {','.join(p3)}" if p3 else ""
    lines.append(f'!python sweep.py P3 --outdir $OUT --best_E $BEST_E --best_ctrl "$BEST_CTRL"{p3only}')
    return lines


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def report(path, ctx):
    D, P = ctx["decision"], ctx["plan"]
    f = lambda v: f"{v:.2f}"
    L = ["# P1 selection (pre-registered rule)\n", f"Source `{ctx['stage_dir']}`; generated {ctx['date']}. Numbers: mean robust AUC over the 3 development seeds (analyze.py JSON); "
         "no CIs or p-values with n = 3 (CLAUDE.md rule 6).\n", "**Rule (not to be changed):** " + RULE + "\n"]
    L += ["## 1. Integrity\n", "All 19 configs x 3 seeds (ids 100-102), args equal to `sweep.py P1`, one JAX runtime.\n" if not ctx["problems"] else "**PROBLEMS:**\n" + "\n".join(f"- {p}" for p in ctx["problems"]) + "\n"]
    L += ["## 2. Robust AUC per config (descending)\n", "| config | mean | SD (ddof 1) | per-seed AUC | train-return AUC mean |", "|---|---|---|---|---|"]
    for n in sorted(D["mean_auc"], key=lambda k: -D["mean_auc"][k]):
        L.append(f"| {n} | {f(D['mean_auc'][n])} | {f(D['sd_auc'][n])} | {', '.join(f(x) for x in ctx['auc'][n])} | {f(ctx['train_mean'][n])} |")
    L += ["", "## 3. Selection rule, step by step\n",
          f"1. Highest mean robust AUC: **{D['best']}** = {f(D['best_mean'])}.", f"2. Fixed PQN `E2_none` = {f(D['e2_none_mean'])}; margin = {f(D['margin'])}.",
          f"3. Development-seed SD used = SD of E2_none's three AUCs = {f(D['sd_used'])}.",
          f"4. margin {'<' if D['confirm_fixed_only'] else '>='} SD -> **{'P2 only confirms the fixed PQN (no method claim)' if D['confirm_fixed_only'] else 'P2 may advance the selected configuration'}**.\n",
          "Sensitivity to the SD definition (does the decision change?):\n", "| SD definition | value | margin < SD (confirm fixed only) |", "|---|---|---|"]
    for k, v in D["alternatives_sd"].items():
        L.append(f"| {k} | {f(v)} | {D['decision_under_alternatives'][k]} |")
    r2, r3 = D["rule2"], D["rule3"]
    L += ["", "## 4. Stopping rules 2 and 3\n",
          f"**Rule 2** (no control at E=8 beats E2_none): **{'TRIGGERED' if r2['triggered'] else 'not triggered'}**. E=8 controls above E2_none ({f(D['e2_none_mean'])}): "
          f"{', '.join(r2['controls_above_E2_none']) or 'none'}; above by more than one SD: {', '.join(r2['controls_above_E2_none_by_more_than_sd']) or 'none'}.\n",
          "| E=8 control | mean AUC |", "|---|---|"] + [f"| {n} | {f(m)} |" for n, m in r2["control_means"].items()]
    L += ["", f"**Rule 3** (best control not better than the best 'just lower the lr' arm): **{'TRIGGERED' if r3['triggered'] else 'not triggered'}**. Best E=8 control {r3['best_control']} = "
          f"{f(r3['best_control_mean'])} vs best lr-only arm {r3['lr_only_best']} = {f(r3['lr_only_means'][r3['lr_only_best']])} "
          f"(lr-only: {', '.join(f'{k} {f(v)}' for k, v in r3['lr_only_means'].items())}).\n"]
    mv = ctx["mechanism"]
    L += ["## 5. Mechanism predictions on P1 (`mechanism.py`)\n", f"Prediction 1 (norm law): **{mv['prediction_1_norm_law']['verdict']}**; prediction 2 (angle): **{mv['prediction_2_angle']['verdict']}**; "
          f"prediction 3 (weight-decay equilibrium): **{mv['prediction_3_wd_equilibrium']['verdict']}**. Details and the figures: `{ctx['mech_dir']}`.\n",
          "## 6. Frozen P2 configuration\n", f"- mode: **{P['mode']}**; `BEST_E = {P['best_E']}`; `BEST_CTRL = '{P['best_ctrl']}'`; P2 arms: {', '.join(ctx['p2_arms'])}.",
          "- P2 seeds 1000-1009 were not read or generated.\n", "Colab cells:\n", "```python"] + ctx["colab"] + ["```", ""]
    if ctx["written"]:
        L += ["## 7. Freeze\n", f"`frozen_P2.json` sha256 `{ctx['sha256']}` (recorded in docs/FREEZE.md). Then:\n", "```bash", "git add frozen_P2.json docs/FREEZE.md",
              'git commit -m "freeze P2 configuration"', "git tag freeze-P2", "```", ""]
    else:
        L += ["## 7. Freeze\n", "Dry run: nothing frozen. Use `--write` on the real `results/P1`.\n"]
    open(path, "w").write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_dir"); ap.add_argument("--out", default=None); ap.add_argument("--write", action="store_true")
    ap.add_argument("--results_root", default=os.path.join(ROOT, "results"), help="--write is only allowed for stage dirs inside this directory")
    ap.add_argument("--freeze_dir", default=ROOT, help="where frozen_P2.json and docs/FREEZE.md are written")
    a = ap.parse_args(argv)
    stage_dir = os.path.abspath(a.stage_dir)
    out = os.path.abspath(a.out or os.path.join(os.path.dirname(stage_dir), "P1_selection")); os.makedirs(out, exist_ok=True)
    runs = load_p1(stage_dir)
    problems = integrity(runs)
    if any(p.startswith(("missing", "unexpected")) for p in problems):
        sys.exit("cannot select from an incomplete stage:\n" + "\n".join(problems))
    rob = analyze(stage_dir, "robust_return", out); trn = analyze(stage_dir, "train_return", out)
    auc = {n: o["auc_per_seed"] for n, o in rob.items()}
    D = decide(auc)
    args_by = {n: runs[n][0][1]["args"] for n in runs}
    P = plan(D, args_by)
    arms = p2_arm_names(P)
    mech_dir = os.path.join(out, "mechanism_P1")
    subprocess.run([sys.executable, os.path.join(ROOT, "mechanism.py"), stage_dir, "--out", mech_dir], check=True, capture_output=True)
    mech = json.load(open(os.path.join(mech_dir, "verdicts.json")))
    rel = lambda p: os.path.relpath(p, os.getcwd())
    frozen = dict(best_E=P["best_E"], best_ctrl=P["best_ctrl"], mode=P["mode"], p2_only=P["p2_only"], p2_arms=arms,
                  selection_numbers=dict(mean_robust_auc=D["mean_auc"], sd_robust_auc=D["sd_auc"], best=D["best"], e2_none_mean=D["e2_none_mean"], margin=D["margin"],
                                         sd_used=D["sd_used"], alternatives_sd=D["alternatives_sd"], rule2_triggered=D["rule2"]["triggered"], rule3_triggered=D["rule3"]["triggered"]),
                  rule=RULE, date=datetime.date.today().isoformat(), source_dir=rel(stage_dir),
                  p1_json_sha256={n: sha256_file(runs[n][0][0]) for n in sorted(runs)})
    written, sha = False, None
    if a.write:
        res_root = os.path.realpath(a.results_root)
        if not os.path.realpath(stage_dir).startswith(res_root + os.sep):
            sys.exit(f"refusing --write: {stage_dir} is not inside {res_root} (test/synthetic data must never be frozen)")
        if problems:
            sys.exit("refusing --write with integrity problems:\n" + "\n".join(problems))
        fp = os.path.join(a.freeze_dir, "frozen_P2.json")
        if os.path.exists(fp):
            sys.exit(f"refusing --write: {fp} already exists (a frozen P2 configuration is never overwritten)")
        os.makedirs(os.path.join(a.freeze_dir, "docs"), exist_ok=True)
        json.dump(frozen, open(fp, "w"), indent=1, sort_keys=True, ensure_ascii=False)
        sha = sha256_file(fp); written = True
        open(os.path.join(a.freeze_dir, "docs", "FREEZE.md"), "w").write(
            f"# P2 freeze\n\n- file: `frozen_P2.json`\n- sha256: `{sha}`\n- date: {frozen['date']}\n- git tag: `freeze-P2` (create it right after committing both files)\n"
            f"- mode: {P['mode']}; BEST_E = {P['best_E']}; BEST_CTRL = `{P['best_ctrl']}`; P2 arms: {', '.join(arms)}\n"
            "- verify: `sha256sum frozen_P2.json` must equal the value above, and P2 JSON `args` must equal `sweep.py P2` with these values.\n")
    ctx = dict(stage_dir=rel(stage_dir), problems=problems, decision=D, plan=P, auc=auc, train_mean={n: o["auc_mean"] for n, o in trn.items()}, mechanism=mech,
               mech_dir=rel(mech_dir), p2_arms=arms, colab=colab_commands(P), date=frozen["date"], written=written, sha256=sha)
    report(os.path.join(out, "P1_selection.md"), ctx)
    json.dump(dict(decision=D, plan=P, frozen=frozen, colab=ctx["colab"]), open(os.path.join(out, "selection.json"), "w"), indent=1, default=float, ensure_ascii=False)
    print(f"best = {D['best']} ({D['best_mean']:.2f}); E2_none = {D['e2_none_mean']:.2f}; margin {D['margin']:.2f} vs SD {D['sd_used']:.2f} -> mode {P['mode']}")
    print(f"rule 2 triggered: {D['rule2']['triggered']}; rule 3 triggered: {D['rule3']['triggered']}; written: {written}")
    print("\n".join(ctx["colab"]))
    if written:
        print("\nnext:\n  git add frozen_P2.json docs/FREEZE.md && git commit -m 'freeze P2 configuration' && git tag freeze-P2")
    return ctx


if __name__ == "__main__":
    main()
