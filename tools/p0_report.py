"""P0 analysis and stopping-rule verdicts (round 6; plan sections 6.1 and 10).

Usage:  python tools/p0_report.py results/P0 [--out results/P0_report] [--md results/P0_report.md]
                [--ref_value 123.4 --ref_step 1e7 --ref_source "PQN paper arXiv:2407.04811, Fig. N"]
Nothing is invented: every number comes from the P0 JSONs or from analyze.py / mechanism.py run on them (the statistics are NOT recomputed here).
Reads  <P0>/<config>__<hash>.json  (5 configs x 10 seeds written by sweep.py P0).
Writes <out>/analysis_*.json (analyze.py --json_out), <out>/capped_frac.pdf|png, <out>/mechanism_P0/, and the markdown report.

Checks (all pre-registered in the plan / prompt):
  1  integrity     5 configs, 10 seeds (ids 0-9), JSON args == the effective sweep.py args (all keys but `out`), equal number of eval points, one JAX runtime
  2  analysis      analyze.py: robust_return / greedy_return / train_return vs `public`; robust_return vs `public_gymnax` (protocol effect, C0)
  3  stop rule 1   public_gymnax's public-protocol curve is of the order of the published PQN number: |mean - ref| / ref <= 0.5 at the eval point nearest ref_step.
                   The reference is an INPUT (--ref_value/--ref_source); if none is given the CPU pilot A_orig_E2 (2M steps) is used and labelled as such.
  4  stop rule 4   10-seed 95 % Student-t CI half-width / mean of the primary metric (robust AUC) > 25 % for any config -> report IQM and flag P2 -> 15 seeds
  5  time limits   mean trunc_frac per config; only `fixed` may have --bootstrap_timeout
  6  C0 figure     public-protocol capped fraction vs env steps (+ robust-protocol capped fraction, + the two protocols on the same final models)
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from effective_args import effective_rows

PILOT = os.path.join(ROOT, "pilot_results", "A_orig_E2.json")
ARM_STYLE = {"public_gymnax": ("#55554f", "--"), "public": ("#55554f", "-"), "fix_target_only": ("#eda100", "-"),
             "fix_input_only": ("#e87ba4", "-"), "fixed": ("#2a78d6", "-")}
INK, MUTED, GRID = "#1f1f1c", "#6b6b66", "#e6e5e0"
SKIP_KEYS = {"out"}


def load_stage(d):
    runs = {}
    for fn in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(fn))
        if "evals" not in j:
            continue
        name = os.path.basename(fn).split("__")[0]
        runs.setdefault(name, []).append((fn, j))
    return runs


def integrity(runs):
    """List of problems (empty = OK) plus a per-config table."""
    problems, table = [], []
    expected = {n: a for n, _, a in effective_rows("P0")}
    for n in expected:
        if n not in runs:
            problems.append(f"missing config {n}")
        elif len(runs[n]) != 1:
            problems.append(f"config {n}: {len(runs[n])} JSON files (expected exactly one)")
    for n in runs:
        if n not in expected:
            problems.append(f"unexpected config {n}")
    npts, versions = {}, set()
    for n, exp in expected.items():
        if n not in runs or len(runs[n]) != 1:
            continue
        fn, j = runs[n][0]
        a = j["args"]
        diff = {k: (a.get(k, "<absent>"), v) for k, v in exp.items() if k not in SKIP_KEYS and a.get(k, "<absent>") != v}
        if diff:
            problems.append(f"config {n}: args differ from sweep.py (json, expected): {diff}")
        if j.get("seed_ids") != list(range(10)):
            problems.append(f"config {n}: seed_ids = {j.get('seed_ids')} (expected 0-9)")
        npts[n] = len(j["evals"]); versions.add(json.dumps(j.get("versions"), sort_keys=True))
        table.append(dict(config=n, seeds=len(j["seed_ids"]), eval_points=len(j["evals"]), sticky=a["sticky"], cap=a["max_steps_in_episode"],
                          eval_cap=a["eval_cap"], input_scale255=a["input_scale255"], buggy_target=a["buggy_target"],
                          bootstrap_timeout=a["bootstrap_timeout"], minutes=j["wall_seconds"] / 60))
    if len(set(npts.values())) > 1:
        problems.append(f"different numbers of eval points: {npts}")
    if len(versions) > 1:
        problems.append(f"JAX runtimes differ between runs: {sorted(versions)}")
    return problems, table, (json.loads(next(iter(versions))) if len(versions) == 1 else None)


def run_analyze(stage_dir, metric, ref, out_json, extra=()):
    cmd = [sys.executable, os.path.join(ROOT, "analyze.py"), stage_dir, "--metric", metric, "--ref", ref, "--json_out", out_json, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("analyze.py failed:\n" + r.stderr[-1500:])
    return json.load(open(out_json)), r.stdout


def stop_rule_1(runs, ref_value, ref_step, ref_source):
    fn, j = runs["public_gymnax"][0]
    steps = np.array([e["env_steps"] for e in j["evals"]]); pub = np.array([e["greedy_return"] for e in j["evals"]])      # (points, seeds)
    capped = np.array([e["capped_frac"] for e in j["evals"]])
    if ref_value is None:                                      # fall back to the CPU pilot (labelled as such)
        pj = json.load(open(PILOT)); ps = np.array([e["env_steps"] for e in pj["evals"]]); pr = np.array([e["greedy_return"] for e in pj["evals"]])
        ref_value, ref_step = float(pr[-1].mean()), float(ps[-1])
        ref_source = (f"CPU pilot A_orig_E2 (2 seeds, public protocol, {ref_step / 1e6:.2f}M steps) - NOT a published number. The PQN paper (arXiv:2407.04811, Appendix C, Fig. 9 'Results in MinAtar') "
                      "shows MinAtar learning curves only; the text gives no numeric SpaceInvaders value, so the literature magnitude must be read off the figure by the authors (pass --ref_value/--ref_step/--ref_source)")
    k = int(np.argmin(np.abs(steps - ref_step)))
    m = float(pub[k].mean()); rel = (m - ref_value) / ref_value
    return dict(rule="public_gymnax public-protocol return within 50 % of the reference at the nearest eval point", reference_value=ref_value,
                reference_step=ref_step, reference_source=ref_source, eval_step=float(steps[k]), measured_mean=m, measured_per_seed=pub[k].tolist(),
                relative_difference=rel, capped_frac_at_point=float(capped[k].mean()), final_return_mean=float(pub[-1].mean()), final_capped_frac=float(capped[-1].mean()),
                satisfied=bool(abs(rel) <= 0.5))


def stop_rule_4(ana, thresh=0.25):
    rows = {}
    for n, o in ana["configs"].items():
        lo, hi = o["auc_ci"]; half = (hi - lo) / 2; rows[n] = dict(auc_mean=o["auc_mean"], half_width=half, rel_half_width=half / abs(o["auc_mean"]) if o["auc_mean"] else float("nan"))
    trig = [n for n, r in rows.items() if not (r["rel_half_width"] <= thresh)]
    return dict(rule=f"10-seed 95% CI half-width / mean of robust AUC > {thresh:.0%} for any config -> IQM + P2 with 15 seeds", per_config=rows,
                triggered=bool(trig), triggered_configs=trig)


def time_limits(runs):
    out = {}
    for n, lst in runs.items():
        j = lst[0][1]; tf = np.array(j["hist"]["trunc_frac"], float)
        out[n] = dict(bootstrap_timeout=bool(j["args"]["bootstrap_timeout"]), trunc_frac_mean=float(tf.mean()), trunc_frac_max=float(tf.max()),
                      trunc_frac_final_mean=float(tf[-1].mean()))
    only_fixed = [n for n, v in out.items() if v["bootstrap_timeout"]] == ["fixed"]
    return dict(per_config=out, only_fixed_bootstraps_timeouts=bool(only_fixed),
                max_trunc_frac=max(v["trunc_frac_max"] for v in out.values()))


def figure(runs, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                         "legend.fontsize": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.5))
    for n in ARM_STYLE:
        if n not in runs:
            continue
        j = runs[n][0][1]; x = np.array([e["env_steps"] for e in j["evals"]]) / 1e6; c, ls = ARM_STYLE[n]
        for panel, key in ((0, "capped_frac"), (1, "robust_capped_frac")):
            y = np.array([e[key] for e in j["evals"]])
            ax[panel].plot(x, y.mean(1), color=c, ls=ls, lw=2, label=n)
        pub = np.array(j["evals"][-1]["greedy_return"]); rob = np.array(j["evals"][-1]["robust_return"])
        ax[2].scatter(pub, rob, s=14, color=c, alpha=0.8, lw=0, label=n)
    ax[0].set_title("(a) public protocol: capped", loc="left", fontsize=8.5); ax[1].set_title("(b) robust protocol: capped", loc="left", fontsize=8.5)
    ax[2].set_title("(c) same final models", loc="left", fontsize=8.5)
    for a_ in ax[:2]:
        a_.set_xlabel("env steps (M)"); a_.set_ylim(-0.03, 1.05)
    ax[0].set_ylabel("fraction of eval episodes that hit the cap"); ax[0].legend(loc="upper left")
    ax[2].set_xlabel("greedy return, public protocol"); ax[2].set_ylabel("greedy return, robust protocol")
    lim = max(ax[2].get_xlim()[1], ax[2].get_ylim()[1]); ax[2].plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls=":")
    fig.tight_layout(); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "capped_frac.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(out, "capped_frac.png"), dpi=200, bbox_inches="tight"); plt.close(fig)


def fmt_ci(ci):
    return "n<5" if ci is None or ci[0] is None or np.isnan(ci[0]) else f"[{ci[0]:.1f}, {ci[1]:.1f}]"


def write_report(path, ctx):
    A, R1, R4, TL = ctx["analysis"], ctx["rule1"], ctx["rule4"], ctx["time_limits"]
    rob = A["robust_vs_public"]["configs"]; gx = A["robust_vs_gymnax"]["configs"]
    L = [f"# P0 report: protocol + correction ladder\n", f"Source: `{ctx['stage_dir']}` (10 seeds x 5 configs, {ctx['n_points']} eval points). All numbers are produced by "
         f"`tools/p0_report.py` from the JSONs, `analyze.py` and `mechanism.py` (no hand-copied values). Generated {ctx['date']}.\n"]
    L += ["## 1. Integrity\n", ("All checks passed: 5 configs, seed ids 0-9, JSON args equal the effective `sweep.py P0` arguments, equal number of eval points, one JAX runtime "
          f"(`{json.dumps(ctx['versions'])}`).\n" if not ctx["problems"] else "**PROBLEMS:**\n" + "\n".join(f"- {p}" for p in ctx["problems"]) + "\n"),
          "| config | seeds | eval points | sticky | train cap | eval cap | scale255 | buggy target | timeout bootstrap | minutes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ctx["table"]:
        L.append(f"| {r['config']} | {r['seeds']} | {r['eval_points']} | {r['sticky']} | {r['cap']} | {r['eval_cap']} | {r['input_scale255']} | {r['buggy_target']} | {r['bootstrap_timeout']} | {r['minutes']:.0f} |")
    L += ["", "## 2. Robust-protocol AUC (primary metric), paired by seed vs `public`\n",
          "CIs: Student-t; p: exact sign-flip test, Holm over the 4 comparisons (analyze.py).\n",
          "| config | AUC mean | AUC 95% CI | final | paired dAUC vs public | 95% CI | rel. | Holm p |", "|---|---|---|---|---|---|---|---|"]
    for n, o in rob.items():
        dm = o.get("diff_mean")
        L.append(f"| {n} | {o['auc_mean']:.1f} | {fmt_ci(o['auc_ci'])} | {o['final_mean']:.1f} | " + ("— | — | — | — |" if dm is None else
                 f"{dm:+.1f} | {fmt_ci(o['diff_ci'])} | {o['diff_rel'] * 100:+.0f}% | {o.get('p_holm', float('nan')):.3f} |"))
    L += ["", "### Secondary metrics (same comparisons)\n", "| metric | config | AUC mean | paired d vs public | rel. | Holm p |", "|---|---|---|---|---|---|"]
    for key, label in (("greedy_vs_public", "greedy_return (public protocol)"), ("train_vs_public", "train_return")):
        for n, o in A[key]["configs"].items():
            dm = o.get("diff_mean")
            L.append(f"| {label} | {n} | {o['auc_mean']:.1f} | " + ("— | — | — |" if dm is None else f"{dm:+.1f} | {o['diff_rel'] * 100:+.0f}% | {o.get('p_holm', float('nan')):.3f} |"))
    L += ["", "## 3. Protocol effect (C0) - NOT a method effect\n",
          "Robust-protocol AUC of every arm against `public_gymnax` (trained and, for this metric, evaluated under the robust protocol). Differences between arms trained "
          "under different protocols measure the protocol, not an algorithm.\n", "| config | AUC mean | paired d vs public_gymnax | 95% CI | rel. | Holm p |", "|---|---|---|---|---|---|"]
    for n, o in gx.items():
        dm = o.get("diff_mean")
        L.append(f"| {n} | {o['auc_mean']:.1f} | " + ("— | — | — | — |" if dm is None else f"{dm:+.1f} | {fmt_ci(o['diff_ci'])} | {o['diff_rel'] * 100:+.0f}% | {o.get('p_holm', float('nan')):.3f} |"))
    L += ["", "![capped fraction](P0_report/capped_frac.png)\n",
          f"Public protocol at the last eval point: public_gymnax capped fraction {R1['final_capped_frac']:.2f} (mean over seeds); robust-protocol capped fractions are in panel (b).\n",
          "## 4. Stopping rules\n",
          f"**Rule 1** (public_gymnax reproduces the published order of magnitude): **{'satisfied' if R1['satisfied'] else 'NOT satisfied'}**. Measured public-protocol return "
          f"{R1['measured_mean']:.1f} at {R1['eval_step'] / 1e6:.2f}M steps vs reference {R1['reference_value']:.1f} ({R1['relative_difference'] * 100:+.0f}%). "
          f"Reference source: {R1['reference_source']}.\n",
          f"**Rule 4** (CI half-width / mean > 25 % for any config): **{'TRIGGERED' if R4['triggered'] else 'not triggered'}**.\n",
          "| config | robust AUC mean | CI half-width | half-width / mean |", "|---|---|---|---|"]
    for n, r in R4["per_config"].items():
        L.append(f"| {n} | {r['auc_mean']:.1f} | {r['half_width']:.1f} | {r['rel_half_width'] * 100:.1f}% |")
    if R4["triggered"]:
        L += ["", f"Triggered by: {', '.join(R4['triggered_configs'])}. IQM (descriptive, percentile bootstrap) is in `{ctx['iqm_file']}`; **P2 needs 15 seeds** (cut P3 if compute is short) - decision for the authors.\n"]
    L += ["", "### Time limits (rule 5)\n", f"Only `fixed` bootstraps through time limits: **{TL['only_fixed_bootstraps_timeouts']}**. Largest trunc_frac of any config at any logged point: {TL['max_trunc_frac']:.4f}.\n",
          "| config | timeout bootstrap | mean trunc_frac | max trunc_frac | final trunc_frac |", "|---|---|---|---|---|"]
    for n, r in TL["per_config"].items():
        L.append(f"| {n} | {r['bootstrap_timeout']} | {r['trunc_frac_mean']:.4f} | {r['trunc_frac_max']:.4f} | {r['trunc_frac_final_mean']:.4f} |")
    mv = ctx["mechanism"]
    L += ["", "## 5. Mechanism (predictions 1-3), `mechanism.py` on P0\n", f"P1 norm law: **{mv['prediction_1_norm_law']['verdict']}**; P2 angle: **{mv['prediction_2_angle']['verdict']}**; "
          f"P3 weight-decay equilibrium: **{mv['prediction_3_wd_equilibrium']['verdict']}** (P0 has no NP/weight-decay arms, so only P1 can be judged here). Details: `{ctx['mech_dir']}/verdicts.json`.\n",
          "## 6. Preliminary reading (wording per CLAUDE.md: <5 seeds no p-values; protocol effects are not method gains)\n",
          "- **C0 (protocol):** see section 3 and panel (a): the public protocol's capped fraction goes to "
          f"{R1['final_capped_frac']:.2f} for public_gymnax; AUC differences between arms trained under different protocols are protocol effects.",
          "- **C1 (corrections):** the rungs `fix_target_only`, `fix_input_only`, `fixed` are compared with `public` in section 2 (same protocol, paired seeds). "
          "Report effect sizes with their CIs; use \"substantial improvement\" only under the plan's rule (>= 25 % relative AUC and CI excludes 0 and Holm p < 0.05) - the check is automated in `make_paper_assets.py`.",
          "", "## Reproduce\n", "```bash"] + ctx["commands"] + ["```", ""]
    open(path, "w").write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_dir"); ap.add_argument("--out", default=None, help="directory for analysis files and the figure (default: <stage_dir>/../P0_report)")
    ap.add_argument("--md", default=None, help="report path (default: <out>.md)")
    ap.add_argument("--ref_value", type=float, default=None); ap.add_argument("--ref_step", type=float, default=1e7); ap.add_argument("--ref_source", default="")
    a = ap.parse_args(argv)
    stage_dir = os.path.abspath(a.stage_dir)
    out = os.path.abspath(a.out or os.path.join(os.path.dirname(stage_dir), "P0_report")); os.makedirs(out, exist_ok=True)
    md = a.md or out + ".md"
    runs = load_stage(stage_dir)
    problems, table, versions = integrity(runs)
    if any(m.startswith(("missing", "unexpected")) for m in problems):
        sys.exit("cannot analyse an incomplete stage:\n" + "\n".join(problems))
    A = {}
    for key, metric, ref in (("robust_vs_public", "robust_return", "public"), ("greedy_vs_public", "greedy_return", "public"),
                             ("train_vs_public", "train_return", "public"), ("robust_vs_gymnax", "robust_return", "public_gymnax")):
        A[key], _ = run_analyze(stage_dir, metric, ref, os.path.join(out, f"analysis_{key}.json"))
    iqm_file = os.path.join(out, "analysis_robust_iqm.json")
    run_analyze(stage_dir, "robust_return", "public", iqm_file, ["--iqm"])
    r4 = stop_rule_4(A["robust_vs_public"])
    r1 = stop_rule_1(runs, a.ref_value, a.ref_step, a.ref_source)
    tl = time_limits(runs)
    figure(runs, out)
    mech_dir = os.path.join(out, "mechanism_P0")
    subprocess.run([sys.executable, os.path.join(ROOT, "mechanism.py"), stage_dir, "--out", mech_dir], check=True, capture_output=True)
    mech = json.load(open(os.path.join(mech_dir, "verdicts.json")))
    import datetime
    rel = lambda p: os.path.relpath(p, os.getcwd())
    cmds = [f"python tools/p0_report.py {rel(stage_dir)} --out {rel(out)} --md {rel(md)}" + (f" --ref_value {a.ref_value} --ref_step {a.ref_step:g} --ref_source \"{a.ref_source}\"" if a.ref_value is not None else ""),
            f"# which runs: python analyze.py {rel(stage_dir)} --metric robust_return --ref public --json_out {rel(out)}/analysis_robust_vs_public.json   (and the other three metrics/refs)",
            f"python mechanism.py {rel(stage_dir)} --out {rel(mech_dir)}"]
    ctx = dict(stage_dir=rel(stage_dir), problems=problems, table=table, versions=versions, analysis=A, rule1=r1, rule4=r4, time_limits=tl, mechanism=mech,
               mech_dir=rel(mech_dir), iqm_file=rel(iqm_file), n_points=table[0]["eval_points"] if table else 0, date=datetime.date.today().isoformat(), commands=cmds)
    write_report(md, ctx)
    json.dump(dict(problems=problems, rule1=r1, rule4=r4, time_limits=tl, versions=versions), open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    print(f"integrity problems: {len(problems)}   rule 1 satisfied: {r1['satisfied']}   rule 4 triggered: {r4['triggered']}   only `fixed` bootstraps timeouts: {tl['only_fixed_bootstraps_timeouts']}")
    print("report:", md)
    return ctx


if __name__ == "__main__":
    main()
