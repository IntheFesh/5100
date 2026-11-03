"""Analyse one sweep stage: per-seed AUC (mean over eval points) of a chosen metric, final value, 95% CIs,
paired-seed differences against a reference config, and Holm-adjusted p-values.

Statistics (chosen for n = 5-10 seeds, where the percentile bootstrap is anti-conservative):
  - CI of a mean and of a paired mean difference: Student-t interval (mean +- t_{n-1,0.975} * sd / sqrt(n)).
  - p-value of a paired difference: exact two-sided sign-flip permutation test (all 2^n sign patterns, n <= 16;
    Monte Carlo with 200k patterns otherwise). Valid under symmetry of the paired differences about 0 under H0.
  - Holm step-down over all comparisons against --ref within the (filtered) stage.
  - CIs and p-values are suppressed below 5 seeds (per-seed values are printed instead).
  - Optional --iqm: interquartile mean over seeds with a percentile-bootstrap CI (descriptive only, Agarwal et al. 2021).
Metrics: robust_return (primary: sticky 0.1, cap 10000), greedy_return (public protocol: deterministic, cap 1000),
         train_return (epsilon-greedy training episodes; secondary)
Usage: python analyze.py <stage_dir> [--metric robust_return] [--ref fixed] [--filter Breakout_] [--plot out.png]"""
import argparse, glob, itertools, json, os, sys, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("stage_dir"); ap.add_argument("--ref", default=None); ap.add_argument("--plot", default=None)
ap.add_argument("--metric", default="robust_return", choices=["robust_return", "greedy_return", "train_return"])
ap.add_argument("--filter", default="", help="only configs whose name starts with this prefix (e.g. a P3 game)")
ap.add_argument("--iqm", action="store_true")
ap.add_argument("--json_out", default=None, help="also write all numbers to this JSON file")
a = ap.parse_args()
rng = np.random.default_rng(0)
MIN_N = 5

try:
    from scipy.stats import t as _t
    tq = lambda df: float(_t.ppf(0.975, df))
except Exception:   # two-sided 95% Student-t quantiles, df = 1..30
    _T = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
          2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
    tq = lambda df: _T[min(df, 30) - 1]


def t_ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < MIN_N:
        return (np.nan, np.nan)
    h = tq(n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (x.mean() - h, x.mean() + h)


def signflip_p(diff):
    diff = np.asarray(diff, float); n = len(diff)
    if n < MIN_N:
        return np.nan
    obs = abs(diff.mean())
    if n <= 16:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    else:
        signs = rng.choice([-1.0, 1.0], size=(200_000, n))
    stats = np.abs((signs * diff).mean(1))
    return float(np.mean(stats >= obs - 1e-12))


def iqm(x):
    x = np.sort(np.asarray(x, float)); n = len(x); k = int(np.floor(0.25 * n))
    return x[k:n - k].mean() if n - 2 * k > 0 else x.mean()


def series(d):
    if a.metric == "train_return":
        h = d["hist"]
        return np.array(h["env_steps"]), np.array(h["train_return"])
    ev = [e for e in d["evals"] if a.metric in e]
    return np.array([e["env_steps"] for e in ev]), np.array([e[a.metric] for e in ev])        # (points, seeds)


runs, seen = {}, {}
for fn in sorted(glob.glob(os.path.join(a.stage_dir, "*.json"))):
    d = json.load(open(fn))
    if "evals" not in d:
        continue
    name = os.path.basename(fn).split("__")[0]
    if not name.startswith(a.filter):
        continue
    if name in seen:
        sys.exit(f"ERROR: two runs named '{name}' ({seen[name]} and {os.path.basename(fn)}); remove the wrong one first")
    seen[name] = os.path.basename(fn)
    x, y = series(d)
    if y.size == 0:
        print(f"[skip] {name}: no '{a.metric}' (run predates protocol v5.1)"); continue
    runs[name] = dict(x=x, y=y, auc=y.mean(0), final=y[-1], wall=d["wall_seconds"], wn=np.array(d["hist"]["wn_dense"]),
                      seed_ids=d.get("seed_ids"))

if a.ref and a.ref not in runs:
    sys.exit(f"ERROR: --ref '{a.ref}' not found among {sorted(runs)}")

out, pvals = {}, {}
for name, r in runs.items():
    o = dict(n=len(r["auc"]), auc_mean=float(r["auc"].mean()), auc_ci=t_ci(r["auc"]), auc_per_seed=r["auc"].round(2).tolist(),
             final_mean=float(r["final"].mean()), final_ci=t_ci(r["final"]), minutes=r["wall"] / 60)
    if a.iqm and len(r["auc"]) >= MIN_N:
        b = [iqm(r["auc"][rng.integers(0, len(r["auc"]), len(r["auc"]))]) for _ in range(5000)]
        o["auc_iqm"] = float(iqm(r["auc"])); o["auc_iqm_ci"] = tuple(np.percentile(b, [2.5, 97.5]))
    if a.ref and name != a.ref:
        ref = runs[a.ref]
        if ref["seed_ids"] is not None and r["seed_ids"] is not None and ref["seed_ids"] != r["seed_ids"]:
            print(f"[warn] {name}: seed ids differ from {a.ref}; paired comparison skipped")
        elif len(ref["auc"]) == len(r["auc"]):
            diff = r["auc"] - ref["auc"]
            o["diff_mean"] = float(diff.mean()); o["diff_ci"] = t_ci(diff)
            o["diff_rel"] = float(diff.mean() / max(abs(ref["auc"].mean()), 1e-9))
            o["diff_per_seed"] = diff.round(2).tolist()
            o["p_raw"] = signflip_p(diff); pvals[name] = o["p_raw"]
    out[name] = o

valid = {k: p for k, p in pvals.items() if not np.isnan(p)}
m, running = len(valid), 0.0
for i, (k, p) in enumerate(sorted(valid.items(), key=lambda kv: kv[1])):
    running = max(running, min(1.0, (m - i) * p)); out[k]["p_holm"] = running

f1 = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"
fci = lambda c: "n<5" if c is None or np.isnan(c[0]) else f"[{c[0]:.1f}, {c[1]:.1f}]"
print(f"metric = {a.metric}   (CIs: Student-t; p: exact sign-flip permutation, Holm over {m} comparison(s) vs {a.ref})")
print(f"| config | n | AUC mean | AUC 95% CI | final | paired ΔAUC vs {a.ref} | Δ 95% CI | rel. | Holm p |")
print("|---|---|---|---|---|---|---|---|---|")
for name, o in out.items():
    dm = o.get("diff_mean"); row = f"| {name} | {o['n']} | {o['auc_mean']:.1f} | {fci(o['auc_ci'])} | {o['final_mean']:.1f} | "
    if dm is None:
        row += "— | — | — | — |"
    else:
        ph = o.get("p_holm"); row += f"{dm:+.1f} | {fci(o['diff_ci'])} | {o['diff_rel']*100:+.0f}% | {'n<5' if ph is None else f'{ph:.3f}'} |"
    print(row)
if any(o["n"] < MIN_N for o in out.values()):
    print("\nPer-seed AUC (fewer than 5 seeds somewhere: no CIs or p-values are reported for those configs):")
    for name, o in out.items():
        print(f"  {name}: {o['auc_per_seed']}" + (f"   Δ vs ref per seed: {o['diff_per_seed']}" if "diff_per_seed" in o else ""))
if a.iqm:
    for name, o in out.items():
        if "auc_iqm" in o:
            print(f"  IQM {name}: {o['auc_iqm']:.1f} [{o['auc_iqm_ci'][0]:.1f}, {o['auc_iqm_ci'][1]:.1f}] (percentile bootstrap, descriptive)")
if a.json_out:
    json.dump({"metric": a.metric, "ref": a.ref, "filter": a.filter, "configs": out}, open(a.json_out, "w"), indent=1, default=float)

if a.plot:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for name, r in runs.items():
        n = r["y"].shape[1]; m_ = r["y"].mean(1)
        h = tq(n - 1) * r["y"].std(1, ddof=1) / np.sqrt(n) if n >= 2 else 0 * m_
        ax[0].plot(r["x"] / 1e6, m_, label=name); ax[0].fill_between(r["x"] / 1e6, m_ - h, m_ + h, alpha=0.2)
        ax[1].plot(np.linspace(0, r["x"][-1] / 1e6, len(r["wn"])), r["wn"].mean(1), label=name)
    ax[0].set_xlabel("env steps (M)"); ax[0].set_ylabel(a.metric.replace("_", " ")); ax[0].legend(fontsize=6, frameon=False)
    ax[1].set_xlabel("env steps (M)"); ax[1].set_ylabel("||W|| dense (LN-preceded)")
    fig.tight_layout(); fig.savefig(a.plot, dpi=150); print("saved", a.plot)
