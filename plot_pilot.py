import json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
runs = [("A_orig_E2", "public PQN (x/255, buggy Q(λ)), E=2", "#888888"),
        ("B_fixed_E2", "fixed PQN, E=2", "#1f5fa8"),
        ("C_fixed_E8", "fixed PQN, E=8", "#d0703a"),
        ("D_fixed_E8_NP", "fixed PQN, E=8 + norm projection", "#2a9d5c")]
out = sys.argv[1] if len(sys.argv) > 1 else "fig"
os.makedirs(out, exist_ok=True)
data = {k: json.load(open(f"{k}.json")) for k, _, _ in runs if os.path.exists(f"{k}.json")}

def smooth(y, k=3):
    y = np.asarray(y, float)
    return np.array([np.nanmean(y[max(0, i - k + 1):i + 1], 0) for i in range(len(y))])

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
summary = {}
for k, lab, c in runs:
    if k not in data: continue
    h = data[k]["hist"]; x = np.array(h["env_steps"]) / 1e6
    tr = np.array(h["train_return"])                       # (points, seeds)
    s = smooth(tr)
    for j in range(s.shape[1]):
        ax[0].plot(x, s[:, j], color=c, lw=0.6, alpha=0.45)
    ax[0].plot(x, s.mean(1), color=c, lw=1.6, label=lab)
    ev = data[k]["evals"]
    ex = np.array([e["env_steps"] for e in ev]) / 1e6; ey = np.array([e["greedy_return"] for e in ev])
    ax[1].errorbar(ex, ey.mean(1), yerr=np.abs(ey - ey.mean(1, keepdims=True)).max(1), color=c, marker="o", ms=3, lw=1.2, capsize=2)
    wn = np.array(h["wn_dense"])
    ax[2].plot(x, wn.mean(1), color=c, lw=1.4)
    summary[k] = dict(final_train=float(s[-1].mean()), final_train_seeds=s[-1].tolist(), auc_train=float(np.nanmean(tr)),
                      final_eval=ey[-1].tolist(), eval_steps=ex.tolist(), eval_mean=ey.mean(1).tolist(),
                      wn_dense_final=wn[-1].tolist(), wall_seconds=data[k]["wall_seconds"], grad_steps=data[k]["grad_steps"])
    if "rad_cum" in h and len(h["rad_cum"]):
        summary[k]["rad_over_quad"] = (np.array(h["rad_cum"][-1]) / np.array(h["quad_cum"][-1])).tolist()
ax[0].set_title("(a) training return (ε-greedy)"); ax[0].set_xlabel("env steps (M)")
ax[1].set_title("(b) greedy evaluation, 64 episodes"); ax[1].set_xlabel("env steps (M)")
ax[2].set_title("(c) ‖W‖ of LN-preceded dense layer"); ax[2].set_xlabel("env steps (M)")
ax[0].legend(frameon=False, fontsize=6.3, loc="upper left")
fig.tight_layout(); fig.savefig(f"{out}/pilot_curves.pdf"); fig.savefig(f"{out}/pilot_curves.png", dpi=160)
json.dump(summary, open(f"{out}/pilot_summary.json", "w"), indent=1)
for k, v in summary.items():
    print(k, {kk: (np.round(vv, 2).tolist() if isinstance(vv, (list, float)) else vv) for kk, vv in v.items()})
