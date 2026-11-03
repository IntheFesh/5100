"""Pilot figure for the v5.1 plan (2x2):
(a) epsilon-greedy training return, (b) greedy return under the PUBLIC protocol with the cap-induced ceiling,
(c) self-attenuation factor Lambda = ||W||/||W0|| of the LN-preceded dense layer,
(d) the same saved models evaluated under the public vs the robust protocol (+ sticky-trained pilot if present)."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1f1f1c", "#6b6b66", "#e6e5e0"
RUNS = [("A_orig_E2", "public PQN, E=2", "#55554f", "--"),
        ("B_fixed_E2", "fixed PQN, E=2", "#2a78d6", "-"),
        ("C_fixed_E8", "fixed PQN, E=8", "#eb6834", "-"),
        ("D_fixed_E8_NP", "fixed PQN, E=8 + NP", "#1baf7a", "-")]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
(a, b), (c, d) = ax

for name, lab, col, ls in RUNS:
    dd = json.load(open(f"{name}.json")); h = dd["hist"]
    x = np.array(h["env_steps"]) / 1e6; tr = np.array(h["train_return"])
    for s in range(tr.shape[1]):
        a.plot(x, tr[:, s], color=col, lw=0.6, alpha=0.35, ls=ls)
    a.plot(x, tr.mean(1), color=col, lw=2, ls=ls, label=lab)
    ev = dd["evals"]; xe = np.array([e["env_steps"] for e in ev]) / 1e6; ye = np.array([e["greedy_return"] for e in ev])
    b.plot(xe, ye.mean(1), color=col, lw=2, ls=ls, marker="o", ms=4)
    for s in range(ye.shape[1]):
        b.scatter(xe, ye[:, s], color=col, s=9, alpha=0.45, lw=0)
    wn = np.array(h["wn_dense"]) / 16.0
    c.plot(x, wn.mean(1), color=col, lw=2, ls=ls)
    off = {"A_orig_E2": 5, "B_fixed_E2": -5}.get(name, 0)
    c.annotate(f"{wn.mean(1)[-1]:.2f}", (x[-1], wn.mean(1)[-1]), xytext=(4, off), textcoords="offset points",
               va="center", fontsize=7, color=INK)

a.set_title("(a) training return (ε-greedy, 2 seeds)", loc="left", fontsize=8.5, color=INK)
a.set_xlabel("env steps (M)"); a.set_ylabel("episode return")
a.legend(frameon=False, fontsize=7, loc="lower right")
b.axhspan(140, 156, color="#f3e3c2", zorder=0)
b.set_ylim(15, 178)
b.text(0.80, 166, "shaded: 1000-step cap reached; at 2M every eval\nepisode is capped and all episodes are identical", fontsize=6.3, color=INK, va="center")
b.set_title("(b) greedy return, public protocol", loc="left", fontsize=8.5, color=INK)
b.set_xlabel("env steps (M)"); b.set_ylabel("return (dots = seeds)")
c.axhline(1, color=MUTED, lw=0.8, ls=":")
c.set_title("(c) self-attenuation Λ = ||W|| / ||W₀||, dense layer", loc="left", fontsize=8.5, color=INK)
c.set_xlabel("env steps (M)"); c.set_ylabel("Λ"); c.set_xlim(0, 2.25)

# (d) protocol comparison on saved models (values from check_protocol.py, 16 episodes per seed)
P = json.load(open("proto_summary.json"))
rows = P["rows"]
short = {"public code, deterministic training": "public, det. train", "fixed code, deterministic training": "fixed, det. train",
         "public code, sticky training": "public, sticky train", "fixed code, sticky training": "fixed, sticky train"}
for r in rows:
    r["label"] = short.get(r["label"], r["label"])
for i, r in enumerate(rows):
    y = len(rows) - 1 - i
    d.scatter(r["public"], [y] * len(r["public"]), marker="o", s=26, facecolor="white", edgecolor=r["color"], lw=1.4, zorder=3)
    d.scatter(r["robust"], [y] * len(r["robust"]), marker="o", s=26, color=r["color"], zorder=3)
    for pu, ro in zip(r["public"], r["robust"]):
        d.plot([pu, ro], [y, y], color=r["color"], lw=0.8, alpha=0.5, zorder=2)
d.set_yticks(range(len(rows))); d.set_yticklabels([r["label"] for r in rows][::-1], fontsize=7)
d.set_ylim(-0.6, len(rows) - 0.4)
d.set_xlabel("greedy return at 2M, per training seed\n○ public protocol    ● robust (sticky 0.1, cap 10 000, 16 ep.)", fontsize=7)
d.set_title("(d) same models, two evaluation protocols", loc="left", fontsize=8.5, color=INK)
d.grid(axis="y", visible=False)
fig.tight_layout(h_pad=1.6, w_pad=1.6)
os.makedirs("fig", exist_ok=True)
fig.savefig("fig/pilot_v51.pdf"); fig.savefig("fig/pilot_v51.png", dpi=200)
print("saved fig/pilot_v51.pdf")
