"""Synthetic results trees for the asset/report tests: results/{P0,P1,P2,P3} + frozen_P2.json + docs/FREEZE.md in a temp dir.
Fixtures only - never results. Names, flags, seeds are exactly those of sweep.py (stage_from_sweep)."""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "tools"))
import synth

S_BY_E = {2: 1.6e5, 4: 2.0e5, 8: 2.3e5}
Q0 = {"public_gymnax": 0.95, "public": 0.9, "fix_target_only": 0.93, "fix_input_only": 0.95, "fixed": 1.0}
Q2 = {"public": 0.9, "fixed_E2": 1.0, "fixed_E8": 1.05, "ours": 1.4}


def build_p0(R, q=Q0, noise=0.02, seed_sd=0.02):
    synth.stage_from_sweep(os.path.join(R, "P0"), "P0", params=lambda n, a: dict(q=q[n], s=1.8e5, noise=noise, seed_sd=seed_sd))


def build_p1(R):
    synth.stage_from_sweep(os.path.join(R, "P1"), "P1", params=lambda n, a: dict(q=1.0, s=S_BY_E[a["epochs"]], noise=0.02, seed_sd=0.02))


def build_p3(R, best_E=8, best_ctrl="--norm_ctrl np --lr 2.5e-4"):
    synth.stage_from_sweep(os.path.join(R, "P3"), "P3", best_E=best_E, best_ctrl=best_ctrl,
                           params=lambda n, a: dict(q=1.0 if "ours" not in n else 1.2, s=2e5, noise=0.02, seed_sd=0.03))


def build_p2(root, q2=None, best_E=8, best_ctrl="--norm_ctrl np --lr 2.5e-4", mode="with_control", p2_only=None, noise=0.02, seed_sd=0.03):
    """P2 tree + frozen_P2.json + docs/FREEZE.md under `root`; `p2_only` emulates `sweep.py --only` (the other configs are not run)."""
    R = os.path.join(root, "results"); os.makedirs(R, exist_ok=True)
    q2 = q2 or Q2

    def params(n, a):
        return dict(q=q2["ours" if n.startswith("ours_") else n], s=S_BY_E[a["epochs"]], noise=noise, seed_sd=seed_sd)
    names = synth.stage_from_sweep(os.path.join(R, "P2"), "P2", best_E=best_E, best_ctrl=best_ctrl, params=params)
    if p2_only:
        for f in os.listdir(os.path.join(R, "P2")):
            if f.split("__")[0] not in p2_only:
                os.remove(os.path.join(R, "P2", f))
        names = [n for n in names if n in p2_only]
    fz = dict(best_E=best_E, best_ctrl=best_ctrl, mode=mode, p2_only=p2_only, p2_arms=names, rule="rule", date="2026-11-03")
    fp = os.path.join(root, "frozen_P2.json"); json.dump(fz, open(fp, "w"), indent=1, sort_keys=True)
    os.makedirs(os.path.join(root, "docs"), exist_ok=True)
    open(os.path.join(root, "docs", "FREEZE.md"), "w").write(f"- sha256: `{hashlib.sha256(open(fp, 'rb').read()).hexdigest()}`\n")
    return R, fp


def build(root, with_p3=True, **kw):
    R, fp = build_p2(root, **kw)
    build_p0(R); build_p1(R)
    if with_p3:
        build_p3(R, kw.get("best_E", 8), kw.get("best_ctrl", "--norm_ctrl np --lr 2.5e-4"))
    return R, fp


def assets_args(root):
    return ["--results", os.path.join(root, "results"), "--frozen", os.path.join(root, "frozen_P2.json"),
            "--freeze_md", os.path.join(root, "docs", "FREEZE.md"), "--paper", os.path.join(root, "paper")]
