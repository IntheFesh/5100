"""Resumable sweep runner for the v5.1 plan (works on Colab; results go to Google Drive).

Usage (Colab):
  python sweep.py P0 --outdir /content/drive/MyDrive/drl_v5
  python sweep.py P1 --outdir ...
  python sweep.py P2 --outdir ... --best_E 8 --best_ctrl "--norm_ctrl np --lr 2.5e-4"   # frozen from P1 (required)
  python sweep.py P3 --outdir ... --best_E 8 --best_ctrl "--norm_ctrl np --lr 2.5e-4"
Each finished run is saved as <outdir>/<stage>/<name>__<hash>.json (+ <name>__<hash>.log, + params pickles);
finished runs are skipped, so a disconnected session simply re-runs the same command. A second config with the
same name but different flags (e.g. P2 re-run with another --best_ctrl) is refused.
"""
import argparse, glob, hashlib, json, os, shlex, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["P0", "P1", "P2", "P3", "LIST"])
ap.add_argument("--outdir", required=True)
ap.add_argument("--env", default="SpaceInvaders-MinAtar")
ap.add_argument("--steps", type=float, default=1e7)
ap.add_argument("--base", default="", help="extra flags added to every non-public config of P1-P3")
ap.add_argument("--best_E", type=int, default=None)
ap.add_argument("--best_ctrl", default=None, help='frozen P2 control incl. its lr, e.g. "--norm_ctrl np --lr 2.5e-4"')
ap.add_argument("--dry", action="store_true")
ap.add_argument("--protocol", default="--sticky 0.1 --max_steps_in_episode 10000 --eval_cap 10000 --eval_sticky 0.1",
                help="environment protocol for every run (v5.1). The pilot showed the gymnax default (no sticky actions, "
                     "1000-step cap) saturates: every greedy policy hits the cap and all eval episodes are identical. "
                     "Only value-flags here (no store_true flags), so per-config flags can override them.")
a = ap.parse_args()

PUBLIC = "--input_scale255 --buggy_target"
GYMNAX_PROTOCOL = "--sticky 0 --max_steps_in_episode 1000"     # appended last -> overrides --protocol (argparse: last wins)
BOOT = "--bootstrap_timeout"                                  # both code fixes are on by default in pqn_v5.py


def ctrl_tag(flags):
    """Readable name for a control string, e.g. '--norm_ctrl np --lr 2.5e-4' -> 'np_lr2.5e-4'."""
    t = shlex.split(flags); kv = {t[i].lstrip("-"): t[i + 1] for i in range(0, len(t) - 1, 2)}
    tag = kv.get("norm_ctrl", "none") + (kv["wd"] if kv.get("norm_ctrl") == "wd" and "wd" in kv else "")
    return tag + (f"_lr{kv['lr']}" if "lr" in kv else "")


def stage_configs(stage):
    B = f"{BOOT} {a.base}"
    if stage == "P0":   # protocol + correction ladder, 10 seeds; each single-fix rung differs from 'public' in ONE thing
        s = "--seeds 10 --seed0 0"
        return [("public_gymnax", f"{PUBLIC} {s} {GYMNAX_PROTOCOL}"),     # replication of the published setting
                ("public", f"{PUBLIC} {s}"),                               # same code, main protocol
                ("fix_target_only", f"--input_scale255 {s}"),
                ("fix_input_only", f"--buggy_target {s}"),
                ("fixed", f"{BOOT} {s}")]                                  # both fixes + timeout bootstrap (= P1/P2 base)
    if stage == "P1":   # screening on development seeds 100-102
        s = "--seeds 3 --seed0 100"
        cfgs = []
        # wd values bracket wd* = s*eta0/(2 r0^2) ~ 0.15-0.23 (pilot slope); 1e-3 would have a 2M-step relaxation time
        for E in (2, 4, 8):
            for cname, cflag in [("none", "--norm_ctrl none"), ("np", "--norm_ctrl np"), ("radial", "--norm_ctrl radial"),
                                 ("wd0.1", "--norm_ctrl wd --wd 0.1"), ("wd0.3", "--norm_ctrl wd --wd 0.3")]:
                cfgs.append((f"E{E}_{cname}", f"{B} --epochs {E} {cflag} {s}"))
        for lr in ("2.5e-4", "1.25e-4"):    # 'just lower the learning rate' controls at E=8
            cfgs.append((f"E8_none_lr{lr}", f"{B} --epochs 8 --lr {lr} {s}"))
        for E in (4, 8):                    # NP keeps the effective LR ~Lambda x higher late in training -> matched-LR NP
            cfgs.append((f"E{E}_np_lr2.5e-4", f"{B} --epochs {E} --norm_ctrl np --lr 2.5e-4 {s}"))
        return cfgs
    if stage in ("P2", "P3"):
        if a.best_E is None or a.best_ctrl is None:
            if a.stage == "LIST":
                a.best_E, a.best_ctrl = 8, "--norm_ctrl np"
            else:
                sys.exit("P2/P3 need --best_E and --best_ctrl (frozen from P1; see frozen_P2.json)")
        ours = f"ours_E{a.best_E}_{ctrl_tag(a.best_ctrl)}"
    if stage == "P2":   # confirmation on fresh seeds 1000-1009; configs frozen before running
        s = "--seeds 10 --seed0 1000 --save_params --ckpt_at 1,3"
        return [("public", f"{PUBLIC} {s}"),
                ("fixed_E2", f"{B} {s}"),
                (f"fixed_E{a.best_E}", f"{B} --epochs {a.best_E} {s}"),
                (ours, f"{B} --epochs {a.best_E} {a.best_ctrl} {s}")]
    if stage == "P3":   # robustness on other MinAtar games, 5 seeds
        s = "--seeds 5 --seed0 2000"
        out = []
        for env in ("Breakout-MinAtar", "Asterix-MinAtar", "Freeway-MinAtar"):
            tag = env.split("-")[0]
            if tag == "Freeway":
                # Freeway's 2500-step limit is part of the game rules (score = crossings within the limit): keep the
                # cap AND treat it as terminal (no timeout bootstrapping), in every arm
                cap, BB = "--max_steps_in_episode 2500 --eval_cap 2500", a.base
            else:
                cap, BB = "", B
            out += [(f"{tag}_public", f"--env {env} {PUBLIC} {s} {cap}"),
                    (f"{tag}_fixed_E2", f"--env {env} {BB} {s} {cap}"),
                    (f"{tag}_fixed_E{a.best_E}", f"--env {env} {BB} --epochs {a.best_E} {s} {cap}"),
                    (f"{tag}_{ours}", f"--env {env} {BB} --epochs {a.best_E} {a.best_ctrl} {s} {cap}")]
        return out
    return []


def run(stage):
    d = os.path.join(a.outdir, stage); os.makedirs(d, exist_ok=True)
    cfgs = stage_configs(stage)
    names = [n for n, _ in cfgs]
    assert len(names) == len(set(names)), "duplicate config names in stage definition"
    failed = []
    for i, (name, flags) in enumerate(cfgs):
        if "--env" not in flags:
            flags = f"--env {a.env} " + flags
        flags = f"--steps {a.steps:g} {a.protocol} " + " ".join(flags.split())
        h = hashlib.sha1(flags.encode()).hexdigest()[:10]
        out = os.path.join(d, f"{name}__{h}.json")
        if os.path.exists(out):
            print(f"[{i+1}/{len(cfgs)}] skip (done) {name}"); continue
        clash = [f for f in glob.glob(os.path.join(d, f"{name}__*.json")) if f != out]
        if clash:
            print(f"[{i+1}/{len(cfgs)}] REFUSED {name}: a run with the same name but different flags exists: "
                  f"{[os.path.basename(c) for c in clash]}. Frozen configs must not change; delete it only if it was a mistake.")
            failed.append(name); continue
        cmd = [sys.executable, "-W", "ignore", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pqn_v5.py")] \
            + shlex.split(flags) + ["--out", out + ".tmp"]
        print(f"[{i+1}/{len(cfgs)}] run {name}: {flags}", flush=True)
        if a.dry: continue
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        with open(out.replace(".json", ".log"), "w") as f:                 # per-run log (stdout + stderr)
            f.write(" ".join(cmd) + "\n\n" + r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0 or not os.path.exists(out + ".tmp"):
            print("   FAILED (see log):", r.stderr[-800:]); failed.append(name); continue
        base = out.replace(".json", "")
        for pk in sorted(set(glob.glob(base + "_*params.pkl.tmp"))):      # params + snapshots FIRST ...
            os.replace(pk, pk[:-4])
        os.replace(out + ".tmp", out)                                        # ... the JSON (= 'done' marker) LAST
        with open(os.path.join(d, "index.jsonl"), "a") as f:
            f.write(json.dumps({"name": name, "flags": flags, "file": os.path.basename(out), "minutes": (time.time() - t) / 60}) + "\n")
        print(f"   done in {(time.time()-t)/60:.1f} min", flush=True)
    if failed:
        print(f"\n{len(failed)} config(s) failed or refused: {failed}")
        sys.exit(1)


if a.stage == "LIST":
    for st in ("P0", "P1", "P2", "P3"):
        print(st, len(stage_configs(st)), "configs:", ", ".join(n for n, _ in stage_configs(st)))
else:
    run(a.stage)
