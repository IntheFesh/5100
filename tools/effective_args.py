"""Print the EFFECTIVE pqn_v5.py arguments (argparse: last occurrence wins) of every config of a sweep stage.

Usage: python tools/effective_args.py P0 [--best_E 8 --best_ctrl "--norm_ctrl np"] [--md]
Reads the stage definition from sweep.py via `sweep.py <stage> --dry` (no training is started) and resolves the flags
with the very parser defined in pqn_v5.py (the source above 'a = p.parse_args()' is executed, nothing else)."""
import argparse, os, re, shlex, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("stage"); ap.add_argument("--best_E", default=None); ap.add_argument("--best_ctrl", default=None)
ap.add_argument("--md", action="store_true", help="markdown table")
ap.add_argument("--keys", default="sticky,max_steps_in_episode,eval_cap,eval_sticky,input_scale255,buggy_target,bootstrap_timeout,epochs,norm_ctrl,wd,lr,seeds,seed0,env",
                help="comma list of arguments to show")
o = ap.parse_args()

src = open(os.path.join(ROOT, "pqn_v5.py")).read().split("a = p.parse_args()")[0]
ns = {}
exec(src, ns)
parser = ns["p"]
for act in parser._actions:                      # --out is required in pqn_v5.py; irrelevant here
    if act.dest == "out":
        act.required = False

cmd = [sys.executable, os.path.join(ROOT, "sweep.py"), o.stage, "--outdir", tempfile.mkdtemp(), "--dry"]
if o.best_E is not None:
    cmd += ["--best_E", o.best_E]
if o.best_ctrl is not None:
    cmd += ["--best_ctrl", o.best_ctrl]
txt = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
keys = o.keys.split(",")
rows = []
for line in txt.splitlines():
    m = re.match(r"\[(\d+)/(\d+)\] run (\S+): (.*)$", line)
    if m:
        ns_ = parser.parse_args(shlex.split(m.group(4)))
        rows.append((m.group(3), {k: getattr(ns_, k) for k in keys}))
if o.md:
    print("| config | " + " | ".join(keys) + " |"); print("|---|" + "---|" * len(keys))
    for n, d in rows:
        print(f"| {n} | " + " | ".join(str(d[k]) for k in keys) + " |")
else:
    for n, d in rows:
        print(n, d)
