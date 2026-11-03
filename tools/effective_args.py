"""Print the EFFECTIVE pqn_v5.py arguments (argparse: last occurrence wins) of every config of a sweep stage.

CLI:   python tools/effective_args.py P0 [--best_E 8 --best_ctrl "--norm_ctrl np"] [--md]
Module: from effective_args import effective_rows   ->   [(config_name, {arg: value, ...}), ...]
The stage definition comes from `sweep.py <stage> --dry` (no training is started); the flags are resolved with the very
parser defined in pqn_v5.py (only the source above 'a = p.parse_args()' is executed)."""
import argparse, os, re, shlex, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARSER = None


def _parser():
    global _PARSER
    if _PARSER is None:
        src = open(os.path.join(ROOT, "pqn_v5.py")).read().split("a = p.parse_args()")[0]
        ns = {}
        exec(src, ns)
        for act in ns["p"]._actions:             # --out is required in pqn_v5.py; irrelevant here
            if act.dest == "out":
                act.required = False
        _PARSER = ns["p"]
    return _PARSER


def effective_rows(stage, best_E=None, best_ctrl=None, extra=()):
    """[(name, flags_string, {arg: value})] for every config of `stage` (via sweep.py --dry)."""
    cmd = [sys.executable, os.path.join(ROOT, "sweep.py"), stage, "--outdir", tempfile.mkdtemp(), "--dry", *extra]
    if best_E is not None:
        cmd += ["--best_E", str(best_E)]
    if best_ctrl is not None:
        cmd += ["--best_ctrl", best_ctrl]
    txt = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in txt.splitlines():
        m = re.match(r"\[(\d+)/(\d+)\] run (\S+): (.*)$", line)
        if m:
            ns_ = _parser().parse_args(shlex.split(m.group(4)))
            rows.append((m.group(3), m.group(4), vars(ns_)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("--best_E", default=None); ap.add_argument("--best_ctrl", default=None)
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--keys", default="sticky,max_steps_in_episode,eval_cap,eval_sticky,input_scale255,buggy_target,"
                                      "bootstrap_timeout,epochs,norm_ctrl,wd,lr,seeds,seed0,env", help="comma list of arguments to show")
    o = ap.parse_args()
    keys = o.keys.split(",")
    rows = effective_rows(o.stage, o.best_E, o.best_ctrl)
    if o.md:
        print("| config | " + " | ".join(keys) + " |"); print("|---|" + "---|" * len(keys))
        for n, _, d in rows:
            print(f"| {n} | " + " | ".join(str(d[k]) for k in keys) + " |")
    else:
        for n, _, d in rows:
            print(n, {k: d[k] for k in keys})
