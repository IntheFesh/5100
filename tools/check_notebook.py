"""Check DRL_v5_colab.ipynb against the code and against a results tree.

STATIC (always):   every `!python X.py ...` command refers to a script that make_zip.sh ships, and every --flag it passes is
                   defined by that script's argparse (so a renamed flag or a forgotten file is caught before Colab time is spent).
DYNAMIC (--run):   executes the analysis cells of section 5 (P2) and the demo cells of section 7 with IPython - exactly as Colab would,
                   `!` commands, `$var` / `{var}` substitution and all - against --out/P2 (a real P2 directory or a fake one made with
                   `sweep.py P2 --steps 8192 --protocol "... --eval_episodes 2 ..."`), skipping the training sweep line, then checks
                   that every expected output file exists.
Usage:  python tools/check_notebook.py                       # static only
        python tools/check_notebook.py --run --out /tmp/fakeP2     # + execute against /tmp/fakeP2/P2
"""
import argparse, io, json, os, re, shlex, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "DRL_v5_colab.ipynb")


def cells():
    return json.load(open(NB))["cells"]


def src(c):
    return "".join(c["source"])


def shipped_files():
    txt = open(os.path.join(ROOT, "make_zip.sh")).read()
    m = re.search(r"FILES=\((.*?)\)", txt, re.S)
    return m.group(1).split()


def script_flags(script):
    text = open(os.path.join(ROOT, script)).read()
    return set(re.findall(r"add_argument\(\s*\"(--[A-Za-z0-9_]+)\"", text))


def notebook_commands():
    """[(cell_index, script, [flags])] for every `!python ...` line (variables like {PROTO} are expanded when defined in the cell)."""
    out = []
    for i, c in enumerate(cells()):
        if c["cell_type"] != "code":
            continue
        text = src(c)
        proto = re.search(r"PROTO\s*=\s*'([^']*)'", text)
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("!python"):
                continue
            if proto:
                line = line.replace("{PROTO}", proto.group(1))
            line = line.split("#")[0]
            line = re.sub(r"\$\w+|\{[^}]*\}", "X", line)           # variable references -> placeholder
            toks = shlex.split(line[1:])
            script = next(t for t in toks if t.endswith(".py"))
            out.append((i, script, [t for t in toks if t.startswith("--")]))
    return out


def static_check():
    problems = []
    ship = shipped_files()
    for f in ship:
        if not os.path.exists(os.path.join(ROOT, f)):
            problems.append(f"make_zip.sh lists a missing file: {f}")
    n = 0
    for i, script, flags in notebook_commands():
        n += 1
        if script not in ship:
            problems.append(f"cell {i}: {script} is not shipped by make_zip.sh")
            continue
        known = script_flags(script)
        for fl in flags:
            if fl not in known:
                problems.append(f"cell {i}: {script} has no argument {fl}")
    return n, problems


STAGE_MARK = {"P0": "sweep.py P0", "P1": "sweep.py P1", "P2": "sweep.py P2", "P3": "sweep.py P3"}
DEMO_MARK = ("median_seed", "demo_public_vs_ours")             # the two demo cells (section 7)
EXPECTED = {
    "P0": ["P0_robust_return.png", "P0_greedy_return.png", "P0_train_return.png", "P0_analysis_robust_return.json",
           "P0_analysis_greedy_return.json", "P0_analysis_train_return.json", "P0_analysis_robust_vs_public_gymnax.json",
           "mechanism_P0/verdicts.json", "mechanism_P0/fig_lambda.pdf"],
    "P1": ["P1.png", "P1_analysis_robust.json", "P1_analysis_train.json", "mechanism_P1/verdicts.json", "mechanism_P1/fig_lambda.pdf"],
    "P2": ["P2_vs_fixed.json", "P2.png", "P2_vs_public.json", "P2_train_vs_fixed.json", "P2_public_protocol_vs_public.json",
           "mechanism_P2/verdicts.json", "mechanism_P2/fig_lambda.pdf"],
    "P3": ["mechanism_P3_Breakout/verdicts.json", "mechanism_P3_Asterix/verdicts.json", "mechanism_P3_Freeway/verdicts.json"],
    "demo": ["demo_progression.mp4", "demo_progression.png", "demo_public_vs_ours.mp4", "demo_public_vs_ours.png",
             "demo_fixed_vs_ours.mp4", "demo_fixed_vs_ours.png"],
}


def select(stage):
    if stage == "demo":
        return [c for c in cells() if c["cell_type"] == "code" and (DEMO_MARK[0] in src(c) or DEMO_MARK[1] in src(c))]
    return [c for c in cells() if c["cell_type"] == "code" and STAGE_MARK[stage] in src(c)]


def dynamic_check(out, stages=("P2", "demo")):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(ROOT)                                           # the notebook runs inside the code directory (/content/code)
    sh = InteractiveShell.instance()
    sh.user_ns["OUT"] = out
    problems = []
    for st in stages:
        sel = select(st)
        want = 2 if st == "demo" else 1
        if len(sel) != want:
            problems.append(f"{st}: expected {want} notebook cell(s), found {len(sel)}")
        for c in sel:
            code = "\n".join(l for l in src(c).splitlines() if "sweep.py" not in l)      # the training sweep itself is not re-run here
            with capture_output() as cap:
                r = sh.run_cell(code)
            text = cap.stdout + cap.stderr
            if r.error_in_exec is not None or r.error_before_exec is not None:
                problems.append(f"{st}: cell raised {r.error_in_exec or r.error_before_exec!r}")
            for bad in ("Traceback", "FAILED", "ERROR:", "error: unrecognized", "usage:"):
                if bad in text:
                    problems.append(f"{st}: output contains '{bad}':\n" + "\n".join(l for l in text.splitlines() if bad in l or "Error" in l)[:1200])
                    break
            print(text[-1200:])
        for f in EXPECTED[st]:
            p = os.path.join(out, f)
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                problems.append(f"{st}: expected output missing or empty: {p}")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--stages", default="P2,demo", help="comma list of P0,P1,P2,P3,demo (cells are executed in this order)")
    a = ap.parse_args()
    n, problems = static_check()
    print(f"static check: {n} `!python` commands inspected, {len(problems)} problem(s)")
    if a.run:
        assert a.out, "--run needs --out (a directory that contains P2/)"
        problems += dynamic_check(os.path.abspath(a.out), a.stages.split(','))
    for p in problems:
        print("PROBLEM:", p)
    print("NOTEBOOK OK" if not problems else "NOTEBOOK HAS PROBLEMS")
    sys.exit(1 if problems else 0)
