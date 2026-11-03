"""Load pqn_v5.py WITHOUT running training (same trick as test_v5.py: execute the source up to 'eval_fn = ').

pqn_v5.py parses the command line and builds everything at import time, so it cannot be imported as a module.
`load_pqn(argv)` executes the definition part in a fresh namespace with a faked sys.argv and returns that namespace;
`ns["a"]` is the argparse Namespace the functions read their hyper-parameters from (functions look `a.*` up at call
time, so tests may flip e.g. `ns["a"].buggy_target` to exercise both modes)."""
import os, sys, contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pqn_source():
    return open(os.path.join(ROOT, "pqn_v5.py")).read()


def load_pqn(argv=(), cut="eval_fn = "):
    src = pqn_source().split(cut)[0]
    old = sys.argv
    sys.argv = ["pqn_v5.py", "--out", "/tmp/_unused.json", *argv]
    ns = {"__name__": "pqn_v5_under_test", "__file__": os.path.join(ROOT, "pqn_v5.py")}
    try:
        exec(compile(src, "pqn_v5.py", "exec"), ns)
    finally:
        sys.argv = old
    return ns


@contextlib.contextmanager
def flags(ns, **kw):
    """Temporarily override attributes of ns['a'] (the parsed args)."""
    a = ns["a"]; old = {k: getattr(a, k) for k in kw}
    for k, v in kw.items():
        setattr(a, k, v)
    try:
        yield a
    finally:
        for k, v in old.items():
            setattr(a, k, v)
