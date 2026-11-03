"""sweep.py: the stage definitions are frozen. tests/golden_sweep.json was recorded from the ORIGINAL sweep.py (before the --only option
and the best_E == 2 handling were added); every config name and every flag string (hence every run hash) must still be identical."""
import json, os, re, subprocess, sys, tempfile
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = json.load(open(os.path.join(ROOT, "tests", "golden_sweep.json")))


def dry(stage, extra=()):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "sweep.py"), stage, "--outdir", tempfile.mkdtemp(), "--dry", *extra],
                         capture_output=True, text=True)
    return out


@pytest.mark.parametrize("key", list(GOLD))
def test_stage_definitions_are_unchanged(key):
    stage, *rest = key.split(" ", 1)
    import shlex
    extra = []
    if rest and rest[0].strip():
        toks = rest[0].split(" --")                                     # ['--best_E 8', 'best_ctrl --norm_ctrl np --lr 2.5e-4'] style
        m = re.match(r"--best_E (\d+) --best_ctrl (.*)$", rest[0])
        if m:
            extra = ["--best_E", m.group(1), "--best_ctrl", m.group(2)]
        else:
            m = re.match(r"--base (.*?) --steps (\S+)$", rest[0]); extra = ["--base", m.group(1), "--steps", m.group(2)]
    r = dry(stage, extra)
    assert r.returncode == 0, r.stderr
    got = [[m.group(1), m.group(2)] for m in re.finditer(r"\] run (\S+): (.*)", r.stdout)]
    assert got == GOLD[key]


def test_only_filters_configs_and_rejects_unknown_names():
    r = dry("P2", ["--best_E", "8", "--best_ctrl", "--norm_ctrl np --lr 2.5e-4", "--only", "public,fixed_E2"])
    assert re.findall(r"\] run (\S+):", r.stdout) == ["public", "fixed_E2"]
    bad = dry("P2", ["--best_E", "8", "--best_ctrl", "--norm_ctrl np", "--only", "public,nope"])
    assert bad.returncode != 0 and "unknown config" in (bad.stderr + bad.stdout)


def test_best_E_2_does_not_create_a_duplicate_fixed_E2():
    r = dry("P2", ["--best_E", "2", "--best_ctrl", "--norm_ctrl np"])
    assert r.returncode == 0, r.stderr
    assert re.findall(r"\] run (\S+):", r.stdout) == ["public", "fixed_E2", "ours_E2_np"]
    r3 = dry("P3", ["--best_E", "2", "--best_ctrl", "--norm_ctrl np"])
    names = re.findall(r"\] run (\S+):", r3.stdout)
    assert r3.returncode == 0 and len(names) == 9 and len(set(names)) == 9
