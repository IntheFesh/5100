"""DRL_v5_colab.ipynb vs the code (round 5): static consistency (scripts shipped, flags exist) and execution of the notebook's analysis /
demo cells (via IPython, as Colab would) against SYNTHETIC P0/P1/P3 trees (tests/synth.py) and, if present, a real tiny P2 tree.

Real tiny P2 tree:  python sweep.py P2 --outdir /tmp/fakeP2 --best_E 8 --best_ctrl "--norm_ctrl np --lr 2.5e-4" --steps 8192 \
                      --protocol "--sticky 0.1 --max_steps_in_episode 10000 --eval_cap 300 --eval_sticky 0.1 --eval_episodes 2 --num_envs 32 --minibatches 8"
                    (+ stand-in 1M/3M snapshots, see tools/check_notebook.py) then:  python tools/check_notebook.py --run --out /tmp/fakeP2
"""
import os, subprocess, sys, zipfile
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "tools"))
import synth
import check_notebook as cn


def test_static_notebook_commands_match_the_scripts_that_are_shipped():
    n, problems = cn.static_check()
    assert n >= 20 and problems == []


def test_make_zip_is_flat_and_complete(tmp_path):
    out = tmp_path / "pkg.zip"
    subprocess.run([os.path.join(ROOT, "make_zip.sh"), str(out)], check=True, capture_output=True)
    names = zipfile.ZipFile(out).namelist()
    assert "pqn_v5.py" in names and "DRL_v5_colab.ipynb" in names and "mechanism.py" in names and "tests/test_stats.py" in names
    assert not any(n.endswith((".pkl", ".mp4")) or "third_party" in n or ".venv" in n for n in names)
    assert all("/" not in n for n in names if n != "tests/test_stats.py")                    # flat: no wrapper folder
    for script in {s for _, s, _ in cn.notebook_commands()} | {"test_v5.py"}:
        assert script in names, script


def _p0(d, n_seeds=10):
    ids = tuple(range(n_seeds)); base = dict(seed_ids=ids, steps=2e6, s=1.8e5)
    runs = {"public_gymnax": dict(base, public=True, protocol_gymnax=True, q=0.9),
            "public": dict(base, public=True, q=0.9), "fix_target_only": dict(base, q=0.95), "fix_input_only": dict(base, q=0.95),
            "fixed": dict(base, q=1.0)}
    synth.make_stage(os.path.join(d, "P0"), runs)


def _p1(d):
    ids = (100, 101, 102); base = dict(seed_ids=ids, steps=2e6, s=2.0e5); runs = {}
    for E in (2, 4, 8):
        runs[f"E{E}_none"] = dict(base, E=E); runs[f"E{E}_np"] = dict(base, E=E, ctrl="np"); runs[f"E{E}_radial"] = dict(base, E=E, ctrl="radial")
        for lam in (0.1, 0.3):
            runs[f"E{E}_wd{lam}"] = dict(base, E=E, ctrl="wd", wd=lam)
    for lr in (2.5e-4, 1.25e-4):
        runs[f"E8_none_lr{lr:g}".replace("e-04", "e-4")] = dict(base, E=8, lr=lr)
    for E in (4, 8):
        runs[f"E{E}_np_lr2.5e-4"] = dict(base, E=E, ctrl="np", lr=2.5e-4)
    assert len(runs) == 19
    synth.make_stage(os.path.join(d, "P1"), runs)


def _p3(d):
    runs = {}
    for env in ("Breakout", "Asterix", "Freeway"):
        base = dict(seed_ids=tuple(range(2000, 2005)), steps=2e6, s=1.8e5, env=f"{env}-MinAtar")
        runs[f"{env}_public"] = dict(base, public=True); runs[f"{env}_fixed_E2"] = dict(base)
        runs[f"{env}_fixed_E8"] = dict(base, E=8); runs[f"{env}_ours_E8_np_lr2.5e-4"] = dict(base, E=8, ctrl="np", lr=2.5e-4)
    synth.make_stage(os.path.join(d, "P3"), runs)


def test_analysis_cells_run_on_synthetic_p0_p1_p3(tmp_path):
    pytest.importorskip("IPython")
    out = str(tmp_path); _p0(out); _p1(out); _p3(out)
    problems = cn.dynamic_check(out, ("P0", "P1", "P3"))
    assert problems == [], problems


def test_demo_and_p2_cells_run_on_a_real_tiny_p2_tree_if_available():
    pytest.importorskip("IPython")
    out = os.environ.get("FAKE_P2", "/tmp/fakeP2")
    if not os.path.isdir(os.path.join(out, "P2")):
        pytest.skip(f"no tiny P2 tree at {out} (see the module docstring)")
    problems = cn.dynamic_check(out, ("P2", "demo"))
    assert problems == [], problems
