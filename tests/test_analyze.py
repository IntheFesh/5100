"""analyze.py against an INDEPENDENT recomputation (scipy t quantiles, brute-force sign flips, own Holm) on synthetic stages, including the new
--pairs comparison family (one Holm correction over explicitly listed pairs). The synthetic data are fixtures, not results."""
import itertools, json, os, subprocess, sys, glob
import numpy as np, pytest
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import synth


def make(tmp, names_q, n_seeds=10, seed0=0, seed_sd=0.05):
    d = str(tmp / "stage")
    runs = {n: dict(q=q, seed_ids=tuple(range(seed0, seed0 + n_seeds)), steps=2e6, seed_sd=seed_sd, noise=0.02) for n, q in names_q.items()}
    synth.make_stage(d, runs)
    return d


def run_analyze(d, *args, tmp=None):
    out = os.path.join(tmp or d, "out.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "analyze.py"), d, "--json_out", out, *args], capture_output=True, text=True)
    return r, (json.load(open(out)) if os.path.exists(out) else None)


def raw_auc(d, name, metric="robust_return"):
    j = json.load(open(glob.glob(os.path.join(d, f"{name}__*.json"))[0]))
    return np.array([e[metric] for e in j["evals"]]).mean(0)


def signflip(diff):
    n = len(diff); obs = abs(diff.mean())
    S = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    return float(np.mean(np.abs((S * diff).mean(1)) >= obs - 1e-12))


def holm(ps):
    order = np.argsort(ps); m = len(ps); out = np.zeros(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); out[i] = run
    return out


def tci(x):
    n = len(x); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean() - h, x.mean() + h


Q = {"public": 0.9, "fixed_E2": 1.0, "fixed_E8": 1.05, "ours": 1.2}


def test_ref_mode_matches_independent_recomputation(tmp_path):
    d = make(tmp_path, Q)
    r, j = run_analyze(d, "--metric", "robust_return", "--ref", "fixed_E2")
    assert r.returncode == 0, r.stderr
    assert "pairs" not in j                                                         # default behaviour unchanged
    ref = raw_auc(d, "fixed_E2"); names = [n for n in Q if n != "fixed_E2"]
    ps = np.array([signflip(raw_auc(d, n) - ref) for n in names]); ph = holm(ps)
    for n, p, pho in zip(names, ps, ph):
        o = j["configs"][n]; diff = raw_auc(d, n) - ref
        assert o["auc_mean"] == pytest.approx(raw_auc(d, n).mean()) and o["diff_mean"] == pytest.approx(diff.mean())
        assert o["diff_ci"] == pytest.approx(list(tci(diff)), rel=1e-9) and o["auc_ci"] == pytest.approx(list(tci(raw_auc(d, n))), rel=1e-9)
        assert o["p_raw"] == pytest.approx(p) and o["p_holm"] == pytest.approx(pho)
        assert o["diff_rel"] == pytest.approx(diff.mean() / ref.mean())


def test_pairs_family_has_one_holm_over_the_listed_pairs(tmp_path):
    d = make(tmp_path, Q)
    pairs = [("fixed_E2", "public"), ("ours", "fixed_E2"), ("ours", "fixed_E8")]
    r, j = run_analyze(d, "--metric", "robust_return", "--pairs", ",".join(f"{a}:{b}" for a, b in pairs))
    assert r.returncode == 0, r.stderr
    assert "Comparison family" in r.stdout and [(p["config"], p["ref"]) for p in j["pairs"]] == pairs
    diffs = [raw_auc(d, a) - raw_auc(d, b) for a, b in pairs]
    ps = np.array([signflip(x) for x in diffs]); ph = holm(ps)
    for p, x, p_raw, p_h in zip(j["pairs"], diffs, ps, ph):
        assert p["n"] == 10 and p["diff_mean"] == pytest.approx(x.mean()) and p["diff_ci"] == pytest.approx(list(tci(x)), rel=1e-9)
        assert p["p_raw"] == pytest.approx(p_raw) and p["p_holm"] == pytest.approx(p_h)


def test_pairs_with_different_seed_ids_is_refused(tmp_path):
    d = str(tmp_path / "stage")
    synth.make_stage(d, {"a": dict(seed_ids=tuple(range(0, 10)), steps=2e6), "b": dict(seed_ids=tuple(range(10, 20)), steps=2e6)})
    r, _ = run_analyze(d, "--pairs", "a:b", tmp=str(tmp_path))
    assert r.returncode != 0 and "different seed ids" in r.stderr + r.stdout


def test_fewer_than_five_seeds_report_no_ci_and_no_p(tmp_path):
    d = make(tmp_path, {"a": 1.0, "b": 1.2}, n_seeds=3, seed0=100)
    r, j = run_analyze(d, "--pairs", "b:a", "--ref", "a")
    assert r.returncode == 0, r.stderr
    p = j["pairs"][0]
    assert np.isnan(p["diff_ci"][0]) and np.isnan(p["p_raw"]) and "p_holm" not in p and len(p["diff_per_seed"]) == 3
    assert np.isnan(j["configs"]["b"]["p_raw"]) if "p_raw" in j["configs"]["b"] else True


def test_unknown_config_in_pairs_is_an_error(tmp_path):
    d = make(tmp_path, {"a": 1.0, "b": 1.2})
    r, _ = run_analyze(d, "--pairs", "a:zzz", tmp=str(tmp_path))
    assert r.returncode != 0 and "not found" in r.stderr + r.stdout


def test_train_return_and_public_protocol_metrics_are_selectable(tmp_path):
    d = make(tmp_path, Q)
    for metric in ("train_return", "greedy_return"):
        r, j = run_analyze(d, "--metric", metric, "--ref", "public")
        assert r.returncode == 0 and j["metric"] == metric and "fixed_E2" in j["configs"]
