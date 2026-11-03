"""tools/p0_report.py on SYNTHETIC P0 trees (fixtures, not results): integrity checks catch planted problems, stop rules 1/4 fire when they should,
the report and figure are written, and nothing is recomputed outside analyze.py / mechanism.py."""
import glob, json, os, shutil, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "tools"))
import synth, p0_report

Q = {"public_gymnax": 0.95, "public": 0.9, "fix_target_only": 0.93, "fix_input_only": 0.95, "fixed": 1.0}


def make_p0(root, noise=0.03, q=Q):
    d = os.path.join(str(root), "P0")
    synth.stage_from_sweep(d, "P0", params=lambda n, a: dict(q=q[n], s=1.8e5, noise=noise))
    return d


def edit(d, name, fn):
    f = glob.glob(os.path.join(d, f"{name}__*.json"))[0]
    j = json.load(open(f)); fn(j); json.dump(j, open(f, "w"))


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    return make_p0(tmp_path_factory.mktemp("p0good"))


def test_clean_stage_passes_integrity(good):
    problems, table, versions = p0_report.integrity(p0_report.load_stage(good))
    assert problems == [] and len(table) == 5 and versions["jax"] == "0.4.38"


def test_integrity_catches_wrong_args_missing_seeds_and_mixed_runtimes(good, tmp_path):
    d = str(tmp_path / "P0"); shutil.copytree(good, d)
    edit(d, "public", lambda j: j["args"].__setitem__("sticky", 0.0))                                   # wrong protocol in one arm
    edit(d, "fixed", lambda j: j.__setitem__("seed_ids", list(range(9))))                              # a seed missing
    edit(d, "fix_input_only", lambda j: j["versions"].__setitem__("jax", "0.4.99"))                     # another runtime
    edit(d, "fix_target_only", lambda j: (j["evals"].pop(), None))                                      # different number of eval points
    problems, _, versions = p0_report.integrity(p0_report.load_stage(d))
    text = "\n".join(problems)
    assert "config public: args differ" in text and "sticky" in text
    assert "config fixed: seed_ids" in text and "JAX runtimes differ" in text and "different numbers of eval points" in text
    assert versions is None


def test_missing_config_is_reported_and_aborts_the_report(good, tmp_path):
    d = str(tmp_path / "P0"); shutil.copytree(good, d)
    os.remove(glob.glob(os.path.join(d, "fixed__*.json"))[0])
    problems, _, _ = p0_report.integrity(p0_report.load_stage(d))
    assert "missing config fixed" in problems
    with pytest.raises(SystemExit):
        p0_report.main([d, "--out", str(tmp_path / "rep")])


def test_stop_rule_4_triggers_on_noisy_seeds(tmp_path):
    d = make_p0(tmp_path, noise=0.0)
    for n in Q:                                                                                         # seed-to-seed spread of the robust returns
        def spread(j):
            for e in j["evals"]:
                e["robust_return"] = [v * (0.2 + 0.2 * i) for i, v in enumerate(e["robust_return"])]
        edit(d, n, spread)
    ctx = p0_report.main([d, "--out", str(tmp_path / "rep")])
    assert ctx["rule4"]["triggered"] and set(ctx["rule4"]["triggered_configs"]) == set(Q)
    assert "TRIGGERED" in open(str(tmp_path / "rep") + ".md").read() and "15 seeds" in open(str(tmp_path / "rep") + ".md").read()
    assert os.path.exists(str(tmp_path / "rep" / "analysis_robust_iqm.json"))


def test_full_report_on_clean_stage(good, tmp_path):
    out = str(tmp_path / "rep")
    ctx = p0_report.main([good, "--out", out])
    md = open(out + ".md").read()
    assert ctx["problems"] == [] and ctx["rule4"]["triggered"] is False and ctx["time_limits"]["only_fixed_bootstraps_timeouts"]
    for f in ("capped_frac.pdf", "capped_frac.png", "analysis_robust_vs_public.json", "analysis_robust_vs_gymnax.json", "summary.json", "mechanism_P0/verdicts.json"):
        assert os.path.exists(os.path.join(out, f)), f
    assert "NOT a published number" in md                                                               # fallback reference is labelled as the pilot
    assert "## 4. Stopping rules" in md and "Holm" in md and "protocol effect" in md.lower()
    # every AUC mean printed in the report equals the analyze.py JSON (no recomputation)
    ana = json.load(open(os.path.join(out, "analysis_robust_vs_public.json")))["configs"]
    for n, o in ana.items():
        assert f"| {n} | {o['auc_mean']:.1f} |" in md


def test_stop_rule_1_uses_a_given_reference_and_can_fail(good, tmp_path):
    ctx = p0_report.main([good, "--out", str(tmp_path / "rep"), "--ref_value", "400", "--ref_step", "1e7", "--ref_source", "unit-test value"])
    assert ctx["rule1"]["satisfied"] is False and ctx["rule1"]["reference_source"] == "unit-test value"
    assert "NOT satisfied" in open(str(tmp_path / "rep") + ".md").read()
