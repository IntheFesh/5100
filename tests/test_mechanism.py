"""Unit and integration tests for mechanism.py (round 4).

 * closed forms vs step-by-step sums, slope recovery from synthetic data with known slope, equilibrium / window formulas
 * end-to-end verdicts on SYNTHETIC stages (tests/synth.py): the pre-registered rules must say 'supported' when the data follow the
   model and 'not supported' for each planted violation (negative controls) - synthetic data are test fixtures, not results
 * the CPU pilot (pilot_results/A-D: no ang_cum, no weight-decay arm) reproduces the plan's slopes and yields 'cannot judge' for P2/P3
 * RUN_SLOW=1: real smoke-level training runs (E=2/8 x none/np/wd, 1 seed, 65k steps) produce ang_cum and all figures
"""
import glob, json, os, shutil, subprocess, sys
import numpy as np, optax, pytest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import mechanism as m
import synth

BASE = dict(steps=1e7, seed_ids=(100, 101, 102), s=2.0e5)          # 10M env steps as in the real stages


# ----------------------------------------------------------------------------------------------------------------------
# numerics
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("G,eta0", [(1000, 5e-4), (3000, 2.5e-4), (777, 1e-3)])
def test_sum_eta2_exact_form_equals_stepwise_sum(G, eta0):
    eta = np.array([eta0 * (1 - t / G) for t in range(G)])
    for k in (0, 1, 7, G // 3, G // 2, G - 1, G):
        assert m.sum_eta2(k, G, eta0) == pytest.approx(float(np.sum(eta[:k] ** 2)), rel=1e-10, abs=1e-18)
        assert m.sum_eta(k, G, eta0) == pytest.approx(float(np.sum(eta[:k])), rel=1e-10, abs=1e-18)


def test_sum_eta2_matches_the_optax_schedule_pqn_uses():
    G, eta0 = 2000, 5e-4
    sched = optax.linear_schedule(eta0, 1e-20, G)
    eta = np.array([float(sched(t)) for t in range(G)])
    assert m.sum_eta2(G, G, eta0) == pytest.approx(float(np.sum(eta ** 2)), rel=1e-6)


@pytest.mark.parametrize("G", [1000, 30000, 156224, 624896])
def test_closed_form_of_the_plan_agrees_with_the_discrete_sum(G):
    """Plan 3.4(b): sum_{s<phi G} eta_s^2 ~ eta0^2 G [1 - (1-phi)^3] / 3.  Relative difference is O(1/(G phi))."""
    eta0 = 5e-4
    for phi in (0.1, 0.25, 0.5, 0.9, 1.0):
        exact = float(m.sum_eta2(round(phi * G), G, eta0)); closed = float(m.sum_eta2_closed(phi, G, eta0))
        assert closed == pytest.approx(exact, rel=3.0 / (G * phi))


@pytest.mark.parametrize("noise", [0.0, 0.001])
def test_slope_recovery_from_synthetic_data_ols(noise):
    """Known slope + multiplicative noise on y -> the OLS slope on the second half (the estimator used for s, as in the plan's
    norm_slope.py) is recovered to < 5 %. NB the OLS slope amplifies noise because the late-training range of S is narrow: at 0.3 % noise
    the 95th percentile of the error is already 6 % (measured: 0.1 % -> 2 %, 0.5 % -> 9 %, 1 % -> 18 %). The pilot's relative residuals
    are ~0.3-0.5 % (R^2 >= 0.996), so per-seed s carries a few-percent uncertainty; the prediction-3 band (factor 2 in Lambda = 4 in s) is wide enough."""
    rng = np.random.default_rng(0)
    G, eta0, s_true = 156224, 5e-4, 1.7e5
    S = m.sum_eta2(np.linspace(0.05, 1.0, 20) * G, G, eta0)
    half = slice(len(S) // 2, None)
    errs = []
    for _ in range(20):
        y = s_true * S * (1 + noise * rng.standard_normal(len(S)))
        s_hat, c_hat, r2 = m.fit_linear(S[half], y[half])
        errs.append(abs(s_hat - s_true) / s_true)
    assert max(errs) < 0.05
    if noise == 0.0:                                                                    # exact data: exact slope, zero intercept, R^2 = 1
        assert s_hat == pytest.approx(s_true, rel=1e-9) and abs(c_hat) < 1e-6 and r2 == pytest.approx(1.0)


@pytest.mark.parametrize("noise", [0.0, 0.01, 0.03])
def test_slope_recovery_from_synthetic_data_through_origin(noise):
    rng = np.random.default_rng(1)
    G, eta0, s_true = 156224, 5e-4, 1.7e5
    S = m.sum_eta2(np.linspace(0.05, 1.0, 20) * G, G, eta0)
    for _ in range(20):
        y = s_true * S * (1 + noise * rng.standard_normal(len(S)))
        assert abs(m.fit_origin(S[10:], y[10:]) - s_true) / s_true < 0.05


def test_equilibrium_formulas_reproduce_the_plans_numbers():
    """Plan 3.6/3.8: s = 1.54-2.32e5, ||W0||^2 = 256, eta0 = 5e-4  ->  lambda* = 0.15-0.23; Lambda_eq(eta0) = 1.2-1.5 (lam 0.1), 0.7-0.9 (lam 0.3)."""
    for s, lo, hi in ((1.54e5, 0.14, 0.16), (2.32e5, 0.22, 0.24)):
        assert lo < m.lambda_star(s, 5e-4, 256.0) < hi
    assert 1.2 < m.lambda_eq(5e-4, 1.54e5, 0.1, 256.0) < 1.5 and 1.2 < m.lambda_eq(5e-4, 2.32e5, 0.1, 256.0) < 1.6
    assert 0.7 < m.lambda_eq(5e-4, 1.54e5, 0.3, 256.0) < 0.95
    assert m.lambda_eq(5e-4, 2e5, 0.1, 256.0) == pytest.approx(np.sqrt(2e5 * 5e-4 / (2 * 0.1 * 256)))
    assert m.lambda_eq(0.0, 2e5, 0.1, 256.0) == 0.0                                       # eta -> 0 : equilibrium norm -> 0
    assert m.lambda_eq(5e-4, 2e5, 0.1, 256.0) / m.lambda_eq(1.25e-4, 2e5, 0.1, 256.0) == pytest.approx(2.0)   # ~ sqrt(eta)


def test_wd_window_is_the_relaxed_but_not_yet_frozen_phase():
    G, eta0, lam = 156224, 5e-4, 0.1
    k = np.round(np.arange(1, 21) / 20 * G)                                                 # integer gradient-step counts, as in the logs
    win, D, R = m.wd_window(k, G, eta0, lam)
    eta = eta0 * (1 - np.arange(G) / G)
    for kk, d, r in zip(k[[0, 5, 12, 19]], D[[0, 5, 12, 19]], R[[0, 5, 12, 19]]):
        kk = int(round(kk)); assert d == pytest.approx(2 * lam * eta[:kk].sum(), rel=1e-6); assert r == pytest.approx(2 * lam * eta[kk:].sum(), rel=1e-4, abs=1e-3)
    assert not win[0] and not win[-1] and win[4:9].all()                                    # transient at the start, frozen at the end


def test_spearman():
    assert m.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert m.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(m.spearman([1, 1, 1], [1, 2, 3]))                                       # constant input: undefined
    assert m.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)                      # ties get average ranks


# ----------------------------------------------------------------------------------------------------------------------
# end-to-end on synthetic stages
# ----------------------------------------------------------------------------------------------------------------------
def stage_supported(tmp_path_factory, **over):
    """E = 2, 4, 8 unconstrained + NP (E=2, 8) + weight decay 0.1 / 0.3 (E=2, 8) with a common growth law."""
    d = tmp_path_factory.mktemp("stage")
    runs = {}
    for E, s in ((2, 1.6e5), (4, 2.0e5), (8, 2.3e5)):
        runs[f"E{E}_none"] = dict(BASE, E=E, s=s)
    for E, s in ((2, 1.6e5), (8, 2.3e5)):
        runs[f"E{E}_np"] = dict(BASE, E=E, s=s, ctrl="np")
        for lam in (0.1, 0.3):
            runs[f"E{E}_wd{lam}"] = dict(BASE, E=E, s=s, ctrl="wd", wd=lam)
    for k, v in over.items():
        runs[k] = dict(runs.get(k, {}), **v)
    return synth.make_stage(str(d), runs)


def run_mech(stage, out):
    m.main([stage, "--out", str(out)])
    return json.load(open(os.path.join(str(out), "verdicts.json")))                     # test the written artifact, not the in-memory dict


def test_synthetic_stage_that_follows_the_model_is_supported(tmp_path_factory, tmp_path):
    v = run_mech(stage_supported(tmp_path_factory), tmp_path / "o")
    p1, p2, p3 = v["prediction_1_norm_law"], v["prediction_2_angle"], v["prediction_3_wd_equilibrium"]
    assert p1["p1a"]["verdict"] == "支持" and p1["p1a"]["pass_fraction"] == 1.0
    assert p1["p1a"]["slope_s_median"] == pytest.approx(2.0e5, rel=0.25)
    assert p1["p1b"]["verdict"] == "支持" and p1["p1b"]["ratios_lambda2m1_vs_E2"]["8"] > 4.0
    assert p2["verdict"] == "支持", p2
    assert p2["late_ratio_mean"] < 0.4 and p2["spearman"] >= 0.9 and 0.5 <= p2["median_ratio_times_lambda"] <= 2
    assert p3["verdict"] == "支持", p3
    assert all(a["verdict"] == "支持" for a in p3["per_arm"].values()) and len(p3["per_arm"]) == 4
    assert all(c["ok"] for c in v["w0_sanity"].values())
    for f in ("verdicts.json", "slopes.json", "slopes.md", "angle_ratio.json", "wd_equilibrium.json", "fig_lambda.pdf", "fig_lambda.png",
              "fig_angle_ratio.pdf", "fig_wd_equilibrium.pdf", "fig_wd_equilibrium.png"):
        assert (tmp_path / "o" / f).exists(), f
    json.load(open(tmp_path / "o" / "verdicts.json"))                                      # valid JSON with the criteria included


def test_negative_control_weight_decay_equilibrium_off_by_more_than_2x(tmp_path_factory, tmp_path):
    """Growth in the wd arm is 6x the partner's s -> measured Lambda_eq is sqrt(6) = 2.4x the prediction (> 2) -> must be 'not supported'."""
    stage = stage_supported(tmp_path_factory, **{"E8_wd0.1": dict(s_wd_factor=6.0)})
    v = run_mech(stage, tmp_path / "o")["prediction_3_wd_equilibrium"]
    assert v["per_arm"]["E8_wd0.1"]["verdict"] == "不支持" and v["verdict"] == "不支持"
    assert v["per_arm"]["E8_wd0.3"]["verdict"] == "支持"


def test_negative_control_angle_that_does_not_track_the_norm(tmp_path_factory, tmp_path):
    stage = stage_supported(tmp_path_factory, **{"E8_none": dict(ang_tracks_norm=False)})
    v = run_mech(stage, tmp_path / "o")["prediction_2_angle"]
    assert v["verdict"] == "不支持" and not v["two_a_late_ratio_below_0p4"] and v["late_ratio_mean"] > 0.8


def test_negative_control_norm_growth_that_is_not_proportional_to_sum_eta2(tmp_path_factory, tmp_path):
    over = {f"E{E}_none": dict(growth_decay=4.0) for E in (2, 4, 8)}
    v = run_mech(stage_supported(tmp_path_factory, **over), tmp_path / "o")["prediction_1_norm_law"]
    assert v["p1a"]["verdict"] in ("不支持", "无法判断") and v["p1a"]["pass_fraction"] < 0.8


def test_negative_control_superlinear_growth_in_E_fails_when_growth_is_only_linear(tmp_path_factory, tmp_path):
    """If s did not grow with E, Lambda^2-1 would grow ~ E/2 per env step (U ~ E, same eta): ratio == E/2, not > E/2."""
    d = tmp_path_factory.mktemp("lin")
    runs = {f"E{E}_none": dict(BASE, E=E, s=2.0e5) for E in (2, 4, 8)}
    # same s for all E gives growth ~ sum eta^2 ~ G ~ E (linear in E) -> ratio ~= E/2 (up to eval-point effects): must NOT be 'supported'
    v = run_mech(synth.make_stage(str(d), runs), tmp_path / "o")["prediction_1_norm_law"]["p1b"]
    assert v["ratios_lambda2m1_vs_E2"]["8"] == pytest.approx(4.0, rel=0.1) and v["verdict"] == "不支持"


def test_missing_arms_give_cannot_judge_not_a_crash(tmp_path_factory, tmp_path):
    d = tmp_path_factory.mktemp("only_none")
    synth.make_stage(str(d), {"E2_none": dict(BASE, E=2)})
    v = run_mech(str(d), tmp_path / "o")
    assert v["prediction_2_angle"]["verdict"] == "无法判断" and v["prediction_3_wd_equilibrium"]["verdict"] == "无法判断"
    assert v["prediction_1_norm_law"]["p1b"]["verdict"] == "无法判断"


def test_run_without_ang_cum_is_skipped_gracefully(tmp_path_factory, tmp_path):
    d = tmp_path_factory.mktemp("noang")
    synth.make_stage(str(d), {"E8_none": dict(BASE, E=8, s=2.3e5, ang=False), "E8_np": dict(BASE, E=8, ctrl="np", ang=False)})
    v = run_mech(str(d), tmp_path / "o")["prediction_2_angle"]
    assert v["verdict"] == "无法判断" and "ang_cum" in v["reason"]


# ----------------------------------------------------------------------------------------------------------------------
# the real CPU pilot
# ----------------------------------------------------------------------------------------------------------------------
def test_pilot_reproduces_the_slopes_printed_in_the_plan(tmp_path):
    stage = tmp_path / "pilot"; stage.mkdir()
    for f in glob.glob(os.path.join(ROOT, "pilot_results", "[A-D]_*.json")):
        shutil.copy(f, stage)
    v = run_mech(str(stage), tmp_path / "o")
    per = v["prediction_1_norm_law"]["p1a"]["per_config"]
    # plan table 5: s = 2.26e5 (A), 1.54e5 (B), 2.32e5 (C); Lambda^2-1 ratio E=8 / E=2 about 5
    assert per["A_orig_E2"]["slope_s_mean"] == pytest.approx(2.26e5, rel=0.01)
    assert per["B_fixed_E2"]["slope_s_mean"] == pytest.approx(1.54e5, rel=0.01)
    assert per["C_fixed_E8"]["slope_s_mean"] == pytest.approx(2.32e5, rel=0.01)
    assert v["prediction_1_norm_law"]["p1b"]["ratios_lambda2m1_vs_E2"]["8"] == pytest.approx(5.1, abs=0.2)
    assert v["prediction_2_angle"]["verdict"] == "无法判断" and "ang_cum" in v["prediction_2_angle"]["reason"]
    assert v["prediction_3_wd_equilibrium"]["verdict"] == "无法判断"
    assert v["w0_sanity"]["D_fixed_E8_NP"]["ok"] and v["w0_sanity"]["D_fixed_E8_NP"]["max_abs_dev"] < 0.01    # NP keeps ||W|| = ||W0|| ~ 16


# ----------------------------------------------------------------------------------------------------------------------
# real (tiny) training runs - slow
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 (about 3 minutes on 4 CPU cores)")
def test_smoke_training_runs_produce_ang_cum_and_all_figures(tmp_path):
    out = tmp_path / "smoke"
    subprocess.run([os.path.join(ROOT, "tools", "make_mech_smoke.sh"), str(out)], check=True, capture_output=True)
    d = json.load(open(glob.glob(str(out / "E8_none__*.json"))[0]))
    assert len(d["hist"]["ang_cum"]) == len(d["hist"]["env_steps"]) >= 8
    assert np.all(np.diff(np.array(d["hist"]["ang_cum"]), axis=0) > 0)                  # angular travel accumulates
    v = run_mech(str(out), tmp_path / "o")
    assert v["prediction_2_angle"]["verdict"] in ("无法判断", "支持", "不支持")
    for f in ("fig_lambda.pdf", "fig_angle_ratio.pdf", "fig_wd_equilibrium.pdf"):
        assert (tmp_path / "o" / f).exists()
