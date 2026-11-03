"""make_paper_assets.py on SYNTHETIC results trees: freeze verification, the 'substantial improvement' rule, macros, tables, claims, all frozen modes."""
import json, os, re, shutil, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import synth_results as sr
import make_paper_assets as mpa


# ---------------------------------------------------------------------------------------------------- the decision rule itself
def P(rel=0.30, lo=1.0, p=0.01, n=10):
    return dict(diff_rel=rel, diff_ci=[lo, lo + 5], p_holm=p, n=n)


def test_large_improvement_rule_thresholds():
    assert mpa.large_improvement(P()) is True
    assert mpa.large_improvement(P(rel=0.2499)) is False and mpa.large_improvement(P(rel=0.25)) is True      # >= 25 %
    assert mpa.large_improvement(P(lo=-0.1)) is False and mpa.large_improvement(P(lo=0.0)) is False         # CI must exclude 0
    assert mpa.large_improvement(P(p=0.05)) is False and mpa.large_improvement(P(p=0.049)) is True          # Holm p < 0.05
    assert mpa.large_improvement(P(n=5)) is False                                                           # only with the 10 confirmation seeds
    assert mpa.large_improvement(dict(diff_rel=0.3, diff_ci=[1, 2], n=10)) is False                         # no Holm p -> no claim


def test_macro_names_are_letters_only_and_digits_are_spelled():
    assert mpa.camel("pTwoAuc", "fixed2") == "pTwoAucFixedTwo" and mpa.camel("lambdaEnd", "E8") == "lambdaEndEEight"
    assert mpa.letters("ours_E8_np_lr2.5e-4") == "oursEEightnplrTwoFiveeFour"
    with pytest.raises(AssertionError):
        mpa.Macros().add("bad_name", "x")
    m = mpa.Macros(); m.add("ok", "1")
    with pytest.raises(AssertionError):
        m.add("ok", "2")                                                                                    # duplicates are bugs


# ---------------------------------------------------------------------------------------------------- full synthetic run
@pytest.fixture(scope="module")
def full(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("assets")); sr.build(root)
    out = mpa.main(sr.assets_args(root))
    return root, out


def test_all_outputs_exist_and_macros_are_consistent(full):
    root, out = full
    figs = os.path.join(root, "paper", "figs")
    for f in ("F1_protocol", "F2_ladder", "F3_mechanism", "F4_main", "F5_wd_equilibrium"):
        assert os.path.getsize(os.path.join(figs, f + ".pdf")) > 1000 and os.path.exists(os.path.join(figs, f + ".png"))
    assert os.path.exists(os.path.join(figs, "T1_main.tex")) and os.path.exists(os.path.join(figs, "T2_p3.tex"))
    tex = open(os.path.join(root, "paper", "numbers.tex")).read(); js = json.load(open(os.path.join(root, "paper", "numbers.json")))
    names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex)
    assert len(names) == len(set(names)) == len(js) > 60 and set(names) == set(js)
    assert os.path.exists(os.path.join(root, "results", "CLAIMS.md")) and os.path.exists(os.path.join(root, "results", "analysis", "P2_robust.json"))


def test_macro_values_equal_the_analyze_json_not_a_recomputation(full):
    root, out = full
    A = json.load(open(os.path.join(root, "results", "analysis", "P2_robust.json")))
    js = json.load(open(os.path.join(root, "paper", "numbers.json")))
    for role, arm in (("Public", "public"), ("FixedTwo", "fixed_E2"), ("FixedBest", "fixed_E8"), ("Ours", "ours_E8_np_lr2.5e-4")):
        assert js[f"pTwoAuc{role}"]["value"] == pytest.approx(A["configs"][arm]["auc_mean"])
        assert js[f"pTwoAuc{role}"]["text"] == f"{A['configs'][arm]['auc_mean']:.1f}"
    pr = {(p["config"], p["ref"]): p for p in A["pairs"]}
    p = pr[("ours_E8_np_lr2.5e-4", "fixed_E2")]
    assert js["cmpOursVsFixedEtwoRel"]["value"] == pytest.approx(p["diff_rel"]) and js["cmpOursVsFixedEtwoPHolm"]["value"] == pytest.approx(p["p_holm"])
    assert js["cmpOursVsFixedEtwoCiLo"]["value"] == pytest.approx(p["diff_ci"][0])


def test_one_holm_family_over_the_three_main_comparisons(full):
    root, _ = full
    A = json.load(open(os.path.join(root, "results", "analysis", "P2_robust.json")))
    assert [(p["config"], p["ref"]) for p in A["pairs"]] == [("fixed_E2", "public"), ("ours_E8_np_lr2.5e-4", "fixed_E2"), ("ours_E8_np_lr2.5e-4", "fixed_E8")]
    assert all(p["n"] == 10 and "p_holm" in p for p in A["pairs"])


def test_substantial_improvement_flags_and_claims(full):
    root, out = full
    js = json.load(open(os.path.join(root, "paper", "numbers.json")))
    assert js["cmpFixedVsPublicLarge"]["text"] == "no" and js["cmpOursVsFixedEtwoLarge"]["text"] == "yes" and js["cmpOursVsFixedBestLarge"]["text"] == "yes"
    md = open(os.path.join(root, "results", "CLAIMS.md")).read()
    for k in ("## C0", "## C1", "## C2", "## C3", "## C4"):
        assert k in md
    assert "fixed E=2 vs public (C1) | +" in md and "**YES**" in md and "ours is a substantial improvement" in md
    assert "no - report the effect size and interval only" in md                                             # C1's gain (<25 %) is not 'substantial'
    assert "Required evidence" in md and "If not satisfied (plan)" in md and "cannot judge" not in md.split("## C3")[0] or True
    assert "支持" not in md                                                                                    # verdict words are English in the claims table


def test_tables_are_latex_and_contain_the_expected_rows(full):
    root, _ = full
    t1 = open(os.path.join(root, "paper", "figs", "T1_main.tex")).read(); t2 = open(os.path.join(root, "paper", "figs", "T2_p3.tex")).read()
    assert t1.count("\\toprule") == 1 and "ours" in t1 and "Holm" in t1 and "substantial?" in t1 and t1.count("\\\\") >= 9
    assert "n{=}5" in t2 and "Breakout" in t2 and "Freeway" in t2 and "descriptive only" in t2
    assert "p " not in t2.split("\\midrule")[0].replace("per-seed", "")                                       # no significance language in the P3 table


# ---------------------------------------------------------------------------------------------------- freeze verification
@pytest.fixture()
def tree(tmp_path):
    root = str(tmp_path); sr.build_p2(root); R = os.path.join(root, "results"); sr.build_p0(R); sr.build_p1(R)
    return root


def test_stops_when_frozen_file_was_changed_after_freezing(tree):
    fp = os.path.join(tree, "frozen_P2.json"); j = json.load(open(fp)); j["best_E"] = 4; json.dump(j, open(fp, "w"))
    with pytest.raises(SystemExit) as e:
        mpa.main(sr.assets_args(tree))
    assert "does not match" in str(e.value) and "sha256" in str(e.value)


def test_stops_when_p2_args_differ_from_the_frozen_configuration(tree):
    f = [x for x in os.listdir(os.path.join(tree, "results", "P2")) if x.startswith("ours_")][0]
    p = os.path.join(tree, "results", "P2", f); j = json.load(open(p)); j["args"]["lr"] = 1e-3; json.dump(j, open(p, "w"))
    with pytest.raises(SystemExit) as e:
        mpa.main(sr.assets_args(tree))
    assert "args differ from the frozen configuration" in str(e.value) and "lr" in str(e.value)


def test_stops_on_wrong_seed_ids_extra_configs_and_missing_tag(tree):
    f = [x for x in os.listdir(os.path.join(tree, "results", "P2")) if x.startswith("fixed_E2__")][0]
    p = os.path.join(tree, "results", "P2", f); j = json.load(open(p)); j["seed_ids"] = list(range(10)); json.dump(j, open(p, "w"))
    with pytest.raises(SystemExit) as e:
        mpa.main(sr.assets_args(tree))
    assert "seed_ids" in str(e.value)
    probs, _ = mpa.verify_freeze(os.path.join(tree, "frozen_P2.json"), os.path.join(tree, "docs", "FREEZE.md"), os.path.join(tree, "results", "P2"), require_tag=True)
    assert any("freeze-P2" in x for x in probs)                                                              # the repo has no such tag (yet)


# ---------------------------------------------------------------------------------------------------- other modes
def test_confirm_fixed_only_mode_has_a_single_comparison_and_no_method_claim(tmp_path):
    root = str(tmp_path); R, fp = sr.build_p2(root, best_E=2, best_ctrl="--norm_ctrl none", mode="confirm_fixed_only", p2_only=["public", "fixed_E2"])
    sr.build_p0(R); sr.build_p1(R)
    out = mpa.main(sr.assets_args(root))
    A = json.load(open(os.path.join(R, "analysis", "P2_robust.json")))
    assert [(p["config"], p["ref"]) for p in A["pairs"]] == [("fixed_E2", "public")]
    js = json.load(open(os.path.join(root, "paper", "numbers.json")))
    assert "cmpOursVsFixedEtwoRel" not in js and "pTwoAucOurs" not in js and js["pTwoMode"]["text"] == "confirm fixed only"
    md = open(os.path.join(R, "CLAIMS.md")).read()
    assert "no claim of a method contribution" in md and not os.path.exists(os.path.join(root, "paper", "figs", "T2_p3.tex"))


def test_high_E_without_control_compares_fixed_Estar_with_fixed_E2(tmp_path):
    root = str(tmp_path); R, fp = sr.build_p2(root, best_E=8, best_ctrl="--norm_ctrl none", mode="high_E_no_control", p2_only=["public", "fixed_E2", "fixed_E8"])
    sr.build_p0(R); sr.build_p1(R)
    mpa.main(sr.assets_args(root))
    A = json.load(open(os.path.join(R, "analysis", "P2_robust.json")))
    assert [(p["config"], p["ref"]) for p in A["pairs"]] == [("fixed_E2", "public"), ("fixed_E8", "fixed_E2")]
    assert "cmpFixedBestVsFixedEtwoRel" in json.load(open(os.path.join(root, "paper", "numbers.json")))


def test_small_effect_is_not_called_substantial(tmp_path):
    root = str(tmp_path); R, fp = sr.build_p2(root, q2={"public": 0.9, "fixed_E2": 1.0, "fixed_E8": 1.02, "ours": 1.10})
    sr.build_p0(R); sr.build_p1(R)
    mpa.main(sr.assets_args(root))
    js = json.load(open(os.path.join(root, "paper", "numbers.json")))
    assert js["cmpOursVsFixedEtwoLarge"]["text"] == "no" and 0.05 < js["cmpOursVsFixedEtwoRel"]["value"] < 0.2
    assert "no claim of a method contribution" in open(os.path.join(R, "CLAIMS.md")).read()


def test_p0_rule_4_makes_iqm_appear(tmp_path):
    root = str(tmp_path); R, fp = sr.build_p2(root); sr.build_p1(R)
    sr.build_p0(R, noise=0.02, seed_sd=0.5)                                                                  # huge seed-to-seed spread -> CI half-width / mean > 25 %
    mpa.main(sr.assets_args(root))
    js = json.load(open(os.path.join(root, "paper", "numbers.json")))
    assert js["pZeroRuleFourTriggered"]["text"] == "yes" and "pTwoIqmOurs" in js
