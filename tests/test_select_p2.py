"""tools/select_p2.py on SYNTHETIC P1 trees: the pre-registered rule, stopping rules 2 and 3, the frozen-file guards. Fixtures only."""
import json, os, shutil, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "tools"))
import synth, select_p2

BASE_Q = {"E2_none": 1.0}


def make_p1(root, q, seed_sd=0.03):
    d = os.path.join(str(root), "P1")
    synth.stage_from_sweep(d, "P1", params=lambda n, a: dict(q=q.get(n, 0.9), s=2.0e5, noise=0.02, seed_sd=seed_sd, R=100.0, seed_ids=(100, 101, 102)))
    return d


def run(d, tmp, *extra):
    return select_p2.main([d, "--out", str(tmp / "sel"), "--results_root", str(tmp / "results"), "--freeze_dir", str(tmp / "freeze"), *extra])


def test_lr_str_and_ctrl_string():
    assert select_p2.lr_str(2.5e-4) == "2.5e-4" and select_p2.lr_str(1.25e-4) == "1.25e-4" and select_p2.lr_str(5e-4) == "5e-4"
    assert select_p2.ctrl_string(dict(norm_ctrl="np", wd=0.0, lr=2.5e-4)) == "--norm_ctrl np --lr 2.5e-4"
    assert select_p2.ctrl_string(dict(norm_ctrl="wd", wd=0.1, lr=5e-4)) == "--norm_ctrl wd --wd 0.1"
    assert select_p2.ctrl_string(dict(norm_ctrl="none", wd=0.0, lr=1.25e-4)) == "--norm_ctrl none --lr 1.25e-4"


def test_clear_winner_with_control(tmp_path):
    d = make_p1(tmp_path, {"E8_np_lr2.5e-4": 1.4, "E8_np": 1.1})
    ctx = run(d, tmp_path)
    D, P = ctx["decision"], ctx["plan"]
    assert D["best"] == "E8_np_lr2.5e-4" and D["confirm_fixed_only"] is False and D["margin"] > 5 * D["sd_used"]
    assert (P["mode"], P["best_E"], P["best_ctrl"], P["p2_only"]) == ("with_control", 8, "--norm_ctrl np --lr 2.5e-4", None)
    assert ctx["p2_arms"] == ["public", "fixed_E2", "fixed_E8", "ours_E8_np_lr2.5e-4"]
    assert D["rule2"]["triggered"] is False and D["rule3"]["triggered"] is False
    assert ctx["colab"][0] == "BEST_E = 8" and ctx["colab"][1] == "BEST_CTRL = '--norm_ctrl np --lr 2.5e-4'" and "--only" not in ctx["colab"][2]
    md = open(tmp_path / "sel" / "P1_selection.md").read()
    assert "pre-registered" in md.lower() and "Sensitivity" in md and "not read or generated" in md


def test_best_within_one_sd_of_fixed_pqn_only_confirms_fixed(tmp_path):
    d = make_p1(tmp_path, {"E8_np": 1.001}, seed_sd=0.08)
    ctx = run(d, tmp_path)
    D, P = ctx["decision"], ctx["plan"]
    assert D["margin"] < D["sd_used"] and D["confirm_fixed_only"] is True
    assert (P["mode"], P["best_E"], P["best_ctrl"]) == ("confirm_fixed_only", 2, "--norm_ctrl none") and ctx["p2_arms"] == ["public", "fixed_E2"]
    assert "--only public,fixed_E2" in ctx["colab"][2] and "Breakout_public,Breakout_fixed_E2" in ctx["colab"][3]


def test_all_high_E_controls_worse_triggers_rule_2_and_confirms_fixed(tmp_path):
    d = make_p1(tmp_path, {n: 0.8 for n in select_p2.E8_CONTROLS})
    ctx = run(d, tmp_path)
    D = ctx["decision"]
    # (the 'best' config may be a noisy E=2 variant with q = 0.9; what matters: no E=8 control beats E2_none, and the margin is below one SD)
    assert D["rule2"]["triggered"] is True and D["rule2"]["controls_above_E2_none"] == [] and D["confirm_fixed_only"] and ctx["plan"]["mode"] == "confirm_fixed_only"


def test_lr_only_arm_beats_every_control_triggers_rule_3(tmp_path):
    d = make_p1(tmp_path, {"E8_none_lr2.5e-4": 1.3, "E8_np_lr2.5e-4": 1.1, "E8_np": 1.05})
    ctx = run(d, tmp_path)
    r3 = ctx["decision"]["rule3"]
    assert r3["triggered"] is True and r3["lr_only_best"] == "E8_none_lr2.5e-4" and r3["best_control"] == "E8_np_lr2.5e-4"
    assert ctx["plan"]["best_ctrl"] == "--norm_ctrl none --lr 2.5e-4" and ctx["plan"]["mode"] == "with_control"       # the winner IS the lr-only arm


def test_plain_high_E_without_control_has_no_ours_arm(tmp_path):
    d = make_p1(tmp_path, {"E8_none": 1.4})
    ctx = run(d, tmp_path)
    P = ctx["plan"]
    assert (P["mode"], P["best_E"], P["best_ctrl"]) == ("high_E_no_control", 8, "--norm_ctrl none") and ctx["p2_arms"] == ["public", "fixed_E2", "fixed_E8"]


def test_best_E2_with_control_gives_three_arms_without_duplicate(tmp_path):
    d = make_p1(tmp_path, {"E2_np": 1.4})
    ctx = run(d, tmp_path)
    assert ctx["plan"]["best_E"] == 2 and ctx["p2_arms"] == ["public", "fixed_E2", "ours_E2_np"]


def test_integrity_problems_and_incomplete_stage(tmp_path):
    d = make_p1(tmp_path, {})
    f = [x for x in os.listdir(d) if x.startswith("E4_np__")][0]
    j = json.load(open(os.path.join(d, f))); j["args"]["sticky"] = 0.0; json.dump(j, open(os.path.join(d, f), "w"))
    ctx = run(d, tmp_path)
    assert any("E4_np" in p and "sticky" in p for p in ctx["problems"])
    os.remove(os.path.join(d, f))
    with pytest.raises(SystemExit):
        run(d, tmp_path)


def test_write_is_refused_outside_results_and_never_overwrites(tmp_path):
    d = make_p1(tmp_path, {"E8_np_lr2.5e-4": 1.4})
    with pytest.raises(SystemExit) as e:
        run(d, tmp_path, "--write")                                           # synthetic tree is not under results_root
    assert "not inside" in str(e.value) and not (tmp_path / "freeze" / "frozen_P2.json").exists()
    # a stage dir inside the results root is accepted
    res = tmp_path / "results"; res.mkdir(); shutil.move(d, str(res / "P1"))
    ctx = select_p2.main([str(res / "P1"), "--write", "--results_root", str(res), "--freeze_dir", str(tmp_path / "freeze")])
    fz = json.load(open(tmp_path / "freeze" / "frozen_P2.json"))
    assert fz["best_E"] == 8 and fz["best_ctrl"] == "--norm_ctrl np --lr 2.5e-4" and fz["rule"] == select_p2.RULE and len(fz["p1_json_sha256"]) == 19
    assert select_p2.sha256_file(str(tmp_path / "freeze" / "frozen_P2.json")) in open(tmp_path / "freeze" / "docs" / "FREEZE.md").read()
    with pytest.raises(SystemExit) as e2:
        select_p2.main([str(res / "P1"), "--write", "--results_root", str(res), "--freeze_dir", str(tmp_path / "freeze")])
    assert "already exists" in str(e2.value)
