"""Demo pipeline (round 9): pre-registered selection rules, notebook/script agreement, captions, duration budget, cards, ffmpeg assembly.
RUN_SLOW=1 additionally trains four tiny real P2-like models (E=2, 1M/3M snapshots) and runs tools/make_demo_video.py end to end."""
import glob, json, os, re, subprocess, sys
import numpy as np, pytest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, ROOT)
import demo_select as ds
import make_demo_video as mdv


def write_json(path, auc_per_seed, ids=None, n_points=4):
    auc = np.asarray(auc_per_seed, float)
    ev = [{"env_steps": 100 * (i + 1), "robust_return": (auc + (0.0 if i % 2 == 0 else 0.0)).tolist()} for i in range(n_points)]   # constant per-seed curve: AUC == auc
    json.dump({"evals": ev, "seed_ids": ids or list(range(1000, 1000 + len(auc))), "ckpts": [], "hist": {"env_steps": [100 * (n_points)]}}, open(path, "w"))


# ------------------------------------------------------------------------------------------------ selection rule
def test_median_seed_index_odd_even_ties():
    assert ds.median_seed_index([5, 1, 9]) == 0                                          # odd: the middle value (5)
    a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]                                       # even: 5th/6th = 50/60, mean = 145 -> closer to the mean is 60
    assert ds.median_seed_index(a) == 5
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                                                  # mean 5.5: 5 and 6 equally close -> the lower one (5)
    assert ds.median_seed_index(b) == 4
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.normal(size=10); i = ds.median_seed_index(x); o = np.sort(x)
        assert x[i] in (o[4], o[5])                                                      # always one of the two middle seeds


def test_notebook_median_seed_is_the_same_rule(tmp_path):
    nb = json.load(open(os.path.join(ROOT, "DRL_v5_colab.ipynb")))
    cell = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "def median_seed" in "".join(c["source"]))
    src = cell[cell.index("def median_seed"):]; src = src[:src.index("\ndef final_steps") if "\ndef final_steps" in src else src.index("\nSP, SF, SO")]
    ns = {"json": json, "np": np}; exec(src, ns)
    rng = np.random.default_rng(1)
    for k in range(40):
        n = [3, 5, 9, 10, 10, 10][k % 6]; auc = rng.normal(80, 15, size=n)
        p = tmp_path / f"cfg{k}__abc_params.pkl"; write_json(str(p).replace("_params.pkl", ".json"), auc)
        assert ns["median_seed"](str(p)) == ds.median_seed_index(auc)
    info = ds.median_seed_info(str(tmp_path / "cfg0__abc.json"))
    assert info["index"] == ds.median_seed_index(info["auc_per_seed"]) and info["seed_id"] == 1000 + info["index"] and info["n_seeds"] == 3


def test_find_one_only_matches_final_params_and_asserts_uniqueness(tmp_path):
    for f in ("ours_E8_np_lr2.5e-4__0123abcdef_params.pkl", "ours_E8_np_lr2.5e-4__0123abcdef_ckpt1M_params.pkl", "ours_E8_np_lr2.5e-4__0123abcdef_ckpt3M_params.pkl",
              "fixed_E2__99aa_params.pkl", "fixed_E2__99aa_ckpt1M_params.pkl", "fixed_E8__77bb_params.pkl", "public__11cc_params.pkl"):
        (tmp_path / f).write_bytes(b"x")
    d = str(tmp_path)
    assert ds.final_params(d, "ours_E8_np_lr2.5e-4").endswith("0123abcdef_params.pkl") and ds.final_params(d, "fixed_E2").endswith("fixed_E2__99aa_params.pkl")
    assert ds.snapshot_params(d, "ours_E8_np_lr2.5e-4", "1M").endswith("_ckpt1M_params.pkl") and ds.snapshot_params(d, "ours_E8_np_lr2.5e-4", "3M").endswith("_ckpt3M_params.pkl")
    with pytest.raises(AssertionError):
        ds.find_one(d, "fixed_E*__*_params.pkl")                                         # ambiguous -> refuse instead of picking one


def test_snapshot_steps_come_from_the_json_not_from_the_label(tmp_path):
    j = tmp_path / "ours__ab12.json"
    json.dump({"ckpts": [{"label": "1M", "update": 245, "env_steps": 1003520}, {"label": "3M", "update": 733, "env_steps": 3002368}], "hist": {"env_steps": [9998336]}}, open(j, "w"))
    p1, p3, pf = (str(tmp_path / n) for n in ("ours__ab12_ckpt1M_params.pkl", "ours__ab12_ckpt3M_params.pkl", "ours__ab12_params.pkl"))
    assert ds.snapshot_steps(p1) == 1003520 and ds.snapshot_steps(p3) == 3002368 and ds.final_steps(pf) == 9998336
    assert ds.json_of(p1) == ds.json_of(pf) == str(j)


# ------------------------------------------------------------------------------------------------ captions, budget, cards
def test_caption_rule_for_group_b():
    pair = lambda p: {"pairs": [{"config": "ours_x", "ref": "fixed_E2", "p_holm": p}]}
    assert mdv.footer_for(pair(0.30), "ours_x") == "difference not statistically significant (10 seeds)"
    assert mdv.footer_for(pair(0.05), "ours_x") is not None and mdv.footer_for(pair(0.049), "ours_x") is None
    assert mdv.footer_for(None, "ours_x") is not None                                    # no analysis -> be conservative


def test_comparison_max_steps_keeps_the_total_within_the_limit():
    assert mdv.comparison_max_steps(210, 100, 2, 2400) == int(np.floor((210 - 12 - 100) / 2 * 24) - 1) == 1175
    assert mdv.comparison_max_steps(210, 20, 2, 2400) == 2135                            # (210-12-20)/2 s = 89 s -> 2135 steps
    assert mdv.comparison_max_steps(400, 20, 2, 2400) == 2400                            # plenty of room (larger limit): default kept
    assert mdv.comparison_max_steps(210, 195, 2, 2400) == 120                            # floor
    for prog in (10, 60, 100):
        ms = mdv.comparison_max_steps(210, prog, 2, 2400)
        assert 12 + prog + 2 * (ms + 1) / 24 <= 210 + 1e-9 or ms == 120


def test_conclusion_text_uses_the_numbers_and_the_rule_wording():
    nums = {"cmpOursVsFixedEtwoRel": {"text": "+43.1\\%"}, "cmpOursVsFixedEtwoCiLo": {"text": "+27.6"}, "cmpOursVsFixedEtwoCiHi": {"text": "+32.6"},
            "cmpOursVsFixedEtwoPHolm": {"text": "0.006"}, "cmpOursVsFixedEtwoLarge": {"text": "yes"}}
    head, body = mdv.conclusion({"mode": "with_control"}, nums, None, "ours_x")
    assert "+43.1%" in body and "0.006" in body and "Not a substantial" not in body and "norm control" in head
    nums["cmpOursVsFixedEtwoLarge"]["text"] = "no"
    assert "Not a substantial improvement" in mdv.conclusion({"mode": "with_control"}, nums, None, "ours_x")[1]
    nums2 = {"cmpFixedVsPublicRel": {"text": "+11.0\\%"}, "cmpFixedVsPublicCiLo": {"text": "+5"}, "cmpFixedVsPublicCiHi": {"text": "+9"}, "cmpFixedVsPublicPHolm": {"text": "0.008"}}
    assert "No gain from more updates" in mdv.conclusion({"mode": "confirm_fixed_only"}, nums2, None, None)[1]


def test_middle_card_states_the_mechanism_only_when_prediction_1_is_supported():
    assert "shrinks" in mdv.card2_text({"predOneVerdict": {"text": "supported"}})[0]
    for nums in (None, {}, {"predOneVerdict": {"text": "not supported"}}, {"predOneVerdict": {"text": "cannot judge"}}):
        title, body = mdv.card2_text(nums)
        assert "shrinks" not in title and "not confirmed" in body


def test_cards_are_1920x1080_and_text_fits(tmp_path):
    from PIL import Image
    p = mdv.card(str(tmp_path / "c.png"), "PQN trains on each batch only twice. Why doesn't more learning per batch help?", "Replay-free Q-learning " * 8, image=None, foot="x")
    assert Image.open(p).size == (1920, 1080)
    long_foot = "Same env steps, same environment seeds; median training seed per config and median of 9 full episodes (rule fixed in advance). " * 2
    pf = mdv.card(str(tmp_path / "c3.png"), "T", "B", foot=long_foot)
    px = np.asarray(Image.open(pf).convert("L"))
    assert px[:, 1800:].max() < 40                                                       # background ~12 grey, text >= 130: nothing drawn in the right margin
    assert px[900:, :1800].max() > 100                                                   # ... but the (wrapped) footer is there
    fig = tmp_path / "f.png"; Image.new("RGB", (1360, 480), (200, 200, 200)).save(fig)
    assert Image.open(mdv.card(str(tmp_path / "c2.png"), "Title", "Body", image=str(fig))).size == (1920, 1080)


def test_assembly_of_cards_and_videos_stays_within_the_limit(tmp_path):
    """render_demo (random-init network) -> 1920x1080 -> concat with cards; the same helpers make_demo_video uses."""
    import imageio.v2 as iio
    v = str(tmp_path / "v.mp4")
    subprocess.run([sys.executable, "-W", "ignore", os.path.join(ROOT, "render_demo.py"), v, "--random_init", "--panel", "x", "A", "--panel", "x", "B", "--episodes", "3",
                    "--max_steps", "60", "--game_cap", "300"], check=True, capture_output=True)
    c = mdv.card(str(tmp_path / "c.png"), "T", "B"); mdv.card_video(c, str(tmp_path / "c.mp4"), seconds=1); mdv.to_hd(v, str(tmp_path / "v_hd.mp4"))
    out = str(tmp_path / "final.mp4"); mdv.concat([str(tmp_path / "c.mp4"), str(tmp_path / "v_hd.mp4"), str(tmp_path / "c.mp4")], out, str(tmp_path / "list.txt"))
    r = iio.get_reader(out); meta = r.get_meta_data(); frame = r.get_data(0); r.close()
    n_v = len(iio.mimread(v)); assert frame.shape == (1080, 1920, 3)
    assert abs(meta["duration"] - (2 * 1.0 + n_v / 24)) < 0.3 and meta["duration"] <= mdv.LIMIT_S


# ------------------------------------------------------------------------------------------------ end to end
def test_full_pipeline_on_tiny_real_models(tmp_path):
    """Three tiny REAL trainings (8 192 steps, 3 seeds) stand in for the P2 models; the 1M/3M snapshot files (unreachable in 8 192 steps) are copies of
    the final parameters with matching `ckpts` entries - this exercises file names, labels, seed selection, rendering, captions, cards, ffmpeg and DEMO.md."""
    import shutil
    d = tmp_path / "P2"; d.mkdir()
    base = ["--steps", "8192", "--num_envs", "32", "--minibatches", "8", "--seeds", "3", "--seed0", "1000", "--eval_every", "4096", "--eval_episodes", "4", "--eval_cap", "200",
            "--sticky", "0.1", "--max_steps_in_episode", "10000", "--save_params"]
    arms = {"public": ["--input_scale255", "--buggy_target"], "fixed_E2": ["--bootstrap_timeout"], "ours_E8_np_lr2.5e-4": ["--bootstrap_timeout", "--epochs", "2", "--norm_ctrl", "np"]}
    procs = [subprocess.Popen([sys.executable, "-W", "ignore", os.path.join(ROOT, "pqn_v5.py"), *base, *x, "--out", str(d / f"{n}__0123456789.json")],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for n, x in arms.items()]
    for p in procs:
        assert p.wait() == 0, p.stderr.read()[-1500:]
    for n in arms:
        base_ = str(d / f"{n}__0123456789"); j = json.load(open(base_ + ".json"))
        for lab in ("1M", "3M"):
            shutil.copy(base_ + "_params.pkl", f"{base_}_ckpt{lab}_params.pkl")
        j["ckpts"] = [{"label": "1M", "update": 245, "env_steps": 245 * 4096}, {"label": "3M", "update": 733, "env_steps": 733 * 4096}]
        json.dump(j, open(base_ + ".json", "w"))
    fz = tmp_path / "frozen.json"; json.dump({"best_E": 8, "best_ctrl": "--norm_ctrl np --lr 2.5e-4", "mode": "with_control", "p2_arms": ["public", "fixed_E2", "fixed_E8", "ours_E8_np_lr2.5e-4"]}, open(fz, "w"))
    ana = tmp_path / "ana.json"; json.dump({"pairs": [{"config": "ours_E8_np_lr2.5e-4", "ref": "fixed_E2", "p_holm": 0.4}]}, open(ana, "w"))
    out = mdv.main(["--p2", str(d), "--frozen", str(fz), "--analysis", str(ana), "--numbers", str(tmp_path / "none.json"), "--fig", str(tmp_path / "nofig.png"),
                    "--out", str(tmp_path / "demo"), "--figs", str(tmp_path / "figs"), "--demo_md", str(tmp_path / "DEMO.md"), "--max_steps", "200", "--episodes", "5", "--limit_s", "60"])
    assert out["total"] <= 60.5 and out["footer"] == "difference not statistically significant (10 seeds)" and out["final_arm"] == "ours_E8_np_lr2.5e-4"
    md = open(tmp_path / "DEMO.md").read()
    for key in ("1.00M steps", "3.00M steps", "Learning progression", "Public PQN vs ours", "Fixed PQN (E=2) vs ours", "chosen seed_index", "full-episode scores"):
        assert key in md
    for f in ("demo_progression.png", "demo_public_vs_ours.png", "demo_fixed_vs_ours.png"):
        assert os.path.exists(tmp_path / "figs" / f)
    rep = json.load(open(tmp_path / "demo" / "A.json"))
    assert [p["label"] for p in rep["panels"]] == ["Public PQN", "Ours"] and rep["panels"][0]["scale255"] is True and rep["panels"][1]["scale255"] is False
    assert all(len(p["episode_scores_sorted"]) == 5 for p in rep["panels"])
