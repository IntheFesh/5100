"""tools/budget.py: timing-log parsing, E interpolation, stage totals and the pre-registered reduction ladder.
The timings below are HYPOTHETICAL numbers made up for arithmetic checks - not measurements."""
import os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import budget

LOG2 = """[   40.0s]  0.50M  public [1.0 2.0] (capped [0. 0.])  robust [1.0 2.0] (capped [0. 0.])  wn_dense [16. 16.]
[   70.0s]  1.00M  public [1.0 2.0] (capped [0. 0.])  robust [1.0 2.0] (capped [0. 0.])  wn_dense [16. 16.]
done 70.5 s"""
LOG8 = LOG2.replace("40.0", "100.0").replace("70.0", "220.0")


def test_parse_timing_separates_compile_from_steady_state():
    t = budget.parse_timing(LOG2)
    assert t["dt"] == pytest.approx(30.0) and t["overhead"] == pytest.approx(10.0) and t["n_points"] == 2
    with pytest.raises(ValueError):
        budget.parse_timing("[ 5.0s]  0.50M only one line")


def test_config_time_and_interpolation():
    t2, t8 = budget.parse_timing(LOG2), budget.parse_timing(LOG8)             # dt: 30 s and 120 s
    assert budget.config_seconds(2, t2, t8) == pytest.approx(10 + 20 * 30)
    assert budget.config_seconds(8, t2, t8) == pytest.approx(t8["overhead"] + 20 * 120)
    assert budget.config_seconds(4, t2, t8) == pytest.approx((2 / 3) * budget.config_seconds(2, t2, t8) + (1 / 3) * budget.config_seconds(8, t2, t8))


def test_stage_totals_and_ladder_arithmetic():
    t2, t8 = budget.parse_timing(LOG2), budget.parse_timing(LOG8)
    res = budget.build(t2, t8, limit=20.0, best_E=8)
    rows = budget.stage_rows(8)
    assert [sum(1 for r in rows if r[0] == s) for s in ("P0", "P1", "P2", "P3")] == [5, 19, 4, 12]
    c = lambda E: budget.config_seconds(E, t2, t8) / 3600
    p1 = sum(c(E) for st, n, E, s in rows if st == "P1")
    assert res["stages"]["P1"] == pytest.approx(p1)
    p3 = res["stages"]["P3"]; fw = sum(c(E) for st, n, E, s in rows if st == "P3" and n.startswith("Freeway_"))
    steps = res["ladder"]
    assert steps[0]["saved"] == pytest.approx(fw) and steps[1]["saved"] == pytest.approx(p3 - fw)           # Freeway first, then the rest of P3
    e4 = sum(c(4) for st, n, E, s in rows if st == "P1" and n in ("E4_none", "E4_np", "E4_radial", "E4_wd0.1", "E4_wd0.3"))
    assert steps[2]["saved"] == pytest.approx(e4)
    wd03 = sum(c(E) for st, n, E, s in rows if st == "P1" and n.endswith("_wd0.3") and not n.startswith("E4_"))
    assert steps[3]["saved"] == pytest.approx(wd03)
    assert res["total"] == pytest.approx(sum(res["stages"].values()))
    assert steps[-1]["total_after"] == pytest.approx(res["total"] - sum(s["saved"] for s in steps))
    assert [s["n_configs"] for s in steps] == [5 + 19 + 4 + 8, 5 + 19 + 4, 5 + 14 + 4, 5 + 12 + 4]
    md = budget.to_md(res, t2, t8)
    assert "| P3 | 12 |" in md and "GPU hours" in md and ("reduction ladder" in md or "no reduction needed" in md)


def test_seed_exponent_scales_small_stages_only():
    t2, t8 = budget.parse_timing(LOG2), budget.parse_timing(LOG8)
    a = budget.build(t2, t8, seed_exp=0.0); b = budget.build(t2, t8, seed_exp=0.5)
    assert b["stages"]["P0"] == pytest.approx(a["stages"]["P0"])               # 10-seed stages unchanged
    assert b["stages"]["P1"] == pytest.approx(a["stages"]["P1"] * (3 / 10) ** 0.5)
