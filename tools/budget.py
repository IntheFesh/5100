"""GPU-hour budget for P0-P3 from the two Colab timing commands of notebook section 2 (round 5).

Input: the printed output of
    pqn_v5.py --steps 1e6 --seeds 10 --epochs 2 --eval_every 5e5 ...     (--e2 FILE)
    pqn_v5.py --steps 1e6 --seeds 10 --epochs 8 --eval_every 5e5 ...     (--e8 FILE)
i.e. lines like  `[ 123.4s]  0.50M  public [...] ...`  and `done 250.1 s`.  (Both runs include the robust evaluation.)

Cost model (stated so it can be criticised):
  * dt(E)  = seconds per 0.5M env steps in steady state = (t_last - t_first) / (M_last - M_first) * 0.5   [first interval contains jit compile]
  * c(E)   = compile/first-call overhead = t_first - dt(E)
  * one config at 10M steps takes  c(E) + 20 dt(E);  E = 4 is interpolated linearly in E between the measured E = 2 and E = 8 (the measured
    runs are 10-seed runs; the plan says the cost is dominated by #configs and E, so the 10-seed timing is applied to every config unless
    --seed_exp > 0, which scales by (seeds / 10)^seed_exp)
  * GPU hours = sum over configs; stage configs and their E / seeds come from `sweep.py <stage> --dry` (effective_args).
Reduction ladder (pre-registered; P0 and P2 are never touched), applied in this order, each step cumulative:
    1 drop Freeway from P3   2 drop the whole of P3   3 drop the five E=4 control configs of P1   4 keep only wd 0.1 in P1
Usage:  python tools/budget.py --e2 t2.txt --e8 t8.txt [--limit 20] [--best_E 8] [--seed_exp 0] [--md out.md]
"""
import argparse, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from effective_args import effective_rows

LINE = re.compile(r"\[\s*([0-9.]+)s\]\s+([0-9.]+)M")
BEST_CTRL = "--norm_ctrl np --lr 2.5e-4"
LADDER = [("drop Freeway from P3", lambda st, n: not (st == "P3" and n.startswith("Freeway_"))),
          ("drop the whole of P3", lambda st, n: st != "P3"),
          ("drop the five E=4 control configs of P1 (E4_none, E4_np, E4_radial, E4_wd0.1, E4_wd0.3)",
           lambda st, n: not (st == "P1" and n in ("E4_none", "E4_np", "E4_radial", "E4_wd0.1", "E4_wd0.3"))),
          ("keep only wd 0.1 in P1 (drop every *_wd0.3 config)", lambda st, n: not (st == "P1" and n.endswith("_wd0.3")))]


def parse_timing(text):
    pts = [(float(m), float(t)) for t, m in LINE.findall(text)]
    if len(pts) < 2:
        raise ValueError("need at least two evaluation lines '[  123.4s]  0.50M ...' to separate compile time from steady state")
    (m0, t0), (m1, t1) = pts[0], pts[-1]
    dt = (t1 - t0) / (m1 - m0) * 0.5
    return dict(dt=dt, overhead=max(0.0, t0 - dt * (m0 / 0.5)), n_points=len(pts))


def config_seconds(E, t2, t8, steps=1e7):
    w = (E - 2) / 6.0                                           # linear interpolation in E between the two measured points
    dt = (1 - w) * t2["dt"] + w * t8["dt"]; ov = (1 - w) * t2["overhead"] + w * t8["overhead"]
    return ov + (steps / 5e5) * dt


def stage_rows(best_E):
    rows = []
    for st, kw in (("P0", {}), ("P1", {}), ("P2", dict(best_E=best_E, best_ctrl=BEST_CTRL)), ("P3", dict(best_E=best_E, best_ctrl=BEST_CTRL))):
        for name, _, d in effective_rows(st, **kw):
            rows.append((st, name, d["epochs"], d["seeds"]))
    return rows


def total_hours(rows, t2, t8, seed_exp=0.0):
    per = {}
    for st, n, E, seeds in rows:
        h = config_seconds(E, t2, t8) * (seeds / 10.0) ** seed_exp / 3600.0
        per[st] = per.get(st, 0.0) + h
    return per


def build(t2, t8, limit=20.0, best_E=8, seed_exp=0.0):
    rows = stage_rows(best_E)
    base = total_hours(rows, t2, t8, seed_exp)
    out = dict(stages=base, total=sum(base.values()), ladder=[], limit=limit,
               counts={st: sum(1 for r in rows if r[0] == st) for st in ("P0", "P1", "P2", "P3")})
    cur, tot = rows, sum(base.values())
    for label, keep in LADDER:
        nxt = [r for r in cur if keep(r[0], r[1])]
        new_tot = sum(total_hours(nxt, t2, t8, seed_exp).values())
        out["ladder"].append(dict(step=label, saved=tot - new_tot, total_after=new_tot, n_configs=len(nxt), within_limit=new_tot <= limit))
        cur, tot = nxt, new_tot
    return out


def to_md(res, t2, t8):
    L = [f"Timing input: E=2: {t2['dt']:.1f} s / 0.5M steps (+{t2['overhead']:.0f} s compile), E=8: {t8['dt']:.1f} s / 0.5M steps (+{t8['overhead']:.0f} s compile) "
         f"(10 seeds, robust evaluation included)\n", "| stage | configs | GPU hours |", "|---|---|---|"]
    for st in ("P0", "P1", "P2", "P3"):
        L.append(f"| {st} | {res['counts'][st]} | {res['stages'][st]:.2f} |")
    L.append(f"| **total** | {sum(res['counts'].values())} | **{res['total']:.2f}** |\n")
    if res["total"] > res["limit"]:
        L += [f"Total exceeds {res['limit']:g} GPU hours; pre-registered reduction ladder (cumulative, P0 and P2 untouched):\n",
              "| step | saves (h) | total after (h) | configs left | <= limit |", "|---|---|---|---|---|"]
        for s in res["ladder"]:
            L.append(f"| {s['step']} | {s['saved']:.2f} | {s['total_after']:.2f} | {s['n_configs']} | {'yes' if s['within_limit'] else 'no'} |")
        first = next((s for s in res["ladder"] if s["within_limit"]), None)
        L.append("\n" + (f"Smallest sufficient prefix of the ladder ends with: *{first['step']}*." if first else "Even after all four steps the total exceeds the limit: P0 + P2 + the reduced P1 alone do not fit - needs your decision."))
    else:
        L.append(f"Total is within {res['limit']:g} GPU hours; no reduction needed.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", required=True); ap.add_argument("--e8", required=True)
    ap.add_argument("--limit", type=float, default=20.0); ap.add_argument("--best_E", type=int, default=8)
    ap.add_argument("--seed_exp", type=float, default=0.0); ap.add_argument("--md", default=None)
    a = ap.parse_args()
    t2, t8 = parse_timing(open(a.e2).read()), parse_timing(open(a.e8).read())
    res = build(t2, t8, a.limit, a.best_E, a.seed_exp)
    md = to_md(res, t2, t8)
    print(md)
    if a.md:
        open(a.md, "w").write(md)
