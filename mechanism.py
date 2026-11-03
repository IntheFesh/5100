"""mechanism.py - pre-registered tests of the plan's predictions 1-3 (plan v5.1, sections 3.3-3.8) on a finished stage.

Usage:   python mechanism.py <stage_dir> --out results/mechanism_<stage>/ [--filter PREFIX] [--w0sq 256]
Input:   <stage_dir>/*.json written by sweep.py / pqn_v5.py (hist.wn_dense, hist.ang_cum, hist.update, args, grad_steps).
Output:  fig_lambda, fig_angle_ratio, fig_wd_equilibrium (.pdf for the paper, .png for a quick look),
         slopes.json / slopes.md (slope s of ||W||^2-||W0||^2 against sum eta^2, per config and seed, with R^2),
         angle_ratio.json, wd_equilibrium.json, verdicts.json (support / not supported / cannot judge + the numbers used).

Quantities (all from the logs; nothing is re-simulated)
  eta_t        = eta0 (1 - t/G)                      PQN's linear schedule over the G gradient steps (eta0 = args.lr)
  S(k)         = sum_{t<k} eta_t^2                    exact discrete sum (sum_eta2); closed form eta0^2 G [1-(1-phi)^3]/3
  Lambda       = ||W_t|| / ||W_0||                    dense layer incl. bias; ||W_0||^2 = 256 (He init, 1024 -> 128; --w0sq)
  s            = d(||W||^2 - ||W0||^2) / dS            fitted per (unconstrained run, seed) on the second half of training
  A_j          = angular travel of the dense layer per eval interval = difference of hist.ang_cum
  Lambda_eq    = sqrt(s eta_t / (2 lam ||W0||^2))     decoupled weight decay equilibrium (prop. 5 iii), s from the same-E,
                                                      same-lr unconstrained run of the same seed

Pre-registered decision rules (fixed here, BEFORE any stage data exist; they are written into verdicts.json -> "criteria").
The plan gives numbers only for prediction 2 (ratio < 0.4 at E=8) and prediction 3 (factor 2); every other threshold below is
the analyst's choice and is stated so it can be criticised, not tuned after the fact.
  P1a norm law       a (unconstrained run, seed) passes if on the second half of the eval points (the last P//2 of P points, as in the
                     pilot script norm_slope.py)  R^2 >= 0.95 of the OLS line y = s x + c  (y = ||W||^2-||W0||^2, x = S),
                     the through-origin fit y = s_o x has max |y - s_o x| / y <= 0.15  ("proportional"; the OLS intercept c is only
                     reported: extrapolating to S = 0 from the narrow late-training range of S amplifies noise, which is why
                     |c| is NOT used), and, when each half of that second half has >= 3 points, slope(later)/slope(earlier) in [0.5, 2]
                     ("stable"). The 0.15 was set after looking at the 2-seed, 2M-step CPU pilot (max residual 0.05), which is development
                     data; no P0/P1/P2 data were available.
                     supported if >= 80 % of the runs pass, not supported if < 50 %, else cannot judge.
  P1b superlinear    same other settings, unconstrained, default lr: R_E = mean_seeds(Lambda_E^2-1)/mean_seeds(Lambda_2^2-1) at the last
                     eval point; supported iff R_E > E/2 for every E > 2 available (linear growth in E would give exactly E/2).
  P2 angle           pair = (unconstrained, NP) with the same E and lr, same seeds. r_j = A_none,j / A_NP,j per eval interval.
                     Only judged if the unconstrained run's final Lambda >= 1.25 (otherwise 1/Lambda ~ 1 and nothing can be tested).
                     2a (E = 8 only): ratio of the angles accumulated over the second half of training, per seed, < 0.4
                        for >= 80 % of the seeds and for the seed mean;
                     2b tracking: Spearman(mean_seeds r_j, mean_seeds 1/Lambda_none,j) >= 0.9, mean r over the first quarter of
                        the intervals > mean r over the last quarter, and median(r_j * Lambda_none,j) in [0.5, 2].
                     supported iff 2a and 2b hold for the E = 8 pair; cannot judge if there is no such pair or no ang_cum.
  P3 equilibrium     per weight-decay arm (lam) and seed: window = eval points with accumulated damping D = 2 lam sum_{t<k} eta_t >= 2
                     and remaining damping R = 2 lam sum_{t>=k} eta_t >= 2 (relaxation has happened and has not yet frozen,
                     the plan expects the norm to freeze as eta -> 0). Not supported if any in-window ratio
                     Lambda_measured / Lambda_eq(eta_t) is outside [1/2, 2]; supported if all are inside (>= 3 points per seed);
                     else cannot judge. Overall: supported iff every judged arm is supported (none judged -> cannot judge).
"""
import argparse, glob, json, os, sys
from dataclasses import dataclass
import numpy as np

PALETTE = dict(public="#55554f", E2="#2a78d6", E4="#8d6bd1", E8="#eb6834", np="#1baf7a", radial="#eda100", wd0p1="#e87ba4", wd0p3="#8d3d6b")
INK, MUTED, GRID = "#1f1f1c", "#6b6b66", "#e6e5e0"
W_SINGLE, W_DOUBLE = 3.3, 6.8

CRITERIA = {
    "w0sq_default": 256.0,
    "p1a": {"r2_min": 0.95, "origin_residual_max": 0.15, "later_over_earlier_slope": [0.5, 2.0], "pass_fraction_support": 0.8,
            "pass_fraction_reject": 0.5, "fit_window": "the last P//2 of the P eval points (second half)"},
    "p1b": {"rule": "R_E = mean(Lambda_E^2-1)/mean(Lambda_2^2-1) at the last eval point > E/2 for all E>2 available"},
    "p2": {"min_lambda_end_testable": 1.25, "ratio_E8_second_half_max": 0.4, "seed_fraction": 0.8, "spearman_min": 0.9, "median_r_times_lambda": [0.5, 2.0],
           "plan_text": "E=8: late-training ratio A_none/A_NP < 0.4 (plan 3.8 prediction 2)"},
    "p3": {"ratio_band": [0.5, 2.0], "damping_accumulated_min": 2.0, "damping_remaining_min": 2.0, "min_points": 3,
           "plan_text": "measured vs predicted differs by more than a factor 2 => not supported (plan 3.8 prediction 3)"},
}


# ======================================================================================================================
# numerics (unit-tested in tests/test_mechanism.py)
# ======================================================================================================================
def lr_at(k, G, eta0):
    """Learning rate of gradient step k of G (PQN: linear decay to ~0)."""
    return eta0 * (1.0 - np.asarray(k, float) / G)


def sum_eta2(k, G, eta0):
    """Exact discrete sum_{t=0}^{k-1} eta_t^2 with eta_t = eta0 (1 - t/G)."""
    k = np.asarray(k, float)
    return eta0 ** 2 * (k - k * (k - 1) / G + (k - 1) * k * (2 * k - 1) / (6.0 * G ** 2))


def sum_eta2_closed(phi, G, eta0):
    """Plan's closed form eta0^2 G [1 - (1-phi)^3] / 3 (continuous limit; phi = k/G)."""
    return eta0 ** 2 * G * (1.0 - (1.0 - np.asarray(phi, float)) ** 3) / 3.0


def sum_eta(k, G, eta0):
    """Exact discrete sum_{t=0}^{k-1} eta_t."""
    k = np.asarray(k, float)
    return eta0 * (k - k * (k - 1) / (2.0 * G))


def fit_linear(x, y):
    """OLS y = s x + c. Returns (s, c, R^2). R^2 = 1 - SSres/SStot (nan when y is constant)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.stack([x, np.ones_like(x)], 1)
    (s, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res, ss_tot = float(np.sum((y - (s * x + c)) ** 2)), float(np.sum((y - y.mean()) ** 2))
    return float(s), float(c), (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def fit_origin(x, y):
    """Least-squares slope of y = s x through the origin."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else float("nan")


def lambda_eq(eta, s, lam, w0sq):
    """Equilibrium of decoupled weight decay (prop. 5 iii): Lambda_eq^2 = s eta / (2 lam ||W0||^2)."""
    return np.sqrt(np.asarray(s, float) * np.asarray(eta, float) / (2.0 * lam * w0sq))


def lambda_star(s, eta0, w0sq):
    """Corollary 2: lam* with Lambda_eq(eta0) = 1:  lam* = s eta0 / (2 ||W0||^2)."""
    return s * eta0 / (2.0 * w0sq)


def _ranks(x):
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, float); order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        idx = np.where(x == v)[0]
        if len(idx) > 1:
            r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def wd_window(k, G, eta0, lam, acc_min=2.0, rem_min=2.0):
    """Eval points where the decay has acted for >= acc_min e-folds (D = 2 lam sum_{t<k} eta_t) and still has >= rem_min e-folds
    left (R = 2 lam sum_{t>=k} eta_t): the norm can track the moving equilibrium there; outside, it is transient or frozen."""
    k = np.asarray(k, float)
    D = 2.0 * lam * sum_eta(k, G, eta0)
    R = 2.0 * lam * (sum_eta(G, G, eta0) - sum_eta(k, G, eta0))
    return (D >= acc_min) & (R >= rem_min), D, R


# ======================================================================================================================
# loading
# ======================================================================================================================
@dataclass
class Run:
    name: str
    path: str
    args: dict
    E: int
    ctrl: str
    wd: float
    lr: float
    G: int
    M: int
    seed_ids: list
    env_steps: np.ndarray       # (P,)
    k: np.ndarray               # (P,) gradient steps done at each eval point
    phi: np.ndarray             # (P,) fraction of training
    wn: np.ndarray              # (P, S) ||W_dense||
    ang: object                 # (P, S) cumulative angular travel or None
    num_updates: int

    def lam(self, w0sq):
        return np.sqrt(self.wn ** 2 / w0sq)

    @property
    def default_lr(self):
        return abs(self.lr - 5e-4) < 1e-12


def ctrl_of(args):
    if "norm_ctrl" in args:
        return args["norm_ctrl"]
    return "np" if args.get("norm_proj") else "none"           # pilot JSONs (older code) used --norm_proj


def load_runs(stage_dir, prefix=""):
    runs = {}
    for fn in sorted(glob.glob(os.path.join(stage_dir, "*.json"))):
        d = json.load(open(fn))
        if "hist" not in d or "args" not in d or "wn_dense" not in d["hist"]:
            continue
        name = os.path.basename(fn).split("__")[0].replace(".json", "")
        if not name.startswith(prefix):
            continue
        if name in runs:
            sys.exit(f"ERROR: two runs named '{name}'")
        a, h = d["args"], d["hist"]
        E, M = int(a["epochs"]), int(a["minibatches"])
        G = int(d["grad_steps"]); NU = int(d["num_updates"])
        upd = np.asarray(h["update"], float)
        k = upd * E * M
        ids = d.get("seed_ids") or list(range(int(a.get("seed0", 0)), int(a.get("seed0", 0)) + int(a["seeds"])))
        runs[name] = Run(name, fn, a, E, ctrl_of(a), float(a.get("wd", 0.0)), float(a["lr"]), G, M, ids,
                         np.asarray(h["env_steps"], float), k, k / G, np.asarray(h["wn_dense"], float),
                         np.asarray(h["ang_cum"], float) if h.get("ang_cum") else None, NU)
    return runs


def config_key(r):
    """Everything except the number of epochs (for the cross-E comparison)."""
    a = r.args
    keys = ("env", "steps", "num_envs", "num_steps", "minibatches", "lr", "input_scale255", "buggy_target", "bootstrap_timeout",
            "sticky", "max_steps_in_episode")
    return tuple(a.get(k) for k in keys) + (r.ctrl, r.wd)


# ======================================================================================================================
# analyses
# ======================================================================================================================
def seed_common(runs):
    ids = set(runs[0].seed_ids)
    for r in runs[1:]:
        ids &= set(r.seed_ids)
    return sorted(ids)


def col(r, ids):
    return [r.seed_ids.index(i) for i in ids]


def fit_run(r, w0sq):
    """Per-seed fit of ||W||^2 - w0sq against S on the second half of the eval points; returns a list of dicts (one per seed)."""
    S = sum_eta2(r.k, r.G, r.lr)
    P = len(S); idx = np.arange(P)
    second = idx >= P // 2                                           # same window as the pilot script norm_slope.py
    sec_idx = idx[second]; h = len(sec_idx) // 2
    earlier, later = np.isin(idx, sec_idx[:h]), np.isin(idx, sec_idx[h:])
    c1 = CRITERIA["p1a"]
    out = []
    for j, sid in enumerate(r.seed_ids):
        y = r.wn[:, j] ** 2 - w0sq
        row = dict(config=r.name, seed=int(sid), E=r.E, ctrl=r.ctrl, wd=r.wd, lr=r.lr, n_fit=int(second.sum()))
        if second.sum() >= 3:
            s_, c, r2 = fit_linear(S[second], y[second])
            so = fit_origin(S[second], y[second])
            res = float(np.max(np.abs(y[second] - so * S[second]) / np.maximum(np.abs(y[second]), 1e-12)))
            row.update(slope_s=s_, intercept=c, r2=r2, slope_origin_2nd=so, origin_residual_max=res)
        else:
            row.update(slope_s=float("nan"), intercept=float("nan"), r2=float("nan"), slope_origin_2nd=float("nan"), origin_residual_max=float("nan"))
        row["slope_origin_all"] = fit_origin(S, y)
        s_e = fit_linear(S[earlier], y[earlier])[0] if earlier.sum() >= 3 else float("nan")
        s_l = fit_linear(S[later], y[later])[0] if later.sum() >= 3 else float("nan")
        row["slope_earlier"], row["slope_later"] = s_e, s_l
        row["later_over_earlier"] = s_l / s_e if np.isfinite(s_e) and np.isfinite(s_l) and s_e != 0 else float("nan")
        row["lambda_end"] = float(np.sqrt(r.wn[-1, j] ** 2 / w0sq))
        ok = bool(np.isfinite(row["r2"]) and row["r2"] >= c1["r2_min"] and row["origin_residual_max"] <= c1["origin_residual_max"])
        if np.isfinite(row["later_over_earlier"]):
            ok = ok and c1["later_over_earlier_slope"][0] <= row["later_over_earlier"] <= c1["later_over_earlier_slope"][1]
        row["pass_p1a"] = bool(ok)
        out.append(row)
    return out


def verdict_p1(runs, w0sq):
    unc = {n: r for n, r in runs.items() if r.ctrl == "none"}
    rows = [row for r in unc.values() for row in fit_run(r, w0sq)]
    v = {"p1a": None, "p1b": None}
    c = CRITERIA["p1a"]
    if not rows or all(not np.isfinite(x["r2"]) for x in rows):
        v["p1a"] = dict(verdict="无法判断", reason="no unconstrained run with >= 3 eval points in the second half")
    else:
        frac = float(np.mean([x["pass_p1a"] for x in rows]))
        verdict = "支持" if frac >= c["pass_fraction_support"] else ("不支持" if frac < c["pass_fraction_reject"] else "无法判断")
        v["p1a"] = dict(verdict=verdict, pass_fraction=frac, n_runs=len(rows),
                        slope_s_median=float(np.nanmedian([x["slope_s"] for x in rows])),
                        r2_median=float(np.nanmedian([x["r2"] for x in rows])),
                        per_config={n: dict(slope_s_mean=float(np.nanmean([x["slope_s"] for x in rows if x["config"] == n])),
                                            r2_min=float(np.nanmin([x["r2"] for x in rows if x["config"] == n])),
                                            n_pass=int(sum(x["pass_p1a"] for x in rows if x["config"] == n)),
                                            n=int(sum(1 for x in rows if x["config"] == n))) for n in unc})
    # P1b: superlinear in E (same settings except E, unconstrained, default lr)
    groups = {}
    for r in unc.values():
        groups.setdefault(config_key(r), {}).setdefault(r.E, r)
    best = None
    for key, byE in groups.items():
        if 2 in byE and any(E > 2 for E in byE) and byE[2].default_lr:
            best = byE if best is None or len(byE) > len(best) else best
    if best is None:
        v["p1b"] = dict(verdict="无法判断", reason="no group of unconstrained default-lr runs that contains E=2 and a larger E")
    else:
        lam2 = lambda r: np.mean(r.wn[-1] ** 2 / w0sq - 1.0)
        base = lam2(best[2]); ratios = {E: float(lam2(best[E]) / base) for E in sorted(best) if E > 2}
        ok = all(ratios[E] > E / 2.0 for E in ratios)
        v["p1b"] = dict(verdict="支持" if ok else "不支持", ratios_lambda2m1_vs_E2=ratios, linear_in_E_reference={E: E / 2.0 for E in ratios},
                        lambda2m1_E2=float(base), configs={E: best[E].name for E in best})
    vs = [v["p1a"]["verdict"], v["p1b"]["verdict"]]
    v["verdict"] = "不支持" if "不支持" in vs else ("支持" if all(x == "支持" for x in vs) else "无法判断")
    return v, rows


def angle_pairs(runs):
    pairs = []
    for n, r in runs.items():
        if r.ctrl != "none" or r.ang is None:
            continue
        for m, q in runs.items():
            if q.ctrl == "np" and q.ang is not None and q.E == r.E and abs(q.lr - r.lr) < 1e-15 and config_key(q)[:-2] == config_key(r)[:-2]:
                pairs.append((r, q))
    return pairs


def analyze_angles(runs, w0sq):
    res = []
    for r, q in angle_pairs(runs):
        ids = seed_common([r, q]); jr, jq = col(r, ids), col(q, ids)
        if len(r.env_steps) != len(q.env_steps) or not np.allclose(r.env_steps, q.env_steps):
            continue
        dA = lambda A: np.diff(np.vstack([np.zeros((1, A.shape[1])), A]), axis=0)
        An, Aq = dA(r.ang[:, jr]), dA(q.ang[:, jq])
        ratio = An / np.where(Aq > 0, Aq, np.nan)                                       # (P, S) per eval interval
        lam = r.lam(w0sq)[:, jr]
        lam_mid = 0.5 * (lam + np.vstack([np.ones((1, lam.shape[1])), lam[:-1]]))          # Lambda at the interval midpoint
        mid = int(np.max(np.where(r.phi <= 0.5)[0])) if np.any(r.phi <= 0.5) else 0
        late = (r.ang[-1, jr] - r.ang[mid, jr]) / (q.ang[-1, jq] - q.ang[mid, jq])      # accumulated over the second half
        P = ratio.shape[0]; qn = max(P // 4, 1)
        mean_ratio, mean_inv = np.nanmean(ratio, 1), np.nanmean(1.0 / lam_mid, 1)
        c = CRITERIA["p2"]
        rho = spearman(mean_ratio, mean_inv)
        trend = float(np.nanmean(mean_ratio[:qn]) - np.nanmean(mean_ratio[-qn:]))
        med = float(np.nanmedian(ratio * lam_mid))
        two_a = bool(np.mean(late < c["ratio_E8_second_half_max"]) >= c["seed_fraction"] and late.mean() < c["ratio_E8_second_half_max"])
        two_b = bool(np.isfinite(rho) and rho >= c["spearman_min"] and trend > 0 and c["median_r_times_lambda"][0] <= med <= c["median_r_times_lambda"][1])
        res.append(dict(none=r.name, np=q.name, E=r.E, lr=r.lr, seeds=[int(i) for i in ids], env_steps=r.env_steps.tolist(),
                        ratio_mean=mean_ratio.tolist(), inv_lambda_mean=mean_inv.tolist(), ratio_per_seed=ratio.tolist(),
                        late_ratio_per_seed=late.tolist(), late_ratio_mean=float(late.mean()), spearman=rho, trend_first_minus_last_quarter=trend,
                        median_ratio_times_lambda=med, two_a=two_a, two_b=two_b))
    return res


def verdict_p2(angles, runs):
    if not any(r.ang is not None for r in runs.values()):
        return dict(verdict="无法判断", reason="no run has hist.ang_cum (logged only by pqn_v5.py from v5.1 on; the CPU pilot A-D lacks it)")
    e8 = [a for a in angles if a["E"] == 8 and abs(a["lr"] - 5e-4) < 1e-12]
    if not e8:
        return dict(verdict="无法判断", reason="no (unconstrained, NP) pair with E=8 and the default lr in this stage",
                    pairs_found=[(a["none"], a["np"], a["E"]) for a in angles],
                    tracking_only={f"{a['none']}|{a['np']}": dict(two_b=a["two_b"], spearman=a["spearman"], late_ratio_mean=a["late_ratio_mean"]) for a in angles})
    a = e8[0]
    lam_end = float(np.mean([np.sqrt(r.wn[-1] ** 2 / CRITERIA["w0sq_default"]) for r in runs.values() if r.name == a["none"]][0]))
    if lam_end < CRITERIA["p2"]["min_lambda_end_testable"]:
        return dict(verdict="无法判断", reason=f"the unconstrained run's final Lambda is {lam_end:.2f} < {CRITERIA['p2']['min_lambda_end_testable']}: "
                    "the norm barely grew, so 1/Lambda is ~1 and the predicted ratio cannot be told apart from 1 (smoke-level data?)", pair=[a["none"], a["np"]],
                    late_ratio_mean=a["late_ratio_mean"])
    ok = a["two_a"] and a["two_b"]
    return dict(verdict="支持" if ok else "不支持", pair=[a["none"], a["np"]], two_a_late_ratio_below_0p4=a["two_a"], two_b_tracking_1_over_lambda=a["two_b"],
                late_ratio_mean=a["late_ratio_mean"], late_ratio_per_seed=a["late_ratio_per_seed"], spearman=a["spearman"],
                median_ratio_times_lambda=a["median_ratio_times_lambda"], trend_first_minus_last_quarter=a["trend_first_minus_last_quarter"],
                other_pairs={f"{b['none']}|{b['np']}": dict(late_ratio_mean=b["late_ratio_mean"], spearman=b["spearman"], two_b=b["two_b"]) for b in angles if b is not a})


def analyze_wd(runs, w0sq):
    res = []
    for n, r in runs.items():
        if r.ctrl != "wd" or r.wd <= 0:
            continue
        partner = next((u for u in runs.values() if u.ctrl == "none" and u.E == r.E and abs(u.lr - r.lr) < 1e-15
                        and config_key(u)[:-2] == config_key(r)[:-2]), None)
        entry = dict(config=n, E=r.E, lam=r.wd, lr=r.lr, partner=partner.name if partner else None, env_steps=r.env_steps.tolist())
        if partner is None:
            entry.update(verdict="无法判断", reason="no unconstrained run with the same E and lr to take the slope s from")
            res.append(entry); continue
        ids = seed_common([r, partner]); jr = col(r, ids)
        s_by_seed = {x["seed"]: x["slope_s"] for x in fit_run(partner, w0sq)}
        s = np.array([s_by_seed[i] for i in ids])
        eta = lr_at(r.k, r.G, r.lr)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            Leq = lambda_eq(eta, np.where(s > 0, s, np.nan)[None, :], r.wd, w0sq)               # (P, S); nan when the slope is unusable
        Lm = r.lam(w0sq)[:, jr]
        win, D, R = wd_window(r.k, r.G, r.lr, r.wd, CRITERIA["p3"]["damping_accumulated_min"], CRITERIA["p3"]["damping_remaining_min"])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = Lm / Leq
        lo, hi = CRITERIA["p3"]["ratio_band"]
        npts = int(win.sum())
        entry.update(seeds=[int(i) for i in ids], slope_s=s.tolist(), lambda_star=lambda_star(s, r.lr, w0sq).tolist(),
                     lambda_eq_at_eta0=lambda_eq(r.lr, s, r.wd, w0sq).tolist(), n_window_points=npts,
                     window=win.tolist(), damping_accumulated=D.tolist(), damping_remaining=R.tolist(),
                     lambda_measured=Lm.tolist(), lambda_predicted=Leq.tolist(), ratio=ratio.tolist(), phi=r.phi.tolist())
        if not (np.all(np.isfinite(s)) and np.all(s > 0)):
            entry.update(verdict="无法判断", reason="fitted slope s of the unconstrained partner is not positive/finite")
        elif npts < CRITERIA["p3"]["min_points"]:
            entry.update(verdict="无法判断", reason=f"only {npts} eval points inside the equilibrium window (need >= {CRITERIA['p3']['min_points']})")
        else:
            rw = ratio[win]
            entry.update(ratio_in_window_min=float(rw.min()), ratio_in_window_max=float(rw.max()),
                         verdict="支持" if (rw.min() >= lo and rw.max() <= hi) else "不支持")
        res.append(entry)
    return res


def verdict_p3(wd):
    if not wd:
        return dict(verdict="无法判断", reason="no weight-decay arm (--norm_ctrl wd --wd lam) in this stage")
    judged = [w for w in wd if w["verdict"] in ("支持", "不支持")]
    per = {w["config"]: dict(verdict=w["verdict"], **{k: w[k] for k in ("ratio_in_window_min", "ratio_in_window_max", "n_window_points", "lam", "E") if k in w},
                             **({"reason": w["reason"]} if "reason" in w else {})) for w in wd}
    if not judged:
        return dict(verdict="无法判断", per_arm=per)
    verdict = "支持" if all(w["verdict"] == "支持" for w in judged) else "不支持"
    return dict(verdict=verdict, per_arm=per)


def w0_checks(runs, w0sq):
    out = {}
    for n, r in runs.items():
        if r.ctrl == "np":                                   # NP keeps ||W|| == ||W0|| exactly: the logged norm IS the per-seed initial norm
            dev = np.abs(r.wn[0] ** 2 / w0sq - 1.0)
            out[n] = dict(kind="np: logged norm^2 / 256 - 1", max_abs_dev=float(dev.max()), ok=bool(dev.max() <= 0.02))
        else:
            l0 = r.wn[0] / np.sqrt(w0sq)
            out[n] = dict(kind="first logged Lambda", min=float(l0.min()), max=float(l0.max()), ok=bool(0.5 <= l0.min() and l0.max() <= 4.0))
    return out


# ======================================================================================================================
# figures
# ======================================================================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 2,
                         "pdf.fonttype": 42, "figure.dpi": 120})
    return plt


def is_public(r):
    return bool(r.args.get("input_scale255")) and bool(r.args.get("buggy_target"))


def ctrl_color(r):
    if is_public(r):
        return PALETTE["public"]
    if r.ctrl == "none":
        return PALETTE.get(f"E{r.E}", PALETTE["E4"])
    if r.ctrl == "wd":
        return PALETTE["wd0p1"] if abs(r.wd - 0.1) < 1e-9 else PALETTE["wd0p3"]
    return PALETTE[r.ctrl]


def ctrl_label(r):
    s = "public PQN" if is_public(r) else {"none": "unconstrained", "np": "NP", "radial": "radial proj.", "wd": f"wd {r.wd:g}"}[r.ctrl]
    return s + ("" if r.default_lr else f", lr {r.lr:g}")


def save(fig, out, stem):
    fig.savefig(os.path.join(out, stem + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(out, stem + ".png"), dpi=200, bbox_inches="tight")


def fig_lambda(runs, w0sq, out):
    plt = _mpl()
    Es = sorted({r.E for r in runs.values()})
    fig, axs = plt.subplots(1, len(Es), figsize=(W_DOUBLE, 3.1), sharey=True, squeeze=False)
    for ax, E in zip(axs[0], Es):
        for r in sorted((r for r in runs.values() if r.E == E), key=lambda r: (not is_public(r), r.ctrl != "none", r.ctrl, r.wd, -r.lr)):
            x = np.concatenate([[0.0], r.env_steps / 1e6]); L = np.vstack([np.ones((1, r.wn.shape[1])), r.lam(w0sq)])   # Lambda_0 = 1
            c = ctrl_color(r)
            ls = "--" if (is_public(r) or (r.ctrl == "none" and not r.default_lr)) else "-"
            for j in range(L.shape[1]):
                ax.plot(x, L[:, j], color=c, lw=0.6, alpha=0.35, ls=ls)
            ax.plot(x, L.mean(1), color=c, lw=2, ls=ls, label=ctrl_label(r))
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"E = {E}", loc="left", fontsize=8.5, color=INK); ax.set_xlabel("env steps (M)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, handlelength=1.6, columnspacing=1.0)   # below the axes: never covers a curve
    axs[0][0].set_ylabel(r"$\Lambda_t=\|W_t\|/\|W_0\|$ (dense)")
    fig.tight_layout(); save(fig, out, "fig_lambda"); plt.close(fig)


def fig_angle(angles, out):
    plt = _mpl()
    fig, axs = plt.subplots(1, len(angles), figsize=(W_DOUBLE, 2.3), sharey=True, squeeze=False)
    for ax, a in zip(axs[0], angles):
        x = np.asarray(a["env_steps"]) / 1e6; c = PALETTE.get(f"E{a['E']}", PALETTE["E4"])
        for rs in np.asarray(a["ratio_per_seed"]).T:
            ax.plot(x, rs, color=c, lw=0.6, alpha=0.35)
        ax.plot(x, a["ratio_mean"], color=c, lw=2, label=r"$A_{none}/A_{NP}$ (per interval)")
        ax.plot(x, a["inv_lambda_mean"], color=INK, lw=1.6, ls="--", label=r"$1/\Lambda_{none}$")
        ax.axhline(0.4, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"E = {a['E']}" + ("" if abs(a["lr"] - 5e-4) < 1e-12 else f", lr {a['lr']:g}"), loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("env steps (M)"); ax.set_ylim(0, 1.15); ax.legend(loc="best", handlelength=1.6)
    axs[0][0].set_ylabel("angle ratio / $1/\\Lambda$")
    fig.tight_layout(); save(fig, out, "fig_angle_ratio"); plt.close(fig)


def fig_wd(wd, runs, out):
    plt = _mpl()
    arms = [w for w in wd if "lambda_measured" in w]
    fig, axs = plt.subplots(1, len(arms), figsize=(W_DOUBLE, 2.3), sharey=True, squeeze=False)
    for ax, w in zip(axs[0], arms):
        x = np.asarray(w["env_steps"]) / 1e6; Lm, Lp = np.asarray(w["lambda_measured"]), np.asarray(w["lambda_predicted"])
        c = PALETTE["wd0p1"] if abs(w["lam"] - 0.1) < 1e-9 else PALETTE["wd0p3"]
        win = np.asarray(w["window"])
        ax.fill_between(x, 0.5 * Lp.mean(1), 2.0 * Lp.mean(1), color=GRID, alpha=0.8, lw=0, label="predicted $\\times/\\div 2$")
        for j in range(Lm.shape[1]):
            ax.plot(x, Lm[:, j], color=c, lw=0.6, alpha=0.35)
        ax.plot(x, Lm.mean(1), color=c, lw=2, label="measured $\\Lambda_t$")
        ax.plot(x, Lp.mean(1), color=INK, lw=1.6, ls="--", label=r"$\Lambda_{eq}(\eta_t)$")
        if win.any():
            ax.plot(x[win], Lm.mean(1)[win], "o", ms=3, color=INK, label="verdict window")
        ax.set_title(f"E = {w['E']}, $\\lambda$ = {w['lam']:g}", loc="left", fontsize=8.5, color=INK); ax.set_xlabel("env steps (M)")
        ax.legend(loc="best", handlelength=1.6)
    axs[0][0].set_ylabel(r"$\Lambda_t$")
    fig.tight_layout(); save(fig, out, "fig_wd_equilibrium"); plt.close(fig)


# ======================================================================================================================
def to_md(rows):
    head = ("| config | seed | E | ctrl | lr | s (2nd half, OLS) | intercept | R² | s (origin, 2nd half) | max resid. (origin) | later/earlier slope | "
            "s (origin, all pts) | Λ_end | pass P1a |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    f = lambda v, p=3: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{p}g}" if isinstance(v, float) else str(v))
    return head + "\n".join(f"| {r['config']} | {r['seed']} | {r['E']} | {r['ctrl']}{(' ' + str(r['wd'])) if r['ctrl'] == 'wd' else ''} | {r['lr']:g} | "
                            f"{f(r['slope_s'])} | {f(r['intercept'])} | {f(r['r2'])} | {f(r['slope_origin_2nd'])} | {f(r['origin_residual_max'])} | "
                            f"{f(r['later_over_earlier'])} | {f(r['slope_origin_all'])} | {f(r['lambda_end'])} | {r['pass_p1a']} |" for r in rows) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--filter", default="", help="only configs whose name starts with this prefix (e.g. a P3 game)")
    ap.add_argument("--w0sq", type=float, default=CRITERIA["w0sq_default"], help="||W_0||^2 of the dense layer (He init: 256)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.stage_dir, a.filter)
    if not runs:
        sys.exit(f"no usable run JSON in {a.stage_dir}")
    v1, slope_rows = verdict_p1(runs, a.w0sq)
    angles = analyze_angles(runs, a.w0sq)
    v2 = verdict_p2(angles, runs)
    wd = analyze_wd(runs, a.w0sq)
    v3 = verdict_p3(wd)
    checks = w0_checks(runs, a.w0sq)
    lam_star = {n: dict(lambda_star_per_seed=[lambda_star(x["slope_s"], r.lr, a.w0sq) for x in slope_rows if x["config"] == n]) for n, r in runs.items() if r.ctrl == "none"}
    json.dump(slope_rows, open(os.path.join(a.out, "slopes.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "slopes.md"), "w").write(to_md(slope_rows))
    json.dump(angles, open(os.path.join(a.out, "angle_ratio.json"), "w"), indent=1, default=float)
    json.dump(wd, open(os.path.join(a.out, "wd_equilibrium.json"), "w"), indent=1, default=float)
    fig_lambda(runs, a.w0sq, a.out)
    if angles:
        fig_angle(angles, a.out)
    if any("lambda_measured" in w for w in wd):
        fig_wd(wd, runs, a.out)
    verdicts = {"stage_dir": os.path.abspath(a.stage_dir), "filter": a.filter, "w0sq": a.w0sq, "configs": sorted(runs),
                "versions": sorted({json.dumps(json.load(open(r.path)).get("versions"), sort_keys=True) for r in runs.values()}),
                "prediction_1_norm_law": v1, "prediction_2_angle": v2, "prediction_3_wd_equilibrium": v3,
                "corollary_2_lambda_star": lam_star, "w0_sanity": checks, "criteria": CRITERIA}
    json.dump(verdicts, open(os.path.join(a.out, "verdicts.json"), "w"), indent=1, default=float, ensure_ascii=False)
    print(f"{a.stage_dir}: {len(runs)} runs")
    for k, v in (("P1 norm law", v1), ("P2 angle", v2), ("P3 wd equilibrium", v3)):
        print(f"  {k:20s} -> {v['verdict']}" + (f"   ({v['reason']})" if "reason" in v else ""))
    bad = [n for n, c in checks.items() if not c["ok"]]
    if bad:
        print("  WARNING ||W0|| sanity check failed for:", bad)
    print("  wrote", a.out)
    return verdicts


if __name__ == "__main__":
    main()
