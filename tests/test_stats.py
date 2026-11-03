"""Calibration of analyze.py statistics under H0 (no true difference) and coverage of the t-interval.
Run: python tests/test_stats.py   (numpy only; ~30 s)"""
import itertools, numpy as np
rng = np.random.default_rng(1)
T = {4: 2.776, 9: 2.262}                     # t_{0.975} for df = 4, 9

def signflip_p(diff):
    n = len(diff); obs = abs(diff.mean())
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    return float(np.mean(np.abs((signs * diff).mean(1)) >= obs - 1e-12))

ok = True
for n in (5, 10):
    S = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    rej = cov = 0; R = 3000
    for _ in range(R):
        diff = rng.standard_t(5, size=n) * 7.0            # heavy-ish tails, true mean 0
        obs = abs(diff.mean()); p = float(np.mean(np.abs((S * diff).mean(1)) >= obs - 1e-12))
        rej += p < 0.05
        h = T[n - 1] * diff.std(ddof=1) / np.sqrt(n); cov += abs(diff.mean()) <= h
    fpr, coverage = rej / R, cov / R
    print(f"n={n}: sign-flip false-positive rate at 5% = {fpr:.3f};  t-interval coverage = {coverage:.3f}")
    ok &= fpr <= 0.065 and coverage >= 0.93
# power sanity: a real shift of 1 sd should usually be detected with n=10
pw = np.mean([signflip_p(rng.normal(1.0, 1.0, 10)) < 0.05 for _ in range(500)])
print(f"power, n=10, effect = 1 sd: {pw:.2f}"); ok &= pw > 0.6
print("ALL PASS" if ok else "FAILED"); raise SystemExit(0 if ok else 1)
