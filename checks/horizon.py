"""(1) Mean effective n-step horizon of chunk-truncated lambda-returns, closed form vs brute force.
   (2) Mean per-target weight of the Q(lambda) off-by-one error vs rollout length T."""
import numpy as np
g, lam = 0.99, 0.65
def Hbar_closed(T, lam):
    return (T - lam * (1 - lam**T) / (1 - lam)) / (T * (1 - lam))
def Hbar_brute(T, lam):
    hs = []
    for t in range(T):
        L = T - t                      # steps until chunk end (bootstrap at s_T)
        w = [(1 - lam) * lam**(k - 1) for k in range(1, L)] + [lam**(L - 1)]
        hs.append(sum(k * wk for k, wk in zip(range(1, L + 1), w)))
    return np.mean(hs)
for T in [1, 2, 4, 8, 16, 32, 64]:
    bug = (g * (1 - lam) / T) * sum((g * lam)**k for k in range(T - 1)) if T >= 2 else 0.0
    print(f"T={T:3d}  Hbar closed={Hbar_closed(T,lam):.4f} brute={Hbar_brute(T,lam):.4f}  "
          f"mean bug weight per unit dQ={bug:.4f}")
