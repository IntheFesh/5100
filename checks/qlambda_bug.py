"""Replays purejaxql's MinAtar Q(lambda) target recurrence (no dones) and compares it
with the correct Peng's Q(lambda) return; checks the closed-form error
e_t = gamma*(1-lam)*(gamma*lam)^(T-2-t) * (maxQ(s_T) - maxQ(s_{T-1})) for t <= T-2."""
import numpy as np
rng = np.random.default_rng(0)
T, A, g = 32, 4, 0.99
for lam in [0.0, 0.65, 0.9]:
    r = rng.normal(size=T); q = rng.normal(size=(T, A)); qT = rng.normal(size=A)  # q[t]=Q(s_t,.), qT=Q(s_T,.)
    m = q.max(1); mT = qT.max()
    # correct: G_{T-1} = r_{T-1} + g*mT ; G_t = r_t + g*[(1-lam)*m[t+1] + lam*G_{t+1}]
    G = np.zeros(T); G[-1] = r[-1] + g * mT
    for t in range(T - 2, -1, -1):
        G[t] = r[t] + g * ((1 - lam) * m[t + 1] + lam * G[t + 1])
    # code: carry=(G_{T-1}, next_q=mT); scan reverse over t=T-2..0 using transition t (q_val=q[t])
    B = np.zeros(T); B[-1] = r[-1] + g * mT; lr, nq = B[-1], mT
    for t in range(T - 2, -1, -1):
        lr = r[t] + g * nq + g * lam * (lr - nq); nq = m[t]; B[t] = lr
    err = B - G
    pred = np.array([g * (1 - lam) * (g * lam) ** (T - 2 - t) * (mT - m[T - 1]) if t <= T - 2 else 0.0 for t in range(T)])
    print(f"lam={lam}: n_wrong(|err|>1e-9)={np.sum(np.abs(err)>1e-9)}  max|err-pred|={np.abs(err-pred).max():.2e}  "
          f"|err| at t=T-2,T-5,T-10: {abs(err[T-2]):.3f},{abs(err[T-5]):.4f},{abs(err[T-10]):.2e}")
