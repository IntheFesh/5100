import sys, numpy as np
pkg = sys.argv[1]
for seed in [0,1,2]:
    z = np.load(f"{pkg}/results/fields_seed{seed}.npz"); A = z['A']; M, p, _ = A.shape
    Am = A.mean(0)
    for eta in [1.0, 10.0]:
        P = np.linalg.inv(np.eye(p)[None] + eta * A); Pm = P.mean(0)
        cov = np.einsum('mij,mjk->ik', P - Pm, A - Am) / M
        lhs = np.eye(p) - eta * Pm @ Am; rhs = Pm + eta * cov
        print(f"seed {seed} eta {eta:5.1f}: ||(I-eta*Pbar*Abar)-(Pbar+eta*Cov)||={np.linalg.norm(lhs-rhs):.1e}  "
              f"||Pbar||2={np.linalg.norm(Pm,2):.4f}  eta*||Cov||2={eta*np.linalg.norm(cov,2):.2e}")
A = np.array([0.01, 100.]); P = 1/(1+A); cov = np.mean((P-P.mean())*(A-A.mean()))
print("scalar counterexample: Pbar + eta*Cov =", P.mean()+cov, " vs 1-eta*Pbar*Abar =", 1-P.mean()*A.mean())
