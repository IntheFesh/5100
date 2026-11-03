import sys, json, numpy as np
from scipy.linalg import solve
pkg = sys.argv[1]
out = {}
for seed in [0,1,2]:
    z = np.load(f"{pkg}/results/fields_seed{seed}.npz")
    A, b, X = z['A'], z['b'], z['X'].reshape(-1, 13)
    M, p, _ = A.shape
    Am, bm = A.mean(0), b.mean(0); ws = solve(Am, bm)
    ref = np.sqrt(np.mean((X @ ws) ** 2))
    Fs = np.einsum('mij,j->mi', A, ws) - b           # per-block residual at the mean root
    covA = np.einsum('mij,mj->i', A - Am, Fs) / M      # first-order coupling: (1/M) sum (A_i-Abar) F_i(w*)
    rows = []
    for eta in [0.1, 1.0, 10.0, 50.0, 200.0]:
        P = np.linalg.inv(np.eye(p)[None] + eta * A); Pm = P.mean(0)
        H = np.einsum('mij,mjk->mik', P, A).mean(0); h = np.einsum('mij,mj->mi', P, b).mean(0)
        w_eta = solve(H, h)
        # (1) identity H_eta = (I - Pbar)/eta
        idres = np.linalg.norm(H - (np.eye(p) - Pm) / eta)
        # (2) first-order prediction of the shift: w_eta - w* ~ eta * Abar^{-1} covA
        shift = w_eta - ws
        fo = eta * solve(Am, covA)
        rel_shift = np.sqrt(np.mean((X @ shift) ** 2)) / ref
        rel_fo = np.sqrt(np.mean((X @ fo) ** 2)) / ref
        rows.append(dict(eta=eta, rel_shift=float(rel_shift), first_order=float(rel_fo),
                         H_identity_residual=float(idres), rho_same=float(np.max(np.abs(np.linalg.eigvals(Pm)))),
                         rho_cross=float(np.max(np.abs(np.linalg.eigvals(np.eye(p) - eta * Pm @ Am))))))
    # block heterogeneity: relative Frobenius spread of A_i
    het = np.sqrt(np.mean([np.linalg.norm(a - Am, 'fro') ** 2 for a in A])) / np.linalg.norm(Am, 'fro')
    out[seed] = dict(block_relative_heterogeneity=float(het), rows=rows)
    print(f"seed {seed}: block heterogeneity ||A_i-Abar||_F/||Abar||_F = {het:.3f}")
    for r in rows:
        print(f"  eta={r['eta']:6.1f} shift={100*r['rel_shift']:.4f}%  first-order={100*r['first_order']:.4f}%  "
              f"H-identity={r['H_identity_residual']:.1e}  rho_same={r['rho_same']:.4f} rho_cross={r['rho_cross']:.4f}")
json.dump(out, open('verify.json', 'w'), indent=1)
