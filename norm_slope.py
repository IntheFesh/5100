"""Slope of ||W||^2 - ||W_0||^2 against cumulative sum of eta_t^2 (prop. 3b), per run.
eta_t follows PQN's linear schedule over total gradient steps G; r0^2 taken as the He-init expectation 256
for the 1024->128 dense layer (checked against the first logged values)."""
import json, sys, numpy as np
eta0 = 5e-4
for name in sys.argv[1:]:
    d = json.load(open(f"{name}.json")); h = d["hist"]
    upd = np.array(h["update"]); E = d["args"]["epochs"]; M = d["args"]["minibatches"]
    G = d["num_updates"] * E * M
    tau = upd * E * M                                    # gradient steps done
    S = eta0 ** 2 * G / 3 * (1 - (1 - tau / G) ** 3)     # sum of eta_t^2
    w2 = np.array(h["wn_dense"]) ** 2                    # (points, seeds)
    growth = w2.mean(1) - 256.0
    k = len(S) // 2                                      # fit on second half (steady state)
    slope = np.polyfit(S[k:], growth[k:], 1)[0]
    lam_end = np.sqrt(w2[-1].mean() / 256.0)
    msg = f"{name:16s} E={E}  slope d||W||^2/d(sum eta^2) = {slope:9.0f}  (per-param: {slope/131200:.2f})  Lambda_end = {lam_end:.2f}"
    if "rad_cum" in h and len(h["rad_cum"]):
        msg += f"  rad/quad = {np.mean(np.array(h['rad_cum'][-1]) / np.array(h['quad_cum'][-1])):.1f}"
    print(msg)
