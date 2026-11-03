"""Unit tests for pqn_v5.py target functions and optimizer equivalence (no training loop)."""
import sys, numpy as np
sys.argv = ["pqn_v5.py", "--out", "x.json", "--steps", "4096"]
src = open("pqn_v5.py").read().split("eval_fn = ")[0]
exec(src)
import jax, jax.numpy as jnp, optax

rng = np.random.default_rng(0)
T, n, A_ = 32, 5, 6
R = jnp.array(rng.normal(size=(T, n)), jnp.float32)
D = jnp.array((rng.random((T, n)) < 0.05), jnp.float32)
QV = jnp.array(rng.normal(size=(T, n, A_)), jnp.float32)
last_q = jnp.array(rng.normal(size=n), jnp.float32)
vfin = jnp.array(rng.normal(size=(T, n)), jnp.float32)

# 1. fixed targets with no truncation == public recursion with the fix
g_pub = targets_public(R, D, QV, last_q)
g_fix = targets_fixed(R, D, jnp.zeros_like(D), QV, last_q, vfin)
print("1. fixed(no trunc) vs public-fixed  max diff:", float(jnp.max(jnp.abs(g_pub - g_fix))))

# 2. brute-force standard Q(lambda) with truncation
TR = D * jnp.array(rng.random((T, n)) < 0.5, jnp.float32)
g_fix2 = np.asarray(targets_fixed(R, D, TR, QV, last_q, vfin))
Rn, Dn, TRn, QVn, Ln, VF = map(np.asarray, (R, D, TR, QV, last_q, vfin))
G = np.zeros((T, n))
for j in range(n):
    for t in reversed(range(T)):
        vn = Ln[j] if t == T - 1 else QVn[t + 1, j].max()
        cont = 0.0 if t == T - 1 else G[t + 1, j]
        if Dn[t, j]:
            G[t, j] = Rn[t, j] + 0.99 * TRn[t, j] * VF[t, j]
        elif t == T - 1:
            G[t, j] = Rn[t, j] + 0.99 * vn
        else:
            G[t, j] = Rn[t, j] + 0.99 * ((1 - 0.65) * vn + 0.65 * cont)
print("2. fixed(with trunc) vs brute force max diff:", float(np.max(np.abs(G - g_fix2))))

# 3. public buggy recursion error matches the closed form of Proposition 1 (general form, with terminations):
#    e_t = g(1-lam)(g lam)^{T-2-t} [(1-d_{T-1}) V(s_T) - V(s_{T-1})] * prod_{j=t}^{T-2} (1-d_j),  t <= T-2
Dd = np.array(D); Dd[-1, :2] = 1.0; Dd[-5, 2] = 1.0; Dd = jnp.asarray(Dd)       # force terminations incl. at T-1
results = {}
for name, DD in [("no dones", jnp.zeros_like(D)), ("with dones", Dd)]:
    a.buggy_target = True; gb = np.asarray(targets_public(R, DD, QV, last_q))
    a.buggy_target = False; gf = np.asarray(targets_public(R, DD, QV, last_q))
    Dn_ = np.asarray(DD); vbar = (1 - Dn_[-1]) * np.asarray(last_q); vT1 = np.asarray(QV[-1]).max(-1)
    pred = np.zeros((T, n))
    for j in range(n):
        for t in range(T - 1):
            alive = np.prod(1 - Dn_[t:T - 1, j])
            pred[t, j] = 0.99 * 0.35 * (0.99 * 0.65) ** (T - 2 - t) * (vbar[j] - vT1[j]) * alive
    results[name] = float(np.max(np.abs((gb - gf) - pred)))
print("3. buggy-fixed vs closed form max diff:", results)

# 3c. buggy mode == literal transcription of purejaxql pqn_minatar.py _get_target (commit 47af6d7, L237-256)
def purejaxql_targets(rew, done, qv, lq, g=0.99, lam=0.65):
    rew, done, qv = np.asarray(rew), np.asarray(done), np.asarray(qv)
    lq = np.asarray(lq) * (1 - done[-1])                       # L252
    lr = rew[-1] + g * lq                                      # L253
    nq = lq; out = [lr]
    for t in reversed(range(T - 1)):                           # scan(reverse=True) over transitions[:-1]
        boot = rew[t] + g * (1 - done[t]) * nq
        lr = boot + g * lam * (lr - nq)
        lr = (1 - done[t]) * lr + done[t] * rew[t]
        nq = qv[t].max(-1)
        out.append(lr)
    return np.stack(out[::-1])
a.buggy_target = True
lit = float(np.max(np.abs(np.asarray(targets_public(R, Dd, QV, last_q)) - purejaxql_targets(R, Dd, QV, last_q))))
a.buggy_target = False
print("3c. buggy mode vs literal purejaxql (with dones incl. T-1) max diff:", lit)

# 4. optimizer chain == optax.radam
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 10, 10, 6)))
ref = optax.chain(optax.clip_by_global_norm(10.0), optax.radam(lr_sched))
s1, s2 = tx.init(params), ref.init(params); p1 = p2 = params; worst = 0.0
for i in range(30):
    gr = jax.tree_util.tree_map(lambda x: jnp.array(rng.normal(size=x.shape), x.dtype), params)
    u1, s1 = tx.update(gr, s1, p1); u2, s2 = ref.update(gr, s2, p2)
    p1, p2 = optax.apply_updates(p1, u1), optax.apply_updates(p2, u2)
    worst = max(worst, max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2))))
print("4. tx vs optax.radam max param diff:", worst)

# 5. radial removal makes the update orthogonal; projection keeps norm and function
upd = jax.tree_util.tree_map(lambda x: jnp.array(rng.normal(size=x.shape), x.dtype), params)
ur = remove_radial(upd, params)
dots = [float(jnp.sum(params["params"][k]["kernel"] * ur["params"][k]["kernel"]) + jnp.sum(params["params"][k]["bias"] * ur["params"][k]["bias"])) for k in LN_LAYERS]
print("5a. <W, radial-free update> per layer:", dots)
r0 = {k: jnorm(params["params"], k) for k in LN_LAYERS}
pb = optax.apply_updates(params, jax.tree_util.tree_map(lambda x: 0.5 * x, upd))
pp = project(pb, r0)
x = jnp.array(rng.random((8, 10, 10, 6)) < 0.2, jnp.float32)
print("5b. norms after projection:", [float(jnorm(pp["params"], k)) for k in LN_LAYERS], "targets", [float(v) for v in r0.values()])
print("5c. function change from projection (LN eps only):", float(jnp.max(jnp.abs(net.apply(pp, x)[0] - net.apply(pb, x)[0]))))

fails = []
if float(jnp.max(jnp.abs(g_pub - g_fix))) > 1e-5: fails.append("1")
if float(np.max(np.abs(G - g_fix2))) > 1e-5: fails.append("2")
if max(results.values()) > 1e-5: fails.append("3")
if lit > 1e-5: fails.append("3c")
if worst > 1e-6: fails.append("4")
if max(abs(v) for v in dots) > 1e-3 * max(float(jnorm(params["params"], k)) for k in LN_LAYERS) ** 2: fails.append("5a")
if max(abs(float(jnorm(pp["params"], k)) - float(r0[k])) for k in LN_LAYERS) > 1e-4: fails.append("5b")
if float(jnp.max(jnp.abs(net.apply(pp, x)[0] - net.apply(pb, x)[0]))) > 1e-3: fails.append("5c")
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
