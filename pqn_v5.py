"""PQN for MinAtar (gymnax) — research code for the v5 plan.

Mirrors purejaxql pqn_minatar.py (commit 47af6d7) by default except for the two fixes,
which are switchable so the "correction ladder" can be run:
  --input_scale255      reproduce public x/255 input scaling           (default off = fixed)
  --buggy_target        reproduce public Q(lambda) off-by-one          (default off = fixed)
Norm control on LayerNorm-preceded (scale-invariant) layers conv+dense, kernel and bias jointly:
  --norm_ctrl none|np|radial|wd   np: project to initial norm after every step
                                  radial: remove the radial component of every update (AdamP-style)
                                  wd: decoupled weight decay --wd on those layers only (AdamW-style)
Time limits:
  --bootstrap_timeout   bootstrap through time-limit truncation (Pardo et al. 2018) instead of treating it as terminal
  --max_steps_in_episode N   override gymnax episode cap (default: env default, 1000 for SpaceInvaders)
Protocol (v5.1, after the pilot showed the public evaluation saturates):
  --sticky p            sticky actions during TRAINING (MinAtar default 0.1; gymnax omits them; default 0 = public)
  Every eval point logs TWO protocols:
    public : deterministic dynamics, gymnax default cap (1000)     -> comparable to the public PQN curves
    robust : sticky --eval_sticky (0.1), cap --eval_cap (10000)    -> primary metric; no ceiling, episodes differ
Logging: greedy evaluation every --eval_every env steps; norm diagnostics; exact norm-increment split.
Seeds are vmapped (keep --seeds <= 10 per process; see purejaxql issue #9). Training seed i uses PRNGKey(seed0+i),
so a seed's stream does not depend on how many seeds share the process (it does depend on the JAX version/threefry
mode, which is recorded in the output JSON - never mix runtimes within one stage).
"""
import argparse, json, time, pickle
from functools import partial
import jax, jax.numpy as jnp, numpy as np, optax, flax.linen as nn, gymnax

p = argparse.ArgumentParser()
p.add_argument("--env", default="SpaceInvaders-MinAtar")
p.add_argument("--steps", type=float, default=1e7)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--num_steps", type=int, default=32)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--minibatches", type=int, default=32)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--lam", type=float, default=0.65)
p.add_argument("--eps_decay", type=float, default=0.1)
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--seed0", type=int, default=0)
p.add_argument("--input_scale255", action="store_true")
p.add_argument("--buggy_target", action="store_true")
p.add_argument("--norm_ctrl", default="none", choices=["none", "np", "radial", "wd"])
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--bootstrap_timeout", action="store_true")
p.add_argument("--max_steps_in_episode", type=int, default=0)
p.add_argument("--eval_every", type=float, default=5e5)
p.add_argument("--eval_episodes", type=int, default=64)
p.add_argument("--sticky", type=float, default=0.0)
p.add_argument("--eval_sticky", type=float, default=0.1)
p.add_argument("--eval_cap", type=int, default=10000)
p.add_argument("--out", required=True)
p.add_argument("--save_params", action="store_true")
p.add_argument("--ckpt_at", default="", help="extra param snapshots, env steps in millions, e.g. 1,3 (for the progression demo)")
a = p.parse_args()
assert not (a.buggy_target and a.bootstrap_timeout), "buggy target replicates the public code, which has no timeout bootstrapping"

env, env_params = gymnax.make(a.env)
PUBLIC_CAP = int(env_params.max_steps_in_episode)          # gymnax default cap, used by the 'public' eval protocol
if a.max_steps_in_episode:
    env_params = env_params.replace(max_steps_in_episode=a.max_steps_in_episode)
MAXT = int(env_params.max_steps_in_episode)
n_act = env.action_space(env_params).n
STEPS_PER_UPDATE = a.num_envs * a.num_steps
NUM_UPDATES = int(a.steps // STEPS_PER_UPDATE)
GRAD_STEPS = NUM_UPDATES * a.epochs * a.minibatches
EVAL_EVERY_U = max(1, int(round(a.eval_every / STEPS_PER_UPDATE)))
LN_LAYERS = ("conv", "dense")


class QNet(nn.Module):
    n_act: int
    scale255: bool

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        if self.scale255:
            x = x / 255.0
        x = nn.Conv(16, (3, 3), 1, padding="VALID", kernel_init=nn.initializers.he_normal(), name="conv")(x)
        x = nn.relu(nn.LayerNorm(name="ln0")(x)).reshape((x.shape[0], -1))
        h = nn.relu(nn.LayerNorm(name="ln1")(nn.Dense(128, kernel_init=nn.initializers.he_normal(), name="dense")(x)))
        return nn.Dense(self.n_act, name="head")(h), h


net = QNet(n_act, a.input_scale255)
lr_sched = optax.linear_schedule(a.lr, 1e-20, GRAD_STEPS)   # same clock as purejaxql: total gradient steps
eps_sched = optax.linear_schedule(1.0, 0.05, max(1, int(a.eps_decay * NUM_UPDATES)))


def ln_mask(params):
    return {"params": {k: jax.tree_util.tree_map(lambda _: k in LN_LAYERS, v) for k, v in params["params"].items()}}


chain = [optax.clip_by_global_norm(10.0), optax.scale_by_radam()]
if a.norm_ctrl == "wd":
    chain.append(optax.add_decayed_weights(a.wd, mask=ln_mask))
chain.append(optax.scale_by_learning_rate(lr_sched))
tx = optax.chain(*chain)                                      # == optax.radam(lr) when wd is off


def jnorm(pr, k):
    return jnp.sqrt(jnp.sum(pr[k]["kernel"] ** 2) + jnp.sum(pr[k]["bias"] ** 2))


def project(params, r0):
    pr = dict(params["params"])
    for k in LN_LAYERS:
        s = r0[k] / (jnorm(pr, k) + 1e-12)
        pr[k] = jax.tree_util.tree_map(lambda x: x * s, pr[k])
    return {"params": pr}


def remove_radial(upd, params):
    u = dict(upd["params"])
    for k in LN_LAYERS:
        W, d = params["params"][k], u[k]
        c = (jnp.sum(W["kernel"] * d["kernel"]) + jnp.sum(W["bias"] * d["bias"])) / (jnorm(params["params"], k) ** 2 + 1e-12)
        u[k] = {"kernel": d["kernel"] - c * W["kernel"], "bias": d["bias"] - c * W["bias"]}
    return {"params": u}


def targets_public(rew, done, qv, last_q):
    """Exact replica of purejaxql pqn_minatar.py (incl. the off-by-one when a.buggy_target)."""
    g, lam = a.gamma, a.lam
    lam_ret = rew[-1] + g * (1 - done[-1]) * last_q
    nq0 = last_q * (1 - done[-1]) if a.buggy_target else jnp.max(qv[-1], -1)   # purejaxql L252 masks last_q

    def body(c, x):
        lr_, nq = c
        r, d, q = x
        lr_ = r + g * (1 - d) * nq + g * lam * (lr_ - nq)
        lr_ = (1 - d) * lr_ + d * r
        return (lr_, jnp.max(q, -1)), lr_

    _, tg = jax.lax.scan(body, (lam_ret, nq0), (rew[:-1], done[:-1], qv[:-1]), reverse=True)
    return jnp.concatenate([tg, lam_ret[None]], 0)


def targets_fixed(rew, done, trunc, qv, last_q, vfin):
    """Standard finite Q(lambda): G_t = r_t + g[(1-d_t)((1-lam) V(s_{t+1}) + lam G_{t+1}) + trunc_t V(final_t)]."""
    g, lam = a.gamma, a.lam
    vnext = jnp.concatenate([jnp.max(qv[1:], -1), last_q[None]], 0)       # V(s_{t+1}) for t<T-1, V(s_T) at T-1
    G_last = rew[-1] + g * ((1 - done[-1]) * vnext[-1] + trunc[-1] * vfin[-1])

    def body(Gn, x):
        r, d, tr, vn, vf = x
        G = r + g * ((1 - d) * ((1 - lam) * vn + lam * Gn) + tr * vf)
        return G, G

    _, tg = jax.lax.scan(body, G_last, (rew[:-1], done[:-1], trunc[:-1], vnext[:-1], vfin[:-1]), reverse=True)
    return jnp.concatenate([tg, G_last[None]], 0)


def dormant_frac(h, tau=0.025):
    s = jnp.mean(jnp.abs(h), 0)
    return jnp.mean(s / (jnp.mean(s) + 1e-9) <= tau)


def sticky_select(u, p, prev, chosen):
    """MinAtar sticky actions (Machado et al. 2018): with probability p the previously EXECUTED action `prev` is repeated,
    otherwise the chosen action is executed. `u` ~ U[0,1). Pure function extracted from update()/greedy_eval() for
    tests/test_protocol.py; the expression is unchanged (CHANGELOG)."""
    return jnp.where(u < p, prev, chosen)


def next_prev(done, act):
    """Action memory of the sticky mechanism: the action just EXECUTED, reset to 0 (no-op) when the episode ended."""
    return jnp.where(done, 0, act)


def env_step(keys, est, act):
    """Like gymnax step (auto-reset) but also returns the pre-reset obs and a truncation flag."""
    def one(k, s, ac):
        k1, k2 = jax.random.split(k)
        o_st, s_st, r, d, _ = env.step_env(k1, s, ac, env_params)
        o_re, s_re = env.reset_env(k2, env_params)
        s_new = jax.tree_util.tree_map(lambda x, y: jax.lax.select(d, x, y), s_re, s_st)
        o_new = jax.lax.select(d, o_re, o_st)
        trunc = jnp.logical_and(d, s_st.time >= MAXT)
        return o_new, s_new, r, d, trunc, o_st
    return jax.vmap(one)(keys, est, act)


def make_train(rng):
    rng, k_init, k_env = jax.random.split(rng, 3)
    obs0, est0 = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_env, a.num_envs), env_params)
    params = net.init(k_init, obs0[:1])
    r0 = {k: jnorm(params["params"], k) for k in LN_LAYERS}
    z = jnp.zeros(a.num_envs)
    return (params, tx.init(params), obs0, est0, z, z.astype(jnp.int32), rng, jnp.array(0, jnp.int32)), r0


def update(carry, r0):
    params, opt, obs, est, ep_ret, prev, rng, u = carry
    eps = eps_sched(u)

    def step(c, _):
        obs_, est_, ep_ret_, prev_, rng_ = c
        rng_, ka, ke, ks, kst = jax.random.split(rng_, 5)
        q, _ = net.apply(params, obs_)
        greedy = jnp.argmax(q, -1)
        chosen = jnp.where(jax.random.uniform(ke, greedy.shape) < eps, jax.random.randint(ka, greedy.shape, 0, n_act), greedy)
        # sticky actions live on the environment side, exactly as in MinAtar's Environment.act: with prob p the
        # previously EXECUTED action is repeated; the agent does not observe this and stores its CHOSEN action
        act = sticky_select(jax.random.uniform(kst, greedy.shape), a.sticky, prev_, chosen)
        nobs, nest, rew, done, trunc, fin = env_step(jax.random.split(ks, a.num_envs), est_, act)
        ep_ret_ = ep_ret_ + rew
        fin_ret = jnp.where(done, ep_ret_, jnp.nan)
        ep_ret_ = jnp.where(done, 0.0, ep_ret_)
        prev_ = next_prev(done, act)
        return (nobs, nest, ep_ret_, prev_, rng_), (obs_, chosen, rew, done.astype(jnp.float32), trunc.astype(jnp.float32), fin, q, fin_ret)

    (obs, est, ep_ret, prev, rng), (O, A_, R, D, TR, FIN_OBS, QV, FIN) = jax.lax.scan(step, (obs, est, ep_ret, prev, rng), None, a.num_steps)
    last_q = jnp.max(net.apply(params, obs)[0], -1)
    if a.buggy_target or not a.bootstrap_timeout:
        tgt = targets_public(R, D, QV, last_q)
    else:
        T, n = R.shape
        vfin = jnp.max(net.apply(params, FIN_OBS.reshape((T * n,) + FIN_OBS.shape[2:]))[0], -1).reshape(T, n)
        tgt = targets_fixed(R, D, TR, QV, last_q, vfin)
    N = STEPS_PER_UPDATE
    flat = lambda x: x.reshape((N,) + x.shape[2:])
    O, A_, tgt = flat(O), flat(A_), flat(tgt)

    def loss_fn(pp, o, ac, t):
        qa = jnp.take_along_axis(net.apply(pp, o)[0], ac[:, None], -1)[:, 0]
        return 0.5 * jnp.mean((qa - t) ** 2)

    def epoch(c, _):
        pp0, oo0, rng_ = c
        rng_, kp = jax.random.split(rng_)
        perm = jax.random.permutation(kp, N)
        mb = lambda x: x[perm].reshape((a.minibatches, -1) + x.shape[1:])

        def mstep(c2, batch):
            pp, oo = c2
            l, gr = jax.value_and_grad(loss_fn)(pp, *batch)
            upd, oo = tx.update(gr, oo, pp)
            if a.norm_ctrl == "radial":
                upd = remove_radial(upd, pp)
            Wd, dd = pp["params"]["dense"], upd["params"]["dense"]
            rad = 2 * (jnp.sum(Wd["kernel"] * dd["kernel"]) + jnp.sum(Wd["bias"] * dd["bias"]))
            quad = jnp.sum(dd["kernel"] ** 2) + jnp.sum(dd["bias"] ** 2)
            # exact angle between W and W+d for the dense layer: sin = ||d_perp|| / ||W+d|| (Lemma 2)
            wn2 = jnp.sum(Wd["kernel"] ** 2) + jnp.sum(Wd["bias"] ** 2)
            ang = jnp.arcsin(jnp.clip(jnp.sqrt(jnp.maximum(quad - (rad / 2) ** 2 / wn2, 0.0) / (wn2 + rad + quad)), 0.0, 1.0))
            pp = optax.apply_updates(pp, upd)
            if a.norm_ctrl == "np":
                pp = project(pp, r0)
            return (pp, oo), (l, rad, quad, ang)

        (pp0, oo0), (ls, rads, quads, angs) = jax.lax.scan(mstep, (pp0, oo0), (mb(O), mb(A_), mb(tgt)))
        return (pp0, oo0, rng_), (ls.mean(), rads.sum(), quads.sum(), angs.sum())

    (params, opt, rng), (losses, rads, quads, angs) = jax.lax.scan(epoch, (params, opt, rng), None, a.epochs)
    _, h = net.apply(params, O[:2048])
    diag = dict(loss=losses.mean(), dormant=dormant_frac(h), rad=rads.sum(), quad=quads.sum(), ang=angs.sum(),
                wn_conv=jnorm(params["params"], "conv"), wn_dense=jnorm(params["params"], "dense"),
                wn_head=jnp.sqrt(jnp.sum(params["params"]["head"]["kernel"] ** 2)),
                ret_sum=jnp.nansum(FIN), ret_cnt=jnp.sum(~jnp.isnan(FIN)), trunc_frac=TR.sum() / jnp.maximum(D.sum(), 1),
                qmean=QV.mean())
    return (params, opt, obs, est, ep_ret, prev, rng, u + 1), diag


def greedy_eval(params, rng, cap, sticky):
    """Greedy policy, n episodes; stops as soon as every episode has ended. Returns (mean return, frac capped, mean len)."""
    n = a.eval_episodes
    ep = env_params.replace(max_steps_in_episode=cap)
    k_reset, rng = jax.random.split(rng)
    obs, est = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_reset, n), ep)

    def cond(c):
        return jnp.logical_and(c[-1] < cap, jnp.any(c[4] > 0))

    def body(c):
        obs_, est_, prev, ret, alive, ln, rng_, t = c
        rng_, ks, kst = jax.random.split(rng_, 3)
        greedy = jnp.argmax(net.apply(params, obs_)[0], -1)
        act = sticky_select(jax.random.uniform(kst, (n,)), sticky, prev, greedy)
        nobs, nest, rew, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(ks, n), est_, act, ep)
        return (nobs, nest, act, ret + rew * alive, alive * (1 - done), ln + alive, rng_, t + 1)

    init = (obs, est, jnp.zeros(n, jnp.int32), jnp.zeros(n), jnp.ones(n), jnp.zeros(n), rng, jnp.array(0))
    _, _, _, ret, _, ln, _, _ = jax.lax.while_loop(cond, body, init)
    return ret.mean(), jnp.mean(ln >= cap), ln.mean()


@partial(jax.jit, static_argnums=(2,))
def chunk(carries, r0s, k):
    return jax.vmap(lambda c, r: jax.lax.scan(lambda cc, _: update(cc, r), c, None, k), in_axes=(0, 0))(carries, r0s)


eval_fn = jax.jit(jax.vmap(greedy_eval, in_axes=(0, 0, None, None)), static_argnums=(2, 3))
# seed identity: training seed i uses PRNGKey(seed0 + i), independent of how many seeds share the process
SEED_IDS = np.arange(a.seed0, a.seed0 + a.seeds)
carries, r0s = jax.vmap(make_train)(jax.vmap(jax.random.PRNGKey)(jnp.asarray(SEED_IDS)))
keys = ["loss", "dormant", "wn_conv", "wn_dense", "wn_head", "qmean", "trunc_frac"]
hist = {k: [] for k in ["update", "env_steps", "train_return", "rad_cum", "quad_cum", "ang_cum", "chunk_seconds"] + keys}
cum_rad = np.zeros(a.seeds); cum_quad = np.zeros(a.seeds); cum_ang = np.zeros(a.seeds); evals = []; ckpt_log = []

# chunk boundaries: eval every EVAL_EVERY_U updates, a short remainder (< half a chunk) folded into the last chunk,
# plus exact snapshot boundaries (snapshots do NOT trigger an evaluation and do not add history points)
eval_b = list(range(EVAL_EVERY_U, NUM_UPDATES + 1, EVAL_EVERY_U))
if not eval_b or NUM_UPDATES - eval_b[-1] >= EVAL_EVERY_U / 2:
    eval_b.append(NUM_UPDATES)
else:
    eval_b[-1] = NUM_UPDATES
ckpt_targets = sorted(float(x) for x in a.ckpt_at.split(",") if x)
ckpt_b = {}
for m in ckpt_targets:
    u = int(np.ceil(m * 1e6 / STEPS_PER_UPDATE))
    if 0 < u < NUM_UPDATES:
        ckpt_b[u] = m
bounds = sorted(set(eval_b) | set(ckpt_b))

t0 = time.time(); done_u = 0; pending = None; tc = time.time()
for b in bounds:
    carries, d = chunk(carries, r0s, b - done_u)
    d = jax.tree_util.tree_map(np.asarray, d)                   # blocks until ready; arrays are (seeds, updates)
    done_u = b
    pending = d if pending is None else {k_: np.concatenate([pending[k_], d[k_]], 1) for k_ in d}
    if b in ckpt_b:
        label = f"{ckpt_b[b]:g}M"
        pickle.dump(jax.tree_util.tree_map(np.asarray, carries[0]), open(a.out.replace(".json", f"_ckpt{label}_params.pkl"), "wb"))
        ckpt_log.append({"label": label, "update": b, "env_steps": b * STEPS_PER_UPDATE})
    if b not in eval_b:
        continue
    d, pending = pending, None
    hist["chunk_seconds"].append(time.time() - tc); tc = time.time()
    hist["update"].append(done_u); hist["env_steps"].append(done_u * STEPS_PER_UPDATE)
    hist["train_return"].append((d["ret_sum"].sum(1) / np.maximum(d["ret_cnt"].sum(1), 1)).tolist())
    for key in keys:
        hist[key].append(d[key][:, -1].tolist())
    cum_rad += d["rad"].sum(1); cum_quad += d["quad"].sum(1)
    cum_ang += d["ang"].sum(1)                                   # cumulative angular travel of the dense layer (radians)
    hist["rad_cum"].append(cum_rad.tolist()); hist["quad_cum"].append(cum_quad.tolist()); hist["ang_cum"].append(cum_ang.tolist())
    # evaluation env keys depend only on (eval point, seed id): identical across configs -> paired comparisons
    ek = jax.vmap(lambda sid: jax.random.fold_in(jax.random.PRNGKey(10_000 + done_u), sid))(jnp.asarray(SEED_IDS))
    pub = [np.asarray(x) for x in eval_fn(carries[0], ek, PUBLIC_CAP, 0.0)]
    rob = [np.asarray(x) for x in eval_fn(carries[0], ek, a.eval_cap, a.eval_sticky)]
    evals.append({"env_steps": done_u * STEPS_PER_UPDATE,
                  "greedy_return": pub[0].tolist(), "capped_frac": pub[1].tolist(), "public_len": pub[2].tolist(),
                  "robust_return": rob[0].tolist(), "robust_capped_frac": rob[1].tolist(), "robust_len": rob[2].tolist()})
    print(f"[{time.time()-t0:7.1f}s] {done_u * STEPS_PER_UPDATE/1e6:5.2f}M  public {np.round(pub[0],1)} (capped {np.round(pub[1],2)})  "
          f"robust {np.round(rob[0],1)} (capped {np.round(rob[1],2)})  wn_dense {np.round(d['wn_dense'][:, -1],1)}", flush=True)

if a.save_params:
    pickle.dump(jax.tree_util.tree_map(np.asarray, carries[0]), open(a.out.replace(".json", "_params.pkl"), "wb"))
import importlib.metadata as _md
versions = {p_: _md.version(p_) for p_ in ("jax", "jaxlib", "flax", "optax", "gymnax")}
versions["jax_threefry_partitionable"] = bool(jax.config.jax_threefry_partitionable)
versions["backend"] = jax.default_backend()
json.dump({"args": vars(a), "seed_ids": SEED_IDS.tolist(), "num_updates": NUM_UPDATES, "grad_steps": GRAD_STEPS,
           "max_steps_in_episode": MAXT, "public_cap": PUBLIC_CAP, "ckpts": ckpt_log, "versions": versions,
           "wall_seconds": time.time() - t0, "hist": hist, "evals": evals}, open(a.out, "w"))
print("done", round(time.time() - t0, 1), "s")
