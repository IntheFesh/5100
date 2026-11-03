"""Compact PQN for MinAtar (gymnax), written to mirror purejaxql pqn_minatar.py
(commit 47af6d7) while exposing the switches we need:
  --input_scale255   reproduce the public x/255 input scaling (default: off = fixed)
  --buggy_target     reproduce the public Q(lambda) off-by-one (default: off = fixed)
  --epochs E, --minibatches M, --lr, --norm_proj (weight-norm projection on LN-preceded layers)
Seeds are vmapped. Writes a JSON with learning curves and diagnostics.
"""
import argparse, json, time
from functools import partial
import jax, jax.numpy as jnp, numpy as np, optax, flax.linen as nn, gymnax

p = argparse.ArgumentParser()
p.add_argument("--env", default="SpaceInvaders-MinAtar")
p.add_argument("--steps", type=float, default=5e6)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--num_steps", type=int, default=32)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--minibatches", type=int, default=32)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--lam", type=float, default=0.65)
p.add_argument("--eps_decay", type=float, default=0.1)
p.add_argument("--seeds", type=int, default=2)
p.add_argument("--seed0", type=int, default=0)
p.add_argument("--input_scale255", action="store_true")
p.add_argument("--buggy_target", action="store_true")
p.add_argument("--norm_proj", action="store_true")
p.add_argument("--log_every", type=int, default=50)
p.add_argument("--out", required=True)
p.add_argument("--save_params", action="store_true")
a = p.parse_args()

env, env_params = gymnax.make(a.env)
n_act = env.action_space(env_params).n
NUM_UPDATES = int(a.steps // (a.num_envs * a.num_steps))
GRAD_STEPS = NUM_UPDATES * a.epochs * a.minibatches


class QNet(nn.Module):
    n_act: int
    scale255: bool

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        if self.scale255:
            x = x / 255.0
        x = nn.Conv(16, (3, 3), 1, padding="VALID", kernel_init=nn.initializers.he_normal(), name="conv")(x)
        x = nn.relu(nn.LayerNorm(name="ln0")(x))
        x = x.reshape((x.shape[0], -1))
        h = nn.Dense(128, kernel_init=nn.initializers.he_normal(), name="dense")(x)
        h = nn.relu(nn.LayerNorm(name="ln1")(h))
        return nn.Dense(self.n_act, name="head")(h), h


net = QNet(n_act, a.input_scale255)
lr_sched = optax.linear_schedule(a.lr, 1e-20, GRAD_STEPS)
tx = optax.chain(optax.clip_by_global_norm(10.0), optax.radam(lr_sched))
eps_sched = optax.linear_schedule(1.0, 0.05, int(a.eps_decay * NUM_UPDATES))
PROJ = ("conv", "dense")  # layers followed by LayerNorm (scale-invariant)


def joint_norm(pr, name):
    return jnp.sqrt(jnp.sum(pr[name]["kernel"] ** 2) + jnp.sum(pr[name]["bias"] ** 2))


def project(params, r0):
    pr = dict(params["params"])
    for k in PROJ:
        s = r0[k] / (joint_norm(pr, k) + 1e-12)
        pr[k] = {"kernel": pr[k]["kernel"] * s, "bias": pr[k]["bias"] * s}
    return {"params": pr}


def qlambda_targets(rew, done, qv, last_q):
    """rew, done: (T,n); qv: (T,n,A) Q at collection; last_q: (n,) max Q(s_T)."""
    g, lam = a.gamma, a.lam
    lam_ret = rew[-1] + g * (1 - done[-1]) * last_q
    next_q0 = last_q if a.buggy_target else jnp.max(qv[-1], -1)

    def body(carry, x):
        lr_, nq = carry
        r, d, q = x
        boot = r + g * (1 - d) * nq
        lr_ = boot + g * lam * (lr_ - nq)
        lr_ = (1 - d) * lr_ + d * r
        return (lr_, jnp.max(q, -1)), lr_

    _, tg = jax.lax.scan(body, (lam_ret, next_q0), (rew[:-1], done[:-1], qv[:-1]), reverse=True)
    return jnp.concatenate([tg, lam_ret[None]], 0)


def dormant_frac(h, tau=0.025):
    s = jnp.mean(jnp.abs(h), 0)
    s = s / (jnp.mean(s) + 1e-9)
    return jnp.mean(s <= tau)


def make_train(rng):
    rng, k_init, k_env = jax.random.split(rng, 3)
    obs0, est0 = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_env, a.num_envs), env_params)
    params = net.init(k_init, obs0[:1])
    r0 = {k: joint_norm(params["params"], k) for k in PROJ}
    opt = tx.init(params)
    ep_ret = jnp.zeros(a.num_envs)
    carry = (params, opt, obs0, est0, ep_ret, rng, jnp.array(0, jnp.int32))
    return carry, r0


def update(carry, r0):
    params, opt, obs, est, ep_ret, rng, u = carry
    eps = eps_sched(u)

    def step(c, _):
        params_, obs_, est_, ep_ret_, rng_ = c
        rng_, ka, ke, ks = jax.random.split(rng_, 4)
        q, _ = net.apply(params_, obs_)
        greedy = jnp.argmax(q, -1)
        rand = jax.random.randint(ka, greedy.shape, 0, n_act)
        act = jnp.where(jax.random.uniform(ke, greedy.shape) < eps, rand, greedy)
        nobs, nest, rew, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(ks, a.num_envs), est_, act, env_params)
        ep_ret_ = ep_ret_ + rew
        fin_ret = jnp.where(done, ep_ret_, jnp.nan)
        ep_ret_ = jnp.where(done, 0.0, ep_ret_)
        return (params_, nobs, nest, ep_ret_, rng_), (obs_, act, rew, done.astype(jnp.float32), q, fin_ret)

    (_, obs, est, ep_ret, rng), (O, A_, R, D, QV, FIN) = jax.lax.scan(
        step, (params, obs, est, ep_ret, rng), None, a.num_steps)
    last_q = jnp.max(net.apply(params, obs)[0], -1)
    tgt = qlambda_targets(R, D, QV, last_q)
    N = a.num_steps * a.num_envs
    flat = lambda x: x.reshape((N,) + x.shape[2:])
    O, A_, tgt = flat(O), flat(A_), flat(tgt)

    def loss_fn(pp, o, ac, t):
        q, _ = net.apply(pp, o)
        qa = jnp.take_along_axis(q, ac[:, None], -1)[:, 0]
        return 0.5 * jnp.mean((qa - t) ** 2)

    def epoch(c, _):
        params_, opt_, rng_ = c
        rng_, kp = jax.random.split(rng_)
        perm = jax.random.permutation(kp, N)
        mb = lambda x: x[perm].reshape((a.minibatches, -1) + x.shape[1:])

        def mstep(c2, batch):
            pp, oo = c2
            l, g = jax.value_and_grad(loss_fn)(pp, *batch)
            upd, oo = tx.update(g, oo, pp)
            # exact split of the norm increment of the LN-preceded dense block:
            # ||W+d||^2-||W||^2 = 2<W,d> + ||d||^2   (d = optimizer update)
            Wd = pp["params"]["dense"]; dd = upd["params"]["dense"]
            rad = 2 * (jnp.sum(Wd["kernel"] * dd["kernel"]) + jnp.sum(Wd["bias"] * dd["bias"]))
            quad = jnp.sum(dd["kernel"] ** 2) + jnp.sum(dd["bias"] ** 2)
            pp = optax.apply_updates(pp, upd)
            if a.norm_proj:
                pp = project(pp, r0)
            return (pp, oo), (l, rad, quad)

        (params_, opt_), (ls, rads, quads) = jax.lax.scan(mstep, (params_, opt_), (mb(O), mb(A_), mb(tgt)))
        return (params_, opt_, rng_), (ls.mean(), rads.sum(), quads.sum())

    (params, opt, rng), (losses, rads, quads) = jax.lax.scan(epoch, (params, opt, rng), None, a.epochs)
    # diagnostics on this rollout's observations
    _, h = net.apply(params, O[:2048])
    diag = dict(loss=losses.mean(), dormant=dormant_frac(h), rad=rads.sum(), quad=quads.sum(),
                wn_conv=joint_norm(params["params"], "conv"), wn_dense=joint_norm(params["params"], "dense"),
                wn_head=jnp.sqrt(jnp.sum(params["params"]["head"]["kernel"] ** 2)),
                ret_sum=jnp.nansum(FIN), ret_cnt=jnp.sum(~jnp.isnan(FIN)), qmean=QV.mean())
    return (params, opt, obs, est, ep_ret, rng, u + 1), diag


def greedy_eval(params, rng, n=64, horizon=2000):
    obs, est = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(rng, n), env_params)

    def st(c, _):
        obs_, est_, ret, alive, rng_ = c
        rng_, ks, ka, ke = jax.random.split(rng_, 4)
        q, _ = net.apply(params, obs_)
        act = jnp.where(jax.random.uniform(ke, (n,)) < 0.0, jax.random.randint(ka, (n,), 0, n_act), jnp.argmax(q, -1))
        nobs, nest, rew, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(ks, n), est_, act, env_params)
        ret = ret + rew * alive
        alive = alive * (1 - done)
        return (nobs, nest, ret, alive, rng_), None

    (_, _, ret, _, _), _ = jax.lax.scan(st, (obs, est, jnp.zeros(n), jnp.ones(n), rng), None, horizon)
    return ret.mean()


def run_seed_chunk(carry, r0, k):
    def body(c, _):
        return update(c, r0)
    return jax.lax.scan(body, carry, None, k)


@partial(jax.jit, static_argnums=(2,))
def chunk(carries, r0s, k):
    return jax.vmap(run_seed_chunk, in_axes=(0, 0, None))(carries, r0s, k)


rngs = jax.random.split(jax.random.PRNGKey(a.seed0), a.seeds)
carries, r0s = jax.vmap(make_train)(rngs)
eval_fn = jax.jit(jax.vmap(greedy_eval, in_axes=(0, 0)))

hist = {k: [] for k in ["update", "env_steps", "loss", "dormant", "wn_conv", "wn_dense", "wn_head", "train_return", "qmean", "rad_cum", "quad_cum"]}
cum_rad = np.zeros(a.seeds); cum_quad = np.zeros(a.seeds)
evals = []
t0 = time.time()
done_u = 0
while done_u < NUM_UPDATES:
    k = min(a.log_every, NUM_UPDATES - done_u)
    carries, d = chunk(carries, r0s, k)
    d = jax.tree_util.tree_map(np.asarray, d)
    done_u += k
    tr = d["ret_sum"].sum(1) / np.maximum(d["ret_cnt"].sum(1), 1)
    hist["update"].append(done_u)
    hist["env_steps"].append(done_u * a.num_envs * a.num_steps)
    for key in ["loss", "dormant", "wn_conv", "wn_dense", "wn_head", "qmean"]:
        hist[key].append(d[key][:, -1].tolist())
    hist["train_return"].append(tr.tolist())
    cum_rad += d["rad"].sum(1); cum_quad += d["quad"].sum(1)
    hist["rad_cum"].append(cum_rad.tolist()); hist["quad_cum"].append(cum_quad.tolist())
    if done_u % (a.log_every * 4) == 0 or done_u == NUM_UPDATES:
        ev = np.asarray(eval_fn(carries[0], jax.random.split(jax.random.PRNGKey(999 + done_u), a.seeds)))
        evals.append({"env_steps": done_u * a.num_envs * a.num_steps, "greedy_return": ev.tolist()})
    print(f"[{time.time()-t0:7.1f}s] upd {done_u}/{NUM_UPDATES} train_ret {np.round(tr,1)} "
          f"dorm {np.round(d['dormant'][:, -1],3)} wn_dense {np.round(d['wn_dense'][:, -1],1)}"
          + (f" eval {np.round(evals[-1]['greedy_return'],1)}" if evals and evals[-1]['env_steps'] == done_u * a.num_envs * a.num_steps else ""),
          flush=True)

if a.save_params:
    import pickle
    pickle.dump(jax.tree_util.tree_map(np.asarray, carries[0]), open(a.out.replace(".json", "_params.pkl"), "wb"))
json.dump({"args": vars(a), "num_updates": NUM_UPDATES, "grad_steps": GRAD_STEPS,
           "wall_seconds": time.time() - t0, "hist": hist, "evals": evals}, open(a.out, "w"))
print("done", time.time() - t0)
