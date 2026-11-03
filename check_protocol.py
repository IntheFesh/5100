"""Evaluate saved params under two protocols:
  public : gymnax default (deterministic dynamics, 1000-step cap)  -> what the public PQN code reports
  robust : sticky actions p=0.1 (MinAtar default, Young & Tian 2019; Machado et al. 2018), cap raised to CAP
Reports per training seed: mean/sd/min/max return, mean length, fraction of episodes that hit the cap,
and the number of distinct returns (1 => all episodes identical => evaluation is a single trajectory).
Usage: python check_protocol.py PARAMS.pkl [--scale255] [--cap 10000] [--episodes 16] [--sticky 0.1]"""
import argparse, pickle, numpy as np, jax, jax.numpy as jnp, flax.linen as nn, gymnax

ap = argparse.ArgumentParser()
ap.add_argument("params"); ap.add_argument("--scale255", action="store_true")
ap.add_argument("--cap", type=int, default=10000); ap.add_argument("--episodes", type=int, default=16)
ap.add_argument("--sticky", type=float, default=0.1); ap.add_argument("--label", default=None)
a = ap.parse_args()
env, base = gymnax.make("SpaceInvaders-MinAtar"); n_act = env.action_space(base).n


class QNet(nn.Module):
    scale255: bool

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        if self.scale255:
            x = x / 255.0
        x = nn.Conv(16, (3, 3), 1, padding="VALID", name="conv")(x)
        x = nn.relu(nn.LayerNorm(name="ln0")(x)).reshape((x.shape[0], -1))
        h = nn.relu(nn.LayerNorm(name="ln1")(nn.Dense(128, name="dense")(x)))
        return nn.Dense(n_act, name="head")(h), h


net = QNet(a.scale255)


def evaluate(params, cap, sticky, n, seed=0):
    ep = base.replace(max_steps_in_episode=cap)
    k0 = jax.random.PRNGKey(seed)
    obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k0, n), ep)

    def body(c, _):
        o, s, prev, ret, ln, alive, k = c
        k, ks, kst = jax.random.split(k, 3)
        greedy = jnp.argmax(net.apply(params, o)[0], -1)
        act = jnp.where(jax.random.uniform(kst, (n,)) < sticky, prev, greedy)
        o2, s2, r, d, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(ks, n), s, act, ep)
        return (o2, s2, act, ret + r * alive, ln + alive, alive * (1 - d), k), None

    init = (obs, st, jnp.zeros(n, jnp.int32), jnp.zeros(n), jnp.zeros(n), jnp.ones(n), jax.random.PRNGKey(seed + 7))
    (_, _, _, ret, ln, _, _), _ = jax.lax.scan(body, init, None, cap)
    return np.asarray(ret), np.asarray(ln)



carry = pickle.load(open(a.params, "rb"))
label = a.label or a.params.split("_params")[0]
for si in range(jax.tree_util.tree_leaves(carry)[0].shape[0]):
    p = jax.tree_util.tree_map(lambda x: x[si], carry)
    for name, cap, sticky in [("public", 1000, 0.0), ("robust", a.cap, a.sticky)]:
        r, l = evaluate(p, cap, sticky, a.episodes)
        print(f"{label:12s} seed{si} {name:6s} cap={cap:5d} sticky={sticky:.1f}: return {r.mean():6.1f} ± {r.std():5.1f} "
              f"[{r.min():.0f}, {r.max():.0f}]  len {l.mean():6.0f}  capped {np.mean(l >= cap):4.0%}  "
              f"distinct {len(np.unique(r))}/{len(r)}", flush=True)
