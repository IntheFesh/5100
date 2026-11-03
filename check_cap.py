"""Are greedy SpaceInvaders episodes hitting gymnax's 1000-step cap? Evaluate saved params with the
default cap and with a raised cap; report returns, lengths and the fraction of capped episodes."""
import sys, pickle, numpy as np, jax, jax.numpy as jnp, flax.linen as nn, gymnax

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


def evaluate(params, net, cap, n=32, seed=0):
    ep = base.replace(max_steps_in_episode=cap)
    obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(seed), n), ep)

    def body(c, _):
        o, s, ret, ln, alive, k = c
        k, ks = jax.random.split(k)
        act = jnp.argmax(net.apply(params, o)[0], -1)
        o2, s2, r, d, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(ks, n), s, act, ep)
        return (o2, s2, ret + r * alive, ln + alive, alive * (1 - d), k), None

    (_, _, ret, ln, _, _), _ = jax.lax.scan(body, (obs, st, jnp.zeros(n), jnp.zeros(n), jnp.ones(n), jax.random.PRNGKey(seed + 1)), None, cap)
    return np.asarray(ret), np.asarray(ln)


for path, s255 in [(sys.argv[1], False), (sys.argv[2], True)] if len(sys.argv) > 2 else [(sys.argv[1], False)]:
    carry = pickle.load(open(path, "rb")); net = QNet(s255)
    for si in range(jax.tree_util.tree_leaves(carry)[0].shape[0]):
        p = jax.tree_util.tree_map(lambda x: x[si], carry)
        for cap in (1000, 5000):
            r, l = evaluate(p, net, cap)
            print(f"{path.split('_params')[0]:12s} seed{si} cap={cap:5d}: return {r.mean():7.1f} (min {r.min():.0f}, max {r.max():.0f})  "
                  f"length {l.mean():7.1f}  capped {np.mean(l >= cap):.0%}", flush=True)
