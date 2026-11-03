"""Check: EMA momentum on a scale-invariant block (gradients orthogonal to W).
Theory (stationary, temporally uncorrelated, isotropic grads, E||g||^2 = G):
  E[2<W_t, d_t>] = 2 eta^2 G beta/(1+beta),  E||d_t||^2 = eta^2 G (1-beta)/(1+beta),  sum = eta^2 G.
Also runs real Adam on a LayerNorm layer to measure the constant c in ||W||^2 ~ ||W0||^2 + c d sum eta^2."""
import numpy as np

rng = np.random.default_rng(0)
d, eta, steps = 2000, 1e-3, 40000
for beta in [0.0, 0.5, 0.9, 0.95]:
    W = rng.normal(size=d); W *= 10 / np.linalg.norm(W)
    m = np.zeros(d); rad = quad = 0.0; G = 0.0
    for t in range(steps):
        g = rng.normal(size=d)
        g -= (g @ W) / (W @ W) * W            # orthogonal to current W (scale invariance)
        G += g @ g
        m = beta * m + (1 - beta) * g
        dstep = -eta * m
        if t >= steps // 4:                     # after burn-in
            rad += 2 * W @ dstep; quad += dstep @ dstep
        W = W + dstep
    n = steps - steps // 4; G /= steps
    print(f"beta={beta:4.2f}  rad/quad meas={rad/quad:7.3f} theory={2*beta/(1-beta) if beta<1 else float('inf'):7.3f}  "
          f"(rad+quad)/(n eta^2 G) meas={(rad+quad)/(n*eta**2*G):.3f} theory=1.000")

# real Adam (beta1=0.9, beta2=0.999) on a scale-invariant layer: noisy regression through LayerNorm
import jax, jax.numpy as jnp, optax
key = jax.random.PRNGKey(0)
din, dh = 256, 128
W = jax.random.normal(key, (din, dh)) * jnp.sqrt(2 / din)
V = jax.random.normal(jax.random.PRNGKey(1), (dh, 4)) * 0.2
def ln(h):
    mu = h.mean(-1, keepdims=True); return (h - mu) / jnp.sqrt(((h - mu) ** 2).mean(-1, keepdims=True) + 1e-6)
def loss(W, V, x, y): return jnp.mean((jax.nn.relu(ln(x @ W)) @ V - y) ** 2)
tx = optax.adam(5e-4); st = tx.init(W)
@jax.jit
def step(W, st, k):
    k1, k2 = jax.random.split(k)
    x = (jax.random.uniform(k1, (128, din)) < 0.1).astype(jnp.float32)
    y = jax.random.normal(k2, (128, 4))                      # pure-noise targets: noise-dominated regime
    g = jax.grad(loss)(W, V, x, y)
    u, st = tx.update(g, st, W)
    W2 = optax.apply_updates(W, u)
    return W2, st, 2 * jnp.sum(W * u), jnp.sum(u * u)
W0n = float(jnp.sum(W ** 2)); R = Q = 0.0; k = jax.random.PRNGKey(7); n = 20000
for t in range(n):
    k, kk = jax.random.split(k); W, st, r, q = step(W, st, kk); R += float(r); Q += float(q)
dW = float(jnp.sum(W ** 2)) - W0n
print(f"Adam on LN layer: rad/quad={R/Q:.2f}  ||W||^2 growth={dW:.2f}  d*sum(eta^2)={din*dh*n*(5e-4)**2:.2f}  c={dW/(din*dh*n*(5e-4)**2):.3f}")
