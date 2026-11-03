import jax, jax.numpy as jnp, optax
din, dh = 256, 128
W0 = jax.random.normal(jax.random.PRNGKey(0), (din, dh)) * jnp.sqrt(2 / din)
V = jax.random.normal(jax.random.PRNGKey(1), (dh, 4)) * 0.2
def ln(h):
    mu = h.mean(-1, keepdims=True); return (h - mu) / jnp.sqrt(((h - mu) ** 2).mean(-1, keepdims=True) + 1e-12)
# LN over the hidden dim of x@W is invariant to scaling each COLUMN? no: to scaling all of W jointly
def loss(W, x, y): return jnp.mean((jax.nn.relu(ln(x @ W)) @ V - y) ** 2)
def run(tx, n=8000, lr_note=""):
    W = W0; st = tx.init(W); R = Q = 0.0; k = jax.random.PRNGKey(7)
    @jax.jit
    def step(W, st, k):
        k1, k2 = jax.random.split(k)
        x = (jax.random.uniform(k1, (128, din)) < 0.1).astype(jnp.float32); y = jax.random.normal(k2, (128, 4))
        g = jax.grad(loss)(W, x, y); u, st = tx.update(g, st, W)
        return W + u, st, 2 * jnp.sum(W * u), jnp.sum(u * u), jnp.sum(W * g)
    orth = 0.0
    for t in range(n):
        k, kk = jax.random.split(k); W, st, r, q, o = step(W, st, kk)
        if t > n // 4: R += float(r); Q += float(q); orth = max(orth, abs(float(o)))
    return R / Q, orth
ema = lambda lr, b: optax.chain(optax.trace(decay=b), optax.scale(-lr * (1 - b)))
print("EMA-momentum SGD beta=0.9 : rad/quad = %.2f (theory 18), max|<W,g>| = %.1e" % run(ema(0.5, 0.9)))
print("Adam beta1=0.9            : rad/quad = %.2f" % run(optax.adam(5e-4))[0])
print("Adam beta1=0 (RMSprop-like): rad/quad = %.2f (isotropic theory 0)" % run(optax.adam(5e-4, b1=0.0))[0])
