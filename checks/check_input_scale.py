"""Proposition check: with Adam (no clipping, eps->0), training f(W x/s + b) from (W0, b0) with lr eta
is trajectory-equivalent to training f(W' x + b) from (W0/s, b0) with lr eta/s on W' only.
Also reports how far RAdam (PQN's optimizer) and SGD deviate from that equivalence."""
import jax, jax.numpy as jnp, numpy as np, optax
jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(0)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)
d_in, d_h, n = 54, 16, 256
X = (jax.random.uniform(k1, (n, d_in)) < 0.15).astype(jnp.float64)       # binary, MinAtar-like
Y = jax.random.normal(k2, (n, 3))
W0 = jax.random.normal(k3, (d_in, d_h)) * jnp.sqrt(2 / d_in)
b0 = jnp.zeros(d_h)
V0 = jax.random.normal(k4, (d_h, 3)) * 0.3
s = 255.0

def ln(h):
    mu = h.mean(-1, keepdims=True); var = ((h - mu) ** 2).mean(-1, keepdims=True)
    return (h - mu) / jnp.sqrt(var + 1e-12)

def model(p, scale):
    h = X @ p["W"] / scale + p["b"]
    return jax.nn.relu(ln(h)) @ p["V"]

loss = lambda p, scale: jnp.mean((model(p, scale) - Y) ** 2)

def run(opt_factory, lrW_scaled, steps=600):
    pA = {"W": W0, "b": b0, "V": V0}                      # x/s input, lr eta everywhere
    pB = {"W": W0 / s, "b": b0, "V": V0}                  # raw input, W init / s
    eta = 1e-2
    txA = opt_factory(eta)
    txB = optax.multi_transform({"w": opt_factory(eta / s if lrW_scaled == "s" else eta / s**2), "o": opt_factory(eta)},
                                {"W": "w", "b": "o", "V": "o"})
    sA, sB = txA.init(pA), txB.init(pB)
    gA = jax.jit(jax.grad(lambda p: loss(p, s))); gB = jax.jit(jax.grad(lambda p: loss(p, 1.0)))
    worst = 0.0
    for t in range(steps):
        uA, sA = txA.update(gA(pA), sA, pA); pA = optax.apply_updates(pA, uA)
        uB, sB = txB.update(gB(pB), sB, pB); pB = optax.apply_updates(pB, uB)
        worst = max(worst, float(jnp.max(jnp.abs(model(pA, s) - model(pB, 1.0)))))
    return worst, float(loss(pA, s))

adam0 = lambda lr: optax.adam(lr, eps=1e-300)
print("Adam  (eps~0), lr_W/s : max |f_A - f_B| =", run(adam0, "s"))
print("RAdam (eps~0), lr_W/s : max |f_A - f_B| =", run(lambda lr: optax.radam(lr, eps=1e-300), "s"))
print("SGD,           lr_W/s : max |f_A - f_B| =", run(optax.sgd, "s"))
print("SGD,           lr_W/s^2: max |f_A - f_B| =", run(optax.sgd, "s2"))
