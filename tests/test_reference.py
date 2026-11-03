"""Independent audit of pqn_v5.py against the official PQN-MinAtar implementation (round 2).

Reference: https://github.com/mttga/purejaxql, commit 47af6d7b35c89ddfe633aaf7341bdb8964cb7cce
           purejaxql/pqn_minatar.py  +  purejaxql/config/alg/pqn_minatar.yaml
All official code below is TRANSCRIBED by hand (not imported), with the line numbers of that file in comments.
This file deliberately does not reuse anything from test_v5.py.

Run: python -m pytest -q tests/test_reference.py
"""
import collections, itertools, os, sys
import numpy as np, pytest
import jax, jax.numpy as jnp, optax, flax.linen as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load_pqn, flags

# --- official hyper-parameters: purejaxql/config/alg/pqn_minatar.yaml (L3-L17) -----------------------------------
OFFICIAL = dict(TOTAL_TIMESTEPS=1e7, TOTAL_TIMESTEPS_DECAY=1e7, NUM_ENVS=128, NUM_STEPS=32, EPS_START=1.0, EPS_FINISH=0.05,
                EPS_DECAY=0.1, NUM_MINIBATCHES=32, NUM_EPOCHS=2, LR=0.0005, MAX_GRAD_NORM=10, GAMMA=0.99, LAMBDA=0.65)

Transition = collections.namedtuple("Transition", "reward done q_val")        # fields used by the official _get_target


def official_targets(reward, done, q_val, last_q_values, GAMMA, LAMBDA):
    """Literal transcription of pqn_minatar.py L227-L260 (jax.lax.scan version).

    reward, done, q_val: (T, n[, A]) with done as BOOL (as returned by gymnax); last_q_values: Q(next_obs[-1]), (n, A)."""
    transitions = Transition(reward=reward, done=done, q_val=q_val)
    last_q = jnp.max(last_q_values, axis=-1)                                                   # L235

    def _get_target(lambda_returns_and_next_q, transition):                                    # L237
        lambda_returns, next_q = lambda_returns_and_next_q                                     # L238
        target_bootstrap = transition.reward + GAMMA * (1 - transition.done) * next_q          # L239-241
        delta = lambda_returns - next_q                                                        # L242
        lambda_returns = target_bootstrap + GAMMA * LAMBDA * delta                             # L243-245
        lambda_returns = (1 - transition.done) * lambda_returns + transition.done * transition.reward   # L246-248
        next_q = jnp.max(transition.q_val, axis=-1)                                            # L249
        return (lambda_returns, next_q), lambda_returns                                        # L250

    last_q = last_q * (1 - transitions.done[-1])                                               # L252
    lambda_returns = transitions.reward[-1] + GAMMA * last_q                                   # L253
    _, targets = jax.lax.scan(                                                                 # L254-259
        _get_target,
        (lambda_returns, last_q),
        jax.tree_util.tree_map(lambda x: x[:-1], transitions),
        reverse=True,
    )
    return jnp.concatenate((targets, lambda_returns[np.newaxis]))                              # L260


def random_rollout(rng, T=32, n=24, A=4, p_done=0.08, last_done_frac=0.4):
    R = rng.normal(size=(T, n)).astype(np.float32)
    D = rng.random((T, n)) < p_done
    D[-1] = rng.random(n) < last_done_frac               # force terminations at the final step for many envs
    D[:, 0] = False                                      # env 0 never terminates; env 1 terminates only at T-1
    D[:, 1] = False; D[-1, 1] = True
    QV = rng.normal(size=(T, n, A)).astype(np.float32)
    LQ = rng.normal(size=(n, A)).astype(np.float32)
    return R, D, QV, LQ


# =====================================================================================================================
# (a) Q(lambda) target: public mode == official code, to < 1e-5, incl. mid-rollout and last-step terminations
# =====================================================================================================================
@pytest.mark.parametrize("seed", range(5))
def test_buggy_mode_equals_official_transcription(pqn, seed):
    R, D, QV, LQ = random_rollout(np.random.default_rng(seed))
    off = np.asarray(official_targets(jnp.asarray(R), jnp.asarray(D), jnp.asarray(QV), jnp.asarray(LQ), 0.99, 0.65))
    with flags(pqn, buggy_target=True, gamma=0.99, lam=0.65):
        ours = np.asarray(pqn["targets_public"](jnp.asarray(R), jnp.asarray(D, jnp.float32), jnp.asarray(QV), jnp.asarray(LQ.max(-1))))
    assert D[-1].any() and D[:-1].any()                  # the input really exercises both kinds of termination
    assert np.max(np.abs(off - ours)) < 1e-5


def test_default_mode_differs_from_official_only_through_the_boundary(pqn):
    """With the fix on, the result must differ from the official one (otherwise the 'fix' would be a no-op)."""
    R, D, QV, LQ = random_rollout(np.random.default_rng(11))
    D[:] = False
    off = np.asarray(official_targets(jnp.asarray(R), jnp.asarray(D), jnp.asarray(QV), jnp.asarray(LQ), 0.99, 0.65))
    with flags(pqn, buggy_target=False):
        ours = np.asarray(pqn["targets_public"](jnp.asarray(R), jnp.asarray(D, jnp.float32), jnp.asarray(QV), jnp.asarray(LQ.max(-1))))
    diff = np.abs(off - ours)
    assert diff[-1].max() < 1e-6                         # step T-1 is identical in both
    assert diff[:-1].max() > 1e-3                        # but the earlier steps carry the off-by-one error


# =====================================================================================================================
# (b) brute-force standard finite Q(lambda), written from the definition (float64, plain loops)
# =====================================================================================================================
def brute_force_qlambda(R, D, TR, QV, LQ, VF, gamma, lam):
    """G_{T-1} = r + g[(1-d) V(s_T) + trunc V(final)];  G_t = r_t + g[(1-d_t)((1-lam) V(s_{t+1}) + lam G_{t+1}) + trunc_t V(final_t)]."""
    T, n = R.shape
    V = QV.max(-1)                                       # V(s_t) from the acting network
    G = np.zeros((T, n))
    for j in range(n):
        for t in reversed(range(T)):
            v_next = LQ[j].max() if t == T - 1 else V[t + 1, j]
            if t == T - 1:
                G[t, j] = R[t, j] + gamma * ((1 - D[t, j]) * v_next + TR[t, j] * VF[t, j])
            else:
                G[t, j] = R[t, j] + gamma * ((1 - D[t, j]) * ((1 - lam) * v_next + lam * G[t + 1, j]) + TR[t, j] * VF[t, j])
    return G


@pytest.mark.parametrize("seed,gamma,lam", [(0, 0.99, 0.65), (1, 0.9, 0.3), (2, 0.99, 0.0), (3, 0.97, 1.0)])
def test_fixed_targets_match_brute_force_with_truncation(pqn, seed, gamma, lam):
    rng = np.random.default_rng(100 + seed)
    R, D, QV, LQ = random_rollout(rng)
    Df = D.astype(np.float32)
    TR = Df * (rng.random(D.shape) < 0.5)                # a random subset of the terminations are time-limit truncations
    VF = rng.normal(size=D.shape).astype(np.float32)
    bf = brute_force_qlambda(R.astype(np.float64), Df.astype(np.float64), TR.astype(np.float64), QV.astype(np.float64),
                             LQ.astype(np.float64), VF.astype(np.float64), gamma, lam)
    with flags(pqn, gamma=gamma, lam=lam):
        ours = np.asarray(pqn["targets_fixed"](jnp.asarray(R), jnp.asarray(Df), jnp.asarray(TR), jnp.asarray(QV),
                                               jnp.asarray(LQ.max(-1)), jnp.asarray(VF)))
    assert np.max(np.abs(bf - ours)) < 1e-4              # float32 vs float64 over a 32-step recursion


def test_fixed_without_truncation_equals_public_with_fix(pqn):
    R, D, QV, LQ = random_rollout(np.random.default_rng(5))
    Df = jnp.asarray(D, jnp.float32); lq = jnp.asarray(LQ.max(-1))
    with flags(pqn, buggy_target=False):
        pub = pqn["targets_public"](jnp.asarray(R), Df, jnp.asarray(QV), lq)
    fix = pqn["targets_fixed"](jnp.asarray(R), Df, jnp.zeros_like(Df), jnp.asarray(QV), lq, jnp.zeros_like(Df))
    assert float(jnp.max(jnp.abs(pub - fix))) < 1e-5


# =====================================================================================================================
# (c) Proposition 1 in its general form
#   buggy - fixed = g (1-lam) (g lam)^{T-2-t} [ (1-d_{T-1}) V(s_T) - V(s_{T-1}) ] * prod_{j=t}^{T-2} (1-d_j),  t <= T-2
# =====================================================================================================================
@pytest.mark.parametrize("with_terminations", [False, True])
@pytest.mark.parametrize("gamma,lam", [(0.99, 0.65), (0.95, 0.9), (0.8, 0.2)])
def test_proposition_1_closed_form(pqn, with_terminations, gamma, lam):
    rng = np.random.default_rng(7)
    R, D, QV, LQ = random_rollout(rng, n=16)
    if not with_terminations:
        D[:] = False
    Dj, Rj, QVj, lq = jnp.asarray(D, jnp.float32), jnp.asarray(R), jnp.asarray(QV), jnp.asarray(LQ.max(-1))
    with flags(pqn, gamma=gamma, lam=lam):
        with flags(pqn, buggy_target=True):
            gb = np.asarray(pqn["targets_public"](Rj, Dj, QVj, lq), np.float64)
        with flags(pqn, buggy_target=False):
            gf = np.asarray(pqn["targets_public"](Rj, Dj, QVj, lq), np.float64)
    T, n = R.shape
    bracket = (1 - D[-1]) * LQ.max(-1) - QV[-1].max(-1)
    pred = np.zeros((T, n))
    for t in range(T - 1):
        alive = np.prod(1 - D[t:T - 1].astype(np.float64), axis=0)       # product over j = t..T-2
        pred[t] = gamma * (1 - lam) * (gamma * lam) ** (T - 2 - t) * bracket * alive
    assert np.max(np.abs((gb - gf) - pred)) < 1e-5
    assert np.max(np.abs((gb - gf)[-1])) < 1e-6                           # step T-1 is never affected


# =====================================================================================================================
# (d) schedules
# =====================================================================================================================
def test_lr_schedule_values_equal_official(pqn):
    C = OFFICIAL
    nu_decay = int(C["TOTAL_TIMESTEPS_DECAY"] // C["NUM_STEPS"] // C["NUM_ENVS"])                    # L95-97
    official = optax.linear_schedule(init_value=C["LR"], end_value=1e-20,
                                     transition_steps=nu_decay * C["NUM_MINIBATCHES"] * C["NUM_EPOCHS"])   # L140-146
    G = pqn["GRAD_STEPS"]
    assert G == nu_decay * C["NUM_MINIBATCHES"] * C["NUM_EPOCHS"] == 156224
    for k in (0, G // 2, G - 1):
        assert float(pqn["lr_sched"](k)) == pytest.approx(float(official(k)), rel=1e-6)
    assert float(pqn["lr_sched"](0)) == pytest.approx(5e-4) and float(pqn["lr_sched"](G // 2)) == pytest.approx(2.5e-4, rel=1e-4)


def _set_schedule_count(state, k):
    """Replace the step counter of every optax ScaleByScheduleState inside a (nested) chain state."""
    if isinstance(state, optax.ScaleByScheduleState):
        return state._replace(count=jnp.asarray(k, state.count.dtype))
    if isinstance(state, tuple) and not hasattr(state, "_fields"):
        return tuple(_set_schedule_count(s, k) for s in state)
    return state


def test_optimizer_applies_the_scheduled_lr_at_given_gradient_step(pqn):
    """Official chain = clip_by_global_norm(10) -> radam(lr_scheduler) (L159-162); with identical state and gradient the
    two optimizers must produce the same update at gradient step 0, G/2 and G-1 (checks the schedule clock, not only the values)."""
    C = OFFICIAL; G = pqn["GRAD_STEPS"]
    sched = optax.linear_schedule(C["LR"], 1e-20, int(C["TOTAL_TIMESTEPS_DECAY"] // 32 // 128) * 32 * 2)
    ref = optax.chain(optax.clip_by_global_norm(C["MAX_GRAD_NORM"]), optax.radam(learning_rate=sched))
    params = pqn["net"].init(jax.random.PRNGKey(0), jnp.zeros((1, 10, 10, 6)))
    grads = jax.tree_util.tree_map(lambda x: jnp.asarray(np.random.default_rng(0).normal(size=x.shape), x.dtype), params)
    for k in (0, G // 2, G - 1):
        s_ours = _set_schedule_count(pqn["tx"].init(params), k); s_ref = _set_schedule_count(ref.init(params), k)
        u1, _ = pqn["tx"].update(grads, s_ours, params); u2, _ = ref.update(grads, s_ref, params)
        err = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree_util.tree_leaves(u1), jax.tree_util.tree_leaves(u2)))
        assert err < 1e-12 + 1e-6 * max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(u2))


def test_epsilon_schedule_vs_official(pqn):
    """Official: linear_schedule(1.0, 0.05, EPS_DECAY * NUM_UPDATES_DECAY) with a FLOAT length (L134-138), evaluated at
    n_updates. pqn_v5 uses int(0.1 * NUM_UPDATES) = 244 instead of 244.1: a tiny, unintended rounding difference
    (< 5e-4 in epsilon, confined to updates < 245). Documented in docs/REFERENCE_DIFF.md."""
    C = OFFICIAL; nu = int(C["TOTAL_TIMESTEPS_DECAY"] // 32 // 128)
    official = optax.linear_schedule(C["EPS_START"], C["EPS_FINISH"], C["EPS_DECAY"] * nu)
    assert pqn["NUM_UPDATES"] == nu == 2441
    diffs = {}
    for u in (0, int(0.1 * nu), nu // 2, nu - 1):         # 0, 10 %, 50 %, last update
        diffs[u] = abs(float(pqn["eps_sched"](u)) - float(official(u)))
    print("eps diffs", diffs)
    assert float(pqn["eps_sched"](0)) == 1.0 and diffs[0] == 0.0
    assert float(pqn["eps_sched"](nu // 2)) == pytest.approx(0.05) and diffs[nu // 2] < 1e-9
    assert max(diffs.values()) < 5e-4


# =====================================================================================================================
# (e) network: structure, shapes, function equality, initial norm
# =====================================================================================================================
class CNN(nn.Module):                                         # pqn_minatar.py L24-L51 (layer_norm branch)
    @nn.compact
    def __call__(self, x):
        normalize = lambda x: nn.LayerNorm()(x)
        x = nn.Conv(16, kernel_size=(3, 3), strides=1, padding="VALID", kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class OfficialQNetwork(nn.Module):                                      # pqn_minatar.py L54-L69 (norm_input=False branch)
    action_dim: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        x_dummy = nn.BatchNorm(use_running_average=not train)(x)       # L65 (unused output, adds unused params)
        x = x / 255.0                                                   # L66  <-- the "divide by 255" of Proposition 2
        x = CNN()(x)
        return nn.Dense(self.action_dim)(x)


def _official_to_ours(op):
    c = op["CNN_0"]
    return {"params": {"conv": c["Conv_0"], "ln0": c["LayerNorm_0"], "dense": c["Dense_0"], "ln1": c["LayerNorm_1"], "head": op["Dense_0"]}}


def _obs(rng, n=16):
    return jnp.asarray(rng.random((n, 10, 10, 6)) < 0.15)               # binary MinAtar-like observation (bool as in gymnax)


def test_param_tree_shapes_equal_official(pqn):
    net_o = OfficialQNetwork(4)
    vo = net_o.init(jax.random.PRNGKey(0), jnp.zeros((1, 10, 10, 6)), train=False)
    mapped = _official_to_ours(vo["params"])
    ours = pqn["net"].init(jax.random.PRNGKey(0), jnp.zeros((1, 10, 10, 6)))
    sh = lambda t: jax.tree_util.tree_map(lambda x: tuple(x.shape), t)
    assert sh(mapped) == sh(ours)
    assert sh(ours)["params"]["conv"]["kernel"] == (3, 3, 6, 16) and sh(ours)["params"]["dense"]["kernel"] == (1024, 128)
    assert sh(ours)["params"]["head"]["kernel"] == (128, 4)


@pytest.mark.parametrize("seed", range(3))
def test_network_function_equals_official(pqn, seed):
    """Official net on binary obs == ours with scale255=True on the same obs; and == ours with scale255=False fed obs/255
    (i.e. the official code divides the binary input by 255 *before* the first layer, which pqn_v5 reproduces only with --input_scale255)."""
    rng = np.random.default_rng(seed)
    net_o = OfficialQNetwork(4)
    vo = net_o.init(jax.random.PRNGKey(seed), jnp.zeros((1, 10, 10, 6)), train=False)
    p = jax.tree_util.tree_map(lambda x: x, vo["params"])
    # non-trivial LayerNorm scale/bias and head so the mapping of every parameter is exercised
    for ln in ("LayerNorm_0", "LayerNorm_1"):
        p["CNN_0"][ln] = {"scale": jnp.asarray(1 + 0.3 * rng.normal(size=p["CNN_0"][ln]["scale"].shape), jnp.float32),
                          "bias": jnp.asarray(0.3 * rng.normal(size=p["CNN_0"][ln]["bias"].shape), jnp.float32)}
    p["CNN_0"]["Dense_0"]["bias"] = jnp.asarray(0.1 * rng.normal(size=(128,)), jnp.float32)
    x = _obs(rng)
    q_off = net_o.apply({"params": p, "batch_stats": vo["batch_stats"]}, x, train=False)
    mine = _official_to_ours(p)
    net_pub = pqn["QNet"](4, True)
    net_fix = pqn["QNet"](4, False)
    assert float(jnp.max(jnp.abs(net_pub.apply(mine, x)[0] - q_off))) < 1e-4
    assert float(jnp.max(jnp.abs(net_fix.apply(mine, x.astype(jnp.float32) / 255.0)[0] - q_off))) < 1e-4


def test_he_init_dense_norm_squared(pqn):
    """E||W0||^2 = fan_in * fan_out * 2 / fan_in = 2 * 128 = 256 for the dense kernel (He normal); plan section 3.8."""
    vals = []
    for s in range(20):
        pr = pqn["net"].init(jax.random.PRNGKey(s), jnp.zeros((1, 10, 10, 6)))
        vals.append(float(jnp.sum(pr["params"]["dense"]["kernel"] ** 2) + jnp.sum(pr["params"]["dense"]["bias"] ** 2)))
        assert float(jnp.sum(pr["params"]["dense"]["bias"] ** 2)) == 0.0
    assert abs(np.mean(vals) - 256.0) / 256.0 < 0.05
    # and the conv layer: fan_in = 3*3*6 = 54, 864 weights  ->  E||W0||^2 = 864 * 2 / 54 = 32
    vc = [float(jnp.sum(pqn["net"].init(jax.random.PRNGKey(s), jnp.zeros((1, 10, 10, 6)))["params"]["conv"]["kernel"] ** 2)) for s in range(20)]
    assert abs(np.mean(vc) - 32.0) / 32.0 < 0.05


def test_default_hyperparameters_equal_official_yaml(pqn):
    """pqn_minatar.yaml (L3-L17): NUM_ENVS 128, NUM_STEPS 32, MINIBATCHES 32, EPOCHS 2, LR 5e-4, GAMMA .99, LAMBDA .65,
    EPS .1 decay ratio, clip 10; the update count and gradient-step count follow."""
    a = pqn["a"]
    assert (a.num_envs, a.num_steps, a.minibatches, a.epochs) == (128, 32, 32, 2)
    assert (a.lr, a.gamma, a.lam, a.eps_decay) == (5e-4, 0.99, 0.65, 0.1)
    assert a.steps == 1e7
    assert pqn["NUM_UPDATES"] == 2441 and pqn["STEPS_PER_UPDATE"] == 4096
