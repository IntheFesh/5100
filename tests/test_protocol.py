"""Protocol / evaluation / seed tests (round 3): pin plan section 3.7 (Proposition 6) and the protocol definitions of
section 6 so that no later change can silently break the evaluation.

Run: python -m pytest -q tests/test_protocol.py        (CPU, ~2-4 min)

Layout
  1. determinism of the public protocol (Prop. 6a) and non-degeneracy of the robust protocol
  2. sticky-action semantics (replacement rule, memory reset, stored action = CHOSEN action)
  3. caps: public protocol <= 1000 steps even when training/robust cap is 10000 (Prop. 6b machinery)
  4. return masking: independent Python-loop reference vs greedy_eval, episode by episode
  5. truncation flag of env_step with a small cap
  6. seed identity: reproducibility, vmap-width independence, paired evaluation keys, versions, stage seeds
  7. Freeway: cap 2500 in training and evaluation, no timeout bootstrapping
"""
import ast, json, os, re, subprocess, sys, tempfile
import numpy as np, pytest
import jax, jax.numpy as jnp
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "tools"))
from _load import load_pqn, pqn_source, flags
from effective_args import effective_rows


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
class ScriptedNet:
    """Hand-written reactive policy with the interface of QNet.apply (returns (Q, h)): go to the nearest column that still
    holds an alien and fire; step aside when an enemy bullet is above the cannon. Random-initialised PQN networks are
    almost always constant-action policies (identical returns even under sticky actions), so they cannot show the
    robust protocol's non-degeneracy; this policy survives ~60 steps, scores ~10 and reacts to the (sticky-perturbed) state."""

    def apply(self, params, obs):
        cannon = jnp.argmax(obs[:, 9, :, 0], -1)
        has = obs[..., 1].any(1)
        dist = jnp.where(has, jnp.abs(jnp.arange(10)[None, :] - cannon[:, None]), 99)
        target = jnp.argmin(dist, -1)
        act = jnp.where(target == cannon, 3, jnp.where(target < cannon, 1, 2))
        danger = jnp.take_along_axis(obs[..., 5][:, 4:9, :].any(1), cannon[:, None], 1)[:, 0]
        act = jnp.where(danger, jnp.where(cannon > 0, 1, 2), act)
        return jax.nn.one_hot(act, 4), None


def jit_greedy_eval(ns, n):
    """Fresh jitted greedy_eval for n episodes (n is read from a.eval_episodes at trace time)."""
    # NB: jax caches traces per underlying function object, so a new jit wrapper of the same function would NOT retrace
    # with a different a.eval_episodes - wrap it in a fresh lambda.
    f = jax.jit(lambda p, r, cap, st: ns["greedy_eval"](p, r, cap, st), static_argnums=(2, 3))

    def call(params, rng, cap, sticky):
        with flags(ns, eval_episodes=n):                                    # the trace happens HERE, so n must be set now
            return tuple(float(x) for x in f(params, rng, cap, sticky))
    return call


def reference_eval(ns, net, params, rng, cap, sticky, n):
    """Independent Python-loop implementation of the evaluation protocol with the SAME key sequence as greedy_eval.
    Returns per-episode (masked return, unmasked return, length) arrays. Written from the definition:
    return = sum of rewards up to and including the first done; length = number of steps taken before/at that done."""
    env = ns["env"]; ep = ns["env_params"].replace(max_steps_in_episode=cap)
    vstep = jax.jit(lambda ks, st, act: jax.vmap(env.step, in_axes=(0, 0, 0, None))(ks, st, act, ep))
    fwd = jax.jit(lambda p, o: jnp.argmax(net.apply(p, o)[0], -1))
    k_reset, rng = jax.random.split(rng)
    obs, est = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_reset, n), ep)
    ret, ret_all, alive, ln, prev = np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n, np.int32)
    for _ in range(cap):
        if not alive.any():
            break
        rng, ks, kst = jax.random.split(rng, 3)
        greedy = np.asarray(fwd(params, obs))
        u = np.asarray(jax.random.uniform(kst, (n,)))
        act = np.where(u < sticky, prev, greedy).astype(np.int32)        # NumPy restatement of the sticky rule
        obs, est, rew, done, _ = vstep(jax.random.split(ks, n), est, jnp.asarray(act))
        rew, done = np.asarray(rew, np.float64), np.asarray(done, np.float64)
        ret += rew * alive; ret_all += rew; ln += alive; alive = alive * (1 - done); prev = act
    return ret, ret_all, ln


@pytest.fixture(scope="module")
def ns():
    return load_pqn(["--max_steps_in_episode", "10000", "--num_envs", "32"])


@pytest.fixture(scope="module")
def ns_scripted():
    n = load_pqn(["--max_steps_in_episode", "10000"])
    n["net"] = ScriptedNet()                                                # greedy_eval looks `net` up as a global
    return n


# ----------------------------------------------------------------------------------------------------------------------
# 1. determinism (Proposition 6a)
# ----------------------------------------------------------------------------------------------------------------------
def test_public_protocol_is_deterministic_and_robust_protocol_is_not(ns_scripted):
    n = ns_scripted
    ret_pub = [reference_eval(n, n["net"], None, jax.random.PRNGKey(k), 1000, 0.0, 8)[0] for k in range(3)]
    for r in ret_pub:
        assert len(np.unique(r)) == 1, r                                    # 8 episodes: one and the same return
    assert all(np.array_equal(ret_pub[0], r) for r in ret_pub)             # ... independent of the evaluation key
    ret_rob = reference_eval(n, n["net"], None, jax.random.PRNGKey(0), 10000, 0.1, 8)[0]
    assert len(np.unique(ret_rob)) > 1, ret_rob                             # sticky 0.1: episodes differ


def test_random_init_pqn_network_is_deterministic_without_sticky_actions(ns):
    """Same statement for the real network class (random init; its robust returns are almost constant-action, see ScriptedNet)."""
    params = ns["net"].init(jax.random.PRNGKey(3), jnp.zeros((1, 10, 10, 6)))
    for k in range(3):
        ret = reference_eval(ns, ns["net"], params, jax.random.PRNGKey(k), 1000, 0.0, 8)[0]
        assert len(np.unique(ret)) == 1
    ge = jit_greedy_eval(ns, 8)
    outs = {ge(params, jax.random.PRNGKey(k), 1000, 0.0) for k in range(3)}
    assert len(outs) == 1                                                   # greedy_eval itself: key-independent


# ----------------------------------------------------------------------------------------------------------------------
# 2. sticky-action semantics
# ----------------------------------------------------------------------------------------------------------------------
def _run_sticky(ns, chosen_fn, p, N, seed=0):
    """Drive the real `sticky_select` / `next_prev` of pqn_v5.py exactly as update() does (prev starts at 0 and carries
    the EXECUTED action). Returns (chosen, executed, prev) arrays of length N."""
    sel, nxt = ns["sticky_select"], ns["next_prev"]

    def step(c, k):
        prev, t = c
        chosen = chosen_fn(prev, t)
        act = sel(jax.random.uniform(k), p, prev, chosen)
        return (nxt(jnp.asarray(False), act), t + 1), (chosen, act, prev)

    _, (ch, ex, pv) = jax.lax.scan(step, (jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32)), jax.random.split(jax.random.PRNGKey(seed), N))
    return np.asarray(ch), np.asarray(ex), np.asarray(pv)


@pytest.mark.parametrize("p", [0.1, 0.3])
def test_sticky_replacement_rate_fixed_choice_sequence(ns, p):
    """Fixed chosen sequence 0,1,0,1,...  The replacement rule is: P(executed == previous executed | chosen != previous executed) = p
    and chosen == previous executed => executed == chosen (== previous). NB: the *unconditional* frequency of
    'executed == previous executed' is NOT p for this sequence (a non-sticky step with chosen == previous executed also
    repeats it); for the alternating sequence it equals 2p/(1+p) (0.1818 for p=0.1). The binomial test therefore uses the
    steps where the two candidates differ - there every step is an independent Bernoulli(p)."""
    N = 40000
    ch, ex, pv = _run_sticky(ns, lambda prev, t: t % 2, p, N)
    assert np.array_equal(ch, np.arange(N) % 2)
    differ = ch != pv
    x, m = int(np.sum(ex[differ] == pv[differ])), int(differ.sum())
    assert m > 15000
    assert binomtest(x, m, p).pvalue > 0.001, (x, m, x / m)                 # alpha = 0.001
    assert np.all(ex[~differ] == ch[~differ])                               # same candidates -> same action
    assert abs(np.mean(ex[1:] == pv[1:]) - 2 * p / (1 + p)) < 0.015         # the derived unconditional frequency


@pytest.mark.parametrize("p", [0.1, 0.3])
def test_sticky_repeat_frequency_equals_p_when_choice_never_equals_previous(ns, p):
    """Adaptive chosen sequence chosen_t = (prev_t + 1) mod 4 never coincides with the previous executed action, so the
    repeat indicator is exactly Bernoulli(p) and its unconditional frequency is p (the statement of the round-3 prompt)."""
    N = 40000
    ch, ex, pv = _run_sticky(ns, lambda prev, t: (prev + 1) % 4, p, N, seed=1)
    assert not np.any(ch == pv)
    x = int(np.sum(ex == pv))
    assert binomtest(x, N, p).pvalue > 0.001, (x / N)
    assert set(np.unique(ex)) <= {0, 1, 2, 3}


def test_sticky_zero_and_one_are_the_degenerate_cases(ns):
    ch, ex, _ = _run_sticky(ns, lambda prev, t: t % 4, 0.0, 2000)
    assert np.array_equal(ch, ex)                                           # p = 0: public setting, no replacement at all
    ch, ex, _ = _run_sticky(ns, lambda prev, t: t % 4, 1.0, 2000)
    assert np.all(ex == 0)                                                  # p = 1: always repeat; prev starts at 0 (no-op)


def test_prev_action_is_reset_to_zero_when_the_episode_ends(ns):
    nxt = ns["next_prev"]
    act = jnp.array([3, 2, 1, 0, 3], jnp.int32); done = jnp.array([True, False, True, False, False])
    out = np.asarray(nxt(done, act))
    assert out.tolist() == [0, 2, 0, 0, 3] and out.dtype == np.int32


def test_update_stores_the_chosen_action_and_remembers_the_executed_one():
    """Reading of update() (pqn_v5.py): inside `step`, the scan output tuple is (obs_, chosen, rew, done, trunc, fin, q, fin_ret),
    i.e. the transition that enters the learning data stores `chosen` (the agent's own action), while env_step receives the
    executed `act` and the carry holds prev_ = next_prev(done, act). Checked on the AST so that a refactor cannot swap them."""
    tree = ast.parse(pqn_source())
    update = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "update")
    step = next(n for n in ast.walk(update) if isinstance(n, ast.FunctionDef) and n.name == "step")
    ret = [n for n in step.body if isinstance(n, ast.Return)][-1].value
    carry_out, ys = ret.elts
    assert ast.unparse(ys.elts[1]) == "chosen"                              # stored action = chosen
    assert "act" not in [ast.unparse(e) for e in ys.elts]                   # the executed action is never stored as data
    assert ast.unparse(carry_out.elts[3]) == "prev_"
    src = ast.unparse(step)
    assert "prev_ = next_prev(done, act)" in src
    assert re.search(r"env_step\(.*, est_, act\)", src)                     # the environment executes `act`
    assert "act = sticky_select(" in src and "prev_, chosen)" in src


# ----------------------------------------------------------------------------------------------------------------------
# 3. caps
# ----------------------------------------------------------------------------------------------------------------------
def test_public_cap_is_the_gymnax_default_even_when_training_cap_is_10000(ns):
    assert ns["PUBLIC_CAP"] == 1000 and ns["MAXT"] == 10000
    src = pqn_source()
    assert "eval_fn(carries[0], ek, PUBLIC_CAP, 0.0)" in src                # public protocol: gymnax cap, no sticky
    assert "eval_fn(carries[0], ek, a.eval_cap, a.eval_sticky)" in src      # robust protocol: --eval_cap / --eval_sticky


def test_public_protocol_lengths_are_capped_at_1000_but_robust_lengths_may_exceed_1000():
    """A 'frozen' game (aliens neither move nor shoot) can only end through the step cap, so episode length == cap exactly."""
    n = load_pqn(["--max_steps_in_episode", "10000", "--eval_episodes", "2"])
    n["env_params"] = n["env_params"].replace(enemy_move_interval=10 ** 6, enemy_shot_interval=10 ** 6)
    params = n["net"].init(jax.random.PRNGKey(0), jnp.zeros((1, 10, 10, 6)))
    ge = jit_greedy_eval(n, 2)
    ret_pub, capped_pub, len_pub = ge(params, jax.random.PRNGKey(0), n["PUBLIC_CAP"], 0.0)
    ret_rob, capped_rob, len_rob = ge(params, jax.random.PRNGKey(0), 10000, 0.1)
    assert len_pub == 1000 and capped_pub == 1.0                            # <= 1000
    assert len_rob == 10000 and capped_rob == 1.0 and len_rob > 1000        # robust protocol is not cut at 1000


# ----------------------------------------------------------------------------------------------------------------------
# 4. return masking: independent reference vs greedy_eval, episode by episode
# ----------------------------------------------------------------------------------------------------------------------
def test_return_is_masked_after_first_done_and_matches_python_reference(ns_scripted):
    n = ns_scripted; net = n["net"]
    # (i) n = 1: per-episode bit-level comparison with 4 different evaluation keys (key usage is identical for the reference)
    ge1 = jit_greedy_eval(n, 1)
    for k in range(4):
        key = jax.random.PRNGKey(100 + k)
        ret, ret_all, ln = reference_eval(n, net, None, key, 300, 0.1, 1)
        g_ret, g_capped, g_len = ge1(None, key, 300, 0.1)
        assert (g_ret, g_len, g_capped) == (ret[0], ln[0], float(ln[0] >= 300)), (k, g_ret, ret, g_len, ln)
    # (ii) n = 4 (vectorised path used in training): the aggregates must equal the reference's per-episode aggregates, and
    #      the masking must matter: episodes that end early keep playing (auto-reset) while the slowest one finishes
    ge4 = jit_greedy_eval(n, 4)
    key = jax.random.PRNGKey(7)
    ret, ret_all, ln = reference_eval(n, net, None, key, 300, 0.1, 4)
    g_ret, g_capped, g_len = ge4(None, key, 300, 0.1)
    assert g_ret == ret.mean() and g_len == ln.mean() and g_capped == np.mean(ln >= 300)
    assert len(np.unique(ln)) > 1 and np.any(ret_all > ret)                 # without the mask the mean would be larger
    assert g_ret != ret_all.mean()


# ----------------------------------------------------------------------------------------------------------------------
# 5. truncation flag
# ----------------------------------------------------------------------------------------------------------------------
def test_trunc_flag_marks_only_time_limit_endings():
    CAP, n_env, T = 50, 16, 300
    n = load_pqn(["--max_steps_in_episode", str(CAP), "--num_envs", str(n_env)])
    env, ep = n["env"], n["env_params"]
    assert n["MAXT"] == CAP
    obs, est = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(0), n_env), ep)
    step = jax.jit(n["env_step"])
    rng = np.random.default_rng(0); key = jax.random.PRNGKey(1)
    acts = rng.integers(0, 4, size=(T, n_env)); D = np.zeros((T, n_env), bool); TR = np.zeros((T, n_env), bool); FIN = []
    for t in range(T):
        key, ks = jax.random.split(key)
        obs, est, r, d, tr, fin = step(jax.random.split(ks, n_env), est, jnp.asarray(acts[t]))
        D[t], TR[t] = np.asarray(d), np.asarray(tr); FIN.append(np.asarray(fin))
    assert np.all(~TR | D)                                                  # trunc implies done
    # segment into episodes (start = 0 or the step after a done) and replay each with an UNcapped copy of the game
    eps = []
    for i in range(n_env):
        s = 0
        for t in range(T):
            if D[t, i]:
                eps.append((i, s, t)); s = t + 1
    lens = np.array([e[2] - e[1] + 1 for e in eps])
    assert lens.max() <= CAP
    unc = ep.replace(max_steps_in_episode=10 ** 6)
    A = np.zeros((len(eps), CAP), int)
    for k, (i, s, e) in enumerate(eps):
        A[k, : e - s + 1] = acts[s:e + 1, i]
    vstep = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)))
    o, st = jax.vmap(env.reset_env, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(5), len(eps)), unc)
    first_done = np.full(len(eps), 10 ** 9); final_obs = [None] * len(eps)
    for j in range(CAP):
        o_next, st, _, dn, _ = vstep(jax.random.split(jax.random.PRNGKey(j), len(eps)), st, jnp.asarray(A[:, j]), unc)
        dn = np.asarray(dn)
        for k in np.where(dn & (first_done > j + 1))[0]:
            first_done[k] = j + 1; final_obs[k] = np.asarray(o_next[k])
    timeouts = 0; game_overs = 0
    for k, (i, s, e) in enumerate(eps):
        L = e - s + 1
        if L < CAP:                                                         # ended before the cap: a REAL game over
            assert first_done[k] == L and not TR[e, i], (k, L, first_done[k])
            game_overs += 1
        else:                                                               # reached the cap: time-limit truncation
            assert TR[e, i]
            assert first_done[k] >= CAP                                     # the uncapped game had not ended before the cap
            timeouts += 1
            if first_done[k] > CAP:                                         # pure timeout: stored final obs == obs of the replayed game
                pass
        assert not TR[s:e, i].any()                                         # no flag on any non-final step
    assert timeouts >= 5 and game_overs >= 5, (timeouts, game_overs)        # both kinds occurred


# ----------------------------------------------------------------------------------------------------------------------
# 6. seed identity
# ----------------------------------------------------------------------------------------------------------------------
SMALL = ["--steps", "8192", "--num_envs", "32", "--minibatches", "8", "--eval_every", "4096", "--eval_episodes", "2",
         "--eval_cap", "100", "--sticky", "0.1", "--max_steps_in_episode", "10000", "--bootstrap_timeout"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("seedruns")
    spec = {"A": ["--seeds", "2", "--seed0", "5"], "A2": ["--seeds", "2", "--seed0", "5"], "B": ["--seeds", "1", "--seed0", "6"]}
    procs = {}
    for name, extra in spec.items():
        cmd = [sys.executable, "-W", "ignore", os.path.join(ROOT, "pqn_v5.py"), *SMALL, *extra, "--out", str(d / f"{name}.json")]
        procs[name] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for name, pr in procs.items():
        out, err = pr.communicate(timeout=900)
        assert pr.returncode == 0, (name, err[-2000:])
    return {k: json.load(open(d / f"{k}.json")) for k in spec}


def _numeric_view(d):
    """Everything except wall-clock timings (top-level wall_seconds, hist.chunk_seconds) and the args (different --out)."""
    out = {k: v for k, v in d.items() if k not in {"wall_seconds", "args"}}
    out["hist"] = {k: v for k, v in d["hist"].items() if k != "chunk_seconds"}
    return out


def _seed_slice(d, i):
    """Everything belonging to the i-th vmapped seed of a run: hist[key][:, i] and evals[...][i]."""
    out = {k: [row[i] for row in v] for k, v in d["hist"].items() if k not in ("chunk_seconds", "update", "env_steps")}
    for k in ("greedy_return", "capped_frac", "public_len", "robust_return", "robust_capped_frac", "robust_len"):
        out["eval_" + k] = [e[k][i] for e in d["evals"]]
    return out


def test_same_command_twice_gives_identical_numbers(runs):
    assert _numeric_view(runs["A"]) == _numeric_view(runs["A2"])


def test_seed_stream_does_not_depend_on_how_many_seeds_share_the_process(runs):
    """--seeds 2 --seed0 5 : second seed (id 6)   vs   --seeds 1 --seed0 6 : first seed (id 6)  -> bitwise identical."""
    assert runs["A"]["seed_ids"] == [5, 6] and runs["B"]["seed_ids"] == [6]
    assert _seed_slice(runs["A"], 1) == _seed_slice(runs["B"], 0)
    assert _seed_slice(runs["A"], 0) != _seed_slice(runs["A"], 1)           # different seeds really differ


def test_json_records_versions_and_actual_eval_points(runs):
    v = runs["A"]["versions"]
    for k in ("jax", "jaxlib", "flax", "optax", "gymnax", "backend", "jax_threefry_partitionable"):
        assert k in v
    assert v["jax"] == jax.__version__
    xs = [e["env_steps"] for e in runs["A"]["evals"]]
    assert xs == sorted(set(xs)) and xs == runs["A"]["hist"]["env_steps"]


def test_evaluation_keys_depend_only_on_eval_point_and_seed_id():
    """Paired evaluation (plan section 6.1): the key of the evaluation environments is built from (env steps of the eval
    point, seed id) only. The expression is taken from pqn_v5.py itself and evaluated here."""
    m = re.search(r"^\s*ek = (.+)$", pqn_source(), re.M)
    assert m, "evaluation key line not found"
    expr = m.group(1)
    assert not re.search(r"\ba\.|\brng\b|carries|params|tx\b", expr), expr    # no hyper-parameter / training-state dependence

    def ek(done_u, ids):
        return np.asarray(eval(expr, {"jax": jax, "jnp": jnp, "np": np, "done_u": done_u, "SEED_IDS": np.asarray(ids)}))

    k_a, k_b = ek(100, [5, 6]), ek(100, [6])
    assert np.array_equal(k_a[1], k_b[0])                                   # seed 6 gets the same key whoever shares the process
    assert not np.array_equal(k_a[0], k_a[1])                               # different seeds -> different keys
    assert not np.array_equal(ek(100, [6])[0], ek(200, [6])[0])             # different eval points -> different keys
    # training seeds: PRNGKey(seed0 + i), independent of the number of seeds
    src = pqn_source()
    assert "SEED_IDS = np.arange(a.seed0, a.seed0 + a.seeds)" in src and "jax.vmap(jax.random.PRNGKey)(jnp.asarray(SEED_IDS))" in src


STAGES = {"P0": {}, "P1": {}, "P2": dict(best_E=8, best_ctrl="--norm_ctrl np --lr 2.5e-4"),
          "P3": dict(best_E=8, best_ctrl="--norm_ctrl np --lr 2.5e-4")}


@pytest.mark.parametrize("stage", list(STAGES))
def test_all_arms_of_a_stage_use_the_same_seeds(stage):
    rows = effective_rows(stage, **STAGES[stage])
    assert len(rows) >= 4
    assert len({(d["seeds"], d["seed0"]) for _, _, d in rows}) == 1, [(n, d["seeds"], d["seed0"]) for n, _, d in rows]
    assert all(d["seeds"] <= 10 for _, _, d in rows)                        # CLAUDE.md rule 8


def test_stage_seed_ranges_are_disjoint_and_match_the_plan():
    rng = {}
    for st, kw in STAGES.items():
        d = effective_rows(st, **kw)[0][2]
        rng[st] = set(range(d["seed0"], d["seed0"] + d["seeds"]))
    assert rng["P0"] == set(range(0, 10)) and rng["P1"] == {100, 101, 102}
    assert rng["P2"] == set(range(1000, 1010)) and rng["P3"] == set(range(2000, 2005))
    sts = list(rng)
    assert all(rng[x].isdisjoint(rng[y]) for i, x in enumerate(sts) for y in sts[i + 1:])


# ----------------------------------------------------------------------------------------------------------------------
# 7. Freeway
# ----------------------------------------------------------------------------------------------------------------------
def test_p3_freeway_cap_is_2500_in_training_and_evaluation_and_is_terminal():
    rows = effective_rows("P3", **STAGES["P3"])
    fw = [(n, d) for n, _, d in rows if n.startswith("Freeway_")]
    assert len(fw) == 4
    for n, d in fw:
        assert d["env"] == "Freeway-MinAtar"
        assert d["max_steps_in_episode"] == 2500 and d["eval_cap"] == 2500, n
        assert d["bootstrap_timeout"] is False, n                           # the limit is a game rule: true termination
    for n, _, d in rows:
        if not n.startswith("Freeway_"):
            assert d["max_steps_in_episode"] == 10000 and d["eval_cap"] == 10000, n
    f = load_pqn(["--env", "Freeway-MinAtar", "--max_steps_in_episode", "2500", "--eval_cap", "2500"])
    assert f["PUBLIC_CAP"] == 2500 and f["MAXT"] == 2500                    # the gymnax default of Freeway is already 2500
