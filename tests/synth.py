"""Synthetic stage data with the SAME JSON schema as pqn_v5.py outputs - used only to test analysis scripts.

NOTHING produced here is a result. The files are written to temporary directories, never to results/.
Dynamics of the dense-layer norm follow the plan's model per gradient step k (eta_k = eta0 (1 - k/G)):
    y_{k+1} = y_k + s eta_k^2 - 2 eta_k lam y_k          y = ||W||^2   (lam = 0 for 'none'; NP keeps y = ||W0||^2)
    angle_k = c * eta_k / sqrt(y_k)                       (scale-invariant layer: angular step ~ eta / ||W||)
Learning curves: r(t) = q * R * (1 - exp(-3 t / T)) + noise, so the 'effect' of a configuration is the multiplier q."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

VERSIONS = {"jax": "0.4.38", "jaxlib": "0.4.38", "flax": "0.10.2", "optax": "0.2.4", "gymnax": "0.0.6",
            "jax_threefry_partitionable": False, "backend": "cpu"}


def make_run(path, *, E=2, ctrl="none", wd=0.0, lr=5e-4, seed_ids=(100, 101, 102), steps=2e6, M=32, n_envs=128, T=32, s=2.0e5,
             c_ang=1.0, n_points=20, q=1.0, R=100.0, noise=0.03, rng_seed=0, public=False, protocol_gymnax=False, w0sq=256.0,
             s_wd_factor=1.0, ang=True, env="SpaceInvaders-MinAtar", bootstrap_timeout=True, trunc_frac=0.0, public_cap_curve=True,
             extra_args=None, ckpts=None, ang_tracks_norm=True, growth_decay=0.0, args_full=None, seed_sd=0.0):
    rng = np.random.default_rng(rng_seed)
    S_ = len(seed_ids)
    per_update = n_envs * T
    NU = int(steps // per_update); G = NU * E * M
    eval_u = [int(round(NU * (i + 1) / n_points)) for i in range(n_points)]; eval_u[-1] = NU
    y = w0sq * (1 + 0.003 * rng.standard_normal(S_)); y0 = y.copy()
    wn, an, ang_cum = [], [], np.zeros(S_)
    k = 0
    for u in eval_u:                                                           # vectorised over the gradient steps of one eval interval
        k_end = u * E * M
        kk = np.arange(k, k_end)
        eta = (lr * (1.0 - kk / G))[:, None]                                   # (L, 1)
        n1 = rng.standard_normal((len(kk), S_)); n2 = rng.standard_normal((len(kk), S_))
        if ctrl == "np":
            y_path = np.broadcast_to(y0, (len(kk), S_))
        else:
            shrink = np.exp(-growth_decay * (kk / G))[:, None]                   # growth_decay > 0: growth stops being ~ eta^2 (negative control)
            growth = s * shrink * (s_wd_factor if ctrl == "wd" else 1.0) * eta ** 2 * (1 + noise * n1)
            if ctrl == "wd":                                                     # y_{k+1} = (1 - 2 eta lam) y_k + growth_k, solved with cumprod/cumsum
                A = np.cumprod(1.0 - 2.0 * eta * wd, axis=0)
                y_path = A * (y + np.cumsum(growth / A, axis=0))
            else:
                y_path = y + np.cumsum(growth, axis=0)
            y = y_path[-1]
        norm_for_angle = y_path if (ctrl != "np" and ang_tracks_norm) else y0     # ang_tracks_norm=False: angle ignores Lambda (negative control)
        ang_cum = ang_cum + np.sum(c_ang * eta / np.sqrt(norm_for_angle) * (1 + noise * n2), axis=0)
        k = k_end
        wn.append(np.sqrt(y_path[-1] if ctrl != "np" else y0).tolist()); an.append(ang_cum.copy().tolist())
    env_steps = [u * per_update for u in eval_u]
    frac = np.asarray(env_steps) / (NU * per_update)
    curve = lambda: q * R * (1 - np.exp(-3 * frac))[:, None] * (1 + noise * rng.standard_normal((n_points, S_)))
    seed_fac = 1 + seed_sd * rng.standard_normal(S_)                          # between-seed spread of the whole curve
    train_ret = curve() * seed_fac; rob = curve() * seed_fac
    pub = np.minimum(R * 1.5 * (1 - np.exp(-6 * frac))[:, None] * (1 + 0.5 * noise * rng.standard_normal((n_points, S_))), 150.0)
    capped = np.clip((frac[:, None] * 1.6 - 0.3), 0, 1) * np.ones((1, S_)) if (public_cap_curve and (protocol_gymnax or public)) else np.zeros((n_points, S_))
    args = dict(env=env, steps=steps, num_envs=n_envs, num_steps=T, epochs=E, minibatches=M, lr=lr, gamma=0.99, lam=0.65, eps_decay=0.1,
                seeds=S_, seed0=int(seed_ids[0]), input_scale255=bool(public), buggy_target=bool(public), norm_ctrl=ctrl, wd=wd,
                bootstrap_timeout=bool(bootstrap_timeout and not public), max_steps_in_episode=1000 if protocol_gymnax else 10000,
                sticky=0.0 if protocol_gymnax else 0.1, eval_cap=10000, eval_sticky=0.1, eval_every=steps / n_points, eval_episodes=64,
                save_params=False, ckpt_at="", out=os.path.basename(path))
    args.update(extra_args or {})
    if args_full is not None:                                                  # exact args of a sweep.py config (effective_args.effective_rows)
        args = dict(args_full, out=os.path.basename(path))
    hist = {"update": eval_u, "env_steps": env_steps, "train_return": train_ret.tolist(), "wn_dense": wn,
            "trunc_frac": (np.full((n_points, S_), trunc_frac)).tolist(), "loss": np.zeros((n_points, S_)).tolist(),
            "chunk_seconds": [1.0] * n_points}
    if ang:
        hist["ang_cum"] = an
    evals = [{"env_steps": env_steps[i], "greedy_return": pub[i].tolist(), "capped_frac": capped[i].tolist(),
              "public_len": (np.minimum(1000, 200 + 800 * capped[i])).tolist(), "robust_return": rob[i].tolist(),
              "robust_capped_frac": np.zeros(S_).tolist(), "robust_len": (300 + 0 * rob[i]).tolist()} for i in range(n_points)]
    d = {"args": args, "seed_ids": [int(i) for i in seed_ids], "num_updates": NU, "grad_steps": G, "max_steps_in_episode": args["max_steps_in_episode"],
         "public_cap": 1000, "ckpts": ckpts or [], "versions": VERSIONS, "wall_seconds": 60.0 * E, "hist": hist, "evals": evals}
    json.dump(d, open(path, "w"))
    return d


def make_stage(outdir, runs):
    """runs: {config_name: kwargs for make_run}; files are named <name>__<hash>.json like sweep.py does."""
    os.makedirs(outdir, exist_ok=True)
    for i, (name, kw) in enumerate(runs.items()):
        make_run(os.path.join(outdir, f"{name}__{i:010x}.json"), rng_seed=i, **kw)
    return outdir


def stage_from_sweep(outdir, stage, *, best_E=None, best_ctrl=None, params=None, steps=None):
    """Synthetic stage whose JSON args are EXACTLY those of the real sweep.py configs of `stage` (names, flags, seeds); `params(name, args)`
    returns the make_run keyword overrides (effects). Returns the list of config names."""
    from effective_args import effective_rows
    rows = effective_rows(stage, best_E, best_ctrl)
    runs = {}
    for name, flags, a in rows:
        seeds = tuple(range(a["seed0"], a["seed0"] + a["seeds"]))
        kw = dict(E=a["epochs"], ctrl=a["norm_ctrl"], wd=a["wd"], lr=a["lr"], seed_ids=seeds, steps=steps or a["steps"], M=a["minibatches"],
                  n_envs=a["num_envs"], T=a["num_steps"], public=bool(a["input_scale255"] and a["buggy_target"]),
                  protocol_gymnax=(a["max_steps_in_episode"] == 1000), env=a["env"], bootstrap_timeout=a["bootstrap_timeout"], args_full=dict(a))
        kw.update((params or (lambda n, a_: {}))(name, a))
        runs[name] = kw
    make_stage(outdir, runs)
    return list(runs)
