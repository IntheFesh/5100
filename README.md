# DRL v5.1 code package

See CHANGELOG.md for the fixes made after the independent code review.

PQN (Gallici et al., ICLR 2025) for MinAtar via gymnax, rewritten to make the v5 experiments one flag each.
Defaults mirror purejaxql `pqn_minatar.py` (commit 47af6d7) **with the two fixes applied**.

## What changed in v5.1 (after the CPU pilot)
The public evaluation setting saturates: in gymnax SpaceInvaders (no sticky actions, 1000-step cap) every greedy
episode of every pilot model hit the cap, and all evaluation episodes of a model were identical (deterministic
dynamics + greedy policy). v5.1 therefore:
- trains and evaluates with sticky actions p=0.1 (MinAtar default) and a 10000-step cap (`sweep.py --protocol`);
- logs **two** evaluation protocols at every eval point: `greedy_return` (public: deterministic, cap 1000) and
  `robust_return` (sticky 0.1, cap 10000; primary metric);
- logs the exact per-step angular travel of the dense layer (`ang_cum`) to test the angle predictions;
- uses theory-derived weight-decay strengths {0.1, 0.3} instead of {1e-4, 1e-3};
- adds learning-rate-matched NP arms; saves intermediate snapshots for a learning-progression demo.

| file | purpose |
|---|---|
| `pqn_v5.py` | training. Switches: `--input_scale255`, `--buggy_target` (reproduce the public code), `--norm_ctrl none/np/radial/wd`, `--wd`, `--epochs`, `--lr`, `--bootstrap_timeout`, `--sticky`, `--max_steps_in_episode`, `--eval_cap`, `--eval_sticky`, `--save_params`, `--ckpt_at` |
| `test_v5.py` | unit tests: fixed targets vs brute force, bug vs closed form, optimizer == optax.radam, radial projection orthogonality, projection keeps norm and function |
| `sweep.py` | resumable stages P0 (protocol + correction ladder), P1 (19-config screening), P2 (confirmation), P3 (robustness); one JSON per finished config |
| `analyze.py` | per-seed AUC of a chosen metric (`--metric robust_return/greedy_return/train_return`), Student-t 95% CIs, paired differences with exact sign-flip p-values + Holm (suppressed below 5 seeds), `--filter`, `--iqm`, `--json_out`, curves |
| `tests/test_stats.py` | calibration of the statistics under H0 (false-positive rate, CI coverage) |
| `render_demo.py` | 2-3 panel SpaceInvaders video; sticky 0.1; same env seeds; shows the **median full-episode score** of 9 greedy episodes; the game is never truncated, only the video; `--seed_indices` picks the training seed per panel |
| `check_protocol.py`, `check_cap.py` | re-evaluate any saved model under the public and the robust protocol |
| `DRL_v5_colab.ipynb` | Colab Pro notebook running all of the above, results to Google Drive |
| `pqn.py`, `plot_pilot.py`, `plot_pilot2.py`, `norm_slope.py` | the CPU pilot exactly as run (2 seeds, 2M steps), its plots and norm-slope fit |
| `pilot_results/` | pilot logs, JSONs, `proto_*.txt`, and the two pilot demo videos |
| `checks/` | theory checks: input-scale equivalence, momentum norm-growth split, AR(1) growth factor, preconditioner effect, Q(λ) bug closed form, v4 identities |

Quick start (any machine with JAX):

```bash
pip install "jax==0.4.38" flax==0.10.2 optax==0.2.4 gymnax==0.0.6 six   # add [cuda12] on GPU
python test_v5.py
P="--sticky 0.1 --max_steps_in_episode 10000 --bootstrap_timeout"
python pqn_v5.py --steps 1e6 --seeds 3 $P --out run.json
python pqn_v5.py --steps 1e6 --seeds 3 $P --epochs 8 --norm_ctrl np --lr 2.5e-4 --out run_np.json
python analyze.py <dir_with_jsons> --metric robust_return --ref <config_name>
```

Notes
- Keep `--seeds` ≤ 10 per process (purejaxql issue #9 reports failures with >20 vmapped seeds).
- Training seed i uses `PRNGKey(seed0+i)`; streams depend on the JAX version (recorded in each JSON) — do not switch runtime within a stage.
- Sticky actions follow MinAtar's `Environment.act`: with prob p the previously *executed* action is repeated; the agent stores the action it *chose*.
- `--bootstrap_timeout` bootstraps through time-limit truncation (Pardo et al., ICML 2018). `--buggy_target` cannot be combined with it (the public target has no truncation handling); with a 10000-step cap truncations are rare and `trunc_frac` is logged. In P3, Freeway's limit is a game rule and is treated as termination in every arm.
- `--norm_ctrl` acts only on the LayerNorm-preceded layers (`conv`, `dense`, kernel and bias jointly). The output layer is never constrained.
- Freeway's 2500-step limit is part of the game's rules and is kept in P3.
