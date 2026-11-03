# CHANGELOG

## v5.1-review (2026-11-03) — fixes from an independent code review, before any GPU run
Behaviour changes (all GPU stages start from this version; the CPU pilot used the earlier code):
- `--buggy_target` now replicates purejaxql exactly: the recursion is seeded with `last_q * (1 - done[-1])`
  (pqn_minatar.py L252). Before, termination at the last rollout step bootstrapped step T-2 from V(reset obs).
  Proposition 1 in the plan is stated in its general form; `test_v5.py` 3/3c test it against a literal transcription.
- Seed identity: training seed i uses `PRNGKey(seed0 + i)` (was `split(PRNGKey(seed0), seeds)[i]`, which depended
  on the number of seeds). Eval env keys = `fold_in(PRNGKey(10000 + update), seed_id)` -> paired across configs.
  Output JSON records `seed_ids` and `versions` (jax/jaxlib/flax/optax/gymnax, threefry mode, backend).
- Snapshots (`--ckpt_at`) are saved at the exact update count (were one eval chunk late: "1M" was 1.5M);
  the JSON `ckpts` field records the actual env steps. Snapshots add no eval and no history point.
- The last chunk absorbs a short remainder (10M steps: 2441 updates no longer produce two evals 1 update apart).
- Eval: separate keys for resets and the loop; `public_len` logged; epsilon schedule length >= 1.
- sweep.py: P0 `fix_target_only` no longer turns on `--bootstrap_timeout` (each single-fix rung differs from
  `public` in one thing); Freeway (P3) keeps its 2500-step limit as a true termination (no timeout bootstrap);
  P2/P3 require `--best_E/--best_ctrl` and put the control in the config name; a second run with the same name and
  different flags is refused; params are renamed before the JSON (crash-safe); per-run `.log`; exit code 1 on failure.
- analyze.py: Student-t CIs and exact sign-flip permutation p-values (+ Holm) replace the percentile bootstrap,
  which was anti-conservative at n = 5-10 (9 % false positives at nominal 5 %, n = 10); `--filter`, `--iqm`,
  `--json_out`; refuses duplicate config names and unknown `--ref`. Calibration: `tests/test_stats.py`.
- render_demo.py: the final frame is the real end-of-game board (step_env, no auto-reset); per-panel
  `--seed_indices`; one compiled forward per network.
- Notebook: demo cells match only final params (asserting a unique match), pick each config's median-AUC training
  seed, P3 analysed per game; warning not to switch JAX runtime within a stage.
- Pilot demo videos re-rendered: the snapshots labelled 0.5M/1M were really 1.0M/1.5M steps.

## Test/infrastructure changes after v5.1-review (no change to training or evaluation behaviour)
- tests/test_stats.py: body moved into `run()` and a pytest entry point added; the file previously executed at import
  time and ended with `raise SystemExit`, so `python -m pytest -q tests/` aborted with INTERNALERROR. `python tests/test_stats.py` behaves as before.
- New: tests/_load.py + tests/conftest.py (load pqn_v5.py definitions without training), tests/test_reference.py
  (independent audit against purejaxql 47af6d7), docs/REFERENCE_DIFF.md, tools/effective_args.py, docs/ENV.md.
