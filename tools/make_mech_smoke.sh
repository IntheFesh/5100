#!/usr/bin/env bash
# Tiny CPU runs (1 seed, 65 536 steps) that contain hist.ang_cum, for testing mechanism.py end to end.
# E=2 and E=8, each unconstrained / NP / weight decay 0.1.  Usage: tools/make_mech_smoke.sh OUTDIR
set -euo pipefail
OUT=${1:?output dir}; mkdir -p "$OUT"; cd "$(dirname "$0")/.."
BASE="--steps 65536 --num_envs 32 --minibatches 8 --eval_every 8192 --eval_episodes 2 --eval_cap 100 --sticky 0.1 --max_steps_in_episode 10000 --bootstrap_timeout --seeds 1 --seed0 100"
for E in 2 8; do
  python -W ignore pqn_v5.py $BASE --epochs $E --out "$OUT/E${E}_none__x.json" > "$OUT/E${E}_none.log" 2>&1 &
  python -W ignore pqn_v5.py $BASE --epochs $E --norm_ctrl np --out "$OUT/E${E}_np__x.json" > "$OUT/E${E}_np.log" 2>&1 &
  python -W ignore pqn_v5.py $BASE --epochs $E --norm_ctrl wd --wd 0.1 --out "$OUT/E${E}_wd0.1__x.json" > "$OUT/E${E}_wd0.1.log" 2>&1 &
  wait
done
ls "$OUT"/*.json
