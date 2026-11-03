"""Pre-registered selection rules for the demonstration videos (round 9). Pure functions, unit-tested (tests/test_demo.py).

Training seed (per config): the seed whose robust AUC is the MEDIAN of the config's confirmation seeds; with an even number of seeds (10) the one
among the two middle seeds (5th and 6th smallest AUC) that is closer to the mean AUC (ties -> the lower one).  The index returned is the position
on the vmap axis, i.e. the order of the JSON's `seed_ids` - exactly what `render_demo.py --seed_indices` expects.
Episode: decided inside render_demo.py - the same K = 9 environment seeds for every model, sticky 0.1, the game runs to its cap and the episode whose
FULL-episode score is the median of the 9 is shown (never the best); only the video is cut at --max_steps.
The same function is defined in the Colab notebook (section 7); tests/test_demo.py checks that both give the same answer."""
import glob, json, os
import numpy as np


def median_seed_index(auc):
    auc = np.asarray(auc, float); n = len(auc)
    order = np.argsort(auc, kind="stable")
    if n % 2:
        return int(order[n // 2])
    lo, hi = order[n // 2 - 1], order[n // 2]
    return int(lo if abs(auc[lo] - auc.mean()) <= abs(auc[hi] - auc.mean()) else hi)


def robust_auc_per_seed(json_path):
    d = json.load(open(json_path))
    return np.array([e["robust_return"] for e in d["evals"]]).mean(0), d.get("seed_ids")


def median_seed_info(json_path):
    auc, ids = robust_auc_per_seed(json_path)
    idx = median_seed_index(auc); n = len(auc); order = np.argsort(auc, kind="stable")
    return dict(json=json_path, n_seeds=n, seed_ids=ids, auc_per_seed=auc.tolist(), mean_auc=float(auc.mean()), index=idx, seed_id=(ids[idx] if ids else None),
                rank_order=order.tolist(), middle_pair=[int(order[n // 2 - 1]), int(order[n // 2])] if n % 2 == 0 else [int(order[n // 2])])


def find_one(directory, pattern):
    hits = sorted(glob.glob(os.path.join(directory, pattern)))
    assert len(hits) == 1, (pattern, hits)                    # exactly one file - never a snapshot instead of the final model by accident
    return hits[0]


FINAL = "__*[0-9a-f]_params.pkl"        # final params end in '<hash>_params.pkl'; snapshots in '_ckpt1M_params.pkl' (an upper-case M never matches [0-9a-f])


def final_params(p2_dir, arm):
    return find_one(p2_dir, arm + FINAL)


def snapshot_params(p2_dir, arm, label):
    return find_one(p2_dir, f"{arm}__*_ckpt{label}_params.pkl")


def json_of(params_path):
    return params_path.split("_ckpt")[0].replace("_params.pkl", "") + ".json"


def snapshot_steps(params_path):
    """Real number of env steps of a snapshot, from the JSON `ckpts` field (the requested 1M/3M are only labels)."""
    d = json.load(open(json_of(params_path))); tag = params_path.split("_ckpt")[1].split("_params")[0]
    hit = [c for c in d["ckpts"] if c["label"] == tag]
    assert len(hit) == 1, (tag, d["ckpts"])
    return int(hit[0]["env_steps"])


def final_steps(params_path):
    return int(json.load(open(json_of(params_path)))["hist"]["env_steps"][-1])
