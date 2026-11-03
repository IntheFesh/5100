"""Side-by-side SpaceInvaders-MinAtar gameplay video for 2-3 saved PQN parameter sets.
Episode selection rule (fixed in advance): run K greedy episodes per model with the SAME env seeds,
show the episode whose FULL-episode score is the MEDIAN of that model's K episodes (never the best one).
Protocol: sticky actions p=0.1 by default (with p=0 gymnax SpaceInvaders is deterministic and every episode is
identical, so 'median of K' would be meaningless). The game runs to --game_cap; --max_steps only cuts the video.
Usage:
  python render_demo.py out.mp4 --panel public.pkl "Public PQN" --panel ours.pkl "Ours" --scale255 0
  python render_demo.py prog.mp4 --panel ours_ckpt1M.pkl "1M steps" --panel ours_ckpt3M.pkl "3M" --panel ours.pkl "10M"
"""
import pickle, argparse
import numpy as np, jax, jax.numpy as jnp, flax.linen as nn, gymnax
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--panel", nargs=2, action="append", metavar=("PARAMS_PKL", "LABEL"), required=True)
ap.add_argument("--scale255", type=int, action="append", default=[], help="index of a panel trained with x/255 inputs")
ap.add_argument("--seed_index", type=int, default=0, help="which training seed (position on the vmapped axis) to show")
ap.add_argument("--seed_indices", default=None, help="comma list, one per panel (overrides --seed_index), e.g. 4,7")
ap.add_argument("--episodes", type=int, default=9)
ap.add_argument("--max_steps", type=int, default=2400, help="video length cap (2400 steps @24fps = 100 s)")
ap.add_argument("--game_cap", type=int, default=10000, help="episode cap of the game itself (robust protocol)")
ap.add_argument("--sticky", type=float, default=0.1, help="sticky-action prob (robust protocol); 0 = deterministic gymnax")
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--random_init", action="store_true", help="pipeline test without trained params")
ap.add_argument("--frame_png", default=None, help="also save one still frame (at --frame_step) as PNG")
ap.add_argument("--frame_step", type=int, default=200)
ap.add_argument("--footer", default=None, help="extra caption line at the bottom of the video (default: none, frames unchanged)")
ap.add_argument("--json_out", default=None, help="write what was shown: params, training-seed index, the K full-episode scores, shown score/length")
a = ap.parse_args()

env, env_params = gymnax.make("SpaceInvaders-MinAtar")
n_act = env.action_space(env_params).n
game_params = env_params.replace(max_steps_in_episode=a.game_cap)


class QNet(nn.Module):
    n_act: int
    scale255: bool

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        if self.scale255:
            x = x / 255.0
        x = nn.Conv(16, (3, 3), 1, padding="VALID", name="conv")(x)
        x = nn.relu(nn.LayerNorm(name="ln0")(x)).reshape((x.shape[0], -1))
        h = nn.relu(nn.LayerNorm(name="ln1")(nn.Dense(128, name="dense")(x)))
        return nn.Dense(self.n_act, name="head")(h), h


def load(path, scale255, key, idx):
    net = QNet(n_act, scale255)
    if a.random_init:
        return net, net.init(key, jnp.zeros((1, 10, 10, 6)))
    carry = pickle.load(open(path, "rb"))
    return net, jax.tree_util.tree_map(lambda x: x[idx], carry)


step_fn = jax.jit(env.step_env)          # NO auto-reset: the last frame of a finished game is the real final board
_apply_cache = {}


def play(net, params, seed):
    """One greedy episode with sticky actions. Returns video frames, per-frame score, full score, full length."""
    if id(net) not in _apply_cache:      # one compiled forward per network (not one per episode)
        _apply_cache[id(net)] = jax.jit(lambda p, o, net=net: net.apply(p, o[None])[0][0])
    apply = _apply_cache[id(net)]
    key = jax.random.PRNGKey(seed)
    key, kr = jax.random.split(key)
    obs, st = env.reset(kr, game_params)
    frames, scores, score = [np.asarray(obs)], [0.0], 0.0
    rs, prev, t = np.random.default_rng(seed), 0, 0
    for t in range(a.game_cap):                # the GAME runs to its cap; only the VIDEO is cut at --max_steps
        act = int(jnp.argmax(apply(params, obs)))
        if rs.random() < a.sticky:             # MinAtar sticky actions: repeat the previously executed action
            act = prev
        prev = act
        key, ks = jax.random.split(key)
        obs, st, r, done, _ = step_fn(ks, st, act, game_params)
        score += float(r)
        if t + 1 <= a.max_steps:
            frames.append(np.asarray(obs)); scores.append(score)
        if bool(done):
            break
    return frames, scores, score, t + 1


ALIEN = ["..X...X..", "...X.X...", "..XXXXX..", ".XX.X.XX.", "XXXXXXXXX", "X.XXXXX.X", "X.X...X.X", "...XX.XX."]
CANNON = ["....X....", "...XXX...", "...XXX...", ".XXXXXXX.", "XXXXXXXXX", "XXXXXXXXX"]


def sprite(draw, x0, y0, cell, pattern, color):
    rows, cols = len(pattern), len(pattern[0])
    px = cell * 0.8 / cols
    ox, oy = x0 + cell * 0.1, y0 + (cell - rows * px) / 2
    for r, line in enumerate(pattern):
        for c, ch in enumerate(line):
            if ch == "X":
                draw.rectangle([ox + c * px, oy + r * px, ox + (c + 1) * px - 0.5, oy + (r + 1) * px - 0.5], fill=color)


def to_rgb(obs, cell=40):
    img = Image.new("RGB", (10 * cell, 10 * cell), (10, 12, 24))
    d = ImageDraw.Draw(img)
    for i in range(0, 10 * cell, cell):              # faint grid
        d.line([(i, 0), (i, 10 * cell)], fill=(22, 26, 44)); d.line([(0, i), (10 * cell, i)], fill=(22, 26, 44))
    for r in range(10):
        for c in range(10):
            x0, y0 = c * cell, r * cell
            if obs[r, c, 1] > 0:
                sprite(d, x0, y0, cell, ALIEN, (120, 235, 120))
            if obs[r, c, 0] > 0:
                sprite(d, x0, y0, cell, CANNON, (90, 190, 255))
            if obs[r, c, 4] > 0:
                d.rectangle([x0 + cell * .44, y0 + cell * .15, x0 + cell * .56, y0 + cell * .85], fill=(255, 230, 90))
            if obs[r, c, 5] > 0:
                d.rectangle([x0 + cell * .42, y0 + cell * .2, x0 + cell * .58, y0 + cell * .8], fill=(255, 90, 90))
    return np.asarray(img)


try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except Exception:
    FONT = FONT_S = ImageFont.load_default()


def panel(obs, label, sub, score, t, cell=40):
    g = to_rgb(obs, cell)
    W = g.shape[1]
    canvas = Image.new("RGB", (W, g.shape[0] + 70), (12, 12, 16))
    canvas.paste(Image.fromarray(g), (0, 70))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), label, font=FONT, fill=(235, 235, 235))
    d.text((10, 38), f"score {int(score):4d}   step {t:4d}{sub}", font=FONT_S, fill=(190, 190, 190))
    bar = int(min(score / 300.0, 1.0) * (W - 20))
    d.rectangle([10, 60, W - 10, 64], fill=(40, 40, 52)); d.rectangle([10, 60, 10 + bar, 64], fill=(255, 200, 80))
    return canvas


def median_episode(net, params, seeds):
    eps = [play(net, params, s) for s in seeds]
    finals = [e[2] for e in eps]                 # FULL-episode scores decide the median, not the video-cut ones
    return eps[int(np.argsort(finals)[len(finals) // 2])], finals


seeds = list(range(1000, 1000 + a.episodes))
shown, report = [], []
idxs = [int(v) for v in a.seed_indices.split(",")] if a.seed_indices else [a.seed_index] * len(a.panel)
assert len(idxs) == len(a.panel), "--seed_indices needs one entry per --panel"
for i, (path, label) in enumerate(a.panel):
    net, params = load(path, i in a.scale255, jax.random.PRNGKey(i), idxs[i])
    print(f"panel {i}: {path} training-seed index {idxs[i]}")
    (fr, sc, tot, ln), finals = median_episode(net, params, seeds)
    print(f"{label}: episode scores {sorted(int(f) for f in finals)}  -> median shown {int(tot)} (length {ln})", flush=True)
    shown.append((label, fr, sc, tot, ln))
    report.append(dict(label=label, params=path, seed_index=idxs[i], scale255=(i in a.scale255), episode_scores_sorted=sorted(int(f) for f in finals),
                       episode_scores_by_env_seed={int(s_): int(f) for s_, f in zip(seeds, finals)}, shown_score=int(tot), shown_length=int(ln)))

T = max(len(s[1]) for s in shown)
gap, frames = 16, []
for t in range(T):
    ps = []
    for label, fr, sc, tot, ln in shown:
        i = min(t, len(fr) - 1)
        sub = ""
        if t >= len(fr) - 1:
            sub = "   game over" if ln <= a.max_steps else f"   (final {int(tot)} @ step {ln})"
        ps.append(panel(fr[i], label, sub, sc[i], i))
    foot_h = 24 if a.footer else 0                                  # extra caption strip only when --footer is given (default frames unchanged)
    full = Image.new("RGB", (sum(p.width for p in ps) + gap * (len(ps) - 1), ps[0].height + 34 + foot_h), (12, 12, 16))
    x = 0
    for p in ps:
        full.paste(p, (x, 0)); x += p.width + gap
    d = ImageDraw.Draw(full)
    d.text((10, ps[0].height + 8), f"SpaceInvaders-MinAtar, greedy, sticky p={a.sticky:g}, same env seeds; "
           f"median of {a.episodes} episodes shown", font=FONT_S, fill=(150, 150, 150))
    if a.footer:
        d.text((10, ps[0].height + 34), a.footer, font=FONT_S, fill=(235, 200, 120))
    frames.append(np.asarray(full))
    if a.frame_png and t == min(a.frame_step, T - 1):
        Image.fromarray(frames[-1]).save(a.frame_png)

import imageio
if a.out.endswith(".mp4"):
    imageio.mimsave(a.out, frames, fps=a.fps, macro_block_size=1)
else:
    imageio.mimsave(a.out, frames, duration=1 / a.fps)
print("wrote", a.out, len(frames), "frames")
if a.json_out:
    import json
    json.dump(dict(out=a.out, frames=len(frames), fps=a.fps, max_steps=a.max_steps, game_cap=a.game_cap, sticky=a.sticky, episodes=a.episodes,
                   env_seeds=seeds, footer=a.footer, panels=report), open(a.json_out, "w"), indent=1)
