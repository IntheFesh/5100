"""Demonstration video from the P2 models (round 9), with the PRE-REGISTERED selection rules (tools/demo_select.py).

Usage:  python tools/make_demo_video.py [--p2 results/P2] [--frozen frozen_P2.json] [--analysis results/analysis/P2_robust.json]
            [--numbers paper/numbers.json] [--fig paper/figs/F3_mechanism.png] [--out results/demo] [--figs paper/figs] [--demo_md results/DEMO.md]
Produces  <out>/demo_progression.mp4  (final method at the 1M / 3M snapshots and the final model; labels use the REAL step counts of the JSON `ckpts` field)
          <out>/demo_public_vs_ours.mp4 , <out>/demo_fixed_vs_ours.mp4  (the second gets the caption "difference not statistically significant (10 seeds)"
          when the Holm-corrected paired test of ours vs fixed PQN (E=2) in the P2 analysis has p >= 0.05)
          <out>/title_{1,2,3}.png (1920x1080 cards: the question / the Lambda_t figure / the result), <out>/demo_final.mp4 (cards 4 s each + the three videos, <= 3:30)
          one still frame per video in --figs, and results/DEMO.md (files, seed indices and how they were computed, the 9 full-episode scores, shown score/length).
The videos are not committed (*.mp4 is git-ignored); this script and DEMO.md are.  `ours` = the final method: the 'ours_*' arm if the frozen P2 has one, else
fixed PQN at E* (high_E_no_control), else fixed PQN E=2 (confirm_fixed_only); then group B does not exist and is skipped.
"""
import argparse, datetime, json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import demo_select as ds

LIMIT_S, CARD_S, FPS = 210.0, 4.0, 24
FOOTER_NS = "difference not statistically significant (10 seeds)"


def font(size, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), "/usr/share/fonts/truetype/liberation/LiberationSans-%s.ttf" % ("Bold" if bold else "Regular")):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + [cur]


def card(path, title, body=None, image=None, foot=None):
    """1920x1080 title card: a headline, optional body text and optional picture (e.g. the Lambda_t figure)."""
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), (12, 12, 16)); d = ImageDraw.Draw(img)
    y = 110
    for line in wrap(d, title, font(64, True), W - 240):
        d.text((120, y), line, font=font(64, True), fill=(240, 240, 240)); y += 84
    y += 20
    if body:
        for line in wrap(d, body, font(40), W - 240):
            d.text((120, y), line, font=font(40), fill=(190, 190, 190)); y += 56
    if image is not None:
        pic = Image.open(image).convert("RGB"); room_h = H - y - 120
        s = min((W - 240) / pic.width, room_h / pic.height); pic = pic.resize((int(pic.width * s), int(pic.height * s)), Image.LANCZOS)
        img.paste(pic, ((W - pic.width) // 2, y + 20))
    if foot:                                                           # wrapped: a long footer must not run off the card
        lines = wrap(d, foot, font(30), W - 240)
        for k, line in enumerate(lines):
            d.text((120, H - 60 - 40 * (len(lines) - k)), line, font=font(30), fill=(130, 130, 130))
    img.save(path)
    return path


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("command failed: " + " ".join(cmd) + "\n" + r.stderr[-2500:])
    return r


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def to_hd(src, dst):
    """Scale a render_demo video into a 1920x1080 frame (nearest neighbour keeps the pixel art sharp), 24 fps, yuv420p."""
    vf = "scale=1920:1080:force_original_aspect_ratio=decrease:flags=neighbor,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0c0c10,fps=%d,format=yuv420p" % FPS
    run([ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", dst])


def card_video(png, dst, seconds=CARD_S):
    run([ffmpeg(), "-y", "-loglevel", "error", "-loop", "1", "-t", str(seconds), "-i", png, "-vf", "fps=%d,format=yuv420p" % FPS, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", dst])


def concat(parts, dst, listfile):
    open(listfile, "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in parts))
    run([ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", dst])


def duration(path):
    import imageio.v2 as iio
    r = iio.get_reader(path); m = r.get_meta_data(); r.close()
    return float(m["duration"])


def render(args):
    cmd = [sys.executable, "-W", "ignore", os.path.join(ROOT, "render_demo.py")] + args
    return run(cmd)


def text_from_numbers(numbers, key, default="n/a"):
    if not numbers or key not in numbers:
        return default
    return numbers[key]["text"].replace("\\%", "%").replace("\\_", "_")


def conclusion(fz, numbers, analysis, ours_arm):
    mode = fz["mode"]
    if mode == "confirm_fixed_only" or ours_arm is None:
        return ("Result", f"Corrected PQN vs public PQN: {text_from_numbers(numbers, 'cmpFixedVsPublicRel')} robust AUC "
                f"(95% CI of the AUC difference [{text_from_numbers(numbers, 'cmpFixedVsPublicCiLo')}, {text_from_numbers(numbers, 'cmpFixedVsPublicCiHi')}], Holm p = {text_from_numbers(numbers, 'cmpFixedVsPublicPHolm')}). "
                "No gain from more updates per batch was confirmed.")
    k = "OursVsFixedEtwo" if mode == "with_control" else "FixedBestVsFixedEtwo"
    rel, lo, hi = (text_from_numbers(numbers, f"cmp{k}{x}") for x in ("Rel", "CiLo", "CiHi")); p = text_from_numbers(numbers, f"cmp{k}PHolm")
    big = text_from_numbers(numbers, f"cmp{k}Large", "no")
    head = "More updates per batch + norm control" if mode == "with_control" else "More updates per batch (no control)"
    body = f"{rel} robust AUC vs fixed PQN (E=2); 95% CI of the AUC difference [{lo}, {hi}], Holm p = {p}, 10 confirmation seeds."
    if big != "yes":
        body += " Not a substantial improvement by the pre-registered rule (>= 25% and CI above 0)."
    return head, body


def card2_text(numbers):
    """Middle card: the claim 'weight norm grows -> effective learning rate shrinks' is only stated if mechanism.py supported prediction 1 (numbers.json)."""
    supported = bool(numbers) and numbers.get("predOneVerdict", {}).get("text") == "supported"
    if supported:
        return ("Weight norm grows with E, so the effective learning rate shrinks",
                "Measured Lambda_t = ||W_t|| / ||W_0|| of the dense layer (development seeds): it grows with the number of epochs E; norm projection (NP) keeps Lambda_t = 1 by construction.")
    return ("Measured weight norm Lambda_t for E = 2, 4, 8",
            "Lambda_t = ||W_t|| / ||W_0|| of the dense layer (development seeds); norm projection (NP) keeps Lambda_t = 1 by construction. The predicted norm law was not confirmed as a whole - see the paper.")


def footer_for(analysis, final_arm):
    """Caption of group B: shown unless the Holm-corrected paired test (final method vs fixed PQN E=2, P2 analysis) has p < 0.05."""
    pr = next((p for p in (analysis or {}).get("pairs", []) if p["config"] == final_arm and p["ref"] == "fixed_E2"), None)
    return FOOTER_NS if (pr is None or pr.get("p_holm", 1.0) >= 0.05) else None


def comparison_max_steps(limit_s, progression_s, n_cmp, default, fps=FPS):
    """--max_steps for the comparison videos (the same for both sides, one value for all of them) so that cards + progression + comparisons <= limit."""
    budget = limit_s - 3 * CARD_S - progression_s
    return int(min(default, max(120, np.floor(budget / n_cmp * fps) - 1)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2", default=os.path.join(ROOT, "results", "P2")); ap.add_argument("--frozen", default=os.path.join(ROOT, "frozen_P2.json"))
    ap.add_argument("--analysis", default=os.path.join(ROOT, "results", "analysis", "P2_robust.json"))
    ap.add_argument("--numbers", default=os.path.join(ROOT, "paper", "numbers.json")); ap.add_argument("--fig", default=os.path.join(ROOT, "paper", "figs", "F3_mechanism.png"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "demo")); ap.add_argument("--figs", default=os.path.join(ROOT, "paper", "figs"))
    ap.add_argument("--demo_md", default=os.path.join(ROOT, "results", "DEMO.md"))
    ap.add_argument("--episodes", type=int, default=9); ap.add_argument("--max_steps", type=int, default=2400)
    ap.add_argument("--limit_s", type=float, default=LIMIT_S)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
    fz = json.load(open(a.frozen)); arms = fz["p2_arms"]; E = fz["best_E"]
    numbers = json.load(open(a.numbers)) if os.path.exists(a.numbers) else None
    analysis = json.load(open(a.analysis)) if os.path.exists(a.analysis) else None
    ours_arm = next((n for n in arms if n.startswith("ours_")), None)
    final_arm = ours_arm or (f"fixed_E{E}" if (E != 2 and f"fixed_E{E}" in arms) else "fixed_E2")
    pub, fix2, fin = (ds.final_params(a.p2, x) for x in ("public", "fixed_E2", final_arm))
    s1, s3 = ds.snapshot_params(a.p2, final_arm, "1M"), ds.snapshot_params(a.p2, final_arm, "3M")
    info = {x: ds.median_seed_info(ds.json_of(p)) for x, p in (("public", pub), ("fixed_E2", fix2), (final_arm, fin))}
    S = {k: v["index"] for k, v in info.items()}
    L1, L3, LF = (f"{ds.snapshot_steps(s1) / 1e6:.2f}M steps", f"{ds.snapshot_steps(s3) / 1e6:.2f}M steps", f"{ds.final_steps(fin) / 1e6:.2f}M steps")
    label_final = "Ours" if ours_arm else ("Fixed PQN (E=%d)" % E if final_arm != "fixed_E2" else "Fixed PQN (E=2)")
    j = {k: os.path.join(a.out, f"{k}.json") for k in ("progression", "A", "B")}
    common = ["--episodes", str(a.episodes)]
    # 1. learning progression of the final method (always valid, whatever the comparison shows)
    prog = os.path.join(a.out, "demo_progression.mp4")
    render([prog, "--seed_indices", ",".join([str(S[final_arm])] * 3), "--panel", s1, L1, "--panel", s3, L3, "--panel", fin, LF, "--max_steps", str(a.max_steps),
            "--frame_png", os.path.join(a.figs, "demo_progression.png"), "--json_out", j["progression"], *common])
    d_prog = json.load(open(j["progression"]))["frames"] / FPS
    # 2. comparison videos share one --max_steps on both sides; shortened if the total would exceed the limit
    n_cmp = 2 if (ours_arm or final_arm != "fixed_E2") else 1
    ms = comparison_max_steps(a.limit_s, d_prog, n_cmp, a.max_steps)
    A_ = os.path.join(a.out, "demo_public_vs_ours.mp4")
    render([A_, "--seed_indices", f"{S['public']},{S[final_arm]}", "--panel", pub, "Public PQN", "--panel", fin, label_final, "--scale255", "0", "--max_steps", str(ms),
            "--frame_png", os.path.join(a.figs, "demo_public_vs_ours.png"), "--json_out", j["A"], *common])
    videos = [("progression", prog)]; videos.append(("public_vs_ours", A_))
    footer = None
    if n_cmp == 2:
        B_ = os.path.join(a.out, "demo_fixed_vs_ours.mp4")
        footer = footer_for(analysis, final_arm)
        render([B_, "--seed_indices", f"{S['fixed_E2']},{S[final_arm]}", "--panel", fix2, "Fixed PQN (E=2)", "--panel", fin, label_final, "--max_steps", str(ms),
                "--frame_png", os.path.join(a.figs, "demo_fixed_vs_ours.png"), "--json_out", j["B"], *common] + (["--footer", footer] if footer else []))
        videos.append(("fixed_vs_ours", B_))
    # 3. title cards + concatenation
    head, body = conclusion(fz, numbers, analysis, ours_arm)
    cards = [card(os.path.join(a.out, "title_1.png"), "PQN trains on each batch only twice. Why doesn't more learning per batch help?", "Replay-free Q-learning on MinAtar SpaceInvaders: update ratio, weight norm, effective learning rate.", foot="Yueyi Li, Cheng Qian - WashU DRL"),
             card(os.path.join(a.out, "title_2.png"), *card2_text(numbers), image=a.fig if os.path.exists(a.fig) else None),
             card(os.path.join(a.out, "title_3.png"), head, body, foot="Same env steps, same environment seeds; median training seed per config and median of 9 full episodes (rule fixed in advance).")]
    parts = []
    hd = lambda p: p.replace(".mp4", "_hd.mp4")
    card_mp4 = lambda i: os.path.join(a.out, f"title_{i}.mp4")
    for i in (1, 2, 3):
        card_video(cards[i - 1], card_mp4(i))
    for _, v in videos:
        to_hd(v, hd(v))
    parts = [card_mp4(1), hd(prog), card_mp4(2), hd(A_)] + ([hd(os.path.join(a.out, "demo_fixed_vs_ours.mp4"))] if n_cmp == 2 else []) + [card_mp4(3)]
    final = os.path.join(a.out, "demo_final.mp4")
    concat(parts, final, os.path.join(a.out, "concat.txt"))
    total = duration(final)
    assert total <= a.limit_s + 0.5, f"demo_final.mp4 is {total:.1f} s > {a.limit_s} s"
    # 4. DEMO.md
    write_demo_md(a.demo_md, a, fz, info, S, final_arm, label_final, (L1, L3, LF), (s1, s3, fin, pub, fix2), j, ms, footer, total, n_cmp)
    print(f"demo_final.mp4: {total:.1f} s; comparison max_steps = {ms}; footer = {footer!r}; DEMO.md -> {a.demo_md}")
    return dict(total=total, max_steps=ms, footer=footer, seed_index=S, final_arm=final_arm)


def write_demo_md(path, a, fz, info, S, final_arm, label_final, labels, files, j, ms, footer, total, n_cmp):
    rel = lambda p: os.path.relpath(p, os.getcwd())
    L = ["# DEMO: selection record\n", f"Generated {datetime.date.today().isoformat()} by `tools/make_demo_video.py`. Frozen P2: mode **{fz['mode']}**, arms {', '.join(fz['p2_arms'])}. "
         f"Final method shown as **{label_final}** (`{final_arm}`).\n", "## Rule (fixed before looking at any model)\n",
         "- Training seed per config: the confirmation seed whose robust AUC (mean over the eval points) is the median; with 10 seeds the 5th/6th smallest, whichever AUC is closer to the mean AUC (tie -> lower). "
         "`seed_index` is the position on the vmap axis = the order of `seed_ids` in the JSON.",
         f"- Episodes: the same {a.episodes} environment seeds (1000..{1000 + a.episodes - 1}) for every model, sticky actions 0.1, the game runs to its cap (10000) and the episode with the median FULL score is shown; "
         "only the video is cut at `--max_steps`. Never the best episode.",
         f"- Snapshot labels use the real step counts from the JSON `ckpts` field: {labels[0]}, {labels[1]}, final {labels[2]}.\n", "## Training seeds\n",
         "| config | JSON | seed ids | robust AUC per seed | mean AUC | middle pair (vmap idx) | chosen seed_index | chosen seed id |", "|---|---|---|---|---|---|---|---|"]
    for k, v in info.items():
        L.append(f"| {k} | `{rel(v['json'])}` | {v['seed_ids']} | {', '.join(f'{x:.2f}' for x in v['auc_per_seed'])} | {v['mean_auc']:.2f} | {v['middle_pair']} | **{v['index']}** | {v['seed_id']} |")
    L += ["", "## Videos\n", f"Comparison videos: `--max_steps {ms}` on both sides (limit: total <= {a.limit_s / 60:.1f} min). Total length of demo_final.mp4: {total:.1f} s.\n"]
    for name, key in (("Learning progression", "progression"), ("Public PQN vs ours", "A"), ("Fixed PQN (E=2) vs ours", "B")):
        if not os.path.exists(j[key]):
            continue
        r = json.load(open(j[key]))
        L += [f"### {name}\n", f"- file `{rel(r['out'])}`, {r['frames']} frames at {r['fps']} fps, sticky {r['sticky']}, game cap {r['game_cap']}, env seeds {r['env_seeds'][0]}..{r['env_seeds'][-1]}"
              + (f", caption: \"{r['footer']}\"" if r.get("footer") else ""), "", f"| panel | params | seed_index | {a.episodes} full-episode scores (sorted) | shown score | shown length |", "|---|---|---|---|---|---|"]
        for p in r["panels"]:
            L.append(f"| {p['label']} | `{rel(p['params'])}` | {p['seed_index']} | {p['episode_scores_sorted']} | {p['shown_score']} | {p['shown_length']} |")
        L.append("")
    if n_cmp == 1:
        L.append("Group B (fixed PQN vs ours) is not produced: the frozen P2 has no separate final method (mode confirm_fixed_only).\n")
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
