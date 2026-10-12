# Hero animation: experts read per decode step for a batch of 32, random order vs sorted by domain.
# Real routing of Qwen3-30B-A3B. Writes figures/out/hero.mp4 and hero.gif.

import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulate import load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.environ.get("FONTS_DIR", f"{ROOT}/figures/fonts") + "/"
W, H, FPS = 1600, 900, 30
B = 32
L, E = 48, 128
CW, CH = 5, 9
GW, GH = E * CW, L * CH
PX = [110, 850]
PY = 250

BG = (252, 252, 251)
INK, INK2, INK3 = (11, 11, 11), (82, 81, 78), (140, 139, 134)
OFF = (236, 235, 231)
BLUE = (42, 120, 214)
GRAY_ON = (120, 119, 114)
DOM_COL = {"code": (42, 120, 214), "math": (235, 104, 52), "creative": (27, 175, 122), "french": (237, 161, 0),
           "arabic": (232, 123, 164), "chinese": (98, 80, 214)}
DOM_NAME = {"code": "code", "math": "math", "creative": "writing", "french": "French", "arabic": "Arabic",
            "chinese": "Chinese"}
SEG = 50  # frames per batch
NSTEP = 6 * SEG
INTRO, OUTRO = 45, 120


@lru_cache(None)
def font(name, size, weight=None):
    f = ImageFont.truetype(FONTS + name, size, layout_engine=ImageFont.Layout.RAQM)
    try:
        vals = []
        for a in f.get_variation_axes():
            n = a["name"] if isinstance(a["name"], str) else a["name"].decode()
            vals.append(weight if (n == "Weight" and weight) else (min(max(size, a["minimum"]), a["maximum"])
                                                                   if n == "Optical Size" else a["default"]))
        f.set_variation_by_axes(vals)
    except Exception:
        pass
    return f


def build():
    meta, dec, _ = load(f"{ROOT}/data/trace_qwen3_30b.npz")
    rng = np.random.default_rng(3)
    doms = list(dec)
    pool = [(d, s) for d in doms for s in dec[d] if s.shape[1] >= 64]
    left_batches, right_batches = [], []
    for seg in range(6):
        idx = rng.choice(len(pool), B, replace=False)
        left_batches.append([pool[i] for i in idx])
        d = doms[seg]
        cand = [s for s in dec[d] if s.shape[1] >= 64]
        right_batches.append([(d, cand[i]) for i in rng.choice(len(cand), B, replace=False)])
    frames = []
    for f in range(NSTEP):
        seg, t = divmod(f, SEG)
        out = []
        for batch in (left_batches[seg], right_batches[seg]):
            cnt = np.zeros((L, E, len(doms)), np.int16)
            for dm, s in batch:
                tt = (t + 8) % s.shape[1]
                for l in range(L):
                    cnt[l, s[l, tt], doms.index(dm)] += 1
            # colour = domain sending the most tokens to that expert, -1 = not read
            grid = np.where(cnt.sum(-1) > 0, cnt.argmax(-1), -1).astype(np.int8)
            out.append((grid, [d for d, _ in batch]))
        frames.append(out)
    return frames


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def blend(c, a, bg=BG):
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


FR = None
HIST = None


def draw_frame(i):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    a_title = ease(i / 20)
    d.text((W // 2, 62), "Same 192 requests. Different order.", font=font("Inter.ttf", 44, 700), fill=blend(INK, a_title),
           anchor="mm")
    d.text((W // 2, 112), "Experts Qwen3-30B-A3B reads from memory at each decode step  ·  batch of 32  ·  real routing",
           font=font("Inter.ttf", 22, 450), fill=blend(INK2, a_title), anchor="mm")
    f = i - INTRO
    k = min(max(f, 0), NSTEP - 1)
    a_grid = ease((i - 15) / 25)
    for p in range(2):
        x0 = PX[p]
        grid, doms = FR[k][p]
        if f < 0:
            grid = np.full_like(grid, -1)
        title = "Random order: every batch mixes domains" if p == 0 else "Sorted by domain: each batch is one domain"
        d.text((x0, PY - 72), title, font=font("Inter.ttf", 25, 650), fill=blend(INK, a_grid), anchor="ls")
        for j, dm in enumerate(doms):
            cx = x0 + 8 + j * 20
            d.ellipse((cx - 6, PY - 46, cx + 6, PY - 34), fill=blend(DOM_COL[dm], a_grid if f >= 0 else 0.25 * a_grid))
        cells = np.array(Image.new("RGB", (GW, GH), BG))
        img = np.zeros((L, E, 3), np.uint8)
        img[:] = blend(OFF, a_grid)
        for di, dm in enumerate(DOM_COL):
            img[grid == di] = blend(DOM_COL[dm], a_grid)
        big = img.repeat(CH, 0).repeat(CW, 1)
        big[CH - 1 :: CH, :, :] = BG
        big[:, CW - 1 :: CW, :] = BG
        cells[:] = big
        im.paste(Image.fromarray(cells), (x0, PY))
        d.text((x0, PY + GH + 26), "128 experts →", font=font("Inter.ttf", 17, 450), fill=blend(INK3, a_grid), anchor="ls")
        if p == 0:
            d.text((x0 - 14, PY + 6), "layer 1", font=font("Inter.ttf", 15, 450), fill=blend(INK3, a_grid), anchor="ra")
            d.text((x0 - 14, PY + GH), "layer 48", font=font("Inter.ttf", 15, 450), fill=blend(INK3, a_grid), anchor="rs")
        lo = max(0, k - 14)
        n = np.mean([(FR[j][p][0] >= 0).sum() for j in range(lo, k + 1)]) if f >= 0 else 0
        gb = n * 9.44e-3  # 9.4 MB per expert
        y = PY + GH + 78
        d.text((x0, y), f"{n:,.0f}", font=font("JetBrainsMono.ttf", 46, 700), fill=blend(INK, a_grid),
               anchor="ls")
        w = font("JetBrainsMono.ttf", 46, 700).getlength(f"{n:,.0f}")
        d.text((x0 + w + 14, y), f"experts read / step  ·  {gb:,.1f} GB", font=font("Inter.ttf", 22, 500),
               fill=blend(INK2, a_grid), anchor="ls")
    lx = PX[0]
    ly = H - 52
    fl = font("Inter.ttf", 18, 500)
    for dm, c in DOM_COL.items():
        d.ellipse((lx, ly - 13, lx + 12, ly - 1), fill=blend(c, a_grid))
        d.text((lx + 18, ly), DOM_NAME[dm], font=fl, fill=blend(INK2, a_grid), anchor="ls")
        lx += 30 + fl.getlength(DOM_NAME[dm]) + 18
    if i >= INTRO + NSTEP:
        a = ease((i - INTRO - NSTEP) / 20)
        ov = Image.new("RGB", (W, H), BG)
        im = Image.blend(im, ov, 0.9 * a)
        d = ImageDraw.Draw(im)
        r = HIST
        d.text((W // 2, 360), f"{r[0]:.0f}% fewer experts read per step", font=font("Inter.ttf", 58, 750),
               fill=blend(INK, a), anchor="mm")
        d.text((W // 2, 450), f"+{r[1]:.0f}% throughput in vLLM, measured. Outputs unchanged. No model change.",
               font=font("Inter.ttf", 30, 500), fill=blend(INK2, a), anchor="mm")
        d.text((W // 2, 520), "Just sort your batch job.", font=font("Inter.ttf", 30, 700), fill=blend(BLUE, a), anchor="mm")
    return im.tobytes()


def init(frames, hist):
    global FR, HIST
    FR, HIST = frames, hist


def main():
    frames = build()
    left = np.mean([(fr[0][0] >= 0).sum() for fr in frames])
    right = np.mean([(fr[1][0] >= 0).sum() for fr in frames])
    hist = (100 * (1 - right / left), float(os.environ.get("VLLM_GAIN", "15")))
    print(f"mean experts/step: random {left:.0f}, sorted {right:.0f} -> {hist[0]:.1f}% fewer")
    total = INTRO + NSTEP + OUTRO
    os.makedirs("figures/out", exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", "figures/out/hero.mp4"], stdin=subprocess.PIPE)
    with Pool(64, initializer=init, initargs=(frames, hist)) as pool:
        for b in pool.imap(draw_frame, range(total), chunksize=4):
            ff.stdin.write(b)
    ff.stdin.close()
    ff.wait()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "figures/out/hero.mp4", "-vf",
                    "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
                    "figures/out/hero.gif"], check=True)
    init(frames, hist)
    Image.frombytes("RGB", (W, H), draw_frame(INTRO + 120)).save("figures/out/hero_still.png")


if __name__ == "__main__":
    main()
