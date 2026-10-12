# 1200x630 thumbnail

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_hero as R  # noqa: E402

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frames = R.build()
gl, _ = frames[100][0]
gr, _ = frames[100][1]
W, H = 1200, 630
im = Image.new("RGB", (W, H), R.BG)
d = ImageDraw.Draw(im)
d.text((60, 92), "Sort your prompts.", font=R.font("Inter.ttf", 62, 780), fill=R.INK, anchor="ls")
d.text((60, 140), "Same MoE, same requests, different order: up to +17% throughput in vLLM.",
       font=R.font("Inter.ttf", 25, 500), fill=R.INK2, anchor="ls")
cw, ch = 4, 7
for k, (g, on, lab, n) in enumerate([(gl, None, "random order", (gl >= 0).sum()), (gr, None, "sorted", (gr >= 0).sum())]):
    x0, y0 = 60 + k * 560, 186
    img = np.zeros((48, 128, 3), np.uint8)
    img[:] = R.OFF
    for di, dm in enumerate(R.DOM_COL):
        img[g == di] = R.DOM_COL[dm]
    big = img.repeat(ch, 0).repeat(cw, 1)
    big[ch - 1 :: ch] = R.BG
    big[:, cw - 1 :: cw] = R.BG
    im.paste(Image.fromarray(big), (x0, y0))
    d.text((x0, y0 + 48 * ch + 44), f"{n:,}", font=R.font("Inter.ttf", 36, 760), fill=R.INK, anchor="ls")
    w = R.font("Inter.ttf", 36, 760).getlength(f"{n:,}")
    d.text((x0 + w + 12, y0 + 48 * ch + 44), f"experts read per step · {lab}", font=R.font("Inter.ttf", 20, 500),
           fill=R.INK2, anchor="ls")
im.save("figures/out/thumbnail.png")
print("ok")
