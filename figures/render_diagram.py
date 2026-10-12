# Toy diagram: one MoE layer, one decode step, mixed batch vs batch of similar requests.

import os

from PIL import Image, ImageDraw

from render_hero import BG, BLUE, INK, INK2, INK3, OFF, DOM_COL, font

W, H = 1800, 980
S = 2  # supersampling


def panel(d, y0, title, toks, sub):
    d.text((80 * S, y0 * S), title, font=font("Inter.ttf", 30 * S, 680), fill=INK, anchor="ls")
    d.text((80 * S, (y0 + 34) * S), sub, font=font("Inter.ttf", 20 * S, 450), fill=INK2, anchor="ls")
    ty, ey = y0 + 90, y0 + 300
    tx = [150 + i * 110 for i in range(8)]
    ex = [110 + i * 62 for i in range(16)]
    used = sorted({e for _, es in toks for e in es})
    for i, (dom, es) in enumerate(toks):
        for e in es:
            d.line((tx[i] * S, (ty + 22) * S, (ex[e] + 22) * S, (ey - 4) * S), fill=DOM_COL[dom] + (110,), width=3 * S)
    for i, (dom, _) in enumerate(toks):
        d.rounded_rectangle(((tx[i] - 40) * S, (ty - 22) * S, (tx[i] + 40) * S, (ty + 22) * S), radius=11 * S,
                            fill=DOM_COL[dom])
        d.text((tx[i] * S, ty * S), {"code": "code", "math": "math", "french": "fr", "chinese": "zh", "creative": "story",
                                     "arabic": "ar"}[dom], font=font("Inter.ttf", 18 * S, 600), fill=(255, 255, 255), anchor="mm")
    for e in range(16):
        on = e in used
        d.rounded_rectangle((ex[e] * S, ey * S, (ex[e] + 44) * S, (ey + 92) * S), radius=6 * S,
                            fill=BLUE if on else OFF)
        for k in range(4):
            yy = ey + 16 + k * 18
            d.line(((ex[e] + 9) * S, yy * S, (ex[e] + 35) * S, yy * S), fill=(255, 255, 255) if on else (220, 219, 214),
                   width=4 * S)
    d.text((1150 * S, (ey + 30) * S), f"{len(used)} / 16", font=font("Inter.ttf", 58 * S, 750), fill=INK, anchor="ls")
    d.text((1150 * S, (ey + 66) * S), "experts read from HBM", font=font("Inter.ttf", 22 * S, 500), fill=INK2, anchor="ls")
    d.text((1150 * S, (ey + 96) * S), f"= {len(used) * 9.4:.0f} MB in this layer, this step", font=font("Inter.ttf", 20 * S, 450),
           fill=INK3, anchor="ls")


def main():
    im = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(im, "RGBA")
    mixed = [("code", (0, 3)), ("french", (5, 9)), ("math", (1, 12)), ("chinese", (7, 14)), ("code", (3, 10)),
             ("arabic", (6, 15)), ("math", (2, 12)), ("french", (9, 4))]
    pure = [("code", (0, 3)), ("code", (3, 10)), ("code", (0, 10)), ("code", (3, 8)), ("code", (5, 3)),
            ("code", (10, 13)), ("code", (0, 1)), ("code", (3, 6))]
    panel(d, 70, "Mixed batch", mixed, "8 requests from different domains, each token picks its top-2 experts")
    panel(d, 545, "Batch of similar requests", pure, "8 code requests: their tokens keep reusing the same experts (illustration; real savings ~25%)")
    im = im.resize((W, H), Image.LANCZOS)
    im.save("figures/out/fig_mechanism.png")
    print("wrote fig_mechanism")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
