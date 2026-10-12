# Static figures, read from data/ and written to figures/out/.

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/figures/out"
FONTS = os.environ.get("FONTS_DIR", f"{ROOT}/figures/fonts")
for f in glob.glob(f"{FONTS}/Inter*.ttf"):
    font_manager.fontManager.addfont(f)

SURF, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8c8b86", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BASE = "#9a9994"

plt.rcParams.update({
    "font.family": "Inter", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK3, "ytick.color": INK3, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "axes.titlelocation": "left", "axes.titlesize": 13,
    "axes.titleweight": 600, "axes.titlecolor": INK, "legend.frameon": False, "legend.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2, "lines.markersize": 7,
})


def j(p):
    return json.load(open(f"{ROOT}/data/{p}"))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def fig_experts_vs_batch():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (f, title, E) in zip(axes, [("trace_qwen3_30b_sim.json", "Qwen3-30B-A3B  ·  128 experts, top-8", 128),
                                        ("trace_gptoss20b_sim.json", "gpt-oss-20b  ·  32 experts, top-4", 32)]):
        r = j(f)["decode"]
        Bs = [int(b) for b in r["mixed"] if int(b) >= 2]
        for key, col, lab, ls in [("uniform", GRID, "if routing were uniform", "--"), ("mixed", BASE, "random order (mixed domains)", "-"),
                                  ("pure", BLUE, "sorted (one domain per batch)", "-")]:
            ys = [100 * r[key][str(b)] / E for b in Bs]
            ax.plot(Bs, ys, color=col if key != "uniform" else "#c9c8c2", ls=ls, marker="o" if key != "uniform" else None,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=lab, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Bs, [str(b) for b in Bs])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("requests in the decode batch")
        b = 32
        gap = 100 * (1 - r["pure"][str(b)] / r["mixed"][str(b)])
        ax.annotate(f"−{gap:.0f}%", (b, 100 * r["pure"][str(b)] / E), xytext=(8, -16), textcoords="offset points",
                    color=INK, fontsize=11, fontweight=600)
    axes[0].set_ylabel("experts read per layer, % of all")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="upper left", fontsize=10)
    fig.suptitle("Fewer distinct experts per decode step when a batch holds similar requests", x=0.065, ha="left",
                 fontsize=14, fontweight=650, y=1.03)
    save(fig, "fig_experts_vs_batch")


def fig_per_layer():
    g = j("grouping_qwen3_domains.json")
    r, s = g["per_layer"]["random"], g["per_layer"]["label"]
    sav = [100 * (1 - b / a) for a, b in zip(r, s)]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(range(1, 49), sav, color=BLUE, width=0.75)
    ax.set_xlim(0.2, 48.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("fewer experts read, %")
    ax.set_title("Later layers are more domain-specialised, so they save the most  ·  Qwen3-30B-A3B, batch 32")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_per_layer")


def fig_groupers():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (f, title) in zip(axes, [("grouping_qwen3_domains.json", "6-domain workload"),
                                     ("grouping_qwen3_ultrachat.json", "UltraChat (no labels exist)")]):
        g = j(f)["by_B"]
        Bs = [int(b) for b in g]
        for key, col, lab in [("label", BLUE, "domain label"), ("router", ORANGE, "router fingerprint (k-means)"),
                              ("embedding", AQUA, "text embedding, 0.6B (k-means)")]:
            if key not in g[str(Bs[0])]:
                continue
            ys = [100 * (1 - g[str(b)][key] / g[str(b)]["random"]) for b in Bs]
            ax.plot(Bs, ys, color=col, marker="o", markeredgecolor=SURF, markeredgewidth=1.5, label=lab, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Bs, [str(b) for b in Bs])
        ax.set_xlabel("requests in the decode batch")
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel("fewer experts read vs random, %")
    axes[0].set_ylim(0, 30)
    axes[0].legend(loc="lower left", fontsize=10)
    fig.suptitle("You don't need labels: a small embedding model groups requests almost as well as the router",
                 x=0.065, ha="left", fontsize=14, fontweight=650, y=1.03)
    save(fig, "fig_groupers")


def fig_window():
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for f, col, lab in [("grouping_qwen3_domains.json", BLUE, "6 domains, sorted by label"),
                        ("grouping_qwen3_ultrachat.json", ORANGE, "UltraChat, sorted by router cluster")]:
        g = j(f)
        base = g["by_B"]["32"]["random"]
        Ws = [int(w) for w in g["windows"]]
        ax.plot(Ws, [100 * (1 - g["windows"][str(w)] / base) for w in Ws], color=col, marker="o",
                markeredgecolor=SURF, markeredgewidth=1.5, label=lab, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ws, [str(w) for w in Ws[:-1]] + ["all\n(768)"])
    ax.set_xlabel("reorder window (requests sorted together)")
    ax.set_ylabel("fewer experts read, %")
    ax.set_ylim(0, 28)
    ax.set_title("Sorting only a sliding window keeps most of the gain  ·  batch 32")
    ax.legend(loc="upper left", fontsize=10)
    save(fig, "fig_window")


def fig_vllm():
    def g(f):
        return json.load(open(f"{ROOT}/data/vllm/{f}"))["gain_pct"]

    q = "qwen3-30b-a3b-instruct-2507"
    series = [
        ("Qwen3-30B-A3B · 6 domains", BLUE, {16: g("vllm_domains_b16.json"), 32: g(f"{q}_domains_label_w0_b32_tp1.json"),
                                            64: g("vllm_domains_b64.json"), 128: g("vllm_domains_b128.json")}),
        ("gpt-oss-120b · 6 domains", AQUA, {32: g("gpt-oss-120b_domains_label_w0_b32_tp1.json"),
                                           64: g("gpt-oss-120b_domains_label_w0_b64_tp1.json"),
                                           128: g("gpt-oss-120b_domains_label_w0_b128_tp1.json")}),
        ("Qwen3-30B-A3B · UltraChat, no labels", ORANGE, {16: g("vllm_ultrachat_b16.json"),
                                                          32: g(f"{q}_ultrachat_embedding_w0_b32_tp1.json"),
                                                          64: g("vllm_ultrachat_b64.json")}),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    Bs = [16, 32, 64, 128]
    w = 0.27
    for k, (lab, col, d) in enumerate(series):
        xs = [i + (k - 1) * (w + 0.015) for i, b in enumerate(Bs) if b in d]
        ys = [d[b] for b in Bs if b in d]
        ax.bar(xs, ys, width=w, color=col, label=lab, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.3, f"+{y:.0f}%", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight=600)
    ax.set_xticks(range(len(Bs)), [str(b) for b in Bs])
    ax.set_xlabel("vLLM max_num_seqs (decode batch size)")
    ax.set_ylabel("output throughput gain, %")
    ax.set_ylim(0, 20)
    ax.grid(axis="x", visible=False)
    ax.set_title("Measured in vLLM: same prompts, same work, only the order changes", pad=34)
    ax.legend(loc="lower left", fontsize=9.5, bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1.2, columnspacing=1.2)
    ax.text(0, -0.22, "One MI300X, vLLM 0.27, 768 prompts × 256 tokens, greedy. Run-to-run spread: <0.5% (Qwen), up to 4% (gpt-oss); every sorted run beat every shuffled run.",
            transform=ax.transAxes, fontsize=9, color=INK3)
    save(fig, "fig_vllm")


def fig_kernel():
    import numpy as np

    d = j("trace_qwen3_30b_fused.json")
    rows = [r for r in d["rows"] if r["B"] >= 4]
    fig, ax = plt.subplots(figsize=(6.8, 4))
    for key, col, lab in [("mixed", BASE, "mixed batch"), ("pure", BLUE, "one-domain batch")]:
        ax.scatter([r[f"experts_{key}"] for r in rows], [r[f"us_{key}"] for r in rows], s=46, color=col,
                   edgecolor=SURF, linewidth=1.5, label=lab, zorder=3)
    x = np.array([r[f"experts_{k}"] for r in rows for k in ("mixed", "pure")])
    y = np.array([r[f"us_{k}"] for r in rows for k in ("mixed", "pure")])
    a, b = np.polyfit(x, y, 1)
    xs = np.linspace(0, 128, 50)
    ax.plot(xs, a * xs + b, color="#c9c8c2", lw=1.5, ls="--", zorder=2)
    ax.text(78, a * 78 + b - 95, f"≈ {a:.1f} µs per extra expert\n(9.4 MB at {9.44 / a:.1f} TB/s)", fontsize=10, color=INK2)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, None)
    ax.set_xlabel("distinct experts in the batch (one layer)")
    ax.set_ylabel("fused-MoE layer time, µs")
    ax.set_title("MoE decode time is set by how many experts the batch touches")
    ax.legend(loc="upper left", fontsize=10)
    save(fig, "fig_kernel")


def fig_domain_experts(layer=40):
    import numpy as np
    import sys

    sys.path.insert(0, ROOT)
    from simulate import load

    meta, dec, _ = load(f"{ROOT}/data/trace_qwen3_30b.npz")
    E = meta["experts"]
    names = {"code": "code", "math": "math", "creative": "writing", "french": "French", "arabic": "Arabic",
             "chinese": "Chinese"}
    doms = list(dec)
    F = np.zeros((len(doms), E))
    cover = []
    for i, d in enumerate(doms):
        for s in dec[d]:
            np.add.at(F[i], s[layer].ravel().astype(np.int64), 1)
        F[i] /= F[i].sum()
    for i in range(len(doms)):
        cover.append(np.sort(F[i])[::-1][:32].sum())
    order = np.lexsort((-F.max(0), F.argmax(0)))
    fig, ax = plt.subplots(figsize=(10, 2.9))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("b", [SURF, "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    ax.imshow(F[:, order] / F.max(), aspect="auto", cmap=cmap, interpolation="nearest", vmax=0.6)
    ax.set_yticks(range(len(doms)), [f"{names[d]}" for d in doms])
    ax.set_xticks([])
    ax.grid(False)
    for i, c in enumerate(cover):
        ax.text(E + 1.5, i, f"{100 * c:.0f}%", va="center", fontsize=10, color=INK)
    ax.text(E + 1.5, -0.95, "top-32\nshare", va="bottom", fontsize=9, color=INK3)
    ax.set_xlabel(f"128 experts of layer {layer + 1}, grouped by the domain that uses them most")
    ax.set_title("Each domain lives in its own corner of the experts  ·  Qwen3-30B-A3B, decode tokens", pad=10)
    save(fig, "fig_domain_experts")
    print("top-32 share:", {names[d]: round(float(100 * c), 1) for d, c in zip(doms, cover)}, "uniform = 25%")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_experts_vs_batch()
    fig_per_layer()
    fig_groupers()
    fig_window()
    fig_vllm()
    fig_kernel()
    fig_domain_experts()
