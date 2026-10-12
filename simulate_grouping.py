# Grouping strategies compared on replayed routing: random order, domain label, router clusters,
# text-embedding clusters. Also sorting only inside windows of W requests.

import argparse
import json

import numpy as np
from sklearn.cluster import KMeans

from simulate import load, unique_experts
from simulate_more import fingerprint
from simulate_sorted import batches_from_order


def windowed(order_rand, lab, W):
    out = []
    for w0 in range(0, len(order_rand), W):
        win = order_rand[w0 : w0 + W]
        out.extend(win[np.argsort(lab[win], kind="stable")])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--embed", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    meta, dec, pre = load(args.trace)
    E = meta["experts"]
    seqs, fps, dlab = [], [], []
    for j, d in enumerate(dec):
        for i in range(len(dec[d])):
            seqs.append(dec[d][i])
            fps.append(fingerprint(pre[d][i], E))
            dlab.append(j)
    fps, dlab = np.stack(fps), np.array(dlab)
    emb = np.load(args.embed)
    n = len(seqs)
    k = args.k
    labs = {"router": KMeans(k, n_init=10, random_state=0).fit_predict(fps),
            "embedding": KMeans(k, n_init=10, random_state=0).fit_predict(emb)}
    if len(dec) > 1:
        labs["label"] = dlab
    rng = np.random.default_rng(1)
    rand_orders = [rng.permutation(n) for _ in range(4)]
    res = {"model": meta["model"], "k": k, "by_B": {}, "windows": {}}
    for B in (8, 16, 32, 64, 128):
        r = float(np.mean([batches_from_order(seqs, o, B, E) for o in rand_orders]))
        row = {"random": r}
        for name, lab in labs.items():
            row[name] = batches_from_order(seqs, np.argsort(lab, kind="stable"), B, E)
        res["by_B"][B] = row
        print(f"B={B:3d} " + " | ".join(f"{a} {v:6.1f} ({100 * (1 - v / r):4.1f}%)" for a, v in row.items()), flush=True)
    # reorder only inside windows of W requests, B=32
    best = "label" if "label" in labs else "router"
    for W in (32, 64, 128, 256, 512, n):
        v = float(np.mean([batches_from_order(seqs, windowed(o, labs[best], W), 32, E) for o in rand_orders]))
        res["windows"][W] = v
        print(f"window W={W:4d} ({best}) B=32: {v:6.1f} ({100 * (1 - v / res['by_B'][32]['random']):4.1f}% fewer)")
    L = meta["layers"]

    def per_layer(order, B=32, steps=6):
        acc = np.zeros(L)
        cnt = 0
        for b0 in range(0, n - B + 1, B):
            group = [seqs[i] for i in order[b0 : b0 + B]]
            for _ in range(steps):
                toks = np.stack([s[:, rng.integers(s.shape[1])] for s in group])
                acc += unique_experts(toks, E)
                cnt += 1
        return (acc / cnt).tolist()

    res["per_layer"] = {"random": per_layer(rand_orders[0]), best: per_layer(np.argsort(labs[best], kind="stable"))}
    json.dump(res, open(f"data/grouping_{args.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
