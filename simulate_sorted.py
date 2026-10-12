# Sort requests by router cluster, cut into consecutive batches, compare with random order.

import argparse
import json

import numpy as np
from sklearn.cluster import KMeans

from simulate import RNG, load, unique_experts
from simulate_more import fingerprint


def batches_from_order(seqs, order, B, E, steps=6):
    vals = []
    for b0 in range(0, len(order) - B + 1, B):
        group = [seqs[i] for i in order[b0 : b0 + B]]
        for _ in range(steps):
            toks = np.stack([s[:, RNG.integers(s.shape[1])] for s in group])
            vals.append(unique_experts(toks, E).mean())
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    args = ap.parse_args()
    meta, dec, pre = load(args.trace)
    E = meta["experts"]
    seqs, fps = [], []
    for d in dec:
        for i in range(len(dec[d])):
            seqs.append(dec[d][i])
            fps.append(fingerprint(pre[d][i], E))
    fps = np.stack(fps)
    n = len(seqs)
    print(f"{meta['model']} on {args.trace.split('/')[-1]}: {n} requests (labels not used)")
    rows = []
    for k in (4, 8, 16, 32):
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(fps)
        order_sorted = np.argsort(lab, kind="stable")
        for B in (8, 16, 32, 64):
            rand = np.mean([batches_from_order(seqs, RNG.permutation(n), B, E) for _ in range(3)])
            srt = batches_from_order(seqs, order_sorted, B, E)
            rows.append(dict(k=k, B=B, random=float(rand), sorted=srt))
            print(f"k={k:2d} B={B:3d} | random order {rand:6.1f} experts | cluster-sorted {srt:6.1f} "
                  f"({100 * (1 - srt / rand):4.1f}% fewer)", flush=True)
    json.dump(rows, open(args.trace.replace(".npz", "_sorted.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
