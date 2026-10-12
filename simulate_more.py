# Clustering prompts by their prefill routing (no labels), and the expert-parallel case
# (experts sharded over G GPUs, the step waits for the busiest one).

import argparse
import json

import numpy as np
from sklearn.cluster import KMeans

from simulate import RNG, load, sample_decode_batch, unique_experts


def fingerprint(prefill, E):
    L, P, K = prefill.shape
    f = np.zeros((L, E))
    for l in range(L):
        np.add.at(f[l], prefill[l].reshape(-1).astype(np.int64), 1)
    f = f.ravel()
    return f / np.linalg.norm(f)


def per_gpu_max(tokens, E, G):
    B, L, K = tokens.shape
    shard = E // G
    out = []
    for l in range(L):
        u = np.unique(tokens[:, l])
        out.append(max(((u // shard) == g).sum() for g in range(G)))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--trials", type=int, default=150)
    args = ap.parse_args()
    meta, dec, pre = load(args.trace)
    E = meta["experts"]
    doms = list(dec)
    seqs, labels, fps = [], [], []
    for d in doms:
        n = len(dec[d])
        for i in range(n):
            seqs.append(dec[d][i])
            labels.append(d)
            fps.append(fingerprint(pre[d][i], E))
    fps = np.stack(fps)
    km = KMeans(n_clusters=len(doms), n_init=10, random_state=0).fit(fps)
    clusters = {c: [s for s, k in zip(seqs, km.labels_) if k == c] for c in range(len(doms))}
    purity = sum(max(sum(1 for k, l in zip(km.labels_, labels) if k == c and l == d) for d in doms)
                 for c in range(len(doms))) / len(labels)
    print(f"{meta['model']}: k-means on prompt router fingerprints, purity vs true domains = {purity * 100:.0f}%")
    print("cluster sizes:", sorted(len(v) for v in clusters.values()))

    by_dom = {d: dec[d] for d in doms}
    rows = []
    print(f"{'B':>4} | {'mixed':>7} {'router-clustered':>17} {'domain labels':>14} | {'EP=2 max/GPU: mixed':>20} {'pure':>6}")
    for B in [8, 16, 32, 64]:
        def avg(groups):
            vals = []
            for t in range(args.trials):
                pool = groups[t % len(groups)]
                if len(pool) < B:
                    continue
                vals.append(unique_experts(sample_decode_batch(pool, B), E).mean())
            return float(np.mean(vals))

        m = avg([seqs])
        c = avg([v for v in clusters.values() if len(v) >= B])
        p = avg(list(by_dom.values()))
        ep_m = float(np.mean([per_gpu_max(sample_decode_batch(seqs, B), E, 2) for _ in range(args.trials // 3)]))
        ep_p = float(np.mean([per_gpu_max(sample_decode_batch(by_dom[doms[t % len(doms)]], B), E, 2)
                              for t in range(args.trials // 3)]))
        rows.append(dict(B=B, mixed=m, clustered=c, domain=p, ep2_mixed=ep_m, ep2_pure=ep_p))
        print(f"{B:4d} | {m:7.1f} {c:10.1f} ({100 * (1 - c / m):4.1f}%) {p:7.1f} ({100 * (1 - p / m):4.1f}%) | "
              f"{ep_m:12.1f} {ep_p:12.1f} ({100 * (1 - ep_p / ep_m):4.1f}%)")
    json.dump(dict(purity=purity, rows=rows), open(args.trace.replace(".npz", "_more.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
