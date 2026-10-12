# Distinct experts per layer per decode step, mixed vs single-domain batches.
# A decode batch = B different sequences, each at a random position of its generation.

import argparse
import json

import numpy as np

RNG = np.random.default_rng(0)


def load(path):
    meta = json.load(open(path.replace(".npz", ".json")))
    raw = np.load(path)
    dec, pre = {}, {}
    for d, n in meta["domains"].items():
        dec[d] = [raw[f"{d}/{i}/decode"] for i in range(n) if f"{d}/{i}/decode" in raw]
        pre[d] = [raw[f"{d}/{i}/prefill"] for i in range(n)]
    return meta, dec, pre


def unique_experts(tokens, E):
    B, L, K = tokens.shape
    onehot = np.zeros((L, E), bool)
    li = np.repeat(np.arange(L)[None, :, None], B, 0).repeat(K, 2)
    onehot[li.ravel(), tokens.ravel().astype(np.int64)] = True
    return onehot.sum(1)


def sample_decode_batch(seqs, B):
    idx = RNG.choice(len(seqs), size=B, replace=False)
    out = []
    for i in idx:
        s = seqs[i]
        out.append(s[:, RNG.integers(s.shape[1])])
    return np.stack(out)


def run(meta, dec, sizes, trials=200):
    E, K = meta["experts"], meta["top_k"]
    doms = list(dec)
    pooled = [s for d in doms for s in dec[d]]
    res = {"mixed": {}, "pure": {}, "pure_by_domain": {d: {} for d in doms}, "uniform": {}}
    for B in sizes:
        if B > min(len(dec[d]) for d in doms):
            break
        mixed = [unique_experts(sample_decode_batch(pooled, B), E).mean() for _ in range(trials)]
        res["mixed"][B] = float(np.mean(mixed))
        per = []
        for d in doms:
            u = [unique_experts(sample_decode_batch(dec[d], B), E).mean() for _ in range(trials // len(doms) + 5)]
            res["pure_by_domain"][d][B] = float(np.mean(u))
            per.append(np.mean(u))
        res["pure"][B] = float(np.mean(per))
        res["uniform"][B] = float(E * (1 - (1 - K / E) ** B))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    args = ap.parse_args()
    meta, dec, pre = load(args.trace)
    sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    res = run(meta, dec, sizes)
    E = meta["experts"]
    print(f"{meta['model']}: {meta['layers']} layers x {E} experts, top-{meta['top_k']}  "
          f"(decode tokens: {sum(s.shape[1] for v in dec.values() for s in v):,})")
    print(f"{'B':>4} | {'uniform':>8} {'mixed':>8} {'pure':>8} | experts loaded: pure vs mixed")
    for B in res["mixed"]:
        u, m, p = res["uniform"][B], res["mixed"][B], res["pure"][B]
        print(f"{B:4d} | {u:8.1f} {m:8.1f} {p:8.1f} | {100 * (1 - p / m):5.1f}% fewer")
    print("per domain (pure, B=32):", {d: round(v.get(32, float('nan')), 1) for d, v in res["pure_by_domain"].items()})
    json.dump(dict(meta=meta, decode=res), open(args.trace.replace(".npz", "_sim.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
