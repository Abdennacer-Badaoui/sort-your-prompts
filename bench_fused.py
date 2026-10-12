# MoE layer time on real routing, single-domain vs mixed batches.

import argparse
import json

import numpy as np
import torch

from fused_moe import align_block_size, grouped_gemm
from simulate import load, sample_decode_batch

CONFIGS = [dict(BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s)
           for bn in (64, 128) for bk in (64, 128, 256) for w in (4, 8) for s in (1, 2)]


def prepared(batches, layer, E, H, block_m):
    out = []
    for b in batches:
        ids = torch.as_tensor(b[:, layer].astype(np.int64), device="cuda")
        x = torch.randn(ids.shape[0], H, device="cuda", dtype=torch.bfloat16)
        out.append((x, ids, align_block_size(ids, E, block_m)))
    return out


def run(items, w1, w2, cfg):
    for x, ids, (sid, eid, npad) in items:
        T, K = ids.shape
        h = grouped_gemm(x, w1, sid, eid, npad, K, cfg, T * K)
        g, u = h.chunk(2, dim=-1)
        grouped_gemm(torch.nn.functional.silu(g) * u, w2, sid, eid, npad, 1, cfg, T * K)


def timed(items, w1, w2, cfg, reps=10):
    run(items, w1, w2, cfg)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(items, w1, w2, cfg)
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / len(items))
    return float(np.median(ts))


def try_time(items, w1, w2, cfg):
    try:
        return timed(items, w1, w2, cfg, reps=3)
    except Exception:
        return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768)
    ap.add_argument("--layers", type=int, nargs="*", default=[4, 24, 44])
    ap.add_argument("--sizes", type=int, nargs="*", default=[1, 4, 8, 16, 32, 64, 128])
    args = ap.parse_args()
    meta, dec, _ = load(args.trace)
    E, H, I = meta["experts"], args.hidden, args.inter
    expert_bytes = (H * 2 * I + I * H) * 2
    w1 = torch.randn(E, H, 2 * I, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16) * 0.02
    doms = list(dec)
    pooled = [s for d in doms for s in dec[d]]
    rows = []
    for B in args.sizes:
        mixed_b = [sample_decode_batch(pooled, B) for _ in range(30)]
        pure_b = [sample_decode_batch(dec[doms[i % len(doms)]], B) for i in range(30)]
        for layer in args.layers:
            mixed = prepared(mixed_b, layer, E, H, 16)
            pure = prepared(pure_b, layer, E, H, 16)
            # tuned on mixed batches, same config for both
            best = min(CONFIGS, key=lambda c: try_time(mixed[:8], w1, w2, c))
            tm, tp = timed(mixed, w1, w2, best), timed(pure, w1, w2, best)
            um = np.mean([len(np.unique(b[:, layer])) for b in mixed_b])
            up = np.mean([len(np.unique(b[:, layer])) for b in pure_b])
            bw = um * expert_bytes / (tm * 1e-6) / 1e12
            rows.append(dict(B=B, layer=layer, us_mixed=tm, us_pure=tp, experts_mixed=um, experts_pure=up, tbps_mixed=bw,
                             cfg=best))
            print(f"B={B:4d} L{layer:2d} | experts {um:6.1f} -> {up:6.1f} | {tm:7.1f} us -> {tp:7.1f} us "
                  f"({100 * (1 - tp / tm):+5.1f}%) | {bw:4.2f} TB/s", flush=True)
    json.dump(dict(meta=meta, hidden=H, inter=I, rows=rows), open(args.trace.replace(".npz", "_fused.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
