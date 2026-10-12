# The fused kernel plugged into transformers' Qwen3-MoE (--check compares greedy outputs with the default path).

import argparse
import json
import random
import time

import torch
import triton
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as qm

from fused_moe import fused_moe_kernel

DECODE_CFG = dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=2)
PREFILL_CFG = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=2)


def align_nosync(topk_ids, E, bm):
  : worst-case buffers, the kernel skips blocks past n_padded
    flat = topk_ids.reshape(-1)
    nv = flat.numel()
    counts = torch.zeros(E, dtype=torch.long, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + bm - 1) // bm * bm
    ends = torch.cumsum(padded, 0)
    starts = ends - padded
    order = torch.argsort(flat, stable=True)
    rank = torch.arange(nv, device=flat.device) - (torch.cumsum(counts, 0) - counts)[flat[order]]
    max_total = nv + E * (bm - 1)
    max_total = (max_total + bm - 1) // bm * bm
    sorted_ids = torch.full((max_total,), nv, dtype=torch.int32, device=flat.device)
    sorted_ids[starts[flat[order]] + rank] = order.to(torch.int32)
    block_start = torch.arange(0, max_total, bm, device=flat.device)
    expert_ids = torch.searchsorted(ends, block_start, right=True).clamp_(max=E - 1).to(torch.int32)
    return sorted_ids, expert_ids, ends[-1:].to(torch.int32)


def gemm(a, w, sid, eid, npad, top_k, cfg, rows):
    E, N, K = w.shape
    c = torch.empty(rows, N, device=a.device, dtype=a.dtype)
    EM = sid.numel()
    grid = (triton.cdiv(EM, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    fused_moe_kernel[grid](a, w, c, sid, eid, npad, N, K, EM, rows, a.stride(0), a.stride(1), w.stride(0), w.stride(2),
                           w.stride(1), c.stride(0), c.stride(1), top_k=top_k, BLOCK_M=cfg["BLOCK_M"],
                           BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], GROUP_M=1, num_warps=cfg["num_warps"],
                           num_stages=cfg["num_stages"])
    return c


def fused_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    T, K = top_k_index.shape
    cfg = DECODE_CFG if T <= 256 else PREFILL_CFG
    sid, eid, npad = align_nosync(top_k_index, self.num_experts, cfg["BLOCK_M"])
    h = gemm(hidden_states, self.gate_up_proj, sid, eid, npad, K, cfg, T * K)
    g, u = h.chunk(2, dim=-1)
    y = gemm(self.act_fn(g) * u, self.down_proj, sid, eid, npad, 1, cfg, T * K).view(T, K, -1)
    return (y * top_k_weights[..., None].to(y.dtype)).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--sizes", type=int, nargs="*", default=[16, 32, 64])
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    prompts = json.load(open("data/prompts.json"))
    doms = list(prompts)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    def encode(batch, length):
        msgs = [[{"role": "user", "content": p}] for p in batch]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                       padding="max_length", max_length=length, truncation=True).to("cuda")

    if args.check:
        enc = encode([prompts[d][0] for d in doms], 160)
        ref = model.generate(**enc, max_new_tokens=32, do_sample=False)
        qm.Qwen3MoeExperts.forward = fused_experts_forward
        out = model.generate(**enc, max_new_tokens=32, do_sample=False)
        same = (ref == out).float().mean().item()
        print(f"greedy tokens identical to transformers' implementation: {same * 100:.1f}%")
        print(tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)[:200])
        return

    qm.Qwen3MoeExperts.forward = fused_experts_forward
    results = []
    for B in args.sizes:
        pool = {d: prompts[d][:B] for d in doms}
        pure = [pool[d] for d in doms]
        flat = [p for d in doms for p in pool[d]]
        rng = random.Random(B)
        rng.shuffle(flat)
        mixed = [flat[i * B : (i + 1) * B] for i in range(len(doms))]
        length = max(len(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True))
                     for p in flat)
        length = min(length, 512)
        kw = dict(max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False,
                  pad_token_id=tok.pad_token_id)
        model.generate(**encode(mixed[0], length), max_new_tokens=8, do_sample=False)
        times = {"pure": [], "mixed": []}
        for rep in range(args.repeats):
            for name, batches in (("pure", pure), ("mixed", mixed)) if rep % 2 == 0 else (("mixed", mixed), ("pure", pure)):
                torch.cuda.synchronize()
                t0 = time.time()
                for b in batches:
                    model.generate(**encode(b, length), **kw)
                torch.cuda.synchronize()
                times[name].append(time.time() - t0)
        toks = len(doms) * B * args.new_tokens
        tp = {k: toks / min(v) for k, v in times.items()}
        results.append(dict(B=B, prompt_len=length, tokens=toks, tok_per_s=tp, seconds=times))
        print(f"B={B:3d} prompt_len={length} | mixed {tp['mixed']:7.1f} tok/s | pure {tp['pure']:7.1f} tok/s | "
              f"{100 * (tp['pure'] / tp['mixed'] - 1):+5.1f}%", flush=True)
    json.dump(results, open("data/e2e_qwen3_30b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
