# Records top-k expert ids for every prompt token and generated token.
# Routing only depends on the sequence itself, so any batch composition can be replayed offline from these traces.

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def router_modules(model):
    mods = []
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        r = getattr(mlp, "router", None) or getattr(mlp, "gate", None)
        if r is not None and hasattr(r, "weight") and r.weight.shape[0] > 8:
            mods.append((i, r))
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--per-domain", type=int, default=128)
    ap.add_argument("--prompts", default="data/prompts.json")
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16, device_map="auto")
    if "gpt-oss" in args.model:
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    mods = router_modules(model)
    cfg = model.config
    E = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts")
    K = cfg.num_experts_per_tok
    print(f"{args.model}: {len(mods)} MoE layers, {E} experts, top-{K}")

    calls = []

    def hook(module, inputs, output):
        logits = output[0] if isinstance(output, tuple) else output
        calls[-1].append(torch.topk(logits.float(), K, dim=-1).indices.to(torch.int16).cpu())

    for _, m in mods:
        m.register_forward_hook(hook)

    # one entry per forward call: the first is the prefill, then one per decode step
    orig_forward = model.forward

    def forward(*a, **k):
        calls.append([])
        return orig_forward(*a, **k)

    model.forward = forward

    traces = {}
    t0 = time.time()
    for dom, plist in prompts.items():
        plist = plist[: args.per_domain]
        traces[dom] = []
        for b0 in range(0, len(plist), args.batch):
            batch = plist[b0 : b0 + args.batch]
            msgs = [[{"role": "user", "content": p}] for p in batch]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                          padding=True).to(model.device)
            calls.clear()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            B, S = enc["input_ids"].shape
            mask = enc["attention_mask"].bool().cpu()
            pre = torch.stack(calls[0]).view(len(mods), B, S, K)
            dec = torch.stack([torch.stack(c) for c in calls[1:]], 2) if len(calls) > 1 else None
            gen = out[:, S:].cpu()
            eos = set(np.atleast_1d(model.generation_config.eos_token_id).tolist()) | {tok.pad_token_id}
            for i in range(B):
                g = gen[i].tolist()
                n = next((j for j, t in enumerate(g) if t in eos), len(g))
                n = min(n, dec.shape[2] if dec is not None else 0)
                traces[dom].append(dict(prefill=pre[:, i, mask[i]].numpy(), decode=dec[:, i, :n].numpy() if n else None))
            print(f"[{dom}] {b0 + len(batch)}/{len(plist)}  {time.time() - t0:.0f}s", flush=True)

    flat = {}
    meta = dict(model=args.model, layers=len(mods), experts=E, top_k=K, domains={})
    for dom, ts in traces.items():
        meta["domains"][dom] = len(ts)
        for i, t in enumerate(ts):
            flat[f"{dom}/{i}/prefill"] = t["prefill"]
            if t["decode"] is not None:
                flat[f"{dom}/{i}/decode"] = t["decode"]
    np.savez_compressed(args.out, **flat)
    json.dump(meta, open(args.out.replace(".npz", ".json"), "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
