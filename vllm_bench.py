# Same prompts and same amount of work in vLLM, only the submission order changes.
# The offline engine is FCFS, so the order decides which requests decode together.

import argparse
import json
import os
import time

import numpy as np
from sklearn.cluster import KMeans
from vllm import LLM, SamplingParams


def load_workload(workload, grouping, k):
    if workload == "domains":
        P = json.load(open("data/prompts.json"))
        prompts = [p for d in P for p in P[d]]
        dlab = np.array([j for j, d in enumerate(P) for _ in P[d]])
    else:
        prompts = json.load(open("data/prompts_ultrachat.json"))["chat"]
        dlab = None
    if grouping == "label":
        return prompts, dlab
    if grouping == "embedding":
        X = np.load(f"data/embed_{workload}.npy")
    else:
        from simulate import load
        from simulate_more import fingerprint

        meta, _, pre = load("data/trace_qwen3_30b.npz" if workload == "domains" else "data/trace_qwen3_ultrachat.npz")
        X = np.stack([fingerprint(x, meta["experts"]) for d in pre for x in pre[d]])
    return prompts, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


def make_orders(lab, window, seed):
    n = len(lab)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    if window and window < n:
        srt = np.concatenate([w[np.argsort(lab[w], kind="stable")] for w in np.array_split(shuffled, range(window, n, window))])
    else:
        srt = np.argsort(lab, kind="stable")
    return srt, shuffled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--workload", choices=["domains", "ultrachat"], default="domains")
    ap.add_argument("--grouping", choices=["label", "router", "embedding"], default="label")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--save-outputs", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    prompts, lab = load_workload(args.workload, args.grouping, args.k)
    srt, shf = make_orders(lab, args.window, seed=args.max_num_seqs)
    kw = dict(tensor_parallel_size=args.tp, enable_expert_parallel=args.ep) if args.tp > 1 else {}
    llm = LLM(args.model, max_num_seqs=args.max_num_seqs, max_model_len=2048, gpu_memory_utilization=0.85, seed=0,
              **kw)
    tok = llm.get_tokenizer()
    if args.natural:
        sp = SamplingParams(temperature=0.0, max_tokens=512)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    text = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts]
    # warm up all batch shapes first, otherwise triton compiles show up in the timings
    llm.generate([text[i] for i in shf], SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)
    times = {"sorted": [], "shuffled": []}
    toks = {}
    for rep in range(args.repeats):
        for name in (("sorted", "shuffled") if rep % 2 == 0 else ("shuffled", "sorted")):
            order = srt if name == "sorted" else shf
            t0 = time.perf_counter()
            outs = llm.generate([text[i] for i in order], sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            n = sum(len(o.outputs[0].token_ids) for o in outs)
            times[name].append(dict(tok_s=n / dt, seconds=dt, tokens=n))
            toks[name] = {int(i): list(o.outputs[0].token_ids) for i, o in zip(order, outs)}
            print(f"  rep {rep} {name:8s}: {n / dt:8.0f} output tok/s ({dt:.1f}s, {n} tokens)", flush=True)
    res = {k: dict(tok_s=float(np.mean([r["tok_s"] for r in v])), seconds=float(np.mean([r["seconds"] for r in v])),
                   runs=v) for k, v in times.items()}
    gain = 100 * (res["sorted"]["tok_s"] / res["shuffled"]["tok_s"] - 1)
    wall = 100 * (res["shuffled"]["seconds"] / res["sorted"]["seconds"] - 1)
    out = dict(args=vars(args), result=res, gain_pct=gain, wall_gain_pct=wall)
    if args.save_outputs:
        # control: another random order, since any change of batch composition can flip greedy ties in bf16
        shf2 = np.random.default_rng(12345).permutation(len(text))
        outs = llm.generate([text[i] for i in shf2], sp, use_tqdm=False)
        toks["shuffled2"] = {int(i): list(o.outputs[0].token_ids) for i, o in zip(shf2, outs)}

        def compare(a, b):
            same = [a[i] == b[i] for i in a]
            div = [next((j for j, (x, y) in enumerate(zip(a[i], b[i])) if x != y), None) for i in a]
            div = [d for d in div if d is not None]
            return dict(identical_frac=float(np.mean(same)), median_first_divergence=float(np.median(div)) if div else None)

        out["identity"] = {"sorted_vs_shuffled": compare(toks["sorted"], toks["shuffled"]),
                           "shuffled_vs_shuffled2": compare(toks["shuffled"], toks["shuffled2"])}
        print("IDENTITY", out["identity"], flush=True)
    print(f"RESULT {args.model} {args.workload}/{args.grouping} W={args.window} B={args.max_num_seqs} tp={args.tp} "
          f"ep={args.ep} natural={args.natural}: shuffled {res['shuffled']['tok_s']:.0f} tok/s, sorted "
          f"{res['sorted']['tok_s']:.0f} tok/s -> {gain:+.1f}% (wall {wall:+.1f}%)", flush=True)
    name = os.path.basename(args.model).lower()
    tag = f"_{args.tag}" if args.tag else ""
    os.makedirs("data/vllm", exist_ok=True)
    json.dump(out, open(f"data/vllm/{name}_{args.workload}_{args.grouping}_w{args.window}_b{args.max_num_seqs}"
                        f"_tp{args.tp}{'ep' if args.ep else ''}{'_nat' if args.natural else ''}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
