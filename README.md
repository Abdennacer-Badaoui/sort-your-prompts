# sort-your-prompts

When an MoE model decodes, every expert that at least one token in the batch routes to has to be read from memory,
once per layer. Prompts from the same domain tend to use the same experts. So if you run a big offline job over
mixed data, simply sorting the prompts so that similar ones are decoded together means fewer experts read per step,
and a faster job. No model or kernel change, the outputs are the same requests computed the same way.

This repo has the experiments behind that: routing traces, a replay simulator, a small Triton fused MoE kernel to
check the time vs experts relation, and an end to end benchmark in vLLM.

## Results

All on AMD MI300X, vLLM 0.27, 768 prompts generating 256 tokens each, greedy. Both runs do exactly the same work,
only the order of the prompt list changes.

| model | workload | batch | shuffled tok/s | sorted tok/s | gain |
|---|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 6 domains | 16 | 1154 | 1310 | +13.5% |
| | | 32 | 1949 | 2286 | +17.3% |
| | | 64 | 2641 | 2974 | +12.6% |
| | | 128 | 5300 | 5915 | +11.6% |
| gpt-oss-120b | 6 domains | 32 | 953 | 1069 | +12.2% |
| | | 64 | 1517 | 1677 | +10.6% |
| | | 128 | 2515 | 2743 | +9.1% |
| Qwen3-30B-A3B | UltraChat, embedding clusters | 32 | 2191 | 2307 | +5.3% |
| gpt-oss-120b | UltraChat, embedding clusters | 64 | 1731 | 1773 | +2.4% |
| Qwen3-30B-A3B, 2 GPUs, EP | 6 domains | 64 | 3440 | 3885 | +12.9% |
| | | 128 | 6835 | 7431 | +8.7% |

A few more things that came out of it:

- In replay, a batch of 32 single-domain requests reads about 25% fewer experts than a mixed one on Qwen3-30B-A3B
  (128 experts, top-8). On gpt-oss-20b (32 experts) it's only 8 to 11%, fewer experts means less room.
- MoE layer time is basically linear in the number of distinct experts in the batch, around 2.9 us per expert.
- You don't need labels. K-means on prompt embeddings from Qwen3-Embedding-0.6B groups as well as the true domain
  labels (+13.0% vs +12.6% at batch 64). Clustering on the router's prefill activations works too.
- Sorting only chunks of 128 or 256 prompts still gives +11.7% and +14.8% (full sort: +17.3%).
- With natural output lengths (EOS on), the sorted job finishes 16.6% faster.
- Outputs: sorted vs shuffled differ about as much as two different shuffles do (Qwen: 92% vs 90% identical
  outputs), which is normal batch nondeterminism in bf16.

This only helps decode, and mostly offline batch jobs. Prefill touches almost every expert anyway, and in online
serving holding requests back to group them costs latency.

Related: ELDR (arXiv 2607.00466) uses the same observation to route requests between decode workers in disaggregated
serving. This repo is the simplest version of the idea, one sort before llm.generate.

## Using it

    from sort_prompts import expert_friendly_order

    order = expert_friendly_order(prompts, n_clusters=16)
    outs = llm.generate([prompts[i] for i in order], sampling_params)
    outs = [o for _, o in sorted(zip(order, outs))]

If you already have labels (dataset name, task, language), pass them with labels=... and skip the embedding step.

## Reproducing

Prompts, routing traces and embeddings:

    python build_prompts.py
    python capture_routing.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --out data/trace_qwen3_30b.npz
    python capture_routing.py --model openai/gpt-oss-20b --out data/trace_gptoss20b.npz
    python capture_routing.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --prompts data/prompts_ultrachat.json \
        --per-domain 768 --out data/trace_qwen3_ultrachat.npz
    python embed_prompts.py --name domains
    python embed_prompts.py --prompts data/prompts_ultrachat.json --name ultrachat

Replay:

    python simulate.py data/trace_qwen3_30b.npz
    python simulate_more.py data/trace_qwen3_30b.npz
    python simulate_sorted.py data/trace_qwen3_ultrachat.npz
    python simulate_grouping.py data/trace_qwen3_30b.npz --embed data/embed_domains.npy --name qwen3_domains --k 6
    python simulate_grouping.py data/trace_qwen3_ultrachat.npz --embed data/embed_ultrachat.npy --name qwen3_ultrachat

Kernel:

    python fused_moe.py
    python bench_fused.py data/trace_qwen3_30b.npz
    python e2e_generate.py --check

vLLM (examples, see vllm_bench.py --help):

    ./run_vllm.sh 0 qwen_b32 --max-num-seqs 32 --save-outputs
    ./run_vllm.sh 0 gptoss_b64 --model openai/gpt-oss-120b --max-num-seqs 64 --repeats 3
    ./run_vllm.sh 0 uc_embed --workload ultrachat --grouping embedding --max-num-seqs 32
    ./run_vllm.sh 0 window256 --window 256 --max-num-seqs 32
    ./run_vllm.sh 0,1 ep --max-num-seqs 64 --tp 2 --ep

## Files

- build_prompts.py, capture_routing.py, embed_prompts.py: data
- simulate*.py: replay of the recorded routing
- fused_moe.py, bench_fused.py, e2e_generate.py: Triton kernel and timings
- vllm_bench.py, run_vllm.sh: end to end runs
- sort_prompts.py: the reordering helper
- figures/: plots and animations
