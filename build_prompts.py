# 6 domains x 128 prompts from public datasets, plus 768 UltraChat prompts

import json
import random

from datasets import load_dataset

N = 128
random.seed(0)


def take(rows, n=N):
    rows = [r.strip() for r in rows if r and 20 <= len(r.strip()) <= 1500]
    random.shuffle(rows)
    return rows[:n]


def main():
    out = {}
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    out["code"] = take([r["text"] + " Write it in Python." for r in mbpp])
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    out["math"] = take([r["question"] for r in gsm])
    nr = load_dataset("HuggingFaceH4/no_robots", split="train")
    out["creative"] = take([r["prompt"] for r in nr if r["category"] == "Generation"])
    aya = load_dataset("CohereLabs/aya_dataset", split="train")
    for name, langs in (("french", {"French"}), ("arabic", {"Standard Arabic", "Moroccan Arabic", "Egyptian Arabic"}),
                        ("chinese", {"Simplified Chinese", "Chinese"})):
        out[name] = take([r["inputs"] for r in aya if r["language"] in langs])
    for k, v in out.items():
        print(f"{k:9s} {len(v):4d}  e.g. {v[0][:90]!r}")
    json.dump(out, open("data/prompts.json", "w"), ensure_ascii=False, indent=1)

    # unlabeled chat traffic
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_gen")
    rows = [r["prompt"].strip() for r in uc if 20 <= len(r["prompt"].strip()) <= 1500]
    random.Random(0).shuffle(rows)
    json.dump({"chat": rows[:768]}, open("data/prompts_ultrachat.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
