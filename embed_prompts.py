# Prompt embeddings with Qwen3-Embedding-0.6B (last-token pooling), same row order as the prompt file.

import argparse
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMB = "Qwen/Qwen3-Embedding-0.6B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.json")
    ap.add_argument("--name", default="domains")
    args = ap.parse_args()
    P = json.load(open(args.prompts))
    texts = [p for d in P for p in P[d]]
    tok = AutoTokenizer.from_pretrained(EMB, padding_side="left")
    model = AutoModel.from_pretrained(EMB, dtype=torch.bfloat16).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i : i + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state[:, -1].float()
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    X = np.concatenate(out)
    np.save(f"data/embed_{args.name}.npy", X)
    print(args.name, X.shape)


if __name__ == "__main__":
    main()
