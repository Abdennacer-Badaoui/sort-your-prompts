# Usage:
#   order = expert_friendly_order(prompts)
#   outs = llm.generate([prompts[i] for i in order], sp)
#   outs = [o for _, o in sorted(zip(order, outs))]

import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer


def embed(texts, model_id="Qwen/Qwen3-Embedding-0.6B", batch_size=64, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
            h = model(**enc.to(device)).last_hidden_state[:, -1].float()
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    return np.concatenate(out)


def expert_friendly_order(texts, n_clusters=16, labels=None):
    if labels is None:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(embed(texts))
    return np.argsort(np.asarray(labels), kind="stable").tolist()
