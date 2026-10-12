# bf16 fused MoE kernel, same layout as vLLM's fused_moe_kernel: tokens sorted by expert and padded to BLOCK_M,
# each program only reads the weights of its block's expert.

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, sorted_ids_ptr, expert_ids_ptr, n_padded_ptr,
                     N, K, EM, num_valid, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
                     top_k: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    gsize = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % gsize)
    pid_n = (pid % num_in_group) // gsize
    if pid_m * BLOCK_M >= tl.load(n_padded_ptr):
        return
    offs_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_tok = tl.load(sorted_ids_ptr + offs_id).to(tl.int64)
    tmask = offs_tok < num_valid
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_tok[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + e * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=tmask[:, None] & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=kmask[:, None], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_tok[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=tmask[:, None] & (offs_cn[None, :] < N))


def align_block_size(topk_ids, E, block_m):
    # torch version of moe_align_block_size
    flat = topk_ids.reshape(-1)
    num_valid = flat.numel()
    counts = torch.bincount(flat, minlength=E)
    padded = (counts + block_m - 1) // block_m * block_m
    starts = torch.cumsum(padded, 0) - padded
    order = torch.argsort(flat, stable=True)
    rank_in_expert = torch.arange(num_valid, device=flat.device) - (torch.cumsum(counts, 0) - counts)[flat[order]]
    total = int(padded.sum())
    sorted_ids = torch.full((total,), num_valid, dtype=torch.int32, device=flat.device)
    sorted_ids[starts[flat[order]] + rank_in_expert] = order.to(torch.int32)
    expert_ids = torch.repeat_interleave(torch.arange(E, device=flat.device, dtype=torch.int32), padded // block_m)
    return sorted_ids, expert_ids, torch.tensor([total], device=flat.device, dtype=torch.int32)


def grouped_gemm(a, w, sorted_ids, expert_ids, n_padded, top_k, cfg, out_rows):
    E, K, N = w.shape
    c = torch.empty(out_rows, N, device=a.device, dtype=a.dtype)
    EM = sorted_ids.numel()
    grid = (triton.cdiv(EM, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    fused_moe_kernel[grid](a, w, c, sorted_ids, expert_ids, n_padded, N, K, EM, out_rows,
                           a.stride(0), a.stride(1), w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1),
                           top_k=top_k, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
                           GROUP_M=cfg.get("GROUP_M", 1), num_warps=cfg.get("num_warps", 4),
                           num_stages=cfg.get("num_stages", 2))
    return c


def moe_forward(x, topk_ids, w1, w2, cfg):
    T, K_top = topk_ids.shape
    E = w1.shape[0]
    sorted_ids, expert_ids, n_padded = align_block_size(topk_ids, E, cfg["BLOCK_M"])
    h = grouped_gemm(x, w1, sorted_ids, expert_ids, n_padded, K_top, cfg, T * K_top)
    g, u = h.chunk(2, dim=-1)
    h = torch.nn.functional.silu(g) * u
    y = grouped_gemm(h, w2, sorted_ids, expert_ids, n_padded, 1, cfg, T * K_top)
    return y.view(T, K_top, -1)


if __name__ == "__main__":
    torch.manual_seed(0)
    E, H, I, T, K = 16, 256, 128, 37, 4
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, H, 2 * I, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16) * 0.05
    ids = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(T)])
    y = moe_forward(x, ids, w1, w2, dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64))
    ref = torch.stack([torch.stack([(torch.nn.functional.silu((x[t] @ w1[e]).chunk(2)[0]) * (x[t] @ w1[e]).chunk(2)[1]) @ w2[e]
                                    for e in ids[t].tolist()]) for t in range(T)])
    print("max abs err", (y.float() - ref.float()).abs().max().item(), "ref scale", ref.float().abs().mean().item())
