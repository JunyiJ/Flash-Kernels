import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_on,
    n_ctx, d_model,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
    v_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
    o_ptr = Out + batch_idx * stride_ob + head_idx * stride_oh

    # m_i running max. l_i running sum, acc: weighted sum of V
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    m_mask = offs_m < n_ctx
    d_mask = offs_d < d_model
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = m_mask[:, None] & d_mask[None, :]
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Iterate over key and value blocks (along the sequence dimension)
    hi = tl.minimum((pid_m + 1) * BLOCK_M, n_ctx)
    for start_n in range(0, hi, BLOCK_N):
        # Load K & V tiles
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < n_ctx
        k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        kv_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # Compute Scores & Online softmax
        s = tl.dot(q, tl.trans(k)) * sm_scale
        # Causal mask
        if start_n + BLOCK_N > pid_m * BLOCK_M:
            mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(mask, s, float("-inf"))
        s = tl.where(n_mask[None, :], s, float("-inf"))
        m_ij = tl.max(s, axis=1)
        p = tl.exp(s - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        # Rescale and update accumulator
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        acc = acc * alpha[:, None] + beta[:, None] * tl.dot(p.to(tl.float16), v.to(tl.float16))
        m_i = m_new
        l_i = alpha * l_i + beta * l_ij

    # Store results
    acc = acc / l_i[:, None]
    out_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    out_mask = m_mask[:, None] & d_mask[None, :]
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)

def flash_attn_forward(q, k, v, sm_scale):
    BLOCK_M = 128
    BLOCK_N = 64
    B, H, N_CTX, D_MODEL = q.shape
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
    assert q.ndim == 4, "q, k, v must be 4D tensors: [B, H, N_CTX, D_MODEL]"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "q, k, v must be contiguous"
    assert q.device == DEVICE and k.device == DEVICE and v.device == DEVICE, "q, k, v must be on active Triton device"
    BLOCK_DMODEL = triton.next_power_of_2(D_MODEL)
    out = torch.empty_like(q)
    grid = (triton.cdiv(N_CTX, BLOCK_M), H, B)

    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_CTX, D_MODEL,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )
    return out

torch.manual_seed(42)
BATCH, HEADS, SEQ, DIM = 2, 8, 1024, 64
SCALE = 1.0 / (DIM ** 0.5)

q = torch.randn((BATCH, HEADS, SEQ, DIM), device=DEVICE, dtype=torch.float16)
k = torch.randn((BATCH, HEADS, SEQ, DIM), device=DEVICE, dtype=torch.float16)
v = torch.randn((BATCH, HEADS, SEQ, DIM), device=DEVICE, dtype=torch.float16)

output_torch = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, is_causal=True, scale=SCALE
)

output_triton = flash_attn_forward(q, k, v, SCALE)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
print(torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2))
