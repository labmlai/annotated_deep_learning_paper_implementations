"""
This is based on the flash attention tutorial from [Triton](https://triton-lang.org/main/index.html)
"""

import triton
import triton.language as tl

import torch
from typing import Any, Tuple, Optional

HI_PRES_TL: tl.constexpr = tl.float32
HI_PRES_TORCH: torch.dtype = torch.float32


class AttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                causal: bool, sm_scale: float) -> torch.Tensor:
        """
        Group query attention forward pass. Returns the output in shape `[batch_size, n_heads, q_seq_len, d_head]`.

        :param ctx: is the context for torch gradient descent
        :param q: has shape `[batch_size, n_heads, q_seq_len, d_head]`
        :param q: has shape `[batch_size, n_heads, q_seq_len, d_head]`
        :param k: has shape `[batch_size, k_heads, kv_seq_len, d_head]`
        :param v: has shape `[batch_size, k_heads, kv_seq_len, d_head]`
        :param causal: whether to apply causal attention mask
        :param sm_scale: softmax scale factor
        """
        batch_size, n_heads, q_seq_len, d_head = q.shape
        _, k_heads, kv_seq_len, _ = k.shape
        assert n_heads % k_heads == 0
        n_groups = n_heads // k_heads

        # Shape constraints
        assert d_head == k.shape[-1] == v.shape[-1]
        assert d_head in {16, 32, 64, 128, 256}

        # Change the tensors combining the heads with the batch dimension
        q = q.view(batch_size * k_heads, n_groups, q_seq_len, d_head)
        k = k.view(batch_size * k_heads, kv_seq_len, d_head)
        v = v.view(batch_size * k_heads, kv_seq_len, d_head)

        # Make sure the tensors are contiguous and the strides are same
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert v.is_contiguous()
        assert k.stride() == v.stride()

        # Tensor for the output
        o = torch.empty_like(q)
        # Tensor for $\log_2 \sum_j e^{S_{ij}}$
        lse = torch.empty((batch_size * k_heads, n_groups, q_seq_len), device=q.device, dtype=HI_PRES_TORCH)

        # The forward computation will be parallelized along the batch dimension and the queries in blocks of size `BLOCK_M`
        grid = lambda args: (triton.cdiv(q_seq_len, args["BLOCK_M"]), batch_size * k_heads * n_groups, 1)
        _attn_fwd[grid](
            q, k, v, sm_scale, lse, o,
            n_groups=n_groups,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            d_head=d_head,
            is_causal=causal,
        )

        # Save the reshaped inputs and outputs for the backward pass
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.n_groups = n_groups
        ctx.causal = causal

        # Return the output in shape `[batch_size, n_heads, q_seq_len, d_head]`
        return o.view(batch_size, n_heads, q_seq_len, d_head)

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        The backward pass computes the gradients of the input tensors.

        :param ctx: is the context for torch gradient descent
        :param do: is the gradient tensor of the attention output with shape `[batch_size, n_heads, q_seq_len, d_head]`
        """

        # Get saved tensors and attributes
        n_groups = ctx.n_groups
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        q, k, v, o, lse = ctx.saved_tensors

        # Get shapes
        batch_size, n_heads, q_seq_len, d_head = do.shape
        _, kv_seq_len, _ = k.shape
        k_heads = n_heads // n_groups

        # Combine the heads with the batch dimension of the output gradients tensor
        do = do.view(batch_size * k_heads, n_groups, q_seq_len, d_head)

        # Make sure it's contiguous and the strides are the same
        assert do.is_contiguous()
        assert k.stride() == v.stride()
        assert q.stride() == o.stride() == do.stride()

        # Create tensors for input gradients
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # $\frac{1}{\log_e 2}$
        RCP_LN2 = 1.4426950408889634
        # Multiply $k$ by softmax scale
        k_scaled = k * (sm_scale * RCP_LN2)
        # $D_i = P^T_{i:}dP_{i:} = do^T_io_i$
        pdp = torch.empty_like(lse)
        # We use fixed `BLOCK_M` for backward pass on $D$
        BLOCK_M = 16
        assert q_seq_len % BLOCK_M == 0
        # Compute $D_i$
        #
        # This is parallelized along the batch and query in blocks of size `BLOCK_M`
        pre_grid = (q_seq_len // BLOCK_M, batch_size * k_heads)
        _attn_bwd_d[pre_grid](
            o, do,
            pdp,
            BLOCK_M=16,
            d_head=d_head,
            q_seq_len=q_seq_len,
            n_groups=n_groups,
            num_stages=1,
        )
        # Compute $dK$ and $dV$
        #
        # This is parallelized along the batch and keys in blocks of size `BLOCK_N`
        grid = lambda args: (triton.cdiv(kv_seq_len, args['BLOCK_N']), batch_size * k_heads)
        _attn_bwd_dkdv[grid](
            q, k_scaled, v, sm_scale, do, dk, dv,
            lse, pdp,
            q_seq_len, kv_seq_len, n_groups, d_head,
            is_causal=causal,

        )
        # Compute $dQ$
        #
        # This is parallelized along the batch and queries in blocks of size `BLOCK_M`
        grid = lambda args: (triton.cdiv(q_seq_len, args["BLOCK_M"]), batch_size * k_heads * n_groups)
        _attn_bwd_dq[grid](
            q, k_scaled, v, do,
            dq,
            lse, pdp,
            q_seq_len, kv_seq_len, n_groups, d_head,
            is_causal=causal,
        )

        # Split the combined batch and heads
        dq = dq.view(batch_size, n_heads, q_seq_len, d_head)
        dk = dk.view(batch_size, k_heads, kv_seq_len, d_head)
        dv = dv.view(batch_size, k_heads, kv_seq_len, d_head)

        #
        return dq, dk, dv, None, None


attention = AttentionFunc.apply


def _get_autotune_configs(inner_loop: str) -> list:
    """
    #### Configs for auto-tuning
    """

    configs = []
    # List possible BLOCK_M and BLOCK_N that satisfy BLOCK_M divisible by BLOCK_N
    # and also try to cover a wide range
    for bm in [64, 128, 256]:
        # We'll try bn in [16, 32, 64, 128] that are divisors and <= bm
        for bn in [64, 128, 256]:
            if inner_loop == 'key' and bm % bn != 0:
                continue
            if inner_loop == 'query' and bn % bm != 0:
                continue
            for s in [2, 3, 4]:
                for w in [4, 8]:
                    if bm * bn < 128 * 128 and w == 8:
                        continue

                    configs.append(triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn}, num_stages=s, num_warps=w))

    return configs


@triton.autotune(_get_autotune_configs(inner_loop='key'),
                 key=["q_seq_len", "kv_seq_len", "d_head", "n_groups", "is_causal"])
@triton.jit
def _attn_fwd(t_q, t_k, t_v, sm_scale, t_lse, t_o,
              n_groups: tl.constexpr,
              q_seq_len: tl.constexpr,
              kv_seq_len: tl.constexpr,
              d_head: tl.constexpr,
              is_causal: tl.constexpr,
              BLOCK_M: tl.constexpr,  # q seq len block
              BLOCK_N: tl.constexpr,  # k seq len block
              ):
    """
    :param t_q: query
    :param t_k: keys
    :param t_v: values
    :param sm_scale: softmax scale
    :param t_lse: $\log_2 \sum_j e^{S_{ij}}$ (out)
    :param t_o: output (out)
    :param n_groups: number of groups
    :param q_seq_len: query sequence length
    :param kv_seq_len: key/value sequence length
    :param d_head: size of a head
    :param BLOCK_M: block size  for query sequence length
    :param BLOCK_N: block size  for key sequence length
    :param is_causal: whether causal attention

    Strides `z`, `h`, `m` and  `d` denote the stride of the corresponding dimensions
     (`batch_size`, `n_heads`, `seq_len`, `d_head`) in the query.
    Stride `n` denote the stride on `seq_len` of key.

    """
    i = tl.program_id(0)
    z = tl.program_id(1) // n_groups
    g = tl.program_id(1) % n_groups

    # Create block pointers
    p_q = tl.make_block_ptr(t_q + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                            (q_seq_len, d_head),
                            (d_head, 1),
                            (i * BLOCK_M, 0),
                            (BLOCK_M, d_head),
                            (1, 0))
    p_v = tl.make_block_ptr(t_v + z * kv_seq_len * d_head,
                            (kv_seq_len, d_head),
                            (d_head, 1),
                            (0, 0),
                            (BLOCK_N, d_head),
                            (1, 0))
    p_kT = tl.make_block_ptr(t_k + z * kv_seq_len * d_head,
                             (d_head, kv_seq_len),
                             (1, d_head),
                             (0, 0),
                             (d_head, BLOCK_N),
                             (0, 1))
    p_o = tl.make_block_ptr(t_o + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                            (q_seq_len, d_head),
                            (d_head, 1),
                            (i * BLOCK_M, 0),
                            (BLOCK_M, d_head),
                            (1, 0))
    p_lse = tl.make_block_ptr(t_lse + z * n_groups * q_seq_len + g * q_seq_len,
                              (q_seq_len,),
                              (1,),
                              (i * BLOCK_M,),
                              (BLOCK_M,),
                              (0,))

    # Initialize offsets
    offs_i = i * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = tl.arange(0, BLOCK_N)

    # Initialize $m_i$ and $l_i$
    b_m = tl.zeros([BLOCK_M], dtype=HI_PRES_TL) - float("inf")
    b_l = tl.zeros([BLOCK_M], dtype=HI_PRES_TL) + 1.0
    # Accumulate $O$
    b_acc = tl.zeros([BLOCK_M, d_head], dtype=HI_PRES_TL)

    # softmax scale / log(2)
    sm_scale = sm_scale * 1.44269504
    # Load $Q_i$
    b_q = tl.load(p_q)

    if is_causal:
        # Upto the diagonal block
        b_acc, b_l, b_m = _attn_fwd_inner(b_acc, b_l, b_m, b_q,
                                          p_kT, p_v,
                                          sm_scale,
                                          BLOCK_M, d_head, BLOCK_N,
                                          offs_i, offs_j,
                                          start_n=tl.full([], 0, tl.int32),  # type: ignore
                                          steps=(i * BLOCK_M) // BLOCK_N,
                                          MASK=False,
                                          )
        # Diagonal block with masking within it
        b_acc, b_l, b_m = _attn_fwd_inner(b_acc, b_l, b_m, b_q, p_kT, p_v,
                                          sm_scale,
                                          BLOCK_M, d_head, BLOCK_N,
                                          offs_i, offs_j,
                                          start_n=i * BLOCK_M,
                                          steps=BLOCK_M // BLOCK_N,
                                          MASK=True,
                                          )
    else:
        b_acc, b_l, b_m = _attn_fwd_inner(b_acc, b_l, b_m, b_q, p_kT, p_v,
                                          sm_scale,
                                          BLOCK_M, d_head, BLOCK_N,
                                          offs_i, offs_j,
                                          start_n=tl.full([], 0, tl.int32),  # type: ignore
                                          steps=kv_seq_len // BLOCK_N,
                                          MASK=False,
                                          )

    # Update LSE
    tl.store(p_lse, b_m + tl.math.log2(b_l))
    tl.store(p_o, (b_acc / b_l[:, None]).to(t_o.type.element_ty))


@triton.jit
def _attn_fwd_inner(b_acc, b_l, b_m, b_q,
                    p_kT, p_v,
                    scale,
                    BLOCK_M: tl.constexpr,
                    d_head: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                    offs_m, offs_n,
                    start_n,
                    steps,
                    MASK: tl.constexpr,
                    ):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)

    p_kT = tl.advance(p_kT, (0, start_n))
    p_v = tl.advance(p_v, (start_n, 0))

    # loop over k, v and update accumulator
    for _ in range(steps):
        b_kT = tl.load(p_kT)
        b_s = tl.dot(b_q, b_kT, out_dtype=HI_PRES_TL)

        tl.static_assert(b_s.dtype == HI_PRES_TL)
        b_s = b_s * scale
        if MASK:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            b_s = b_s + tl.where(mask, 0, -1.0e6)

        # $m_{i}^{\text{new}} = \max(m_i, \text{rowmax}(S_{ij}))$
        tl.static_assert(len(b_s.shape) == 2)
        b_m_new = tl.maximum(b_m, tl.max(b_s, -1))
        # $\tilde{P}_{ij} = \exp(S_{ij} - m_i^{\text{new}})$
        b_p = tl.math.exp2(b_s - b_m_new[:, None])
        # $\tilde{l}_ij = \text{rowsum}(\tilde{P}_{ij})$
        b_l_new = tl.sum(b_p, -1)

        # $e^{m_i - m_{i}^{\text{new}}}$
        b_m_m_new = tl.math.exp2(b_m - b_m_new)
        # $l_i \leftarrow e^{m_i - m_{i}^{\text{new}}} l_i + \tilde{l}_{ij}$
        b_l = b_l * b_m_m_new + b_l_new

        # $O_i \leftarrow e^{m_i - m_{i}^{\text{new}}} O_i + \tilde{P}_{ij} * V_j$
        b_v = tl.load(p_v)
        b_acc = b_acc * b_m_m_new[:, None]
        b_p = b_p.to(b_q.dtype)
        b_acc += tl.dot(b_p, b_v, out_dtype=HI_PRES_TL)

        # update $m_i$
        b_m = b_m_new

        # Move pointers
        start_n += BLOCK_N
        p_v = tl.advance(p_v, (BLOCK_N, 0))
        p_kT = tl.advance(p_kT, (0, BLOCK_N))

    tl.static_assert(b_acc.dtype == HI_PRES_TL, "attn_fwd_inner requires accumulator to be in HI_PRES_TL precision")

    return b_acc, b_l, b_m


@triton.jit
def _attn_bwd_d(t_o, t_do,
                t_pdp,
                BLOCK_M: tl.constexpr, d_head: tl.constexpr,
                q_seq_len: tl.constexpr,
                n_groups: tl.constexpr,
                ):
    i = tl.program_id(0) * BLOCK_M
    z = tl.program_id(1)

    # Create block pointers
    p_o = tl.make_block_ptr(t_o + z * n_groups * q_seq_len * d_head,
                            (n_groups, q_seq_len, d_head),
                            (q_seq_len * d_head, d_head, 1),
                            (0, i, 0),
                            (n_groups, BLOCK_M, d_head),
                            (2, 1, 0))
    p_do = tl.make_block_ptr(t_do + z * n_groups * q_seq_len * d_head,
                             (n_groups, q_seq_len, d_head),
                             (q_seq_len * d_head, d_head, 1),
                             (0, i, 0),
                             (n_groups, BLOCK_M, d_head),
                             (2, 1, 0))
    p_pdp = tl.make_block_ptr(t_pdp + z * n_groups * q_seq_len,
                              (n_groups, q_seq_len),
                              (q_seq_len, 1),
                              (0, i),
                              (n_groups, BLOCK_M),
                              (1, 0))

    o = tl.load(p_o)
    do = tl.load(p_do).to(HI_PRES_TL)
    d = tl.sum(o * do, axis=-1)
    tl.store(p_pdp, d)


@triton.autotune(_get_autotune_configs(inner_loop='query'),
                 key=["q_seq_len", "kv_seq_len", "d_head", "n_groups", "is_causal"])
@triton.jit
def _attn_bwd_dkdv(t_q, t_k, t_v, sm_scale,
                   t_do,
                   t_dk, t_dv,
                   t_lse, t_pdp,
                   q_seq_len: tl.constexpr, kv_seq_len: tl.constexpr,
                   n_groups: tl.constexpr, d_head: tl.constexpr,
                   is_causal: tl.constexpr,
                   BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr,
                   ):
    """
    Loop along m query; n % m == 0
    """
    # K is already multiplied by scale
    n = tl.program_id(0)
    z = tl.program_id(1)

    p_k = tl.make_block_ptr(t_k + z * kv_seq_len * d_head,
                            (kv_seq_len, d_head),
                            (d_head, 1),
                            (n * BLOCK_N, 0),
                            (BLOCK_N, d_head),
                            (1, 0))
    p_v = tl.make_block_ptr(t_v + z * kv_seq_len * d_head,
                            (kv_seq_len, d_head),
                            (d_head, 1),
                            (n * BLOCK_N, 0),
                            (BLOCK_N, d_head),
                            (1, 0))
    p_dk = tl.make_block_ptr(t_dk + z * kv_seq_len * d_head,
                             (kv_seq_len, d_head),
                             (d_head, 1),
                             (n * BLOCK_N, 0),
                             (BLOCK_N, d_head),
                             (1, 0))
    p_dv = tl.make_block_ptr(t_dv + z * kv_seq_len * d_head,
                             (kv_seq_len, d_head),
                             (d_head, 1),
                             (n * BLOCK_N, 0),
                             (BLOCK_N, d_head),
                             (1, 0))

    b_dv = tl.zeros([BLOCK_N, d_head], dtype=HI_PRES_TL)
    b_dk = tl.zeros([BLOCK_N, d_head], dtype=HI_PRES_TL)

    # load K and V: they stay in SRAM throughout the inner loop.
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)

    # Iterate through queries that attend to save keys
    for g in range(n_groups):
        # Create block pointers
        p_qT = tl.make_block_ptr(t_q + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                                 (d_head, q_seq_len),
                                 (1, d_head),
                                 (0, 0),
                                 (d_head, BLOCK_M),
                                 (0, 1))

        p_do = tl.make_block_ptr(t_do + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                                 (q_seq_len, d_head),
                                 (d_head, 1),
                                 (0, 0),
                                 (BLOCK_M, d_head),
                                 (1, 0))
        p_lse = tl.make_block_ptr(t_lse + z * n_groups * q_seq_len + g * q_seq_len,
                                  (q_seq_len,),
                                  (1,),
                                  (0,),
                                  (BLOCK_M,),
                                  (0,))
        p_pdp = tl.make_block_ptr(t_pdp + z * n_groups * q_seq_len + g * q_seq_len,
                                  (q_seq_len,),
                                  (1,),
                                  (0,),
                                  (BLOCK_M,),
                                  (0,))

        # $$dk_j = \sum_i dS_{ij} q_i = \sum_i P_{ij} \big( do_i^T v_j - D_i \big) q_i$$
        # $$dv_j = \sum_i P_{ij} do_i$$

        # Compute $dk$ $dv$ and $dv$ along the masked blocks near diagonal.
        # Use smaller block size of MASK_BLOCK_M
        # because there is a little extra computation?
        if is_causal:
            # loop along m
            b_dk, b_dv = _attn_bwd_dkdv_inner(
                b_dk, b_dv,
                p_qT, b_k, b_v, p_do,
                p_lse, p_pdp,
                # You can use a smaller BLOCK_M if BLOCK_N is not divisible by BLOCK_M
                BLOCK_M, BLOCK_N,
                d_head,
                n=n * BLOCK_N, start_m=n * BLOCK_N,
                steps=BLOCK_N // BLOCK_M,
                MASK=True
            )

            # Compute $dk$ and $dv$ for non-masked blocks.
            b_dk, b_dv = _attn_bwd_dkdv_inner(
                b_dk, b_dv,
                p_qT, b_k, b_v, p_do,
                p_lse, p_pdp,
                BLOCK_M, BLOCK_N,
                d_head,
                n=n * BLOCK_N, start_m=(n + 1) * BLOCK_N,
                steps=(q_seq_len - (n + 1) * BLOCK_N) // BLOCK_M,
                MASK=False,
            )
        else:
            b_dk, b_dv = _attn_bwd_dkdv_inner(
                b_dk, b_dv,
                p_qT, b_k, b_v, p_do,
                p_lse, p_pdp,
                BLOCK_M, BLOCK_N,
                d_head,
                n=n * BLOCK_N, start_m=tl.full([], 0, tl.int32),
                steps=q_seq_len // BLOCK_M,
                MASK=False,
            )

    # Save $dv$
    tl.store(p_dv, b_dv.to(t_dv.type.element_ty))

    # Since we used $k = \text{scale} * \hat{k}$ where $\hat{k} are the original keys
    # we multiple by scale again to get gradient on original keys.
    b_dk *= sm_scale

    # Save $dk$
    tl.store(p_dk, b_dk.to(t_dk.type.element_ty))


@triton.jit
def _attn_bwd_dkdv_inner(b_dk, b_dv,
                         p_qT, b_k, b_v, p_do,
                         p_lse, p_pdp,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         d_head: tl.constexpr,
                         n, start_m, steps,
                         MASK: tl.constexpr):
    """Inner loop along m query"""

    # To apply the mask
    tl.static_assert(BLOCK_N % BLOCK_M == 0)

    # Offsets for mask computation
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = n + tl.arange(0, BLOCK_N)

    # Pointers
    p_qT = tl.advance(p_qT, (0, start_m))
    p_do = tl.advance(p_do, (start_m, 0))
    p_lse = tl.advance(p_lse, (start_m,))
    p_pdp = tl.advance(p_pdp, (start_m,))

    # Loop
    for _ in range(steps):
        # Load $$qT$$
        b_qT = tl.load(p_qT)

        # $M_i = log_2 L_i$
        b_m = tl.load(p_lse)

        # $$P_{ij} = \frac{e^{q_i^T k_j}}{L_i} = e^{q_i^T k_j - M_i}$$
        # Not that k is already multiplied by softmax scale.
        # It is also divided by $log_e 2$ so we can use $2^x$ instead of $e^x$
        b_qkT = tl.dot(b_k, b_qT, out_dtype=HI_PRES_TL)
        b_pT = tl.math.exp2(b_qkT - b_m[None, :])

        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            b_pT = tl.where(mask, b_pT, 0.0)

        # $$dv_j = \sum_i P_{ij} do_i$$
        b_do = tl.load(p_do)
        b_dv += tl.dot(b_pT.to(b_do.dtype),
                       b_do,
                       out_dtype=HI_PRES_TL)

        # $$dk_j = \sum_i dS_{ij} q_i = \sum_i P_{ij} \big( dP^T_{i:} - D_i \big) q_i$$
        b_pdp = tl.load(p_pdp)
        # $dP_{ij} = do^T_i v_j$
        b_dpT = tl.dot(b_v, tl.trans(b_do), out_dtype=HI_PRES_TL).to(HI_PRES_TL)
        # $dS_{ij} = P_{ij} \big( dP_{i:} - D_i \big)$
        b_dsT = b_pT * (b_dpT - b_pdp[None, :])
        # $dk_j = \sum_i dS_{ij} q_i$
        b_dk += tl.dot(b_dsT.to(b_qT.dtype),
                       tl.trans(b_qT), out_dtype=HI_PRES_TL)

        # Increment pointers.
        offs_m += BLOCK_M
        p_lse = tl.advance(p_lse, (BLOCK_M,))
        p_pdp = tl.advance(p_pdp, (BLOCK_M,))
        p_qT = tl.advance(p_qT, (0, BLOCK_M))
        p_do = tl.advance(p_do, (BLOCK_M, 0))

    # Return accumulated $dk$ and $dv$
    return b_dk, b_dv


@triton.autotune(_get_autotune_configs(inner_loop='key'),
                 key=["q_seq_len", "kv_seq_len", "d_head", "n_groups", "is_causal"])
@triton.jit
def _attn_bwd_dq(t_q, t_k, t_v, t_do,
                 t_dq,
                 t_lse, t_pdp,
                 q_seq_len: tl.constexpr, kv_seq_len: tl.constexpr,
                 n_groups: tl.constexpr, d_head: tl.constexpr,
                 is_causal: tl.constexpr,
                 BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr,
                 ):
    # $\log_e 2$
    LN2: tl.constexpr = 0.6931471824645996  # type: ignore

    m = tl.program_id(0)
    z = tl.program_id(1) // n_groups
    g = tl.program_id(1) % n_groups

    # Create block pointers
    p_q = tl.make_block_ptr(t_q + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                            (q_seq_len, d_head),
                            (d_head, 1),
                            (m * BLOCK_M, 0),
                            (BLOCK_M, d_head),
                            (1, 0))
    p_dq = tl.make_block_ptr(t_dq + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                             (q_seq_len, d_head),
                             (d_head, 1),
                             (m * BLOCK_M, 0),
                             (BLOCK_M, d_head),
                             (1, 0))
    p_do = tl.make_block_ptr(t_do + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                             (q_seq_len, d_head),
                             (d_head, 1),
                             (m * BLOCK_M, 0),
                             (BLOCK_M, d_head),
                             (1, 0))
    p_kT = tl.make_block_ptr(t_k + z * kv_seq_len * d_head,
                             (d_head, kv_seq_len),
                             (1, d_head),
                             (0, 0),
                             (d_head, BLOCK_N),
                             (0, 1))
    p_vT = tl.make_block_ptr(t_v + z * kv_seq_len * d_head,
                             (d_head, kv_seq_len),
                             (1, d_head),
                             (0, 0),
                             (d_head, BLOCK_N),
                             (0, 1))
    p_lse = tl.make_block_ptr(t_lse + z * n_groups * q_seq_len + g * q_seq_len,
                              (q_seq_len,),
                              (1,),
                              (m * BLOCK_M,),
                              (BLOCK_M,),
                              (0,))
    p_pdp = tl.make_block_ptr(t_pdp + z * n_groups * q_seq_len + g * q_seq_len,
                              (q_seq_len,),
                              (1,),
                              (m * BLOCK_M,),
                              (BLOCK_M,),
                              (0,))

    b_q = tl.load(p_q)
    b_do = tl.load(p_do)
    b_pdp = tl.load(p_pdp)

    b_dq = tl.zeros([BLOCK_M, d_head], dtype=HI_PRES_TL)

    b_lse = tl.load(p_lse)

    # $$dq_i = \sum_j dS_{ij} k_j = \sum_j P_{ij} \big( dP_{ij} - D_i \big) k_j$$

    if is_causal:
        # Compute $dQ$ for masked (diagonal) blocks.
        b_dq = _attn_bwd_dq_inner(b_dq, b_q, p_kT, p_vT,
                                  b_do, b_lse, b_pdp,
                                  BLOCK_M, BLOCK_N,
                                  m=m * BLOCK_M, start_n=m * BLOCK_M,
                                  steps=BLOCK_M // BLOCK_N,
                                  MASK=True
                                  )

        # Other blocks
        b_dq = _attn_bwd_dq_inner(b_dq, b_q, p_kT, p_vT,
                                  b_do, b_lse, b_pdp,
                                  BLOCK_M, BLOCK_N,
                                  m=m * BLOCK_M, start_n=tl.full([], 0, tl.int32),  # type: ignore
                                  steps=(m * BLOCK_M) // BLOCK_N,
                                  MASK=False
                                  )
    else:
        b_dq = _attn_bwd_dq_inner(b_dq, b_q, p_kT, p_vT,
                                  b_do, b_lse, b_pdp,
                                  BLOCK_M, BLOCK_N,
                                  m=m * BLOCK_M, start_n=tl.full([], 0, tl.int32),  # type: ignore
                                  steps=kv_seq_len // BLOCK_N,
                                  MASK=False
                                  )

    # Since $k$ was scaled by $\frac{1}{log_e 2}$, and $dq_j = \sum_j dS_{ij} k_j$
    # got this factor in to computed $dq$ we need to reverse it.
    b_dq *= LN2

    # Save $dq$
    tl.store(p_dq, b_dq.to(t_dq.type.element_ty))


@triton.jit
def _attn_bwd_dq_inner(b_dq, b_q, p_kT, p_vT,
                       b_do, b_lse, b_pdp,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       m, start_n, steps,
                       MASK: tl.constexpr):
    """Inner loop over n key"""
    offs_m = m + tl.arange(0, BLOCK_M)

    p_kT = tl.advance(p_kT, (0, start_n))
    p_vT = tl.advance(p_vT, (0, start_n))

    tl.static_assert(BLOCK_M % BLOCK_N == 0, 'BLOCK_M must be divisible by BLOCK_N')

    for _ in range(steps):
        # $$P_{ij} = \frac{e^{q_i^T k_j}}{L_i} = e^{q_i^T k_j - M_i}$$
        # Not that k is already multiplied by softmax scale.
        # It is also divided by $log_e 2$ so we can use $2^x$ instead of $e^x$
        b_kT = tl.load(p_kT)
        b_vT = tl.load(p_vT)
        b_qk = tl.dot(b_q, b_kT, out_dtype=HI_PRES_TL)
        b_p = tl.math.exp2(b_qk - b_lse[:, None])

        # Autoregressive masking.
        if MASK:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask = (offs_m[:, None] >= offs_n[None, :])
            b_p = tl.where(mask, b_p, 0.0)

        # $$dq_i = \sum_j dS_{ij} k_j = \sum_j P_{ij} \big( dP_{ij} - D_i \big) k_j$$

        # $dP_{ij} = do^T_i v_j$
        b_dp = tl.dot(b_do, b_vT, out_dtype=HI_PRES_TL).to(HI_PRES_TL)
        # $dS_{ij} = P_{ij} \big( dP_{i:} - D_i \big)$
        b_ds = b_p * (b_dp - b_pdp[:, None])
        # $dq_j = \sum_j dS_{ij} k_j$
        b_dq += tl.dot(b_ds.to(b_kT.dtype),
                       tl.trans(b_kT),
                       out_dtype=HI_PRES_TL)

        # Increment pointers.
        start_n += BLOCK_N
        p_kT = tl.advance(p_kT, (0, BLOCK_N))
        p_vT = tl.advance(p_vT, (0, BLOCK_N))

    # Return accumulated $dq$
    return b_dq