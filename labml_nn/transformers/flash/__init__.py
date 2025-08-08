"""
---
title: Flash Attention
summary: >
  This is a PyTorch/Triton implementation of Flash Attention 2
  with explanations.
---

# Flash Attention

Flash attention speeds up transformer attention mechanism by reducing the number of
memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM.

It's introduced in paper
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
and further optimized in paper
[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691).
Official CUDA implementation can be found at [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

Our implementation is based on the
[Triton's example implementation](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).

*Note: You can click on the mathematical symbols or identifiers to highlight them*.

You can run [test.py](./test.html) to see correctness and measure performance of this implementation.

## Forward pass

Here's the attention forward pass. The formulas represent a single attention head.
$Q_i$ is query vector (row vector) at position $i$
and $K_j$ and $V_j$ are the key and value row vectors at position $j$.
$O_i$ is the output vector at position $i$.

\begin{align}
S_{ij} &= \sigma Q_i K_j^T
\\
L_i &= \sum_j e^{S_{ij}}
\\
P_{ij} &= \frac{e^{S_{ij}}}{L_i}
\\
O_i &= \sum_j P_{ij} V_j
\\
&= \frac{1}{L_i} \sum_j  e^{S_{ij}} V_j
\end{align}

$S_{ij}$ is the attention score matrix before softmax,
$L_i$ is the softmax denominator,
and $P_{ij}$ is the attention matrix after softmax.

#### Flash Attention Optimization

You can compute $O_i$, instead of doing the full softmax,
by computing the sum of exponents $l_i$ and the unnormalized output $\tilde{O}_i$
while iterating over keys:

\begin{align}
S_{ij} &= \sigma Q_i K_j^T
\\
l_i &\leftarrow l_i + e^{S_{ij}}
\\
\tilde{O}_i &\leftarrow \tilde{O}_i + e^{S_{ij}} o_j
\end{align}

Finally you can compute,

$$O_i = \frac{\tilde{O}_i}{l_i}$$

To make it numerically stable flash attention subtracts the current max of $S_{ij}$ before exponentiating.

So it maintains the following while iterating over keys:

* $m_i$, the max $S_{ij}$
* $l_i$, the sum of exponents $\sum_j e^{S_{ij} - m_i}$, and
* $\tilde{O}_i$, the unnormalized output

For each block of keys $j_1 \dots j_2$ it updates them:

\begin{align}
m_i^{\text{new}} &= \max(m_i, \max_{j=j1}^{j2} S_{ij})
\\
\tilde{P}_{ij} &= \exp(S_{ij} - m_i^{\text{new}})
\\
l_i &\leftarrow e^{m_i - m_{i}^{\text{new}}} l_i + \sum_{j=j1}^{j2} \tilde{P}_{ij}
\\
\tilde{O}_i &\leftarrow e^{m_i - m_{i}^{\text{new}}} \tilde{O}_i + \tilde{P}_{ij} * V_j
\\
m_i &\leftarrow m_{i}^{\text{new}}
\end{align}

Then finally,

$$O_i = \frac{\tilde{O}_i}{l_i}$$

This reduces the memory usage since we don't have to compute full $S_{ij}$ matrix or $P_{ij}$ matrix.
It also speeds up since we don't have to load these large matrices.
Instead it only loads blocks of $K$ and $V$ as it iterates over them.

## Backward pass

Here's the standard backward pass. $dO_i$ is the gradient vector on the output $O_i$

\begin{align}
dV_j &= \sum_i P_{ij} dO_i
\\
dP_{ij} &= dO_{i} V_j^T
\\
dS_{ij} &= d\text{softmax}(dP_{ij})
\\
&= \sum_k P_{ik} (\delta_{jk} - P_{ij}) dP_{ik}
\\
&= P_{ij} dP_{ij} - P_{ij} \sum P_{ik} dP_{ik}
\\
dQ_i &= \sigma \sum_j dS_{ij} K_j
\\
dK_j &= \sigma \sum_i dS_{ij} Q_i
\end{align}

where $\delta_{jk}$ is $1$ when $j = k$ and $0$ otherwise.

Flash attention paper introduces $D_i$ to simplify $dS$ computation.

\begin{align}
D_i &= \sum_k P_{ik} dP_{ik}
\\
&= \sum_k P_{ik} dO_i V_k^T
\\
&= dO_i \sum_k P_{ik} V_k^T
\\
&= dO_i O_i^T
\end{align}

Then,

\begin{align}
dS_{ij} = P_{ij} dP_{ij} - D_i P_{ij}
\end{align}

Flash attention saves $L_i$ from the forward pass since it doesn't take much memory.
So during the backward pass it doesn't have to keep computing $l_i$ or $m_i$.

It first computes $D_i$.
Then it iterates over the queries and compute (accumulate) $dK_j$ and $dV_j$.
Finally it iterates over the keys and compute (accumulate) $dQ_i$.

In both forward and backward pass we calculate logarithms and exponentials of $2$ instead of $e$ for performance.
"""

from typing import Any, Tuple

import torch
import triton
import triton.language as tl

HI_PRES_TL: tl.constexpr = tl.float32
HI_PRES_TORCH: torch.dtype = torch.float32


class AttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                causal: bool, sm_scale: float) -> torch.Tensor:
        """
        ### Forward pass

        Group query attention forward pass. Returns the output in shape `[batch_size, n_heads, q_seq_len, d_head]`.

        :param ctx: is the context for torch gradient descent
        :param q: has shape `[batch_size, n_heads, q_seq_len, d_head]`
        :param q: has shape `[batch_size, n_heads, q_seq_len, d_head]`
        :param k: has shape `[batch_size, k_heads, kv_seq_len, d_head]`
        :param v: has shape `[batch_size, k_heads, kv_seq_len, d_head]`
        :param causal: whether to apply causal attention mask
        :param sm_scale: softmax scale factor $\sigma$
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
        # Tensor for log of sum of exponentials $\log_2 L_i = \log_2 \sum_j e^{S_{ij}}$
        lse = torch.empty((batch_size * k_heads, n_groups, q_seq_len), device=q.device, dtype=HI_PRES_TORCH)

        # The forward computation will be parallelized along the batch dimension and the queries in blocks of size `BLOCK_Q`
        grid = lambda meta: (triton.cdiv(q_seq_len, meta["BLOCK_Q"]), batch_size * k_heads * n_groups, 1)
        _attn_fwd[grid](
            q, k, v, sm_scale * 1.4426950408889634, lse, o,
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
        ### Backward pass

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

        # Precompute $\sigma (\log_2 e) K_j$
        k_scaled = k * (sm_scale * 1.4426950408889634)
        # $D_i = P^T_{i:}dP_{i:} = do^T_io_i$
        pdp = torch.empty_like(lse)
        # We use fixed `BLOCK_Q` for backward pass on $D$

        # Compute $D_i$
        #
        # This is parallelized along the batch and query in blocks of size `BLOCK_Q`
        BLOCK_Q = 16
        pre_grid = (triton.cdiv(q_seq_len, BLOCK_Q), batch_size * k_heads)
        _attn_bwd_d[pre_grid](
            o, do,
            pdp,
            BLOCK_Q=16,
            d_head=d_head,
            q_seq_len=q_seq_len,
            n_groups=n_groups,
            num_stages=1,
        )

        # Compute $dK$ and $dV$
        #
        # This is parallelized along the batch and keys in blocks of size `BLOCK_K`
        grid = lambda meta: (triton.cdiv(kv_seq_len, meta['BLOCK_K']), batch_size * k_heads)
        _attn_bwd_dkdv[grid](
            q, k_scaled, v, sm_scale, do, dk, dv,
            lse, pdp,
            q_seq_len, kv_seq_len, n_groups, d_head,
            is_causal=causal,

        )

        # Compute $dQ$
        #
        # This is parallelized along the batch and queries in blocks of size `BLOCK_Q`
        grid = lambda meta: (triton.cdiv(q_seq_len, meta['BLOCK_Q']), batch_size * k_heads * n_groups)
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

    # Possible options for `BLOCK_Q`
    for bq in [64, 128, 256]:
        # Possible options for `BLOCK_K`
        for bk in [64, 128, 256]:
            # If the inner loop is along keys the `BLOCK_Q` must be a multiple of `BLOCK_K` for causal masking
            if inner_loop == 'key' and bq % bk != 0:
                continue
            # Similarly when the inner loop is along queries
            if inner_loop == 'query' and bk % bq != 0:
                continue

            # Number of stages and warps
            for s in [2, 3, 4]:
                for w in [4, 8]:
                    if bq * bk < 128 * 128 and w == 8:
                        continue

                    configs.append(triton.Config({'BLOCK_Q': bq, 'BLOCK_K': bk}, num_stages=s, num_warps=w))

    # **Use `return configs` to autotune. Trying all combinations is slow for testing.**
    return configs[:1]


@triton.autotune(_get_autotune_configs(inner_loop='key'),
                 key=["q_seq_len", "kv_seq_len", "d_head", "n_groups", "is_causal"])
@triton.jit
def _attn_fwd(t_q, t_k, t_v, sm_scale_log2e, t_lse, t_o,
              n_groups: tl.constexpr,
              q_seq_len: tl.constexpr,
              kv_seq_len: tl.constexpr,
              d_head: tl.constexpr,
              is_causal: tl.constexpr,
              BLOCK_Q: tl.constexpr,
              BLOCK_K: tl.constexpr,
              ):
    """
    ### Triton kernel for Flash attention forward pass

    :param t_q: queries $Q_i$
    :param t_k: keys $K_j$
    :param t_v: values $V_j$
    :param sm_scale_log2e: $\sigma \log_2 e$ softmax scale multiplied by $\log_2 e$
    :param t_lse: $\log_2 \sum_j e^{S_{ij}}$ (out)
    :param t_o: $O_i$ output
    :param n_groups: number of groups in GQA
    :param q_seq_len: query sequence length
    :param kv_seq_len: key/value sequence length
    :param d_head: number of dimensions in a head
    :param BLOCK_Q: block size for query sequence length
    :param BLOCK_K: block size for key sequence length
    :param is_causal: whether causal attention

    Strides `z`, `h`, `m` and  `d` denote the stride of the corresponding dimensions
     (`batch_size`, `n_heads`, `q_seq_len`, `d_head`) in the query.
    Stride `n` denote the stride on `kv_seq_len` of key.
    """

    # We are computing the attention for $O_i$ for `i` ... `i + BLOCK_Q' in batch/head combination $z$.
    i = tl.program_id(0)
    z = tl.program_id(1) // n_groups
    g = tl.program_id(1) % n_groups

    # #### Create block pointers
    p_q = tl.make_block_ptr(t_q + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                            (q_seq_len, d_head),
                            (d_head, 1),
                            (i * BLOCK_Q, 0),
                            (BLOCK_Q, d_head),
                            (1, 0))
    p_v = tl.make_block_ptr(t_v + z * kv_seq_len * d_head,
                            (kv_seq_len, d_head),
                            (d_head, 1),
                            (0, 0),
                            (BLOCK_K, d_head),
                            (1, 0))
    p_kT = tl.make_block_ptr(t_k + z * kv_seq_len * d_head,
                             (d_head, kv_seq_len),
                             (1, d_head),
                             (0, 0),
                             (d_head, BLOCK_K),
                             (0, 1))
    p_o = tl.make_block_ptr(t_o + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                            (q_seq_len, d_head),
                            (d_head, 1),
                            (i * BLOCK_Q, 0),
                            (BLOCK_Q, d_head),
                            (1, 0))
    p_lse = tl.make_block_ptr(t_lse + z * n_groups * q_seq_len + g * q_seq_len,
                              (q_seq_len,),
                              (1,),
                              (i * BLOCK_Q,),
                              (BLOCK_Q,),
                              (0,))

    # Initialize offsets
    offs_i = i * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_j = tl.arange(0, BLOCK_K)

    # Mask for $Q$ for the last block
    i_mask = offs_i < q_seq_len

    # Initialize $m_i$ and $l_i$. $m_i$ is initialized to $-\inf$ and $l_i$ to $1$. So in the first update,
    # the effect of initial $l_i$ is $e^{m_i - m_{i}^{\text{new}}} l_i = 0$.
    #
    # `b_m` will be storing $m_i \log_2 e$
    b_m = tl.where(i_mask, -float("inf"), 0.0)
    b_l = tl.where(i_mask, 1.0, 0.0)

    # $O_i$
    b_o = tl.zeros([BLOCK_Q, d_head], dtype=HI_PRES_TL)

    # Load $Q_i$ outside the loop since it will be reused through out the loop over $K_j$.
    b_q = tl.load(p_q, boundary_check=(0,), padding_option="zero")

    if is_causal:
        # Inner loop upto the diagonal block
        b_o, b_l, b_m = _attn_fwd_inner(b_o, b_l, b_m, b_q,
                                        p_kT, p_v,
                                        sm_scale_log2e,
                                        BLOCK_Q, d_head, BLOCK_K,
                                        offs_i, offs_j,
                                        j=tl.full([], 0, tl.int32),  # type: ignore
                                        steps=(i * BLOCK_Q) // BLOCK_K,
                                        MASK=False,
                                        q_seq_len=q_seq_len,
                                        kv_seq_len=kv_seq_len
                                        )
        # Diagonal block with masking within it
        b_o, b_l, b_m = _attn_fwd_inner(b_o, b_l, b_m, b_q, p_kT, p_v,
                                        sm_scale_log2e,
                                        BLOCK_Q, d_head, BLOCK_K,
                                        offs_i, offs_j,
                                        j=i * BLOCK_Q,
                                        steps=BLOCK_Q // BLOCK_K,
                                        MASK=True,
                                        q_seq_len=q_seq_len,
                                        kv_seq_len=kv_seq_len
                                        )
    else:
        # Iterate through all $K_j$
        b_o, b_l, b_m = _attn_fwd_inner(b_o, b_l, b_m, b_q, p_kT, p_v,
                                        sm_scale_log2e,
                                        BLOCK_Q, d_head, BLOCK_K,
                                        offs_i, offs_j,
                                        j=tl.full([], 0, tl.int32),  # type: ignore
                                        steps=tl.cdiv(kv_seq_len, BLOCK_K),
                                        MASK=False,
                                        q_seq_len=q_seq_len,
                                        kv_seq_len=kv_seq_len
                                        )

    # Store LSE $\log_2 L_i = \log_2 \big( l_i * e^{m_i} \big) = \log_2 l_i + m_i log 2$
    tl.store(p_lse, b_m + tl.math.log2(b_l), boundary_check=(0,))
    # Store $O_i = \frac{\tilde{O}_i}{l_i}$
    tl.store(p_o, (b_o / b_l[:, None]).to(t_o.type.element_ty), boundary_check=(0,))


@triton.jit
def _attn_fwd_inner(b_o, b_l, b_m, b_q,
                    p_kT, p_v,
                    sm_scale_log2e,
                    BLOCK_Q: tl.constexpr,
                    d_head: tl.constexpr,
                    BLOCK_K: tl.constexpr,
                    offs_i, offs_j,
                    j,
                    steps,
                    MASK: tl.constexpr,
                    q_seq_len: tl.constexpr,
                    kv_seq_len: tl.constexpr
                    ):
    """
    #### Inner loop to calculate $O_i$

    This iterates through keys and values starting from `j` for `steps` number of steps.
    In each step it processes `BLOCK_K` entries of keys/values.
    """
    tl.static_assert(BLOCK_Q % BLOCK_K == 0)

    # Move $K_j$ and $V_j$ pointers
    p_kT = tl.advance(p_kT, (0, j))
    p_v = tl.advance(p_v, (j, 0))

    # Iterate over $K$, $V$ and update $\tilde{O}_i$ and $l_i$
    for _ in range(steps):
        # Load $K_j^T$
        b_kT = tl.load(p_kT, boundary_check=(1,), padding_option="zero")
        # Compute $(\log 2) S_ij  = (\log 2) \sigma Q_i K_j^T$
        b_s = tl.dot(b_q, b_kT, out_dtype=HI_PRES_TL)
        b_s = b_s * sm_scale_log2e

        # Apply causal mask
        if MASK:
            causal_mask = offs_i[:, None] >= (j + offs_j[None, :])
            b_s = tl.where(causal_mask, b_s, -float("inf"))

        # Mask out if the block is beyond the end of $K_j$
        j_mask = (j + offs_j) < kv_seq_len
        b_s = tl.where(j_mask[None, :], b_s, -float("inf"))

        # $(\log_2 e) m_{i}^{\text{new}} = \max((\log_2 e) m_i, \max_{j=j1}^{j2} (\log_2 e) S_{ij})$
        b_m_new = tl.maximum(b_m, tl.max(b_s, -1))
        # \begin{align}
        # \tilde{P}_{ij} &= e^{(S_{ij} - m_i^{\text{new}}}
        # \\
        # &= 2^{(\log_2 e) S_{ij} - (\log_2 e) m_i^{\text{new}}}
        # \end{align}
        b_p = tl.math.exp2(b_s - b_m_new[:, None])

        # $\sum_{j=j1}^{j2} \tilde{P}_{ij}$
        b_l_new = tl.sum(b_p, -1)
        # $e^{m_i - m_{i}^{\text{new}}}$
        b_m_m_new = tl.math.exp2(b_m - b_m_new)
        # $l_i \leftarrow e^{m_i - m_{i}^{\text{new}}} l_i + \sum_{j=j1}^{j2} \tilde{P}_{ij}$
        b_l = b_l * b_m_m_new + b_l_new

        # $O_i \leftarrow e^{m_i - m_{i}^{\text{new}}} O_i + \tilde{P}_{ij} V_j$
        b_o = b_o * b_m_m_new[:, None]
        b_p = b_p.to(b_q.dtype)  # TODO
        b_v = tl.load(p_v, boundary_check=(0,), padding_option="zero")
        b_o += tl.dot(b_p, b_v, out_dtype=HI_PRES_TL)

        # $(\log_2 e) m_i \leftarrow (\log_2 e) m_{i}^{\text{new}}$
        b_m = b_m_new

        # Move pointers
        j += BLOCK_K
        p_v = tl.advance(p_v, (BLOCK_K, 0))
        p_kT = tl.advance(p_kT, (0, BLOCK_K))

    tl.static_assert(b_o.dtype == HI_PRES_TL, "attn_fwd_inner requires accumulator to be in HI_PRES_TL precision")

    return b_o, b_l, b_m


@triton.jit
def _attn_bwd_d(t_o, t_do,
                t_pdp,
                BLOCK_Q: tl.constexpr, d_head: tl.constexpr,
                q_seq_len: tl.constexpr,
                n_groups: tl.constexpr,
                ):
    """
    #### Triton kernel to compute $D_i$
    """
    i = tl.program_id(0) * BLOCK_Q
    z = tl.program_id(1)

    # Create block pointers
    p_o = tl.make_block_ptr(t_o + z * n_groups * q_seq_len * d_head,
                            (n_groups, q_seq_len, d_head),
                            (q_seq_len * d_head, d_head, 1),
                            (0, i, 0),
                            (n_groups, BLOCK_Q, d_head),
                            (2, 1, 0))
    p_do = tl.make_block_ptr(t_do + z * n_groups * q_seq_len * d_head,
                             (n_groups, q_seq_len, d_head),
                             (q_seq_len * d_head, d_head, 1),
                             (0, i, 0),
                             (n_groups, BLOCK_Q, d_head),
                             (2, 1, 0))
    p_pdp = tl.make_block_ptr(t_pdp + z * n_groups * q_seq_len,
                              (n_groups, q_seq_len),
                              (q_seq_len, 1),
                              (0, i),
                              (n_groups, BLOCK_Q),
                              (1, 0))

    # Load $O_i$
    o = tl.load(p_o, boundary_check=(1,), padding_option="zero")
    # Load $dO_i$
    do = tl.load(p_do, boundary_check=(1,), padding_option="zero").to(HI_PRES_TL)
    # Calculate $D_i = dO_i O_i^T$
    d = tl.sum(o * do, axis=-1)
    # Save $D_i$
    tl.store(p_pdp, d, boundary_check=(1,))


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
                   BLOCK_Q: tl.constexpr,
                   BLOCK_K: tl.constexpr,
                   ):
    """
    #### Triton kernel to compute $dK_j$ and $dV_j$
    """

    # Compute $dK_j$ and $dV_j$ for `j` ... `j + BLOCK_K` by iterating over $Q_i$
    j = tl.program_id(0) * BLOCK_K
    z = tl.program_id(1)

    # Create block pointers
    p_k = tl.make_block_ptr(t_k + z * kv_seq_len * d_head,
                            (kv_seq_len, d_head),
                            (d_head, 1),
                            (j, 0),
                            (BLOCK_K, d_head),
                            (1, 0))
    p_v = tl.make_block_ptr(t_v + z * kv_seq_len * d_head,
                            (kv_seq_len, d_head),
                            (d_head, 1),
                            (j, 0),
                            (BLOCK_K, d_head),
                            (1, 0))
    p_dk = tl.make_block_ptr(t_dk + z * kv_seq_len * d_head,
                             (kv_seq_len, d_head),
                             (d_head, 1),
                             (j, 0),
                             (BLOCK_K, d_head),
                             (1, 0))
    p_dv = tl.make_block_ptr(t_dv + z * kv_seq_len * d_head,
                             (kv_seq_len, d_head),
                             (d_head, 1),
                             (j, 0),
                             (BLOCK_K, d_head),
                             (1, 0))

    # Initialize $\frac{1}{\sigma} dK$ and $dV$
    b_dk = tl.zeros([BLOCK_K, d_head], dtype=HI_PRES_TL)
    b_dv = tl.zeros([BLOCK_K, d_head], dtype=HI_PRES_TL)

    # Load $\frac{\sigma}{\log 2} K$ and $V$ outside the loop.
    b_k = tl.load(p_k, boundary_check=(0,), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0,), padding_option="zero")

    # Iterate through queries in GQA
    for g in range(n_groups):
        # Create block pointers
        p_qT = tl.make_block_ptr(t_q + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                                 (d_head, q_seq_len),
                                 (1, d_head),
                                 (0, 0),
                                 (d_head, BLOCK_Q),
                                 (0, 1))

        p_do = tl.make_block_ptr(t_do + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                                 (q_seq_len, d_head),
                                 (d_head, 1),
                                 (0, 0),
                                 (BLOCK_Q, d_head),
                                 (1, 0))
        p_lse = tl.make_block_ptr(t_lse + z * n_groups * q_seq_len + g * q_seq_len,
                                  (q_seq_len,),
                                  (1,),
                                  (0,),
                                  (BLOCK_Q,),
                                  (0,))
        p_pdp = tl.make_block_ptr(t_pdp + z * n_groups * q_seq_len + g * q_seq_len,
                                  (q_seq_len,),
                                  (1,),
                                  (0,),
                                  (BLOCK_Q,),
                                  (0,))

        if is_causal:
            # Inner loop at the diagonal block
            b_dk, b_dv = _attn_bwd_dkdv_inner(
                b_dk, b_dv,
                p_qT, b_k, b_v, p_do,
                p_lse, p_pdp,
                BLOCK_Q, BLOCK_K,
                d_head,
                j=j, i=j,
                steps=BLOCK_K // BLOCK_Q,
                MASK=True,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
            )

            # Inner loop on queries after the diagonal
            b_dk, b_dv = _attn_bwd_dkdv_inner(
                b_dk, b_dv,
                p_qT, b_k, b_v, p_do,
                p_lse, p_pdp,
                BLOCK_Q, BLOCK_K,
                d_head,
                j=j, i=j + BLOCK_K,
                steps=tl.cdiv((q_seq_len - (j + BLOCK_K)), BLOCK_Q),
                MASK=False,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len
            )
        else:
            # Iterate through all queries
            b_dk, b_dv = _attn_bwd_dkdv_inner(
                b_dk, b_dv,
                p_qT, b_k, b_v, p_do,
                p_lse, p_pdp,
                BLOCK_Q, BLOCK_K,
                d_head,
                j=j, i=tl.full([], 0, tl.int32),
                steps=tl.cdiv(q_seq_len, BLOCK_Q),
                MASK=False,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len
            )

    # Save $dV$
    tl.store(p_dv, b_dv.to(t_dv.type.element_ty), boundary_check=(0,))

    # `b_dk` had $\frac{1}{\sigma} dK$
    b_dk *= sm_scale

    # Save $dK$
    tl.store(p_dk, b_dk.to(t_dk.type.element_ty), boundary_check=(0,))


@triton.jit
def _attn_bwd_dkdv_inner(b_dk, b_dv,
                         p_qT, b_k, b_v, p_do,
                         p_lse, p_pdp,
                         BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                         d_head: tl.constexpr,
                         j, i, steps,
                         MASK: tl.constexpr,
                         q_seq_len: tl.constexpr,
                         kv_seq_len: tl.constexpr):
    """
    #### Inner loop to calculate $dK_j$, $dV_j$
    """

    # To apply the mask
    tl.static_assert(BLOCK_K % BLOCK_Q == 0)

    # Offsets and mask
    offs_i = i + tl.arange(0, BLOCK_Q)
    offs_j = j + tl.arange(0, BLOCK_K)

    # Move the pointers
    p_qT = tl.advance(p_qT, (0, i))
    p_do = tl.advance(p_do, (i, 0))
    p_lse = tl.advance(p_lse, (i,))
    p_pdp = tl.advance(p_pdp, (i,))

    # Iterate over $Q$
    for _ in range(steps):
        # Load $Q_i^T$
        b_qT = tl.load(p_qT, boundary_check=(1,), padding_option="zero")

        # $log_2 L_i$
        b_l = tl.load(p_lse, boundary_check=(0,), padding_option="zero")

        # $(\log_2 e) S_{ij}^T = \sigma (\log_2 e) K_j Q_i^T$
        b_sT = tl.dot(b_k, b_qT, out_dtype=HI_PRES_TL)

        # \begin{align}
        # P_{ij} &= \frac{e^{S_{ij}}}{L_i}
        # \\
        # &= \frac{2^{(log_2 e) S_{ij}}}{2^{\log_2 L_i}}
        # \\
        # &= 2^{(log_2 e) S_{ij} - \log_2 L_i}
        # \end{align}
        b_pT = tl.math.exp2(b_sT - b_l[None, :])

        # Autoregressive masking
        if MASK:
            mask = (offs_i[None, :] >= offs_j[:, None])
            b_pT = tl.where(mask, b_pT, 0.0)

        # Mask out if the block is beyond the end of $Q_i$
        #
        # Note: No need to mask out based on $j$
        # because the effects on positions outside boundary will not get stored in $dK$ or $dV$
        # Masking by $i$ may also not be necessary size the tensors have 0 on loading
        i_mask = offs_i < q_seq_len
        b_pT = tl.where(i_mask[None, :], b_pT, 0.0)

        # $dV_j = \sum_i P_{ij} dO_i$
        b_do = tl.load(p_do, boundary_check=(0,), padding_option="zero")
        b_dv += tl.dot(b_pT.to(b_do.dtype), b_do, out_dtype=HI_PRES_TL)

        # $D_i$
        b_pdp = tl.load(p_pdp, boundary_check=(0,), padding_option="zero")
        # $dP_{ij} = V_j dO_i^T$
        b_dpT = tl.dot(b_v, tl.trans(b_do), out_dtype=HI_PRES_TL).to(HI_PRES_TL)
        # $dS_{ij} = P_{ij} \big( dP_{ij} - D_i \big)$
        b_dsT = b_pT * (b_dpT - b_pdp[None, :])
        # $\frac{1}{\sigma} dK_j = \sum_i dS_{ij} Q_i$
        b_dk += tl.dot(b_dsT.to(b_qT.dtype), tl.trans(b_qT), out_dtype=HI_PRES_TL)

        # Increment pointers.
        offs_i += BLOCK_Q
        p_lse = tl.advance(p_lse, (BLOCK_Q,))
        p_pdp = tl.advance(p_pdp, (BLOCK_Q,))
        p_qT = tl.advance(p_qT, (0, BLOCK_Q))
        p_do = tl.advance(p_do, (BLOCK_Q, 0))

    # Return accumulated $dK$ and $dV$
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
                 BLOCK_Q: tl.constexpr,
                 BLOCK_K: tl.constexpr,
                 ):
    """
    #### Triton kernel to compute $dQ_i$
    """

    i = tl.program_id(0) * BLOCK_Q
    z = tl.program_id(1) // n_groups
    g = tl.program_id(1) % n_groups  # TODO

    # Create block pointers
    p_q = tl.make_block_ptr(t_q + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                            (q_seq_len, d_head),
                            (d_head, 1),
                            (i, 0),
                            (BLOCK_Q, d_head),
                            (1, 0))
    p_dq = tl.make_block_ptr(t_dq + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                             (q_seq_len, d_head),
                             (d_head, 1),
                             (i, 0),
                             (BLOCK_Q, d_head),
                             (1, 0))
    p_do = tl.make_block_ptr(t_do + z * n_groups * q_seq_len * d_head + g * q_seq_len * d_head,
                             (q_seq_len, d_head),
                             (d_head, 1),
                             (i, 0),
                             (BLOCK_Q, d_head),
                             (1, 0))
    p_kT = tl.make_block_ptr(t_k + z * kv_seq_len * d_head,
                             (d_head, kv_seq_len),
                             (1, d_head),
                             (0, 0),
                             (d_head, BLOCK_K),
                             (0, 1))
    p_vT = tl.make_block_ptr(t_v + z * kv_seq_len * d_head,
                             (d_head, kv_seq_len),
                             (1, d_head),
                             (0, 0),
                             (d_head, BLOCK_K),
                             (0, 1))
    p_lse = tl.make_block_ptr(t_lse + z * n_groups * q_seq_len + g * q_seq_len,
                              (q_seq_len,),
                              (1,),
                              (i,),
                              (BLOCK_Q,),
                              (0,))
    p_pdp = tl.make_block_ptr(t_pdp + z * n_groups * q_seq_len + g * q_seq_len,
                              (q_seq_len,),
                              (1,),
                              (i,),
                              (BLOCK_Q,),
                              (0,))

    # Load $Q_i$, $dO_i$, $D_i$, and $\log_2 L_i$ outside the loop
    b_q = tl.load(p_q, boundary_check=(0,), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=(0,), padding_option="zero")
    b_pdp = tl.load(p_pdp, boundary_check=(0,), padding_option="zero")
    b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")

    # Initialize $(\log_2 e)dQ$
    b_dq = tl.zeros([BLOCK_Q, d_head], dtype=HI_PRES_TL)

    # $$dq_i = \sum_j dS_{ij} k_j = \sum_j P_{ij} \big( dP_{ij} - D_i \big) k_j$$

    if is_causal:
        # Compute $dQ$ for masked (diagonal) blocks.
        b_dq = _attn_bwd_dq_inner(b_dq, b_q, p_kT, p_vT,
                                  b_do, b_lse, b_pdp,
                                  BLOCK_Q, BLOCK_K,
                                  i=i, j=i,
                                  steps=BLOCK_Q // BLOCK_K,
                                  MASK=True,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=kv_seq_len
                                  )

        # Compute for other blocks
        b_dq = _attn_bwd_dq_inner(b_dq, b_q, p_kT, p_vT,
                                  b_do, b_lse, b_pdp,
                                  BLOCK_Q, BLOCK_K,
                                  i=i, j=tl.full([], 0, tl.int32),  # type: ignore
                                  steps=i // BLOCK_K,
                                  MASK=False,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=kv_seq_len
                                  )
    else:
        # Iterate through all $K$
        b_dq = _attn_bwd_dq_inner(b_dq, b_q, p_kT, p_vT,
                                  b_do, b_lse, b_pdp,
                                  BLOCK_Q, BLOCK_K,
                                  i=i, j=tl.full([], 0, tl.int32),  # type: ignore
                                  steps=tl.cdiv(kv_seq_len, BLOCK_K),
                                  MASK=False,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=kv_seq_len
                                  )

    # `b_dq` stores $(\log_2 e)dQ$ so multiply by $\log_e 2$ to get $dQ$
    b_dq *= 0.6931471824645996

    # Save $dQ$
    tl.store(p_dq, b_dq.to(t_dq.type.element_ty), boundary_check=(0,))


@triton.jit
def _attn_bwd_dq_inner(b_dq, b_q, p_kT, p_vT,
                       b_do, b_lse, b_pdp,
                       BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                       i, j, steps,
                       MASK: tl.constexpr,
                       q_seq_len: tl.constexpr,
                       kv_seq_len: tl.constexpr):
    """
    #### Inner loop to calculate $dQ_i$
    """

    # Offsets
    offs_i = i + tl.arange(0, BLOCK_Q)
    offs_j = j + tl.arange(0, BLOCK_K)

    # Move the pointers
    p_kT = tl.advance(p_kT, (0, j))
    p_vT = tl.advance(p_vT, (0, j))

    tl.static_assert(BLOCK_Q % BLOCK_K == 0, 'BLOCK_Q must be divisible by BLOCK_K')

    # Iterate over $K$
    for _ in range(steps):
        # Load $K_j^T$
        b_kT = tl.load(p_kT, boundary_check=(1,), padding_option="zero")
        # Load $V_j^T$
        b_vT = tl.load(p_vT, boundary_check=(1,), padding_option="zero")

        # $(\log_2 e) S_{ij} = \sigma (\log_2 e) Q_i K_j^T$
        b_s = tl.dot(b_q, b_kT, out_dtype=HI_PRES_TL)

        # \begin{align}
        # P_{ij} &= \frac{e^{S_{ij}}}{L_i}
        # \\
        # &= \frac{2^{(log_2 e) S_{ij}}}{2^{\log_2 L_i}}
        # \\
        # &= 2^{(log_2 e) S_{ij} - \log_2 L_i}
        # \end{align}
        b_p = tl.math.exp2(b_s - b_lse[:, None])

        # Autoregressive masking
        if MASK:
            causal_mask = (offs_i[:, None] >= offs_j[None, :])
            b_p = tl.where(causal_mask, b_p, 0.0)

        # Mask out if the block is beyond the end of $Q_i$
        j_mask = offs_j < kv_seq_len
        b_p = tl.where(j_mask[None, :], b_p, 0.0)

        # $$dq_i = \sum_j dS_{ij} k_j = \sum_j P_{ij} \big( dP_{ij} - D_i \big) k_j$$

        # $dP_{ij} = dO_i V_j^T$
        b_dp = tl.dot(b_do, b_vT, out_dtype=HI_PRES_TL).to(HI_PRES_TL)
        # $dS_{ij} = P_{ij} \big( dP_{ij} - D_i \big)$
        b_ds = b_p * (b_dp - b_pdp[:, None])
        # $(\log_2 e) dQ_i = \sum_j dS_{ij} \sigma (\log_2 e) K_j$
        b_dq += tl.dot(b_ds.to(b_kT.dtype), tl.trans(b_kT), out_dtype=HI_PRES_TL)

        # Increment pointers.
        offs_j += BLOCK_K
        p_kT = tl.advance(p_kT, (0, BLOCK_K))
        p_vT = tl.advance(p_vT, (0, BLOCK_K))

    # Return accumulated $dQ$
    return b_dq
