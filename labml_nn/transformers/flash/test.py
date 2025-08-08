"""
### Test Flash Attention Implementation

This is the code to test and measure performance of our flash attention implementation
"""

import torch
import triton

from labml import logger, monit
from labml_nn.transformers.flash import attention

HI_PRES_TORCH = torch.float32


@torch.no_grad()
def _calc_abs_rel_error(a: torch.Tensor, b: torch.Tensor, atol=1e-2):
    """
    #### Calculate absolute and relative error for reporting
    """
    d = (a - b).abs()
    max_abs = d.max()
    d = (d - atol).clamp(min=0)
    d = d / b.abs()
    max_rel = d.max()

    return max_abs.cpu().item(), max_rel.cpu().item()


def test_fwd_bwd(batch_size, n_heads, k_heads, q_seq_len, kv_seq_len, d_head, causal, dtype, device):
    """
    #### Compare our implementation with naive PyTorch attention
    """

    with monit.section(f'Init {q_seq_len} {kv_seq_len} {d_head}'):
        torch.manual_seed(20)
        q = (torch.empty((batch_size, n_heads, q_seq_len, d_head),
                         dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
        k = (torch.empty((batch_size, k_heads, kv_seq_len, d_head),
                         dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
        v = (torch.empty((batch_size, k_heads, kv_seq_len, d_head),
                         dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
        sm_scale = d_head ** -0.5
        d_out = torch.randn_like(q)
        # reference implementation
        mask = torch.tril(torch.ones((q_seq_len, kv_seq_len), device=device, dtype=torch.bool))
        torch.cuda.synchronize()

    with monit.section('Pytorch'):
        p = torch.matmul(q.view(batch_size, k_heads, -1, q_seq_len, d_head),
                         k.transpose(2, 3)[:, :, None, :, :]) * sm_scale
        if causal:
            p[:, :, :, ~mask] = float("-inf")
        p = torch.softmax(p.to(HI_PRES_TORCH), dim=-1).to(dtype)
        ref_out = torch.matmul(p, v[:, :, None, :, :])
        ref_out = ref_out.view(q.shape)
        ref_out.backward(d_out)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
        torch.cuda.synchronize()

    with monit.section('Triton'):
        assert q.dtype == dtype
        tri_out = attention(q, k, v, causal, sm_scale).to(dtype)
        monit.progress(0.5)

        tri_out.backward(d_out)
        monit.progress(0.9)
        tri_dv, v.grad = v.grad.clone(), None  # type: ignore
        tri_dk, k.grad = k.grad.clone(), None  # type: ignore
        tri_dq, q.grad = q.grad.clone(), None  # type: ignore
        torch.cuda.synchronize()

    with monit.section('Test') as s:
        # compare
        passed = True
        if not torch.allclose(tri_out, ref_out, atol=1e-2, rtol=0.):
            abs_err, rel_err = _calc_abs_rel_error(ref_out, tri_out)
            logger.log(('[FAILED]', logger.Text.danger), f' Out mismatch {abs_err} {rel_err}')
            passed = False
        rtol = 1e-1
        if not torch.allclose(tri_dq, ref_dq, atol=1e-2, rtol=rtol):
            abs_err, rel_err = _calc_abs_rel_error(ref_dq, tri_dq)
            logger.log(('[FAILED]', logger.Text.danger), f' dQ mismatch {abs_err} {rel_err}')
            passed = False
        if not torch.allclose(tri_dv, ref_dv, atol=1e-2, rtol=rtol):
            abs_err, rel_err = _calc_abs_rel_error(ref_dv, tri_dv)
            logger.log(('[FAILED]', logger.Text.danger), f' dV mismatch {abs_err} {rel_err}')
            passed = False
        if not torch.allclose(tri_dk, ref_dk, atol=1e-2, rtol=rtol):
            abs_err, rel_err = _calc_abs_rel_error(ref_dk, tri_dk)
            logger.log(('[FAILED]', logger.Text.danger), f' dK mismatch {abs_err} {rel_err}')
            passed = False

        if passed:
            logger.log('[PASSED]', logger.Text.success)
            s.success = True
        else:
            s.success = False
        torch.cuda.synchronize()


def _perf_triton_fn(*, device, dtype, batch_size, k_heads, n_groups, seq_len, d_head, causal):
    """
    Get a partial function to test performance of our implementation
    """
    q = torch.randn((batch_size, k_heads * n_groups, seq_len, d_head), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((batch_size, k_heads, seq_len, d_head), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((batch_size, k_heads, seq_len, d_head), dtype=dtype, device=device, requires_grad=True)
    sm_scale = d_head ** -0.5
    return lambda: attention(q, k, v, causal, sm_scale)


def _perf_flash(*, batch_size, k_heads, n_groups, seq_len, d_head, causal, device, dtype):
    """
    Get a partial function to test performance of original flash implementation
    """
    q = torch.randn((batch_size, seq_len, k_heads * n_groups, d_head), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((batch_size, seq_len, k_heads, d_head), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((batch_size, seq_len, k_heads, d_head), dtype=dtype, device=device, requires_grad=True)
    from flash_attn import flash_attn_func
    return lambda: flash_attn_func(q, k, v, causal=causal)


def measure_performance(name, fn, *, batch_size, k_heads, n_groups, seq_len, d_head, causal, is_bwd: bool):
    """
    ### Measure the speed
    """
    if is_bwd:
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
    ms = triton.testing.do_bench(fn)

    flops_per_matmul = 2.0 * batch_size * k_heads * n_groups * seq_len * seq_len * d_head
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if is_bwd:
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)

    tf_ps = total_flops * 1e-12 / (ms * 1e-3)
    logger.log((f'{name}', logger.Text.key), ': ', f'{ms :,.1f}ms', ' ', f'{tf_ps :,.2f}TFps')


def main():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    dtype = torch.float16

    # only works on post-Ampere GPUs right now
    test_fwd_bwd(1, 4, 1, 2048, 2048, 128, True, dtype=dtype, device=device)
    test_fwd_bwd(16, 32, 8, 2001, 4001, 128, False, dtype=dtype, device=device)
    test_fwd_bwd(4, 32, 8, 2048, 1024, 128, False, dtype=dtype, device=device)
    test_fwd_bwd(4, 32, 8, 2001, 4001, 128, True, dtype=dtype, device=device)

    _conf = {
        'batch_size': 16,
        'k_heads': 8,
        'n_groups': 4,
        'seq_len': 2048,
        'd_head': 128,
    }

    for _causal in [False, True]:
        for is_bwd in [False, True]:
            logger.log(f'{"Causal" if _causal else "Non-causal"} {" Backward" if is_bwd else ""}', logger.Text.title)
            measure_performance(f'flash', _perf_flash(causal=_causal, device=device, dtype=dtype, **_conf),
                                is_bwd=is_bwd,
                                causal=_causal, **_conf)
            measure_performance(f'triton', _perf_triton_fn(causal=_causal, device=device, dtype=dtype, **_conf),
                                is_bwd=is_bwd,
                                causal=_causal, **_conf)


if __name__ == "__main__":
    main()
