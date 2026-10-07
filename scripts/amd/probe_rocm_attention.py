#!/usr/bin/env python3
"""Check rectangular BF16 attention, padding and independent NHD batches on RDNA4."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from freevideo_engine.rocm_attention import attention
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.manual_seed(20261007)
    assert torch.version.hip and torch.cuda.device_count() == 1
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(':')[0] == 'gfx1201'
    rows = []
    for batch, queries, keys, heads in [(2, 17, 31, 3), (1, 129, 257, 8),
                                        (4, 336, 2041, 16), (1, 2875, 73435, 16),
                                        (4, 5040, 17995, 16)]:
        q = torch.randn(batch, queries, heads, 128, device='cuda', dtype=torch.bfloat16)
        k, v = [torch.randn(batch, keys, heads, 128, device='cuda', dtype=torch.bfloat16)
                for _ in range(2)]
        scale = 128 ** -.5
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            reference = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                      v.transpose(1, 2), scale=scale).transpose(1, 2)
        actual = attention(q, k, v, scale)
        difference = actual.float() - reference.float()
        relative = float((difference.square().mean() / reference.float().square().mean()).sqrt())
        finite = bool(actual.isfinite().all())
        assert finite and relative < .003, (batch, queries, keys, heads, relative)
        assert torch.equal(actual, attention(q, k, v, scale))
        row = dict(query=list(q.shape), key=list(k.shape), finite=finite,
                   relative_rmse=relative, repeat_exact=True,
                   max_absolute_difference=float(difference.abs().max()))
        if queries < 300:
            with sdpa_kernel(SDPBackend.MATH):
                exact = F.scaled_dot_product_attention(q.float().transpose(1, 2), k.float().transpose(1, 2),
                                                       v.float().transpose(1, 2), scale=scale).transpose(1, 2)
            error = float(((actual.float() - exact).square().mean() / exact.square().mean()).sqrt())
            assert error < .003, error
            row['fp32_math_relative_rmse'] = error
        rows.append(row)
    # Strided NHD inputs must read the caller's actual strides, not assume contiguous data.
    q, k, v = [torch.randn(2, 35, 3, 256, device='cuda', dtype=torch.bfloat16)[..., ::2]
               for _ in range(3)]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        reference = F.scaled_dot_product_attention(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                                                  v.transpose(1, 2).contiguous()).transpose(1, 2)
    actual = attention(q, k, v, 128 ** -.5)
    relative = float(((actual.float() - reference.float()).square().mean() / reference.float().square().mean()).sqrt())
    assert relative < .003, relative
    rows.append(dict(strided_inputs=True, relative_rmse=relative))
    try:
        attention(q.float(), k.float(), v.float(), 128 ** -.5)
    except ValueError:
        rows.append(dict(unsupported_dtype_rejected=True))
    else:
        raise AssertionError('Unsupported dtype was accepted')
    report = dict(passed=True, gpu=torch.cuda.get_device_name(), arch=arch,
                  torch=str(torch.__version__), hip=torch.version.hip,
                  reference='AOTriton BF16 Flash Attention; small shapes also checked against FP32 math SDPA',
                  numerical_gate_relative_rmse=.003, cases=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
