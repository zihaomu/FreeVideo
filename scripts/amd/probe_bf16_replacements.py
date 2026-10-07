#!/usr/bin/env python3
"""Exercise real gfx1201 gate calls and BLAS restoration, including exceptions."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-uuid', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    import torch
    import torch.nn.functional as F
    from freevideo_engine.locking import runtime_lock, device_lock_path
    from freevideo_engine.rocm_bf16 import make_gate_linear, _blaslt
    with ExitStack() as leases, torch.inference_mode():
        leases.enter_context(runtime_lock(shared=True))
        leases.enter_context(runtime_lock(device_lock_path(args.device_uuid), inherit=False))
        assert torch.version.hip and torch.cuda.device_count() == 1
        previous = torch.backends.cuda.preferred_blas_library()
        assert 'Cublaslt' not in str(previous)
        policy = SimpleNamespace()
        os.environ['FREEVIDEO_ROCM_GATE_BLAS'] = 'cublaslt'
        linear = make_gate_linear(policy, 16, 1)
        torch.manual_seed(42)
        rows = []
        for m, k, n, bias in [(49, 5376, 16, False), (2053, 5376, 16, True),
                              (2053, 5376, 128, False), (2053, 128, 2048, True)]:
            x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
            w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
            b = torch.randn(n, device='cuda', dtype=torch.bfloat16) if bias else None
            reference = F.linear(x, w, b)
            value = linear(x, w, b)
            assert torch.backends.cuda.preferred_blas_library() == previous
            diff = value.float() - reference.float()
            error = float((diff.square().mean() / reference.float().square().mean()).sqrt())
            assert value.shape == reference.shape and bool(value.isfinite().all()) and error < .01
            rows.append(dict(shape=[m, k, n], bias=bias, relative_rmse=error,
                             exact=bool(torch.equal(reference, value))))
        try:
            linear(torch.ones(2, 3, device='cuda', dtype=torch.bfloat16),
                   torch.ones(4, 5, device='cuda', dtype=torch.bfloat16))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Expected invalid-shape call to raise')
        assert torch.backends.cuda.preferred_blas_library() == previous
        try:
            with _blaslt():
                raise RuntimeError('restoration probe')
        except RuntimeError:
            pass
        assert torch.backends.cuda.preferred_blas_library() == previous
        cpu_x, cpu_w = torch.ones(2, 3), torch.ones(4, 3)
        assert torch.equal(linear(cpu_x, cpu_w), F.linear(cpu_x, cpu_w))
        assert policy.rocm_gate_linear_calls == 4
        os.environ['FREEVIDEO_ROCM_GATE_BLAS'] = 'default'
        assert make_gate_linear(policy, 16, 1) is F.linear
        torch.cuda.synchronize()
        report = dict(success=True, rows=rows, successful_routed_calls=4,
                      normal_and_exception_restore=True, cpu_fallback=True,
                      preferred_blas=str(previous), torch=str(torch.__version__), hip=torch.version.hip,
                      scope='Backend isolation and synthetic numerical sanity; full generation quality requires separate validation.')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
