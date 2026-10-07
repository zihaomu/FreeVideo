#!/usr/bin/env python3
"""Weight-free R9700 checks. Successful kernels do not establish H3 readiness."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.set_num_threads(4)
    torch.manual_seed(171)
    report = dict(
        scope='Small weight-free kernels only; no H3 video or H3 speed measurement',
        time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        python=platform.python_version(), torch=str(torch.__version__),
        hip=torch.version.hip, cuda=torch.version.cuda,
        rocr_visible_devices=os.environ.get('ROCR_VISIBLE_DEVICES'),
        image_id=os.environ.get('FV_PREFLIGHT_IMAGE_ID'),
        storage_root=os.environ.get('FV_STORAGE_ROOT'),
        vdn_root=os.environ.get('FREEVIDEO_VDN_ROOT'),
        kernel_cache=os.environ.get('TRITON_CACHE_DIR'),
        kernels=[], full_video_verified=False,
    )

    def check(name, operation):
        start = time.perf_counter()
        try:
            details = operation() or {}
            torch.cuda.synchronize()
            row = dict(name=name, status='passed', **details)
        except Exception as error:
            row = dict(name=name, status='failed', error=f'{type(error).__name__}: {error}')
        row['wall_seconds_including_first_compile'] = time.perf_counter() - start
        report['kernels'].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    def relative_error(value, reference, limit):
        finite = bool(value.isfinite().all())
        error = ((value.float() - reference.float()).square().mean()
                 / reference.float().square().mean().clamp_min(1e-20)).sqrt().item()
        if not finite or error > limit:
            raise RuntimeError(f'finite={finite}, relative RMSE={error}, limit={limit}')
        return dict(finite=finite, relative_rmse=error, relative_rmse_limit=limit,
                    shape=list(value.shape))

    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        report.update(basic_platform_ready=False, error='Require ROCm and exactly one visible GPU')
    else:
        device = torch.cuda.get_device_properties(0)
        report['gpu'] = dict(name=device.name, arch=getattr(device, 'gcnArchName', ''),
                             capability=list(torch.cuda.get_device_capability(0)),
                             total_bytes=device.total_memory,
                             free_bytes=torch.cuda.mem_get_info(0)[0],
                             pci_bus_id=getattr(device, 'pci_bus_id', None))
        for package in ('triton', 'transformers', 'diffusers', 'safetensors', 'av'):
            try:
                report.setdefault('packages', {})[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report.setdefault('packages', {})[package] = None

        def hardware_identity():
            from freevideo_engine.hardware import _detect_local
            hardware = _detect_local()
            report['freevideo_hardware'] = hardware.to_dict()
            report['architecture_misclassified'] = hardware.architecture.startswith('blackwell')
            return dict(architecture=hardware.architecture,
                        warning='ROCm capability is not an NVIDIA architecture identifier')

        check('unmodified_freevideo_hardware_detection', hardware_identity)
        with torch.no_grad():
            x = torch.randn(701, 512, device='cuda', dtype=torch.bfloat16)
            weight = torch.randn(12288, 512, device='cuda', dtype=torch.bfloat16)
            reference = x.float() @ weight.float().t()
            check('bf16_gemm', lambda: relative_error(x @ weight.t(), reference, .01))

            def fp8_gemm(rowwise):
                xs = x.float().abs().amax(1, keepdim=True) / 448 if rowwise else x.float().abs().amax().reshape(1) / 448
                ws = weight.float().abs().amax(1, keepdim=True) / 448 if rowwise else weight.float().abs().amax().reshape(1) / 448
                xq = (x.float() / xs).to(torch.float8_e4m3fn)
                wq = (weight.float() / ws).to(torch.float8_e4m3fn)
                # PyTorch's FP8 GEMM requires aligned rows. Keep a separate tail probe in W8A16.
                xq = xq[:688].contiguous()
                xs = xs[:688].contiguous() if rowwise else xs
                result = torch._scaled_mm(xq, wq.t(), scale_a=xs,
                                          scale_b=ws.t().contiguous() if rowwise else ws,
                                          out_dtype=torch.bfloat16)
                expected = (xq.float() * xs) @ (wq.float() * ws).t()
                return relative_error(result, expected, .02)

            check('torch_fp8_per_tensor', lambda: fp8_gemm(False))
            check('torch_fp8_rowwise', lambda: fp8_gemm(True))

            def weight_only():
                from freevideo_engine.weight_only import linear
                scale = weight.float().abs().amax().reshape(1, 1) / 448
                packed = (weight.float() / scale).to(torch.float8_e4m3fn)
                return relative_error(linear(x, packed, scale),
                                      x.float() @ (packed.float() * scale).t(), .01)

            check('freevideo_w8a16_triton', weight_only)
            del x, weight, reference
            q, k, v = [torch.randn(1, 487, 4, 128, device='cuda', dtype=torch.bfloat16)
                       for _ in range(3)]
            with sdpa_kernel(SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                         v.transpose(1, 2)).transpose(1, 2)

            def attention(backend):
                from freevideo_engine.backends.cuda_attention import CUDAAttentionKernels
                kernels = CUDAAttentionKernels(backend, backend)
                return relative_error(kernels.batched(backend, q, k, v, 128 ** -.5), expected, .006)

            check('freevideo_torch_flash_attention', lambda: attention('torch-flash'))
            check('freevideo_cudnn_attention', lambda: attention('cudnn'))

            def upstream_probe(backend):
                child = subprocess.run([sys.executable, '-m', 'freevideo_engine.probe', backend],
                                       text=True, capture_output=True, timeout=90)
                if child.returncode:
                    raise RuntimeError(child.stderr[-2000:] or child.stdout[-2000:])
                receipt = json.loads(child.stdout.strip().splitlines()[-1])
                if receipt.get('status') != 'complete':
                    raise RuntimeError(json.dumps(receipt))
                return dict(receipt=receipt)

            check('unmodified_freevideo_hybrid_window_probe', lambda: upstream_probe('torch-flash'))
            check('unmodified_freevideo_fp8_linear_probe', lambda: upstream_probe('linear'))

        passed = {row['name'] for row in report['kernels'] if row['status'] == 'passed'}
        report['basic_platform_ready'] = (
            'gfx1201' in report['gpu']['arch']
            and {'bf16_gemm', 'freevideo_torch_flash_attention'} <= passed)
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(report=str(args.out), basic_platform_ready=report['basic_platform_ready'],
                         full_video_verified=False)), flush=True)
    return 0 if report['basic_platform_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
