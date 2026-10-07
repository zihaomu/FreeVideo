#!/usr/bin/env python3
"""Probe same-precision FreeVideo routes informed by the pinned native H3 port.

Production modules are not patched. Timings are warm operator measurements,
not predictions or measurements of complete video latency.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import shutil
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def build_h3_split(source, out):
    """Compile the original two F32 kernels, with a stream-aware C ABI only."""
    original = (source / 'h3_gpu_hip.cpp').read_text()
    begin = original.index('__global__ static void h3_hip_sdpa_f32_d64_split_qk_kernel(')
    end = original.index('static int h3_gpu_linear(', begin)
    kernels = original[begin:end]
    wrapper = r'''
extern "C" int h3_split(const float *q, const float *k, const float *v,
    float *o, float *scores, float *inverses, unsigned n, unsigned h,
    float scale, hipStream_t stream) {
    hipLaunchKernelGGL(h3_hip_sdpa_f32_d64_split_qk_kernel,
        dim3(n,h), dim3(32), 0, stream, q,k,scores,inverses,1,n,h,scale);
    hipError_t status=hipGetLastError();
    if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(h3_hip_sdpa_f32_d64_split_pv_kernel,
        dim3(n,h), dim3(32), 0, stream, v,scores,inverses,o,1,n,h);
    return int(hipGetLastError());
}
'''
    text = '#include <hip/hip_runtime.h>\n#include <cstdint>\n#include <cmath>\n' + kernels + wrapper
    path = out / 'h3_split.cpp'
    path.write_text('// Kernels from zihaomu/h3-vdn.c, under its retained MIT LICENSE.\n' + text)
    (out / 'h3-LICENSE').write_bytes((source / 'LICENSE').read_bytes())
    compiler = shutil.which('hipcc')
    if compiler is None:raise RuntimeError('The probe environment needs hipcc')
    import torch
    sdk = Path(torch.__file__).resolve().parent.parent / '_rocm_sdk_core' / 'lib'
    runtime = next(sdk.glob('libamdhip64.so.*'), None)
    # The pinned pip SDK has a versioned runtime but no libamdhip64.so alias.
    # Compile HIP first, then link against that exact runtime without changing it.
    if runtime is not None:
        commands = [[compiler, '-O3', '-std=c++17', '--offload-arch=gfx1201',
                     '-fPIC', '-c', str(path), '-o', str(out / 'h3_split.o')],
                    [str(sdk / 'llvm/bin/clang++'), '-shared', str(out / 'h3_split.o'),
                     str(runtime), '-o', str(out / 'h3_split.so')]]
    else:
        commands = [[compiler, '-O3', '-std=c++17', '--offload-arch=gfx1201',
                     '-fPIC', '-shared', str(path), '-o', str(out / 'h3_split.so')]]
    with (out / 'hip-build.log').open('w') as log:
        for command in commands:subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return dict(commands=commands, kernel_sha256=hashlib.sha256(kernels.encode()).hexdigest(),
                original_sha256=hashlib.sha256(original.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--h3-source', type=Path, required=True)
    parser.add_argument('--sections', default='attention,fp8,vae')
    parser.add_argument('--include-projection-controls', action='store_true',
        help='Also compare artificial QKV/output row tilings; these are not production baselines')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--device-uuid', required=True)
    args = parser.parse_args()
    sections = set(args.sections.split(','))
    if sections - {'attention', 'fp8', 'vae'} or args.repeats < 3:
        parser.error('Sections: attention,fp8,vae; at least three repeats')
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import torch.nn.functional as F
    import triton
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from freevideo_engine.locking import runtime_lock, device_lock_path
    from freevideo_engine.rocm_attention import attention, _flash
    from freevideo_engine.paths import add_vdn
    add_vdn()
    from src.models.softmax_attention.decomposed import _plan
    from src.models.softmax_attention.window import window_bounds
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    assert torch.version.hip and torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(':')[0] == 'gfx1201'
    from freevideo_engine.hardware import _detect_local
    assert _detect_local().gpu_uuid == args.device_uuid
    report = dict(state='running', scope='Warm synthetic operator probes; no production dispatch changes or model-quality claim',
        gpu=props.name, arch=props.gcnArchName, uuid=args.device_uuid,
        torch=str(torch.__version__), hip=torch.version.hip, repeats=args.repeats,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git', '-C', str(args.h3_source), 'rev-parse', 'HEAD'], text=True).strip(),
        results=[], window_shapes=[])

    def save():
        (args.out / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')

    def measured(operation):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = operation()
        end.record(); end.synchronize()
        return begin.elapsed_time(end), result

    def compare(case, baseline, candidate, contract, extra=None):
        for _ in range(2):
            baseline(); candidate()
        torch.cuda.synchronize()
        expected, actual = baseline(), candidate()
        if isinstance(expected, list):
            expected, actual = torch.cat(expected), torch.cat(actual)
        delta = actual.float() - expected.float()
        relative = float((delta.square().mean() / expected.float().square().mean().clamp_min(1e-30)).sqrt())
        finite = bool(actual.isfinite().all())
        exact = torch.equal(actual, expected)
        repeated = candidate()
        if isinstance(repeated, list):repeated = torch.cat(repeated)
        deterministic = torch.equal(actual, repeated)
        row = dict(case=case, contract=contract, relative_rmse=relative,
                   max_abs=float(delta.abs().max()), finite=finite,
                   exact=exact, repeat_exact=deterministic, **(extra or {}))
        del expected, actual, delta, repeated
        bt, ct = [], []
        for repeat in range(args.repeats):
            order = ((baseline, bt), (candidate, ct)) if repeat % 2 == 0 else ((candidate, ct), (baseline, bt))
            for operation, times in order:
                elapsed, result = measured(operation)
                times.append(elapsed); del result
        row.update(baseline_ms=bt, candidate_ms=ct,
                   baseline_median_ms=statistics.median(bt), candidate_median_ms=statistics.median(ct),
                   speedup=statistics.median(bt) / statistics.median(ct))
        report['results'].append(row); save()
        print(json.dumps(row), flush=True)
        return row

    def flash(q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                  v.transpose(1, 2)).transpose(1, 2)

    try:
        save()
        if 'vae' in sections:
            report['native_build'] = build_h3_split(args.h3_source, args.out)
            save()
        with ExitStack() as lease:
            lease.enter_context(runtime_lock(shared=True))
            lease.enter_context(runtime_lock(device_lock_path(args.device_uuid), inherit=False))
            if 'attention' in sections:
                for tpf in (252, 1008):
                    layout = SimpleNamespace(seq_len=72*tpf+859, video_start=859,
                                             video_end=72*tpf+859, num_frames=72, tokens_per_frame=tpf)
                    plan = _plan(layout, window_bounds(72, 1, 5), 'both', 'cpu')
                    groups = Counter(zip(torch.diff(plan.cu_q).tolist(), torch.diff(plan.cu_k).tolist()))
                    report['window_shapes'].append(dict(tokens_per_frame=tpf, global_queries=len(plan.dense_q),
                        keys=layout.seq_len, window_groups=[dict(nq=q,nk=k,count=n) for (q,k),n in groups.items()]))
                save()
                configs = [(64, 64, 4, 1), (128, 64, 8, 1),
                           (32, 64, 4, 1), (64, 32, 4, 1), (64, 128, 4, 1),
                           (128, 32, 4, 1), (128, 64, 4, 1), (128, 128, 8, 1),
                           (64, 64, 4, 2), (128, 64, 8, 2), (128, 128, 8, 2)]
                for name,b,nq,nk in [('base_window',4,1260,5143), ('refine_window',4,5040,17995),
                                      ('refine_global',1,2875,73435)]:
                    torch.manual_seed(202610071)
                    q = torch.randn(b,nq,16,128,device='cuda',dtype=torch.bfloat16)
                    k,v = [torch.randn(b,nk,16,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
                    baseline = (lambda: flash(q,k,v)) if 'global' in name else (lambda: attention(q,k,v,128**-.5))
                    for bm,bn,warps,stages in configs:
                        if 'global' not in name and (bm,bn,warps,stages)==((128,64,8,1) if nq>=2048 else (64,64,4,1)):
                            continue
                        def candidate():
                            output = torch.empty_like(q)
                            _flash[(triton.cdiv(nq,bm),b*16)](q,k,v,output,nq,nk,16,128,
                                *q.stride(),*k.stride(),*v.stride(),128**-.5,
                                BM=bm,BN=bn,num_warps=warps,num_stages=stages,matrix_instr_nonkdim=16)
                            return output
                        try:
                            compare(name,baseline,candidate,'BF16 QKV, FP32 online softmax, all keys',
                                dict(shape=[b,nq,nk,16,128],config=[bm,bn,warps,stages]))
                        except triton.OutOfResources as error:
                            report['results'].append(dict(case=name,config=[bm,bn,warps,stages],rejected=str(error)));save()
                    del q,k,v
                    torch.cuda.empty_cache()
            if 'fp8' in sections:
                unit = torch.ones((1,1),device='cuda',dtype=torch.float32)
                cases = [('ff_up',5376,28672,2048), ('ff_down',14336,5376,2048)]
                if args.include_projection_controls:
                    cases.extend([('qkv',5376,2048,1024), ('out',7168,5376,1024)])
                for name,kdim,ndim,baseline_rows in cases:
                    torch.manual_seed(202610072)
                    x = torch.randn(8192,kdim,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                    weight = torch.randn(ndim,kdim,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                    def operation(rows):
                        output = []
                        for start in range(0,8192,rows):
                            stop=min(start+rows,8192)
                            output.append(torch._scaled_mm(x[start:stop],weight.t(),scale_a=unit,
                                scale_b=unit,out_dtype=torch.bfloat16,use_fast_accum=True))
                        return output
                    for chunk in (512,1024,2048,4096,8192):
                        if chunk==baseline_rows:continue
                        compare('fp8_'+name,lambda:operation(baseline_rows),lambda:operation(chunk),
                            'Existing E4M3FN operands and unit per-tensor scales, BF16 output; quantization and output assembly excluded',
                            dict(shape=[8192,kdim,ndim],baseline_chunk=baseline_rows,candidate_chunk=chunk,
                                 production_chunk_control=name in ('ff_up','ff_down')))
                    del x,weight
                    torch.cuda.empty_cache()
            if 'vae' in sections:
                torch.manual_seed(202610073)
                q,k,v = [torch.randn(1,1797,32,64,device='cuda',dtype=torch.float16) for _ in range(3)]
                library=ctypes.CDLL(str(args.out/'h3_split.so'))
                library.h3_split.argtypes=[ctypes.c_void_p]*6+[ctypes.c_uint,ctypes.c_uint,ctypes.c_float,ctypes.c_void_p]
                library.h3_split.restype=ctypes.c_int
                scores=torch.empty(32,1797,1797,device='cuda',dtype=torch.float32)
                inverse=torch.empty(32,1797,device='cuda',dtype=torch.float32)
                def native():
                    qf,kf,vf=[t.float() for t in (q,k,v)]
                    output=torch.empty_like(qf)
                    code=library.h3_split(qf.data_ptr(),kf.data_ptr(),vf.data_ptr(),output.data_ptr(),
                        scores.data_ptr(),inverse.data_ptr(),1797,32,.125,torch.cuda.current_stream().cuda_stream)
                    if code:raise RuntimeError('Native HIP launch error '+str(code))
                    return output.half()
                compare('vae_h3_f32_split',lambda:flash(q,k,v),native,
                    'H3 F32 math and workspace with input/output casts versus current FP16 Flash; different numerical contract',
                    dict(shape=[1,1797,1797,32,64],workspace_bytes=scores.numel()*4+inverse.numel()*4))
                del q,k,v,scores,inverse
                for name,kdim,ndim in [('projection',2048,2048),('ff_up',2048,16384),('ff_down',8192,2048)]:
                    x=torch.randn(1797,kdim,device='cuda',dtype=torch.float16)
                    weight=torch.randn(ndim,kdim,device='cuda',dtype=torch.float16)
                    bias=torch.randn(ndim,device='cuda',dtype=torch.float16)
                    initial=torch.backends.cuda.preferred_blas_library()
                    def linear(backend):
                        torch.backends.cuda.preferred_blas_library(backend)
                        try:return F.linear(x,weight,bias)
                        finally:torch.backends.cuda.preferred_blas_library(initial)
                    compare('vae_'+name,lambda:linear('cublaslt'),lambda:linear('cublas'),
                        'FP16 input/weights/bias and FP16 output, FP32 accumulation',dict(shape=[1797,kdim,ndim]))
                    del x,weight,bias
            report['state']='complete';save()
    except BaseException as error:
        report.update(state='failed',error=repr(error));save();raise
    return 0


if __name__=='__main__':raise SystemExit(main())
