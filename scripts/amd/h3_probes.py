#!/usr/bin/env python3
"""Real H3 projection widths and attention geometry; no full-video claim."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--skip-linears', action='store_true')
    parser.add_argument('--real-weights', action='store_true', help='Use the downloaded first H3 block, rather than synthetic weights')
    args = parser.parse_args()
    import torch
    from freevideo_engine.paths import add_vdn
    add_vdn()
    from src.models.ops.fp8_linear import quantize_tensor
    from freevideo_engine.weight_only import linear
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.manual_seed(2026090901)
    assert torch.version.hip and torch.cuda.device_count() == 1
    device = torch.cuda.get_device_properties(0)
    assert device.gcnArchName.split(':')[0] == 'gfx1201'
    from src.models.ops import fp8_linear
    # Match the engine's verified RDNA4 policy rather than the HIP SM alias.
    fp8_linear._PER_TENSOR = True
    from freevideo_engine.hardware import _detect_local
    identity=_detect_local()
    report = dict(scope='H3 real projection widths; synthetic weights, S0 attention geometry',
                  image_id=os.environ.get('FV_PREFLIGHT_IMAGE_ID'), torch=str(torch.__version__),
                  hip=torch.version.hip, gpu=device.name, arch=device.gcnArchName,
                  gpu_uuid=identity.gpu_uuid, rocr_visible_devices=os.environ.get('ROCR_VISIBLE_DEVICES'),
                  scale_policy='per_tensor selected explicitly for gfx1201',
                  visible_gpus=1, probes=[])
    real={}
    if args.real_weights:
        from safetensors.torch import load_file
        block=Path('/data/models/prepared/edge-5e6ecc39f472f98c/cache/blocks/00.safetensors')
        tensors=load_file(str(block))
        for name,value in tensors.items():
            if name.endswith('.weight_fp8'):
                real.setdefault(tuple(value.shape),(name,value,tensors[name.removesuffix('.weight_fp8')+'.weight_scale']))
        report.update(scope='Downloaded H3 first-block weights, synthetic activations, S0 attention geometry',
                      weight_file=str(block))

    def check(name, operation):
        start = time.perf_counter()
        try:
            row = dict(name=name, status='passed', **operation())
        except Exception as error:
            row = dict(name=name, status='failed', error=str(error), traceback=traceback.format_exc())
        row['wall_seconds_including_compile'] = time.perf_counter()-start
        report['probes'].append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)

    def error(value, reference, limit):
        finite = bool(value.isfinite().all())
        relative = ((value.float()-reference.float()).square().mean() /
                    reference.float().square().mean().clamp_min(1e-20)).sqrt().item()
        assert finite and (limit is None or relative <= limit), (finite, relative, limit)
        return dict(finite=finite, relative_rmse=relative, limit=limit)

    def timed(operation):
        operation()
        torch.cuda.synchronize()
        values=[]
        for _ in range(3):
            torch.cuda.synchronize()
            start=time.perf_counter()
            operation()
            torch.cuda.synchronize()
            values.append(time.perf_counter()-start)
        return dict(seconds=values, median_seconds=statistics.median(values))

    def projection(n, k, m):
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        if args.real_weights and (n,k) in real:
            weight_name,stored,stored_scale=real[(n,k)]
            packed=stored.cuda()
            scale=stored_scale.cuda()
        else:
            assert not args.real_weights, ('Requested shape absent from real block',n,k)
            weight_name='synthetic'
            weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
            scale=(weight.float().abs().amax()/448).reshape(1,1)
            packed=(weight.float()/scale).to(torch.float8_e4m3fn)
        unpacked=(packed.float()*scale).to(torch.bfloat16)
        xq,xs=quantize_tensor(x)
        # Preserve original per-tensor activation quantization, including the tail.
        def fp8():
            q,s=quantize_tensor(x)
            return torch._scaled_mm(q,packed.t(),scale_a=s,scale_b=scale,
                                    out_dtype=torch.bfloat16,use_fast_accum=True)
        native_value=fp8()
        native_ref=(xq.float()*xs) @ (packed.float()*scale).t()
        # Kernel arithmetic and activation quantization are separate questions.
        # Both references use the same stored FP8 weights; this does not measure
        # weight quantization against an unavailable original BF16 checkpoint.
        unquantized_activation_ref=x.float() @ (packed.float()*scale).t()
        stable_value=linear(x,packed,scale)
        stable_ref=x.float() @ unpacked.float().t()
        return dict(activation=[m,k], weight=[n,k], weight_name=weight_name, scale='per_tensor',
                    fp8=error(native_value,native_ref,.01),
                    fp8_vs_unquantized_activations=error(native_value,unquantized_activation_ref,None),
                    activation_quantization=dict(dtype=str(xq.dtype),scale_shape=list(xs.shape),
                        scale=float(xs.item()),reference='FP32 GEMM with original BF16 activations and the same dequantized stored FP8 weights',
                        scope='Observed activation quantization difference; no semantic quality threshold inferred'),
                    w8a16=error(stable_value,stable_ref,.01),
                    bf16=timed(lambda:x @ unpacked.t()),
                    native_fp8_with_quantize=timed(fp8),
                    w8a16_time=timed(lambda:linear(x,packed,scale)))

    if not args.skip_linears:
        manifest=json.loads(Path('/data/models/prepared/edge-5e6ecc39f472f98c/cache/manifest.json').read_text())
        shapes=sorted(set(tuple(row['weight_shape']) for row in manifest['linears'].values()))
        if args.real_weights:
            shapes=sorted(real)
        for n,k in shapes:
            for m in (701,2048):
                check(f'projection_{m}_{n}_{k}',lambda n=n,k=k,m=m:projection(n,k,m))

    def window():
        from freevideo_engine.attention import WindowAttention
        from src.models.sequence_layout import SequenceLayout
        from src.models.softmax_attention.window import window_bounds, window_softmax_reference
        # S0 latent: 12 frames, 9x16 tokens/frame, H3's 56 heads of width 128.
        frames,tokens,heads,dim,text=12,144,56,128,17
        layout=SequenceLayout(seq_len=frames*tokens+text,video_start=text,
                              num_frames=frames,tokens_per_frame=tokens)
        q,k,v=[torch.randn(layout.seq_len,heads,dim,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        bounds=window_bounds(frames,radius=1,chunk=5)
        attention=WindowAttention('torch-flash',window_batch=1)
        value=attention(q,k,v,layout,bounds,dim**-.5,'both')
        reference=window_softmax_reference(q,k,v,layout,bounds,dim**-.5,'both')
        return dict(qkv=list(q.shape),**error(value,reference,.006),calls=attention.backend_calls)
    check('s0_hybrid_window',window)

    def branch():
        from src.models.linear_attention import BidirectionalLinearBranch
        from src.models.softmax_attention.window import window_bounds
        frames,tokens,heads,dim,hidden=12,144,56,128,5376
        model=BidirectionalLinearBranch(hidden,heads,dim,delta_rule='vdn_solve',short_conv=('k','v')).cuda().to(torch.bfloat16)
        # H3 alpha is an explicit FP32 island.
        model.alpha.float()
        # Safetensors loads contiguous checkpoint parameters; constructor init
        # may retain strided temporal-convolution weights.
        for parameter in model.parameters():
            parameter.data=parameter.data.contiguous()
        x=torch.randn(frames*tokens,hidden,device='cuda',dtype=torch.bfloat16)
        qkv=tuple(torch.randn(frames*tokens,heads,dim,device='cuda',dtype=torch.bfloat16) for _ in range(3))
        kwargs=dict(frame_size=(9,16),text_x=x[:17],text_qkv_raw=tuple(t[:17] for t in qkv))
        bounds=window_bounds(frames,radius=1,chunk=5)
        reference=model(x,frames,tokens,bounds,qkv,inference=False,**kwargs)
        value=model(x,frames,tokens,bounds,qkv,inference=True,**kwargs)
        return dict(shape=list(value.shape),delta_rule='vdn_solve',short_conv=['k','v'],**error(value,reference,.03))
    check('s0_vdn_linear_branch',branch)
    report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  passed=all(row['status']=='passed' for row in report['probes']))
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    return not report['passed']


if __name__=='__main__':
    raise SystemExit(main())
