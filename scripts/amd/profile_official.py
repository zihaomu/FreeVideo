#!/usr/bin/env python3
"""Trace a retained official request, with process-local labels and unchanged math."""
import argparse
from contextlib import ExitStack, contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('encode','video'),required=True)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True,help='New directory; never overwrite a retained request')
    parser.add_argument('--conditioning',type=Path,help='Optional fresh encoded conditioning for the video trace')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('PYTORCH_ALLOC_CONF','backend:native')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    import torch
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    from freevideo_engine.hardware import _detect_local
    from freevideo_engine.locking import runtime_lock,device_lock_path
    from freevideo_engine.paths import add_vdn
    add_vdn()
    hardware=_detect_local()
    request=json.loads(args.request.read_text())
    report=dict(state='running',kind=args.kind,source_request=str(args.request),
        source_request_sha256=hashlib.sha256(args.request.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=str(torch.__version__),hip=torch.version.hip,gpu=hardware.gpu_name,
        gpu_uuid=hardware.gpu_uuid,arch=hardware.gcn_arch,
        scope='Same retained prompt/request, all sampling steps and both decoder forwards. VAE loading/postprocessing/mux are not fully traced. Profiled timings include instrumentation. Disk JIT cache is reused; original normal latency is retained separately.',
        profiler_options=dict(record_shapes=False,profile_memory=False,with_stack=False),
        environment={k:os.environ.get(k) for k in ('TORCH_BLAS_PREFER_HIPBLASLT','ROCBLAS_USE_HIPBLASLT',
            'FREEVIDEO_ROCM_SPATIAL_CONV','FREEVIDEO_ROCM_ATTENTION','FREEVIDEO_ROCM_VIDEO_BLAS',
            'FREEVIDEO_ROCM_AUDIO_CONV','FREEVIDEO_ROCM_GATE_BLAS','FREEVIDEO_ROCM_STATE_BLAS')},phases=[])
    # Shape recording across complete timesteps retains large activation tensors.
    # Labels and dispatch/kernel correlation suffice for exact category accounting.
    retained=[]
    active=[False]
    def save():
        (args.out/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    @contextmanager
    def capture(name):
        if active[0]:
            raise RuntimeError('Nested GPU profilers are unsupported: '+name)
        row=dict(name=name,state='running',trace=name+'.json')
        report['phases'].append(row);save()
        print(json.dumps(dict(event='profile_phase_start',name=name)),flush=True)
        started=time.perf_counter()
        prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)
        with prof:
            active[0]=True
            try:
                with torch.profiler.record_function('fv::phase::'+name):
                    yield
            finally:
                torch.cuda.synchronize()
                active[0]=False
        row.update(state='captured',profile_wall_seconds=time.perf_counter()-started)
        retained.append((row,prof));save()
        print(json.dumps(dict(event='profile_phase_captured',**row)),flush=True)
    def label(original,name):
        @wraps(original)
        def wrapped(*a,**kw):
            if not active[0]:
                return original(*a,**kw)
            with torch.profiler.record_function('fv::'+name):
                return original(*a,**kw)
        return wrapped
    def patch(stack,obj,key,value):
        original=getattr(obj,key)
        setattr(obj,key,value)
        stack.callback(setattr,obj,key,original)
    save()
    try:
        with ExitStack() as stack:
            stack.enter_context(runtime_lock(shared=True))
            stack.enter_context(runtime_lock(device_lock_path(hardware.gpu_uuid),inherit=False))
            if args.kind=='encode':
                from freevideo_engine import encode_worker
                request.update(output=str(args.out/'conditioning.pt'),metrics=str(args.out/'encoding.json'),
                    diagnostic_output=str(args.out/'video.mp4'))
                encoder_args=argparse.Namespace(check_library=False,comfy_root=None)
                with capture('text_encoder'):
                    encode_worker.encode(encoder_args,request)
            else:
                from freevideo_engine import worker,latent_upscale
                from freevideo_engine.runtime import Engine
                from freevideo_engine.attention import WindowAttention
                from src.models.linear_attention import branch
                from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
                original_init=Engine.__init__
                def init(engine,*a,**kw):
                    with capture('transformer_load'):
                        original_init(engine,*a,**kw)
                    for i,block in enumerate(engine.transformer.transformer_blocks):
                        for obj,name in ((block,'block'),(block.attn,'attention'),(block.ff,'feed_forward')):
                            patch(stack,obj,'forward',label(obj.forward,f'{name}::block_{i:02d}'))
                    for name,module in engine.transformer.named_modules():
                        if type(module).__name__=='BidirectionalLinearBranch':
                            for method in ('_features','_text_state','_readout_inference'):
                                patch(stack,module,method,label(getattr(module,method),'linear_state::'+method))
                patch(stack,Engine,'__init__',init)
                original_sample=Engine.sample
                def sample(engine,*a,**kw):
                    name='refine_3_steps' if kw.get('initial_latents') is not None else 'base_8_steps'
                    with capture(name):
                        return original_sample(engine,*a,**kw)
                patch(stack,Engine,'sample',sample)
                original_batched=WindowAttention.batched
                def batched(policy,*a,**kw):
                    name='window_attention' if kw.get('window',False) else 'global_attention'
                    return label(original_batched,'softmax::'+name)(policy,*a,**kw)
                patch(stack,WindowAttention,'batched',batched)
                for name in ('frame_statistics','_run_scans_inference','gather_linear_state','linear_epilogue'):
                    patch(stack,branch,name,label(getattr(branch,name),'linear_state::'+name))
                for obj,key,name in ((latent_upscale,'upscale','latent_upscale'),
                    (AutoencoderKLMiniMaxH3,'decode','video_vae'),
                    (AutoencoderKLMiniMaxH3Audio,'decode','audio_vae')):
                    original=getattr(obj,key)
                    def phase_call(*a,_original=original,_name=name,**kw):
                        with capture(_name):
                            return _original(*a,**kw)
                    patch(stack,obj,key,phase_call)
                request.update(output=str(args.out/'video.mp4'),metrics=str(args.out/'engine.json'),
                    artifacts=str(args.out/'video.artifacts'),input_cache_dir=None,
                    resource_attempt='operator-profile')
                if args.conditioning:
                    request['conditioning']=str(args.conditioning)
                (args.out/'video.artifacts').mkdir()
                worker.generate(request)
                shutil.copyfile(request['conditioning'],args.out/'video.artifacts/conditioning.pt')
            (args.out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
        report['state']='exporting';save()
        # Export only after GPU work finishes; serialization does not delay a pass.
        for row,prof in retained:
            prof.export_chrome_trace(str(args.out/row['trace']))
            row['trace_bytes']=(args.out/row['trace']).stat().st_size
            row['state']='complete'
            # The streaming analyzer performs exclusive GPU accounting. Avoid
            # building a second Python event tree for these multi-GB traces.
            save()
            print(json.dumps(dict(event='profile_trace_exported',name=row['name'],bytes=row['trace_bytes'])),flush=True)
        report['state']='complete';save()
    except BaseException as error:
        report.update(state='failed',error=repr(error));save()
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
