#!/usr/bin/env python3
"""Measure GPU-only decoder BLAS candidates on identical retained latents."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latents',type=Path,required=True)
    parser.add_argument('--blas',choices=('cublas','cublaslt'),required=True,
        help='PyTorch API names for rocBLAS or hipBLASLt on HIP')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from freevideo_engine.paths import add_vdn
    from freevideo_engine.locking import runtime_lock
    add_vdn()
    from freevideo_engine.decode import decode_to_file
    from freevideo_engine.hardware import _detect_local
    from freevideo_engine.monitoring import Monitor
    torch.set_num_threads(8);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.version.hip and torch.cuda.device_count()==1
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx1201'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    latents=torch.load(args.latents,map_location='cpu',weights_only=True)
    artifacts=args.out.with_suffix('.artifacts')
    artifacts.mkdir(parents=True,exist_ok=False)
    for name,source in [('latents.pt',args.latents),('conditioning.pt',args.latents.parent/'conditioning.pt')]:
        (artifacts/name).symlink_to(os.path.relpath(source,artifacts))
    with runtime_lock():
        hardware=_detect_local()
        monitor=Monitor(args.out.with_suffix('.gpu.csv'),device=hardware.gpu_uuid).start()
        torch.backends.cuda.preferred_blas_library(args.blas)
        started=time.perf_counter()
        video,audio=latents['video'].cuda(),latents['audio'].cuda()
        torch.cuda.reset_peak_memory_stats()
        metrics=decode_to_file(video,audio,str(args.out),base='/data/models/vdn/h3-base',
            offload=False,prefetch=False,artifacts_dir=artifacts)
        torch.cuda.synchronize()
        metrics.update(probe_seconds=time.perf_counter()-started,
            scope='Decoder phase only, identical retained sampled latents; excludes text encoder and sampling',
            source_latents=str(args.latents),blas=args.blas,
            hip=torch.version.hip,torch=str(torch.__version__),
            hardware=hardware.to_dict(),whole_gpu=monitor.stop(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        args.out.with_suffix('.decode-probe.json').write_text(json.dumps(metrics,indent=2)+'\n')
        print(json.dumps({k:v for k,v in metrics.items() if k.endswith('seconds') or k in ('blas','audio_decoder')}),flush=True)


if __name__=='__main__':
    main()
