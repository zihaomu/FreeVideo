#!/usr/bin/env python3
"""Compare identical retained H3 audio latents in CPU and HIP FP32 decoders."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latents',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import torch
    import numpy as np
    from freevideo_engine.paths import add_vdn
    add_vdn()
    from freevideo_engine.vae_weights import load_audio_vae
    torch.set_num_threads(8);torch.set_grad_enabled(False)
    latent=torch.load(args.latents,map_location='cpu',weights_only=True)['audio']
    report=dict(torch=str(torch.__version__),hip=torch.version.hip,source_latents=str(args.latents),runs=[])
    cpu,_=load_audio_vae('/data/models/vdn/h3-base',device='cpu')
    mean=torch.tensor(cpu.config.latents_mean).view(1,-1,1)
    std=torch.tensor(cpu.config.latents_std).view(1,-1,1)
    value=latent*std+mean
    reference=cpu.decode(value,return_dict=False)[0]
    report['cpu_reference']=dict(rms=reference.square().mean().sqrt().item(),peak=reference.abs().max().item())
    gpu,_=load_audio_vae('/data/models/vdn/h3-base',device='cuda')
    value=value.cuda()
    for miopen in (True,False):
        start=time.perf_counter()
        with torch.backends.cudnn.flags(enabled=miopen):
            result=gpu.decode(value,return_dict=False)[0]
        torch.cuda.synchronize();result=result.cpu()
        row=dict(miopen=miopen,seconds=time.perf_counter()-start,finite=bool(result.isfinite().all()),
                 rms=result.square().mean().sqrt().item(),peak=result.abs().max().item(),
                 saturation_fraction=(result.abs()>=.999).float().mean().item(),
                 relative_rmse=((result-reference).square().mean()/reference.square().mean()).sqrt().item())
        report['runs'].append(row)
        np.save(args.out.with_name(args.out.stem+f'-miopen-{miopen}.npy'),result.numpy())
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
