#!/usr/bin/env python3
"""Run complete independent requests with retained telemetry and decoded media checks."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
STORAGE=Path(os.environ.get('FV_STORAGE_ROOT','/dc1/zihaomu/free_token_mapping'))
EXPERIMENT=STORAGE/'experiments/freevideo-r9700'


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--width',type=int,required=True)
    parser.add_argument('--height',type=int,required=True)
    parser.add_argument('--frames',type=int,required=True)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--seed',type=int,default=2026090901)
    parser.add_argument('--two-pass',action='store_true')
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--conditioning',type=Path)
    parser.add_argument('--prompt',type=Path,default=ROOT/'scripts/amd/prompt.txt')
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--wait-for-models',action='store_true')
    args=parser.parse_args()
    assert re.fullmatch('[a-zA-Z0-9_-]+',args.name)
    assert args.repeats>=1
    report_path=EXPERIMENT/'reports'/f'{args.name}-runs.json'
    if report_path.exists():
        raise FileExistsError('Use a new name to retain previous successful and failed experiments: '+str(report_path))
    # The application also keeps a crash-triggered compatibility setting. Turn
    # that adaptation off explicitly for matched experiments, preserving incident
    # records. --no-tuning alone does not disable this separate setting.
    sys.path.insert(0,str(ROOT))
    from freevideo_engine.compatibility import Store
    hardware=read(args.profile)['hardware']
    compatibility=Store(EXPERIMENT/'runtime').set(
        dict(gpu_uuid=hardware['gpu_uuid'],system=hardware['system']),0,False)
    report=dict(case=vars(args).copy(),state='waiting' if args.wait_for_models else 'running',
                cache_scope='Fresh workers; separate JIT cache for this case. First request JIT cold, then cache-warm independent requests. OS file cache not cleared.',
                timing_scope='FreeVideo request_seconds includes encoding/loading/sampling/decode/mux; launch_seconds also includes container startup.',
                pid=os.getpid(),runs=[])
    report['fixed_compatibility']=compatibility
    report['blas_environment']=dict(
        TORCH_BLAS_PREFER_HIPBLASLT=os.environ.get('FV_TORCH_BLAS_PREFER_HIPBLASLT','0'),
        ROCBLAS_USE_HIPBLASLT=os.environ.get('FV_ROCBLAS_USE_HIPBLASLT','0'))
    report['compile_environment']=dict(TORCH_COMPILE_DISABLE=os.environ.get('FV_TORCH_COMPILE_DISABLE','0'))
    report['video_blas_environment']=dict(FREEVIDEO_ROCM_VIDEO_BLAS=os.environ.get('FV_ROCM_VIDEO_BLAS','default'))
    report['spatial_conv_environment']=dict(FREEVIDEO_ROCM_SPATIAL_CONV=os.environ.get('FV_ROCM_SPATIAL_CONV','miopen'))
    report['attention_environment']=dict(FREEVIDEO_ROCM_ATTENTION=os.environ.get('FV_ROCM_ATTENTION','aotriton'))
    report['audio_conv_environment']=dict(FREEVIDEO_ROCM_AUDIO_CONV=os.environ.get('FV_ROCM_AUDIO_CONV','miopen'))
    report['bf16_environment']={f'FREEVIDEO_ROCM_{kind}_BLAS': os.environ.get(f'FV_ROCM_{kind}_BLAS','default')
                                for kind in ('GATE','STATE')}
    report['case']={k:str(v) if isinstance(v,Path) else v for k,v in report['case'].items()}
    def save():
        staging=report_path.with_suffix('.tmp')
        staging.write_text(json.dumps(report,indent=2)+'\n')
        staging.replace(report_path)
    save()
    if args.wait_for_models:
        dependencies=['download-components.json','download-prepared.json']
        while True:
            states={n:read(EXPERIMENT/'reports'/n)['state'] for n in dependencies}
            if any(s=='failed' for s in states.values()):
                report.update(state='failed',error='Model download failed',dependencies=states)
                save()
                return 1
            if all(s=='complete' for s in states.values()):
                break
            time.sleep(5)
    report['state']='running'
    save()
    def inside(path):
        path=path.resolve()
        if ROOT==path or ROOT in path.parents:
            return str(Path('/workspace')/path.relative_to(ROOT))
        return str(Path('/data')/path.relative_to(STORAGE))
    env=dict(os.environ,FV_CACHE_GROUP=args.name)
    for repeat in range(args.repeats):
        directory=EXPERIMENT/'outputs'/args.name/f'run-{repeat:02d}'
        directory.mkdir(parents=True,exist_ok=False)
        video=directory/'video.mp4'
        command=['bash',str(ROOT/'scripts/amd/run_rocm.sh'),'-m','freevideo_engine','generate',
                 '--cache','/data/models/prepared/edge-5e6ecc39f472f98c/cache',
                 '--base','/data/models/vdn/h3-base','--attention','torch-flash',
                 '--width',str(args.width),'--height',str(args.height),'--frames',str(args.frames),
                 '--base-steps',str(args.steps),'--two-pass' if args.two_pass else '--no-two-pass',
                 '--seed',str(args.seed),'--no-history-placement','--no-tuning','--resource-retries','0',
                 '--profile',inside(args.profile),'--model-paths','/data/experiments/freevideo-r9700/runtime/encoder-paths.yaml',
                 '--out',inside(video)]
        command+=['--conditioning',inside(args.conditioning)] if args.conditioning else ['--prompt-file',inside(args.prompt)]
        run=dict(index=repeat,cache='JIT cold' if repeat==0 else 'cache-warm independent request',command=command,
                 output=str(video),status='running')
        report['runs'].append(run);save()
        print(json.dumps(dict(event='request_start',case=args.name,repeat=repeat,output=str(video))),flush=True)
        started=time.monotonic()
        with (directory/'launch.log').open('w') as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        run.update(returncode=result.returncode,launch_seconds=time.monotonic()-started)
        if video.with_suffix('.request.json').is_file():
            request=read(video.with_suffix('.request.json'))
            run['request_seconds']=request.get('request_seconds')
        if result.returncode:
            run['status']='failed';report['state']='failed';save()
            print(json.dumps(run),flush=True)
            return 1
        # ffprobe is the host tool; AV performs an independent complete decode.
        with video.with_suffix('.ffprobe.json').open('w') as stream:
            probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],stdout=stream)
        with (directory/'verify.log').open('w') as stream:
            verify=subprocess.run(['bash',str(ROOT/'scripts/amd/run_rocm.sh'),'scripts/amd/verify_video.py',inside(video),
                '--width',str(args.width),'--height',str(args.height),'--frames',str(args.frames)],
                env=env,stdout=stream,stderr=subprocess.STDOUT)
        run['verification_returncode']=verify.returncode
        run['status']='passed' if probe.returncode==verify.returncode==0 else 'failed-verification'
        if run['status']!='passed':
            report['state']='failed';save();return 1
        run.update(rtf=run['request_seconds']/(args.frames/24),generated_fps=args.frames/run['request_seconds'])
        save()
        print(json.dumps(dict(event='request_complete',case=args.name,**run)),flush=True)
    report['state']='complete'
    warm=[r['request_seconds'] for r in report['runs'][1:]]
    if warm:
        report['warm_request_seconds']=dict(median=statistics.median(warm),minimum=min(warm),maximum=max(warm),count=len(warm))
    save()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
