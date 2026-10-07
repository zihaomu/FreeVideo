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
DEFAULT_WORKLOAD=json.loads(Path(__file__).with_name('workload.json').read_text())


def read(path):
    return json.loads(path.read_text())


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--width',type=int,default=DEFAULT_WORKLOAD['width'])
    parser.add_argument('--height',type=int,default=DEFAULT_WORKLOAD['height'])
    length=parser.add_mutually_exclusive_group()
    length.add_argument('--frames',type=int)
    length.add_argument('--seconds',type=float,help='Requested duration; default: official 10 seconds')
    parser.add_argument('--steps','--base-steps',dest='steps',type=int,choices=range(1,33),default=DEFAULT_WORKLOAD['base_steps'])
    parser.add_argument('--refine-steps',type=int,choices=range(1,32),default=DEFAULT_WORKLOAD['refine_steps'])
    parser.add_argument('--seed',type=int,default=2026090901)
    parser.add_argument('--two-pass',action=argparse.BooleanOptionalAction,default=DEFAULT_WORKLOAD['two_pass'])
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--conditioning',type=Path)
    parser.add_argument('--prompt',type=Path,default=ROOT/'scripts/amd/prompt.txt')
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--wait-for-models',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Show resolved geometry and sampling plan without running or writing state')
    args=parser.parse_args(argv)
    if args.frames is None and args.seconds is None:
        args.seconds=DEFAULT_WORKLOAD['seconds']
    return args


def resolve_workload(args):
    sys.path.insert(0,str(ROOT))
    from freevideo_engine.geometry import geometry
    from freevideo_engine.two_pass import plan
    canvas=geometry(args.width,args.height,frames=args.frames,seconds=args.seconds)
    return canvas,plan(canvas,args.two_pass,base_steps=args.steps,refine_steps=args.refine_steps)


def main():
    args=parse_args()
    canvas,sampling_plan=resolve_workload(args)
    if args.dry_run:
        print(json.dumps(dict(default_workload=DEFAULT_WORKLOAD,geometry=canvas,sampling_plan=sampling_plan),indent=2))
        return 0
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
                default_workload=DEFAULT_WORKLOAD,geometry=canvas,sampling_plan=sampling_plan,
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
                 '--width',str(args.width),'--height',str(args.height),
                 *(['--frames',str(args.frames)] if args.frames is not None else ['--seconds',str(args.seconds)]),
                 '--base-steps',str(args.steps),'--two-pass' if args.two_pass else '--no-two-pass',
                 '--refine-steps',str(args.refine_steps),
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
        request=None
        if video.with_suffix('.request.json').is_file():
            request=read(video.with_suffix('.request.json'))
            run['request_seconds']=request.get('request_seconds')
        if result.returncode:
            run['status']='failed';report['state']='failed';save()
            print(json.dumps(run),flush=True)
            return 1
        if request is None or not isinstance(run['request_seconds'],(int,float)):
            run['status']='failed-missing-request-receipt';report['state']='failed';save();return 1
        actual_plan=request.get('sampling_plan')
        actual_geometry=request.get('geometry') or {}
        if actual_plan!=sampling_plan or any(actual_geometry.get(key)!=canvas[key] for key in ('width','height','frames','fps')):
            run.update(status='failed-workload-verification',actual_sampling_plan=actual_plan)
            report['state']='failed';save();return 1
        # ffprobe is the host tool; AV performs an independent complete decode.
        with video.with_suffix('.ffprobe.json').open('w') as stream:
            probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],stdout=stream)
        with (directory/'verify.log').open('w') as stream:
            verify=subprocess.run(['bash',str(ROOT/'scripts/amd/run_rocm.sh'),'scripts/amd/verify_video.py',inside(video),
                '--width',str(args.width),'--height',str(args.height),'--frames',str(canvas['frames'])],
                env=env,stdout=stream,stderr=subprocess.STDOUT)
        run['verification_returncode']=verify.returncode
        run['status']='passed' if probe.returncode==verify.returncode==0 else 'failed-verification'
        if run['status']!='passed':
            report['state']='failed';save();return 1
        run.update(rtf=run['request_seconds']/canvas['seconds'],generated_fps=canvas['frames']/run['request_seconds'])
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
