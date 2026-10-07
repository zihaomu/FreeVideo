#!/usr/bin/env python3
"""Run required quality, precision and matched engineering experiments serially."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT=Path(os.environ.get('FV_STORAGE_ROOT','/dc1/zihaomu/free_token_mapping'))/'experiments/freevideo-r9700'


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def warm_samples(name,key):
    values=[]
    for row in read(EXPERIMENT/'reports'/f'{name}-runs.json')['runs'][1:]:
        if row['status']!='passed':
            continue
        video=Path(row['output'])
        source=read(video.with_suffix('.request.json') if key=='request_seconds' else video.with_suffix('.engine.json'))
        values.append(source[key])
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--matrix-prefix',required=True)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--decode-blas-patch',type=Path,
        help='Apply a hash-checked prepared patch only after the P3 matrix completes')
    args=parser.parse_args()
    output=EXPERIMENT/'reports'/f'{args.prefix}-followups.json'
    if output.exists():
        raise FileExistsError(output)
    state=dict(state='waiting-for-matrix',pid=os.getpid(),gate=str(args.gate),cases=[],
        scope='Three extra quality prompts; W8A16 precision comparison; one-lever native FP8 S1 ablations including video-only hipBLASLt; combined S2 candidate against the completed matrix reference.')
    write(output,state)
    while True:
        gate=read(args.gate)
        if gate['state']=='complete':
            break
        if gate['state']=='failed':
            state.update(state='failed',error='Matrix prerequisite failed');write(output,state);return 1
        time.sleep(5)
    if args.decode_blas_patch:
        state['state']='preparing-video-blas';write(output,state)
        try:
            manifest=read(args.decode_blas_patch.with_suffix('.manifest.json'))
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            if digest(args.decode_blas_patch)!=manifest['patch_sha256']:
                raise ValueError('Prepared patch checksum mismatch')
            if any(digest(ROOT/name)!=entry['before_sha256'] for name,entry in manifest['files'].items()):
                raise ValueError('Core source changed since the prepared patch; refusing to overwrite it')
            # Preserve the four core/runner files used throughout the baseline.
            before=EXPERIMENT/'prepared'/f'{args.prefix}-baseline-source'
            before.mkdir(exist_ok=False)
            for name in manifest['files']:
                target=before/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((ROOT/name).read_bytes())
            subprocess.run(['git','apply','--check',str(args.decode_blas_patch)],cwd=ROOT,check=True)
            subprocess.run(['git','apply',str(args.decode_blas_patch)],cwd=ROOT,check=True)
            if any(digest(ROOT/name)!=entry['after_sha256'] for name,entry in manifest['files'].items()):
                raise ValueError('Applied patch source checksum mismatch')
            state['deferred_patch']=dict(manifest=manifest,baseline_source=str(before),applied=True)
            write(output,state)
            with (EXPERIMENT/'reports'/f'{args.prefix}-doctor.json').open('w') as stream, (EXPERIMENT/'reports'/f'{args.prefix}-doctor.log').open('w') as errors:
                subprocess.run(['bash',str(ROOT/'scripts/amd/run_rocm.sh'),'-m','freevideo_engine','doctor','--probe'],
                    stdout=stream,stderr=errors,check=True,
                    env=dict(os.environ,FV_CACHE_GROUP=args.prefix+'-doctor',FV_ROCM_VIDEO_BLAS='default'))
            if not read(EXPERIMENT/'reports'/f'{args.prefix}-doctor.json').get('ready'):
                raise ValueError('Patched core kernel readiness probe failed')
        except Exception as error:
            state.update(state='failed',error=repr(error));write(output,state);return 1
    baseline=read(args.profile)
    reference=args.matrix_prefix+'-s1-single'
    baseline_sample=warm_samples(reference,'sample_seconds')
    baseline_request=warm_samples(reference,'request_seconds')
    baseline_video=warm_samples(reference,'video_decode_seconds')
    if len(baseline_sample)<3 or len(baseline_request)<3:
        raise ValueError('Matched reference requires three verified warm independent requests')
    state.update(state='running',reference_s1=reference,reference_s2=args.matrix_prefix+'-s2-single')
    write(output,state)

    def run(suffix,profile,*,width=768,height=448,frames=124,repeats=4,prompt=None,seed=2026090901,compile_disable='0',video_blas='default',kind='engineering'):
        name=args.prefix+'-'+suffix
        path=EXPERIMENT/'prepared/profiles'/f'{name}.json'
        if path.exists():
            raise FileExistsError(path)
        write(path,profile)
        command=['python3',str(ROOT/'scripts/amd/run_case.py'),'--name',name,'--width',str(width),
            '--height',str(height),'--frames',str(frames),'--steps','8','--seed',str(seed),
            '--profile',str(path),'--repeats',str(repeats)]
        if prompt:
            command+=['--prompt',str(prompt)]
        row=dict(name=name,state='running',kind=kind,profile=str(path),
            change={k:v for k,v in profile['engine'].items() if baseline['engine'].get(k)!=v},
            compile_disable=compile_disable,video_blas=video_blas)
        state['cases'].append(row);write(output,state)
        print(json.dumps(dict(event='case_start',**row)),flush=True)
        with (EXPERIMENT/'reports'/f'{name}-driver.log').open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,FV_TORCH_COMPILE_DISABLE=compile_disable,FV_ROCM_VIDEO_BLAS=video_blas))
        row.update(state='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
        write(output,state)
        return name,result.returncode

    for index,prompt in enumerate(('person-motion','object-motion','complex-scene'),start=2):
        name,code=run('quality-'+prompt,copy.deepcopy(baseline),repeats=1,
            prompt=ROOT/'scripts/amd/prompts'/f'{prompt}.txt',seed=2026090900+index,kind='quality')
        if code:
            state.update(state='failed',failed_case=name);write(output,state);return 1
    precision=copy.deepcopy(baseline);precision['engine']['linear_compute']='bf16-weight-only'
    name,code=run('w8a16',precision,kind='precision')
    if code:
        state.update(state='failed',failed_case=name);write(output,state);return 1

    name,code=run('video-hipblaslt',copy.deepcopy(baseline),video_blas='cublaslt')
    selected_video_blas='default'
    if not code:
        video=warm_samples(name,'video_decode_seconds');request=warm_samples(name,'request_seconds')
        improves=(len(video)>=3 and max(video)<min(baseline_video)
            and max(request)<min(baseline_request)
            and statistics.median(request)<statistics.median(baseline_request)*.99)
        state['cases'][-1].update(candidate_for_combination=improves,
            video_decode_median=statistics.median(video),request_median=statistics.median(request))
        if improves:
            selected_video_blas='cublaslt'
        write(output,state)

    candidates=[('resident0','resident_blocks',0),('resident32','resident_blocks',32),
        ('one-slot','prefetch',False),('head8','head_chunk',8),('head32','head_chunk',32),
        ('projection2048','projection_chunk',2048),('ff4096','ff_chunk',4096),('pin24','pin_host_gb',24.0)]
    winners={}
    for suffix,key,value in candidates:
        profile=copy.deepcopy(baseline);profile['engine'][key]=value
        name,code=run(suffix,profile)
        if code:
            continue  # Retain capacity/kernel failures; do not retry or hide them.
        sample=warm_samples(name,'sample_seconds');request=warm_samples(name,'request_seconds')
        # The measured sample range must separate from the reference and the
        # full-request median must improve. No claim based on tiny timing noise.
        improves=(len(sample)>=3 and max(sample)<min(baseline_sample)
            and statistics.median(request)<statistics.median(baseline_request)*.99)
        row=state['cases'][-1]
        row.update(sample_median=statistics.median(sample),request_median=statistics.median(request),
            candidate_for_combination=improves)
        prior=winners.get(key)
        if improves and (prior is None or row['request_median']<prior['request_median']):
            winners[key]=dict(value=value,name=name,request_median=row['request_median'])
        write(output,state)
    name,code=run('compile-disabled',copy.deepcopy(baseline),compile_disable='1')
    compile_disable='0'
    if not code:
        sample=warm_samples(name,'sample_seconds');request=warm_samples(name,'request_seconds')
        improves=(max(sample)<min(baseline_sample)
            and statistics.median(request)<statistics.median(baseline_request)*.99)
        state['cases'][-1].update(candidate_for_combination=improves,
            sample_median=statistics.median(sample),request_median=statistics.median(request))
        if improves:
            compile_disable='1'
    optimized=copy.deepcopy(baseline)
    for key,winner in winners.items():
        optimized['engine'][key]=winner['value']
    state.update(selected_changes=winners,selected_compile_disable=compile_disable,selected_video_blas=selected_video_blas)
    write(output,state)
    name,code=run('combined-s2',optimized,width=1344,height=768,frames=243,compile_disable=compile_disable,video_blas=selected_video_blas)
    if code:
        state.update(state='failed',failed_case=name);write(output,state);return 1
    reference_s2=warm_samples(args.matrix_prefix+'-s2-single','request_seconds')
    candidate_s2=warm_samples(name,'request_seconds')
    state['matched_s2']=dict(reference=args.matrix_prefix+'-s2-single',candidate=name,
        reference_request_seconds=reference_s2,candidate_request_seconds=candidate_s2,
        speedup=statistics.median(reference_s2)/statistics.median(candidate_s2),
        ranges_separate=max(candidate_s2)<min(reference_s2),
        quality_scope='Automatic decode/finite checks passed; semantic review and contact-sheet comparisons must be recorded separately.')
    state['state']='complete';write(output,state)
    subprocess.run(['python3',str(ROOT/'scripts/amd/summarize_results.py')],check=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
