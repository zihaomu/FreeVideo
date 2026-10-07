#!/usr/bin/env python3
"""Aggregate retained requests; failed or unverified samples never enter medians."""
import argparse
import json
import os
from pathlib import Path
import statistics

EXPERIMENT=Path(os.environ.get('FV_STORAGE_ROOT','/dc1/zihaomu/free_token_mapping'))/'experiments/freevideo-r9700'


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def summary(values):
    values=[v for v in values if isinstance(v,(float,int))]
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values),count=len(values)) if values else None


def extract(run):
    video=Path(run['output'])
    request=read(video.with_suffix('.request.json'))
    engine=read(video.with_suffix('.engine.json'))
    encoding=request.get('encoding') or {}
    passes=engine.get('sampling_passes') or [engine]
    row=dict(index=run['index'],status=run['status'],output=str(video),
        request_seconds=request.get('request_seconds'),launch_seconds=run.get('launch_seconds'),
        encode_work_seconds=encoding.get('work_seconds'),encoder_load_seconds=encoding.get('load_seconds'),
        load_seconds=engine.get('load_seconds'),sample_seconds=engine.get('sample_seconds'),
        base_sample_seconds=passes[0].get('sample_seconds'),
        refine_sample_seconds=passes[1].get('sample_seconds') if len(passes)>1 else None,
        upscale_seconds=engine.get('latent_upscale',{}).get('stage_seconds'),
        vae_load_seconds=engine.get('vae_load_seconds'),video_decode_seconds=engine.get('video_decode_seconds'),
        audio_load_decode_seconds=engine.get('audio_load_decode_seconds'),mux_seconds=engine.get('encode_seconds'),
        decode_save_seconds=engine.get('decode_save_seconds'),
        rtf=run.get('rtf'),generated_fps=run.get('generated_fps'),
        geometry=request.get('geometry'),sampling_plan=request.get('sampling_plan'),
        engine_config=engine.get('config'),profile=request.get('profile'),
        hardware=request.get('runtime_hardware'),runtime_code=request.get('runtime_code'),
        compatibility=request.get('compatibility'),offload=engine.get('offload'),
        audio_decoder=engine.get('audio_decoder'),video_blas=engine.get('video_blas'))
    for metric in ('allocated','reserved'):
        row[f'encoder_peak_{metric}_bytes']=encoding.get(f'torch_peak_{metric}_bytes')
        row[f'sample_peak_{metric}_bytes']=engine.get(f'torch_peak_{metric}_bytes')
        row[f'decode_peak_{metric}_bytes']=engine.get(f'final_stage_peak_{metric}_bytes')
    resources=[request.get('encoding_resources') or {},request.get('resources') or {}]
    for key,section,field in (
        ('whole_gpu_peak_bytes','gpu','gpu_peak_bytes'),
        ('rss_peak_bytes','ram','process_tree_peak_rss_bytes'),
        ('pss_peak_bytes','ram','process_tree_peak_pss_bytes'),
        ('private_working_pss_peak_bytes','ram','inference_peak_working_pss_bytes')):
        values=[r.get(section,{}).get(field) for r in resources]
        row[key]=max((v for v in values if v is not None),default=None)
    verification=read(video.with_suffix('.verification.json'))
    row['verification']={k:verification.get(k) for k in ('success','black_frames','exact_repeated_frames','raw_audio','audio','finite_latent_tensors','finite_conditioning_tensors')}
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',default='')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out is None:
        args.out=EXPERIMENT/'reports'/((args.prefix+'-' if args.prefix else '')+'measurement-summary.json')
    cases=[]
    for path in sorted((EXPERIMENT/'reports').glob(args.prefix+'*-runs.json')):
        source=read(path)
        rows=[extract(r) for r in source['runs'] if 'output' in r]
        warm=[r for r in rows if r['index']>0 and r['status']=='passed'
              and r['verification']['success'] is True
              and (r['verification'].get('raw_audio') or {}).get('finite') is True
              and r['verification']['raw_audio'].get('saturation_fraction',1)<.05]
        keys={k for row in warm for k,v in row.items() if k.endswith(('_seconds','_bytes')) and isinstance(v,(float,int))}|{'rtf','generated_fps'}
        cases.append(dict(case=source['case'],state=source['state'],cache_scope=source['cache_scope'],
            blas_environment=source.get('blas_environment'),source_report=str(path),runs=rows,
            video_blas_environment=source.get('video_blas_environment'),
            compile_environment=source.get('compile_environment'),
            warm_statistics={k:summary([r.get(k) for r in warm]) for k in sorted(keys)},
            satisfies_three_warm_requests=len(warm)>=3))
    report=dict(scope='Independent full requests unless case.conditioning is supplied; phase timings overlap. Cold runs and failures excluded from warm statistics.',cases=cases)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(args.out)
    print(json.dumps(dict(cases=len(cases),complete=sum(c['state']=='complete' for c in cases),out=str(args.out))))


if __name__=='__main__':
    main()
