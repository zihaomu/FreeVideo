#!/usr/bin/env python3
"""Run the plan's required S1/S2/S3 matrix after a validated pipeline gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
STORAGE=Path(os.environ.get('FV_STORAGE_ROOT','/dc1/zihaomu/free_token_mapping'))
EXPERIMENT=STORAGE/'experiments/freevideo-r9700'
CASES=[('s1-single',768,448,124,8,False),('s1-light',768,448,124,8,True),
       ('s2-light',1344,768,243,8,True),('s2-single',1344,768,243,8,False),
       ('s3-medium',1344,768,243,12,False),('s3-high',1344,768,243,16,False),
       ('s3-max',1344,768,243,20,False)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--start-at',choices=[c[0] for c in CASES],default=CASES[0][0])
    parser.add_argument('--stop-after',choices=[c[0] for c in CASES],default=CASES[-1][0])
    parser.add_argument('--probe',action='store_true',help='Refresh kernel receipts before the matrix')
    args=parser.parse_args()
    report=EXPERIMENT/'reports'/f'{args.prefix}-matrix.json'
    names=[c[0] for c in CASES]
    first,last=names.index(args.start_at),names.index(args.stop_after)
    if first>last:
        parser.error('--start-at must not follow --stop-after')
    selected=CASES[first:last+1]
    state=dict(state='waiting-for-gate',pid=os.getpid(),profile=str(args.profile),cases=[],
               gate=str(args.gate),case_order=[c[0] for c in selected])
    def save():
        tmp=report.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(report)
    if report.exists():
        raise FileExistsError(report)
    save()
    while True:
        gate=json.loads(args.gate.read_text())
        if gate['state']=='complete':
            break
        if gate['state']=='failed':
            state.update(state='failed',error='Pipeline prerequisite failed');save();return 1
        time.sleep(5)
    if args.probe:
        state['state']='kernel-probes';save()
        with (EXPERIMENT/'reports'/f'{args.prefix}-doctor.json').open('w') as stream:
            result=subprocess.run(['bash',str(ROOT/'scripts/amd/run_rocm.sh'),'-m','freevideo_engine',
                                   'doctor','--probe'],stdout=stream,
                                  stderr=(EXPERIMENT/'reports'/f'{args.prefix}-doctor.log').open('w'))
        checked=json.loads((EXPERIMENT/'reports'/f'{args.prefix}-doctor.json').read_text())
        if result.returncode or not checked.get('ready'):
            state.update(state='failed',error='Kernel readiness probe failed');save();return 1
    state['state']='running';save()
    for name,width,height,frames,steps,two_pass in selected:
        name=args.prefix+'-'+name
        command=['python3',str(ROOT/'scripts/amd/run_case.py'),'--name',name,
                 '--width',str(width),'--height',str(height),'--frames',str(frames),
                 '--steps',str(steps),'--profile',str(args.profile),'--repeats','4']
        if two_pass:
            command.append('--two-pass')
        row=dict(name=name,state='running');state['cases'].append(row);save()
        print(json.dumps(dict(event='case_start',**row)),flush=True)
        with (EXPERIMENT/'reports'/f'{name}-driver.log').open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,state='complete' if result.returncode==0 else 'failed')
        save()
        if result.returncode:
            state.update(state='failed',failed_case=name);save();return 1
    state['state']='complete';save();return 0


if __name__=='__main__':
    raise SystemExit(main())
