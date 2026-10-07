#!/usr/bin/env python3
"""Pause only the matrix parent and run a GPU probe after its current case ends."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT=Path(os.environ.get('FV_STORAGE_ROOT','/dc1/zihaomu/free_token_mapping'))/'experiments/freevideo-r9700'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:parser.error('GPU probe command required')
    report=EXPERIMENT/'reports'/f'{args.name}-schedule.json'
    if report.exists():raise FileExistsError(report)
    matrix=json.loads(args.matrix.read_text());pid=matrix['pid']
    assert matrix['state']=='running'
    assert b'run_matrix.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    case=json.loads(args.case.read_text());case_pid=case['pid']
    assert case['state']=='running'
    assert b'run_case.py' in Path(f'/proc/{case_pid}/cmdline').read_bytes()
    state=dict(state='waiting-for-case-boundary',pid=os.getpid(),matrix_pid=pid,
               case_pid=case_pid,case=str(args.case),command=command,matrix_resume_sent=False)
    def save():
        temporary=report.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(report)
    save()
    os.kill(pid,signal.SIGSTOP)
    try:
        while True:
            case=json.loads(args.case.read_text())
            if case['state'] in ('complete','failed'):break
            if not Path(f'/proc/{case_pid}').exists():
                raise RuntimeError('Case driver disappeared before publishing its terminal state')
            time.sleep(5)
        if case['state']!='complete':
            raise RuntimeError('Case prerequisite failed; resume matrix to publish the failure')
        state['state']='running-probe';save()
        with (EXPERIMENT/'reports'/f'{args.name}-probe.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,FV_CACHE_GROUP=args.name))
        state.update(state='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
    except Exception as error:
        state.update(state='failed',error=repr(error))
    finally:
        try:
            os.kill(pid,signal.SIGCONT);state['matrix_resume_sent']=True
        except ProcessLookupError:
            state['matrix_resume_error']='Matrix parent already exited'
        save()
    return int(state['state']!='complete')


if __name__=='__main__':
    raise SystemExit(main())
