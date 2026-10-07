#!/usr/bin/env python3
"""Wait for the GPU experiment queue, then score verified audio on the CPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
STORAGE = Path(os.environ.get('FV_STORAGE_ROOT', '/dc1/zihaomu/free_token_mapping'))
EXPERIMENT = STORAGE/'experiments/freevideo-r9700'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    report = EXPERIMENT/'reports'/f'{args.prefix}-queue.json'
    if report.exists():
        raise FileExistsError(report)
    state = dict(state='waiting-for-followups', pid=os.getpid(), gate=str(args.gate),
                 scope='CPU-only semantic scoring after the GPU experiment queue ends')
    def save():
        temporary = report.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(report)
    save()
    while True:
        gate = json.loads(args.gate.read_text())
        if gate['state'] in ('complete', 'failed'):
            break
        time.sleep(5)
    destination = EXPERIMENT/'reports'/f'{args.prefix}-all.json'
    state.update(state='running', output=str(destination)); save()
    with (EXPERIMENT/'reports'/f'{args.prefix}-all.log').open('w') as log:
        result = subprocess.run(['bash', str(ROOT/'scripts/amd/run_rocm.sh'),
            'scripts/amd/audio_semantics.py', '--scan',
            '/data/experiments/freevideo-r9700/outputs', '--out',
            str(Path('/data')/destination.relative_to(STORAGE))],
            stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, FV_CACHE_GROUP=args.prefix))
    state.update(state='complete' if result.returncode == 0 else 'failed',
                 returncode=result.returncode); save()
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
