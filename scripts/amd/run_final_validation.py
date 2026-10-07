#!/usr/bin/env python3
"""Verify the selected engineering stack on three matched quality prompts."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
STORAGE = Path(os.environ.get('FV_STORAGE_ROOT', '/dc1/zihaomu/free_token_mapping'))
EXPERIMENT = STORAGE/'experiments/freevideo-r9700'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--baseline-prefix',
                        help='Existing quality case prefix; defaults to the gate prefix plus -quality')
    args = parser.parse_args()
    report = EXPERIMENT/'reports'/f'{args.prefix}-validation.json'
    if report.exists():
        raise FileExistsError(report)
    state = dict(state='waiting-for-followups', pid=os.getpid(), gate=str(args.gate), cases=[],
                 scope='Selected engineering stack versus baseline on the same three prompts/seeds at S1; CPU-only CLAP after all GPU runs. Agent semantic review remains required.')
    write(report, state)
    while True:
        gate = read(args.gate)
        if gate['state'] == 'complete':
            break
        if gate['state'] == 'failed':
            state.update(state='failed', error='Follow-up prerequisite failed')
            write(report, state)
            return 1
        time.sleep(5)
    profile = copy.deepcopy(read(args.profile))
    for key, winner in gate['selected_changes'].items():
        profile['engine'][key] = winner['value']
    compile_disable = gate['selected_compile_disable']
    video_blas = gate['selected_video_blas']
    profile_path = EXPERIMENT/'prepared/profiles'/f'{args.prefix}.json'
    if profile_path.exists():
        raise FileExistsError(profile_path)
    write(profile_path, profile)
    state.update(state='running', selected_profile=str(profile_path),
                 compile_disable=compile_disable, video_blas=video_blas)
    write(report, state)
    for index, family in enumerate(('person-motion', 'object-motion', 'complex-scene'), start=2):
        name = args.prefix+'-'+family
        row = dict(name=name, family=family, seed=2026090900+index, state='running',
                   baseline=(args.baseline_prefix or args.gate.name.removesuffix('-followups.json')+'-quality')+'-'+family)
        state['cases'].append(row); write(report, state)
        command = ['python3', str(ROOT/'scripts/amd/run_case.py'), '--name', name,
                   '--width', '768', '--height', '448', '--frames', '124', '--steps', '8',
                   '--seed', str(row['seed']), '--repeats', '1', '--profile', str(profile_path),
                   '--prompt', str(ROOT/'scripts/amd/prompts'/f'{family}.txt')]
        with (EXPERIMENT/'reports'/f'{name}-driver.log').open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, FV_ROCM_VIDEO_BLAS=video_blas, FV_TORCH_COMPILE_DISABLE=compile_disable))
        row.update(state='complete' if result.returncode == 0 else 'failed', returncode=result.returncode)
        write(report, state)
    # The semantic scorer never shares GPU work with the benchmark queue.
    destination = EXPERIMENT/'reports'/f'{args.prefix}-audio-semantics.json'
    state.update(state='cpu-audio-semantics', audio_semantics=str(destination)); write(report, state)
    with (EXPERIMENT/'reports'/f'{args.prefix}-audio-semantics.log').open('w') as log:
        result = subprocess.run(['bash', str(ROOT/'scripts/amd/run_rocm.sh'),
            'scripts/amd/audio_semantics.py', '--scan', '/data/experiments/freevideo-r9700/outputs',
            '--out', str(Path('/data')/destination.relative_to(STORAGE))],
            stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, FV_CACHE_GROUP=args.prefix+'-audio'))
    state['audio_semantics_returncode'] = result.returncode
    failed = result.returncode != 0 or any(row['state'] != 'complete' for row in state['cases'])
    state['state'] = 'failed' if failed else 'complete'; write(report, state)
    subprocess.run(['python3', str(ROOT/'scripts/amd/summarize_results.py')], check=True)
    subprocess.run(['python3', str(ROOT/'scripts/amd/capture_worktree.py'), '--name', args.prefix], check=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
