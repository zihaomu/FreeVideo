#!/usr/bin/env python3
"""Audit matched inputs before interpreting an A/B result as engineering speedup."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def identity(video):
    request = read(video.with_suffix('.request.json'))
    engine = read(video.with_suffix('.engine.json'))
    verification = read(video.with_suffix('.verification.json'))
    prompt = video.with_suffix('.artifacts')/'prompt.txt'
    encoding = read(video.with_suffix('.artifacts')/'encode.json')
    provenance = engine.get('sampling_provenance') or {}
    hardware = request.get('runtime_hardware') or {}
    raw_audio = verification.get('raw_audio') or {}
    passes = engine.get('sampling_passes') or [engine]
    fixed = dict(
        seed=request.get('seed'),
        sampling_seed=provenance.get('seed'),
        geometry={key: (request.get('geometry') or {}).get(key)
                  for key in ('width', 'height', 'frames', 'fps')},
        prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
        encoder=encoding.get('encoder'),
        cache=provenance.get('cache'),
        cache_manifest_sha256=provenance.get('cache_manifest_sha256'),
        base=provenance.get('base'),
        sampling_plan=request.get('sampling_plan'),
        hardware={key: hardware.get(key) for key in
                  ('gpu_uuid', 'hip_version', 'torch_version', 'gcn_arch')},
        model_configuration={key: (engine.get('config') or {}).get(key) for key in
            ('source_id', 'adaln_table_identity', 'task', 'precision', 'linear_compute',
             'fp8_scale_granularity', 'attention', 'steps', 'inference_kernels')},
        passes=[dict(geometry={key: (item.get('geometry') or {}).get(key)
                              for key in ('width', 'height', 'frames', 'fps')},
                     steps=len(item.get('step_seconds') or []),
                     refinement=item.get('refinement')) for item in passes],
    )
    # A missing field must not pass merely because it is absent on both sides.
    def missing(value, path=''):
        if isinstance(value, dict):
            return [field for key, item in value.items()
                    for field in missing(item, f'{path}.{key}' if path else key)]
        if isinstance(value, list):
            return [field for index, item in enumerate(value)
                    for field in missing(item, f'{path}[{index}]')]
        return [path] if value is None or value == '' else []

    # Disabled plans legitimately contain null second/upscale fields.
    required = dict(fixed)
    required.pop('sampling_plan')
    required['passes'] = [{key: value for key, value in item.items() if key != 'refinement'}
                          for item in fixed['passes']]
    absent = missing(required)
    if not isinstance(fixed['sampling_plan'], dict):
        absent.append('sampling_plan')
    else:
        for key in ('enabled', 'mode', 'base_steps', 'refine_steps', 'total_steps', 'first'):
            if fixed['sampling_plan'].get(key) is None:
                absent.append('sampling_plan.'+key)
    consistent_seed = fixed['seed'] is not None and fixed['seed'] == fixed['sampling_seed']
    actual_steps = sum(item['steps'] for item in fixed['passes'])
    consistent_steps = actual_steps > 0 and actual_steps == (fixed['sampling_plan'] or {}).get('total_steps')
    passed = (video.is_file() and request.get('success') is True
              and verification.get('success') is True
              and verification.get('black_frames') == []
              and verification.get('exact_repeated_frames') == []
              and raw_audio.get('finite') is True
              and raw_audio.get('saturation_fraction', 1) < .05)
    full_request = (request.get('encoder_process_exited_before_video') is True
                    and (request.get('input_cache') or {}).get('conditioning_hit') is False
                    and engine.get('resident_engine_cache_hit') is False
                    and isinstance(request.get('request_seconds'), (int, float)))
    return dict(fixed=fixed, missing_required_fields=absent, verified=passed,
                internally_consistent=consistent_seed and consistent_steps,
                independent_full_request=full_request,
                conditioning_sha256=provenance.get('conditioning_sha256'),
                runtime_code=request.get('runtime_code'),
                request_seconds=request.get('request_seconds'))


def differences(left, right, path=''):
    if isinstance(left, dict) and isinstance(right, dict):
        return [item for key in sorted(set(left) | set(right))
                for item in differences(left.get(key), right.get(key),
                                        f'{path}.{key}' if path else key)]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, f'{path}[{index}]')]
    return [] if left == right else [dict(field=path, reference=left, candidate=right)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Retained MP4 path')
    parser.add_argument('--candidate', type=Path, required=True, help='Retained MP4 path')
    parser.add_argument('--precision-comparison', action='store_true',
                        help='Permit a linear compute precision change; never classify as engineering speedup')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    reference, candidate = identity(args.reference), identity(args.candidate)
    changed = differences(reference['fixed'], candidate['fixed'])
    permitted, unexpected = [], []
    for item in changed:
        allowed = (args.precision_comparison and
                   item['field'].endswith(('.linear_compute', '.fp8_scale_granularity')))
        (permitted if allowed else unexpected).append(item)
    matched = (not unexpected and
               all(side['verified'] and side['independent_full_request'] and side['internally_consistent'] and
                   not side['missing_required_fields'] for side in (reference, candidate)))
    report = dict(
        reference=str(args.reference), candidate=str(args.candidate),
        kind='precision' if args.precision_comparison else 'engineering',
        matched_inputs=matched, engineering_inputs=matched and not args.precision_comparison,
        unexpected_differences=unexpected, permitted_precision_differences=permitted,
        reference_identity=reference, candidate_identity=candidate,
        identical_conditioning=(reference['conditioning_sha256'] is not None and
                                reference['conditioning_sha256'] == candidate['conditioning_sha256']),
        runtime_code_differences=differences(reference.get('runtime_code'), candidate.get('runtime_code')),
        scope='Checks retained full-request inputs, model manifest identity, actual pass configurations and media/numeric verification. Runtime code changes are disclosed for review. This does not establish quality, timing stability or speedup; those require separate evidence. Conditioning hashes are disclosed rather than required, since independently encoded tensors may differ numerically.',
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out), matched_inputs=matched,
                          unexpected_fields=[item['field'] for item in unexpected])))
    return 0 if matched else 2


if __name__ == '__main__':
    raise SystemExit(main())
