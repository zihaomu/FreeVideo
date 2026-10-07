#!/usr/bin/env python3
"""Export the validation plan's table from retained, verified request summaries."""
import argparse
import csv
import io
import json
from pathlib import Path


TIMINGS = {
    'total': 'request_seconds', 'launch': 'launch_seconds', 'encode': 'encode_work_seconds',
    'encoder_load': 'encoder_load_seconds', 'engine_load': 'load_seconds',
    'sample': 'sample_seconds', 'base_sample': 'base_sample_seconds',
    'upscale': 'upscale_seconds', 'refine_sample': 'refine_sample_seconds',
    'vae_load': 'vae_load_seconds', 'video_decode': 'video_decode_seconds',
    'audio_decode': 'audio_load_decode_seconds', 'mux': 'mux_seconds',
    'decode': 'decode_save_seconds', 'rtf': 'rtf', 'generated_fps': 'generated_fps',
}
PEAKS = (
    'whole_gpu_peak_bytes', 'rss_peak_bytes', 'pss_peak_bytes',
    'private_working_pss_peak_bytes', 'encoder_peak_allocated_bytes',
    'encoder_peak_reserved_bytes', 'sample_peak_allocated_bytes',
    'sample_peak_reserved_bytes', 'decode_peak_allocated_bytes',
    'decode_peak_reserved_bytes',
)


def verified(run):
    verification = run.get('verification') or {}
    audio = verification.get('raw_audio') or {}
    return (run.get('status') == 'passed' and verification.get('success') is True
            and audio.get('finite') is True and audio.get('saturation_fraction', 1) < .05)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def export_case(case, commit, image_id, receipt):
    runs = case['runs']
    successful = [run for run in runs if verified(run)]
    warm = [run for run in successful if run['index'] > 0]
    representative = (successful or runs or [{}])[0]
    config = representative.get('engine_config') or {}
    hardware = representative.get('hardware') or {}
    requested = case['case']
    geometry = representative.get('geometry') or requested
    plan = representative.get('sampling_plan') or geometry.get('sampling_plan') or {}
    cold = next((run for run in runs if run['index'] == 0), {})
    eligible = case['state'] == 'complete' and case['satisfies_three_warm_requests']
    statistics = case['warm_statistics'] if eligible else {}
    row = dict(
        case=requested['name'], state=case['state'], commit=commit, image_id=image_id,
        source_receipt=str(receipt), source_report=case['source_report'],
        request_scope=('Generation with supplied conditioning; encoder omitted'
                       if requested.get('conditioning') else 'Independent full request'),
        conditioning=requested.get('conditioning'),
        torch=hardware.get('torch_version'), hip=hardware.get('hip_version'),
        device_backend='rocm' if hardware.get('hip_version') else config.get('device_backend'),
        gcn_arch=hardware.get('gcn_arch'), gpu_uuid=hardware.get('gpu_uuid'),
        canvas=f"{geometry['width']}x{geometry['height']}", frames=geometry['frames'],
        video_seconds=geometry.get('seconds'), seed=requested.get('seed'),
        base_steps=plan.get('base_steps', config.get('steps', requested.get('steps'))),
        refine_steps=plan.get('refine_steps', 0),
        two_pass=plan.get('enabled', requested.get('two_pass')),
        sampling_plan=compact(plan), precision=config.get('precision'),
        linear_compute=config.get('linear_compute'), attention=config.get('attention'),
        fp8_scale_granularity=config.get('fp8_scale_granularity'),
        resident_blocks=config.get('resident_blocks'), prefetch=config.get('prefetch'),
        head_chunk=config.get('head_chunk'), projection_chunk=config.get('projection_chunk'),
        ff_chunk=config.get('ff_chunk'), pin_host_gb=config.get('pin_host_gb'),
        blas_environment=compact(case.get('blas_environment')),
        video_blas_environment=compact(case.get('video_blas_environment')),
        compile_environment=compact(case.get('compile_environment')),
        cache_scope=case.get('cache_scope'),
        cold_status=cold.get('status'), cold_verified=verified(cold),
        cold_total_seconds=cold.get('request_seconds'),
        successful_requests=len(successful), successful_warm_requests=len(warm),
        warm_statistics_eligible=eligible,
        successful_outputs=compact([run['output'] for run in successful]),
        all_outputs=compact([run['output'] for run in runs]),
        runtime_code_by_run=compact({str(run['index']): run.get('runtime_code') for run in runs}),
    )
    for label, metric in TIMINGS.items():
        values = statistics.get(metric) or {}
        for statistic in ('median', 'minimum', 'maximum'):
            row[f'{label}_{statistic}'] = values.get(statistic)
    for metric in PEAKS:
        row[metric] = (statistics.get(metric) or {}).get('maximum')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--code-receipt', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    receipt = json.loads(args.code_receipt.read_text())
    rows = [export_case(case, receipt['head'], args.image_id, args.code_receipt)
            for case in summary['cases']]
    if not rows:
        raise ValueError('Summary contains no cases')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(stream.getvalue())
    temporary.replace(args.out)
    print(compact(dict(out=str(args.out), cases=len(rows),
                       eligible=sum(row['warm_statistics_eligible'] for row in rows),
                       scope='Warm statistics require a complete case with at least three verified independent warm requests. Phase times overlap; memory columns are warm maxima. Supplied image ID and receipt HEAD identify the experiment environment; per-run runtime hashes identify modified code.')))


if __name__ == '__main__':
    main()
