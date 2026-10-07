#!/usr/bin/env python3
"""Compare retained raw RGB/audio artifacts without decoding or using a GPU."""
import argparse
import json
import math
from pathlib import Path


def compare(reference, candidate):
    import numpy as np
    left = np.load(reference, mmap_mode='r', allow_pickle=False)
    right = np.load(candidate, mmap_mode='r', allow_pickle=False)
    if left.shape != right.shape:
        raise ValueError(f'Artifact shape mismatch: {left.shape} != {right.shape}')
    squared_error = squared_reference = absolute_error = 0.
    changed = 0
    largest = 0.
    frame_errors = []
    for index in range(left.shape[0]):
        a = left[index].astype(np.float64)
        b = right[index].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Non-finite raw artifact')
        difference = b - a
        squared_error += float(np.square(difference).sum())
        squared_reference += float(np.square(a).sum())
        absolute_error += float(np.abs(difference).sum())
        changed += int(np.count_nonzero(difference))
        largest = max(largest, float(np.abs(difference).max()))
        if reference.name == 'rgb.npy':
            frame_errors.append(float(np.abs(difference).mean()))
    rmse = math.sqrt(squared_error / left.size)
    result = dict(shape=list(left.shape), reference_dtype=str(left.dtype), candidate_dtype=str(right.dtype),
                  mae=absolute_error/left.size, rmse=rmse,
                  relative_rmse=math.sqrt(squared_error/max(squared_reference, 1e-30)),
                  max_absolute_difference=largest, fraction_changed=changed/left.size)
    if frame_errors:
        result.update(psnr_db=None if rmse == 0 else 20*math.log10(255/rmse),
                      exact_match=changed == 0,
                      frame_mae_min=min(frame_errors), frame_mae_max=max(frame_errors))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = dict(reference=str(args.reference), candidate=str(args.candidate),
                  scope='Raw decoder artifacts before MP4 compression; numerical differences alone do not establish semantic quality',
                  artifacts={})
    for name in ('rgb.npy', 'audio.npy', 'audio_decoder_input.npy'):
        if (args.reference/name).is_file() and (args.candidate/name).is_file():
            report['artifacts'][name] = compare(args.reference/name, args.candidate/name)
    if not report['artifacts']:
        raise ValueError('No comparable raw artifacts')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
