#!/usr/bin/env python3
"""Check native GPU FP32 audio decode on a retained input, including first call."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from freevideo_engine.paths import add_vdn
    from freevideo_engine.locking import runtime_lock
    add_vdn()
    from freevideo_engine.rocm_compat import audio_convolutions
    from freevideo_engine.vae_weights import load_audio_vae
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    os.environ['FREEVIDEO_ROCM_AUDIO_CONV'] = 'native'
    assert not args.out.exists()
    def flags():
        return tuple(getattr(torch.backends.cudnn, k) for k in
                     ('enabled', 'benchmark', 'deterministic', 'allow_tf32'))
    before = flags()
    with runtime_lock():
        tick = time.perf_counter()
        model, _ = load_audio_vae('/data/models/vdn/h3-base')
        load_seconds = time.perf_counter() - tick
        value = torch.from_numpy(np.load(args.input, allow_pickle=False)).cuda()
        assert value.dtype == torch.float32
        durations = []
        for _ in range(4):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with audio_convolutions() as backend:
                result = model.decode(value, return_dict=False)[0]
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - tick)
            assert flags() == before
        audio = result.float().permute(1, 0, 2)[0].cpu().numpy()
        reference = np.load(args.reference, allow_pickle=False)
        difference = audio.astype(np.float64) - reference.astype(np.float64)
        error = float(np.sqrt(np.mean(difference ** 2) / np.mean(reference.astype(np.float64) ** 2)))
        assert np.isfinite(audio).all() and error < 1e-5, error
        try:
            with audio_convolutions():
                raise RuntimeError('injected audio decode failure')
        except RuntimeError as exc:
            assert str(exc) == 'injected audio decode failure'
        assert flags() == before
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out.with_suffix('.npy'), audio, allow_pickle=False)
        report = dict(status='passed', scope='Same retained S2 input; native FP32 GPU decoder only',
                      backend=backend, load_seconds=load_seconds, first_seconds=durations[0],
                      warm_seconds=durations[1:], warm_median_seconds=statistics.median(durations[1:]),
                      relative_rmse=error, max_abs=float(np.abs(difference).max()),
                      exact=bool(np.array_equal(audio, reference)), flags_restored=True,
                      exception_restoration=True, input=str(args.input), reference=str(args.reference))
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
