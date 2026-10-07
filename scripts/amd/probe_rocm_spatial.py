#!/usr/bin/env python3
"""Check RDNA4 spatial convolution padding, frame boundaries and real H3 sizes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    from freevideo_engine.rocm_spatial import spatial
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.manual_seed(20261007)
    assert torch.version.hip and torch.cuda.device_count() == 1
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(':')[0] == 'gfx1201'
    rows = []
    # Distinct constants per frame expose accidental reads across frame borders;
    # seven channels and a six-pixel frame exercise the final masked launch tile.
    frames, height, width, channels = 3, 2, 3, 7
    volume = torch.arange(1, frames + 1, dtype=torch.float32).view(frames, 1, 1, 1)
    volume = volume.expand(frames, channels, height, width).contiguous()
    weight = torch.ones(channels, 1, 5, 5)
    reference = F.conv2d(volume, weight, padding=2, groups=channels).to(torch.bfloat16)
    tokens = volume.permute(0, 2, 3, 1).reshape(-1, 1, channels).cuda().to(torch.bfloat16)
    actual = spatial(tokens, weight.cuda().to(torch.bfloat16), frames, (height, width))
    expected = reference.permute(0, 2, 3, 1).reshape(frames, height * width, channels)
    assert torch.equal(actual.cpu(), expected)
    rows.append(dict(shape=list(tokens.shape), frame_size=[height, width],
                     constant_frames_and_padding_exact=True))
    for frames, height, width, channels in [(2, 1, 1, 128), (4, 3, 5, 128),
                                            (23, 14, 24, 2048), (70, 24, 42, 2048)]:
        tokens = torch.randn(frames * height * width, 1, channels,
                             device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(channels, 1, 5, 5, device='cuda', dtype=torch.bfloat16) * .2
        volume = tokens.reshape(frames, height, width, channels).permute(0, 3, 1, 2)
        reference = F.conv2d(volume, weight, padding=2, groups=channels)
        reference = reference.permute(0, 2, 3, 1).reshape(frames, height * width, channels)
        actual = spatial(tokens, weight, frames, (height, width))
        difference = actual.float() - reference.float()
        relative = float((difference.square().mean() / reference.float().square().mean()).sqrt())
        finite = bool(actual.isfinite().all())
        assert finite and relative < 2e-5, (frames, height, width, channels, relative)
        assert torch.equal(actual, spatial(tokens, weight, frames, (height, width)))
        rows.append(dict(shape=list(tokens.shape), frame_size=[height, width],
                         finite=finite, relative_rmse=relative, repeat_exact=True,
                         max_absolute_difference=float(difference.abs().max())))
    try:
        spatial(tokens.float(), weight.float(), frames, (height, width))
    except ValueError:
        rows.append(dict(unsupported_dtype_rejected=True))
    else:
        raise AssertionError('Unsupported dtype was accepted')
    report = dict(passed=True, gpu=torch.cuda.get_device_name(), arch=arch,
                  torch=str(torch.__version__), hip=torch.version.hip,
                  reference='Same BF16 inputs/weights with MIOpen Conv2d; small boundary case uses exact FP32 CPU convolution',
                  numerical_gate_relative_rmse=2e-5, cases=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
