"""HIP adaptations for the pinned VDN implementation, preserving its math."""
from functools import wraps
from contextlib import contextmanager
import os


def activate():
    import torch
    if not torch.version.hip:
        return
    from src.models.linear_attention.features import LinearAttentionSepConv
    from src.models.linear_attention.branch import _HeadSliceSepConv

    def wrap(original):
        @wraps(original)
        def spatial(self, *args, **kwargs):
            x, weight = original(self, *args, **kwargs)
            # MIOpen can return NCHW even for the channels-last input. The pinned
            # temporal Triton kernel requires contiguous [frames, tokens, channels].
            return x.contiguous(), weight.contiguous()
        spatial._freevideo_rocm = True
        return spatial

    for kind in (LinearAttentionSepConv, _HeadSliceSepConv):
        if not getattr(kind.spatial, '_freevideo_rocm', False):
            kind.spatial=wrap(kind.spatial)


@contextmanager
def video_blas():
    """Temporarily select HIP video GEMM without changing sampling/audio GEMM."""
    import torch
    requested = os.environ.get('FREEVIDEO_ROCM_VIDEO_BLAS', 'default')
    if not torch.version.hip or requested == 'default':
        yield dict(requested=requested, changed=False)
        return
    if requested not in ('cublas', 'cublaslt'):
        raise ValueError('FREEVIDEO_ROCM_VIDEO_BLAS must be default, cublas, or cublaslt')
    previous = torch.backends.cuda.preferred_blas_library()
    torch.cuda.synchronize()
    torch.backends.cuda.preferred_blas_library(requested)
    try:
        yield dict(requested=requested, changed=True, previous=str(previous),
                   active=str(torch.backends.cuda.preferred_blas_library()),
                   scope='Video VAE only; sampling and audio retain their prior BLAS backend')
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            torch.backends.cuda.preferred_blas_library(previous)
