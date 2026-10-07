"""Opt-in gfx1201 BF16 replacements; reduction rounding can change generation.

Only explicitly selected gate/branch calls use hipBLASLt. FP8, FP32 statistics,
factorization and scans keep their original backend. BLAS preference is restored
after submission (also on exceptions); no synchronization is added per operator.
"""
from contextlib import contextmanager
import os
import torch
import torch.nn.functional as F


def _selected(kind, chunk, parallelism):
    name = f'FREEVIDEO_ROCM_{kind.upper()}_BLAS'
    requested = os.environ.get(name, 'default')
    if requested not in ('default', 'cublaslt'):
        raise ValueError(f'{name} must be default or cublaslt')
    if requested == 'default':
        return False
    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1201':
        raise ValueError('H3 BF16 hipBLASLt replacements are validated only on gfx1201')
    if chunk != 16 or parallelism != 1:
        raise ValueError('H3 BF16 hipBLASLt requires head chunk 16 and serial head groups')
    return True


@contextmanager
def _blaslt():
    previous = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library('cublaslt')
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


def make_gate_linear(policy, chunk, parallelism):
    if not _selected('gate', chunk, parallelism):
        return F.linear

    def linear(x, weight, bias=None):
        if not x.is_cuda or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            return F.linear(x, weight, bias)
        with _blaslt():
            result = F.linear(x, weight, bias)
        policy.rocm_gate_linear_calls = getattr(policy, 'rocm_gate_linear_calls', 0) + 1
        return result
    return linear


def make_state_readout(policy, chunk, parallelism):
    if not _selected('state', chunk, parallelism):
        return None
    # These are the pinned VDN's own compiled preparation, scans and epilogue.
    # Only the two BF16 matmuls below change backend, on this model instance.
    from src.models.linear_attention.scan import _frame_stats_prep, _tf32_matmul
    from src.models.linear_attention.branch import (
        _run_scans_inference, gather_linear_state, linear_epilogue)

    def matmul(a, b):
        if not a.is_cuda or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            return torch.matmul(a, b)
        with _blaslt():
            result = torch.matmul(a, b)
        policy.rocm_state_matmul_calls = getattr(policy, 'rocm_state_matmul_calls', 0) + 1
        return result

    def statistics(kf, vf, beta, a_fp32):
        with torch.autocast(device_type=kf.device.type, enabled=False):
            kf, kf32, scaled32, vb = _frame_stats_prep(kf, vf, beta, inference=True)
            if a_fp32:
                with _tf32_matmul():
                    A = torch.matmul(scaled32.transpose(-1, -2), kf32)
            else:
                A = torch.matmul((kf * beta.unsqueeze(-1).to(kf.dtype)).contiguous()
                                 .transpose(-1, -2), kf).float()
            A = 0.5 * (A + A.transpose(-1, -2))
            B = matmul(vb.transpose(-1, -2), kf).float()
            return A, B

    def readout(self, xv, num_frames, tokens_per_frame, bounds, qkv_raw,
                frame_size=None, text_x=None, text_qkv_raw=None, heads=None,
                beta=None, gate=None, frame_mean=None, text_beta=None):
        # Keep the pinned _readout_inference's order, shapes, dtypes and text path.
        # Calling its helpers also retains the original FP32 inverse/scan island.
        head_dim = self.head_dim
        n_heads = self.num_heads if heads is None else heads.stop - heads.start
        backend = self._delta_backend('backend', tokens_per_frame)
        shape_per_frame = (num_frames, tokens_per_frame, n_heads, head_dim)
        query_by_frame, key, value = self._features(
            qkv_raw, num_frames, frame_size, inference=True,
            query_fhsd=(num_frames, tokens_per_frame), heads=heads)
        key_by_frame = key.view(shape_per_frame).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape_per_frame).permute(0, 2, 1, 3)
        if beta is None:
            beta = torch.sigmoid(self.beta_proj(xv))
        beta = beta.view(num_frames, tokens_per_frame, n_heads).permute(0, 2, 1)
        A, B = statistics(key_by_frame, value_by_frame, beta, self.a_fp32)
        if frame_mean is None:
            frame_mean = xv.view(num_frames, tokens_per_frame, -1).mean(dim=1, dtype=torch.float32)
        alpha = self.alpha(frame_mean, heads=heads)
        text_state = self._text_state(text_x, text_qkv_raw, heads=heads, text_beta=text_beta)
        prefix_states, suffix_states = _run_scans_inference(
            backend, alpha, A, B, text_state=text_state)
        if gate is None:
            gate = self.output_gate(xv)
        linear_state = gather_linear_state(prefix_states, suffix_states, alpha, bounds,
            bridge=self.bridge, text_state=text_state, inference=True, out_dtype=gate.dtype)
        del prefix_states, suffix_states
        readout = matmul(query_by_frame, linear_state.transpose(-1, -2))
        return linear_epilogue(readout, self.norm.weight, gate, self.norm.eps,
                               inference=True, fhsd=True)
    return readout
