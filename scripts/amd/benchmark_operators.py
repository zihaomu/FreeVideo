#!/usr/bin/env python3
"""Measure identical H3 operator shapes on CUDA or ROCm without model weights.

Inputs, scales, strides and timing scope are recorded for cross-device comparison.
This is an operator benchmark, not a complete sampling or quality benchmark.
"""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Shapes from the real first 1344x768/243-frame H3 block, step zero.
CASES = [
    dict(id="fp8_ff_up", kind="gemm", dtype="fp8", m=2048, k=5376, n=28672),
    dict(id="fp8_ff_down", kind="gemm", dtype="fp8", m=2048, k=14336, n=5376),
    dict(id="fp8_qkv", kind="gemm", dtype="fp8", m=73435, k=5376, n=2048),
    dict(id="fp8_qkv_tail", kind="gemm", dtype="fp8", m=73435, k=5376, n=1024),
    dict(id="fp8_attention_out", kind="gemm", dtype="fp8", m=73435, k=7168, n=5376),
    dict(id="bf16_softmax_gate", kind="gemm", dtype="bf16", m=73435, k=5376, n=16, bias=True),
    dict(id="bf16_beta", kind="gemm", dtype="bf16", m=72576, k=5376, n=16),
    dict(id="bf16_text_beta", kind="gemm", dtype="bf16", m=49, k=5376, n=16),
    dict(id="bf16_output_gate_down", kind="gemm", dtype="bf16", m=72576, k=5376, n=128),
    dict(id="bf16_output_gate_up", kind="gemm", dtype="bf16", m=72576, k=128, n=2048, bias=True),
    dict(id="bf16_state_gram", kind="bmm", dtype="bf16", b=1120, m=128, k=1008, n=128, trans_a=True),
    dict(id="bf16_state_readout", kind="bmm", dtype="bf16", b=1120, m=1008, k=128, n=128, trans_b=True),
    dict(id="fp32_state_gram", kind="bmm", dtype="fp32", b=1120, m=128, k=1008, n=128, trans_a=True),
    dict(id="window_large", kind="attention", dtype="bf16", b=4, nq=5040, nk=17995, h=16, d=128),
    dict(id="window_small", kind="attention", dtype="bf16", b=1, nq=1008, nk=8923, h=16, d=128),
    dict(id="global", kind="attention", dtype="bf16", b=1, nq=2875, nk=73435, h=16, d=128),
    dict(id="state_cholesky", kind="cholesky", dtype="fp32", b=16, n=128),
    dict(id="state_solve", kind="solve", dtype="fp32", b=16, n=128, upper=False),
    dict(id="bf16_copy", kind="copy", dtype="bf16", m=72576, n=5376),
    dict(id="state_cholesky_video", kind="cholesky", dtype="fp32", b=1120, n=128, batch_shape=[70, 16]),
    dict(id="state_solve_video", kind="solve", dtype="fp32", b=1120, n=128, batch_shape=[70, 16], upper=False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--cases", help="Comma-separated case ids; default: all")
    parser.add_argument("--blas", default="default,cublaslt", help="GEMM/BMM backends")
    parser.add_argument("--attention", help="Default: torch-flash plus rocm-triton on HIP")
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt-in FP32 tensor math; compare separately from IEEE FP32")
    parser.add_argument("--device-uuid", help="Device lease identity, required on HIP")
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 1:
        parser.error("Need at least three repetitions and one warmup")
    selected = set(args.cases.split(",")) if args.cases else {c["id"] for c in CASES}
    unknown = selected - {c["id"] for c in CASES}
    if unknown:
        parser.error("Unknown cases: " + ",".join(sorted(unknown)))
    blas_choices = args.blas.split(",")
    if set(blas_choices) - {"default", "cublas", "cublaslt"}:
        parser.error("BLAS choices: default,cublas,cublaslt")

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        parser.error("A CUDA or ROCm GPU is required")
    torch.cuda.set_device(args.device)
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if torch.version.hip and not args.device_uuid:
        parser.error("Provide the selected HIP --device-uuid for the runtime lease")
    args.out.mkdir(parents=True, exist_ok=False)
    attention_choices = (args.attention.split(",") if args.attention else
                         ["torch-flash", "rocm-triton"] if torch.version.hip else ["torch-flash"])
    props = torch.cuda.get_device_properties(args.device)
    report = dict(
        schema="freevideo-operator-parity-v1", scope="Synthetic deterministic inputs, real H3 shapes; warm operator only",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=str(torch.__version__), hip=torch.version.hip, cuda=torch.version.cuda,
        gpu=props.name, arch=getattr(props, "gcnArchName", ""), capability=list(torch.cuda.get_device_capability()),
        device_uuid=args.device_uuid or str(getattr(props, "uuid", "unknown")),
        total_memory=props.total_memory, repeats=args.repeats, warmup=args.warmup,
        tf32=args.tf32, bf16_reduced_reduction=True, results=[],
        fp8_contract="E4M3FN, per-tensor FP32 unit scales, BF16 output, use_fast_accum=True; quantization excluded",
        input_contract="CPU int8 randint(-16,17), seeded per case/tensor, converted and divided by 16; SHA256 recorded",
        environment={name: os.environ.get(name) for name in
                     ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "TORCH_BLAS_PREFER_HIPBLASLT", "ROCBLAS_USE_HIPBLASLT")},
    )
    try:
        report["triton"] = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        report["triton"] = None

    def save():
        (args.out / "operators.json").write_text(json.dumps(report, indent=2) + "\n")

    @contextmanager
    def blas_context(name):
        previous = torch.backends.cuda.preferred_blas_library()
        torch.cuda.synchronize()
        if name != "default":
            torch.backends.cuda.preferred_blas_library(name)
        try:
            yield
        finally:
            torch.cuda.synchronize()
            torch.backends.cuda.preferred_blas_library(previous)

    def tensor(shape, dtype, seed, identity):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        host = torch.randint(-16, 17, shape, dtype=torch.int8, generator=generator)
        identity.append(dict(shape=list(shape), seed=seed,
                             int8_sha256=hashlib.sha256(host.numpy().tobytes()).hexdigest()))
        value = host.to(device="cuda", dtype=dtype) / 16
        return value

    def timed(operation):
        for _ in range(args.warmup):
            result = operation()
            del result
        torch.cuda.synchronize()
        values, walls = [], []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tick = time.perf_counter()
            start.record()
            result = operation()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
            walls.append((time.perf_counter() - tick) * 1000)
            del result
        return dict(ms=values, median_ms=statistics.median(values), min_ms=min(values),
                    wall_ms=walls, median_wall_ms=statistics.median(walls))

    with ExitStack() as lease:
        if args.device_uuid:
            from freevideo_engine.locking import runtime_lock, device_lock_path
            lease.enter_context(runtime_lock(shared=True))
            lease.enter_context(runtime_lock(device_lock_path(args.device_uuid), inherit=False))
        for index, case in enumerate(CASES):
            if case["id"] not in selected:
                continue
            identities, tensors = [], []
            seed = 2026100700 + index * 10
            dtype = dict(fp8=torch.bfloat16, bf16=torch.bfloat16, fp32=torch.float32)[case["dtype"]]
            kind = case["kind"]
            if kind == "gemm":
                x = tensor((case["m"], case["k"]), dtype, seed, identities)
                weight = tensor((case["n"], case["k"]), dtype, seed + 1, identities)
                tensors = [x, weight]
                if case["dtype"] == "fp8":
                    x, weight = x.to(torch.float8_e4m3fn), weight.to(torch.float8_e4m3fn)
                    unit = torch.ones((1, 1), device="cuda", dtype=torch.float32)
                    tensors = [x, weight, unit]
                    operation = lambda: torch._scaled_mm(x, weight.t(), scale_a=unit, scale_b=unit,
                                                         out_dtype=torch.bfloat16, use_fast_accum=True)
                else:
                    bias = tensor((case["n"],), dtype, seed + 2, identities) if case.get("bias") else None
                    if bias is not None:
                        tensors.append(bias)
                    operation = lambda: F.linear(x, weight, bias)
                flops = 2 * case["m"] * case["k"] * case["n"]
                traffic = (case["m"] * case["k"] + case["k"] * case["n"]) * x.element_size() + 2 * case["m"] * case["n"]
            elif kind == "bmm":
                ashape = (case["b"], case["k"], case["m"]) if case.get("trans_a") else (case["b"], case["m"], case["k"])
                bshape = (case["b"], case["n"], case["k"]) if case.get("trans_b") else (case["b"], case["k"], case["n"])
                x, weight = tensor(ashape, dtype, seed, identities), tensor(bshape, dtype, seed + 1, identities)
                if case.get("trans_a"):
                    x = x.transpose(-1, -2)
                if case.get("trans_b"):
                    weight = weight.transpose(-1, -2)
                tensors = [x, weight]
                operation = lambda: torch.bmm(x, weight)
                flops = 2 * case["b"] * case["m"] * case["k"] * case["n"]
                traffic = case["b"] * (case["m"] * case["k"] + case["k"] * case["n"] + case["m"] * case["n"]) * x.element_size()
            elif kind == "attention":
                q = tensor((case["b"], case["nq"], case["h"], case["d"]), dtype, seed, identities)
                k, v = [tensor((case["b"], case["nk"], case["h"], case["d"]), dtype, seed + offset, identities) for offset in (1, 2)]
                tensors = [q, k, v]
                flops = 4 * case["b"] * case["h"] * case["nq"] * case["nk"] * case["d"]
                traffic = 2 * (q.numel() + k.numel()) * q.element_size()
            elif kind in ("cholesky", "solve"):
                batch_shape = tuple(case.get("batch_shape", [case["b"]]))
                x = tensor((*batch_shape, case["n"], case["n"]), dtype, seed, identities)
                identity = torch.eye(case["n"], device="cuda").expand(*batch_shape, -1, -1)
                positive = x @ x.transpose(-1, -2) + identity
                if kind == "cholesky":
                    tensors = [positive]
                    operation = lambda: torch.linalg.cholesky(positive)
                else:
                    factor = torch.linalg.cholesky(positive)
                    tensors = [factor, identity]
                    operation = lambda: torch.linalg.solve_triangular(factor, identity, upper=case["upper"])
                flops, traffic = None, None
            else:
                x = tensor((case["m"], case["n"]), dtype, seed, identities)
                destination = torch.empty_like(x)
                tensors = [x, destination]
                operation = lambda: destination.copy_(x)
                flops, traffic = 0, 2 * x.numel() * x.element_size()

            if kind == "attention":
                variants = attention_choices
            elif kind in ("gemm", "bmm") and case["dtype"] != "fp8":
                variants = blas_choices
            else:
                # Global hipBLASLt FP8 is deliberately never selected.
                variants = ["default"]
            reference = None
            for variant in variants:
                row = dict(case=case, variant=variant, inputs=identities,
                           tensor_layouts=[dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype)) for t in tensors],
                           flops=flops, minimum_io_bytes=traffic)
                try:
                    if kind == "attention":
                        if variant == "rocm-triton":
                            if not torch.version.hip or props.gcnArchName.split(":")[0] != "gfx1201":
                                raise ValueError("rocm-triton requires gfx1201")
                            from freevideo_engine.rocm_attention import attention
                            operation = lambda: attention(q, k, v, case["d"] ** -.5)
                        elif variant == "sage2":
                            from sageattention import sageattn
                            operation = lambda: sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=case["d"] ** -.5)
                        elif variant == "torch-flash":
                            def operation():
                                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                                    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=case["d"] ** -.5).transpose(1, 2)
                        else:
                            raise ValueError("Unknown attention variant: " + variant)
                        row["attention_arithmetic"] = "quantized-sage2" if variant == "sage2" else "bf16-flash"
                    with blas_context(variant if kind in ("gemm", "bmm") else "default"):
                        value = operation()
                        row["finite"] = bool(value.isfinite().all())
                        if reference is None:
                            reference = value.detach().clone()
                        else:
                            diff = value.float() - reference.float()
                            row["difference_from_first_variant"] = dict(
                                exact=bool(torch.equal(value, reference)),
                                relative_rmse=float((diff.square().mean() / reference.float().square().mean().clamp_min(1e-20)).sqrt()),
                                max_abs=float(diff.abs().max()))
                            del diff
                        del value
                        row.update(timed(operation))
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                            value = operation()
                            torch.cuda.synchronize()
                            del value
                        trace_path = args.out / (case["id"] + "-" + variant + ".json")
                        prof.export_chrome_trace(str(trace_path))
                        events = json.loads(trace_path.read_text())["traceEvents"]
                        kernels = Counter()
                        for event in events:
                            if event.get("cat") == "kernel":
                                kernels[event["name"]] += event["dur"] / 1000
                        row["trace"] = trace_path.name
                        row["gpu_kernels_ms"] = dict(kernels)
                    row["status"] = "measured"
                    if flops is not None:
                        row["effective_tflops"] = flops / (row["median_ms"] * 1e9)
                    if traffic is not None:
                        row["minimum_io_gbps"] = traffic / (row["median_ms"] * 1e6)
                except Exception as error:
                    row.update(status="failed", error=type(error).__name__ + ": " + str(error))
                report["results"].append(row)
                save()
                print(json.dumps({key: row[key] for key in ("case", "variant", "status", "median_ms", "effective_tflops", "minimum_io_gbps", "difference_from_first_variant", "error") if key in row}), flush=True)
            del reference, tensors, operation
            # Drop per-case closures and tensors before allocating the next shape.
            x = weight = bias = unit = q = k = v = positive = identity = factor = destination = None
            torch.cuda.empty_cache()
    save()


if __name__ == "__main__":
    main()
