#!/usr/bin/env python3
"""Compare operator reports only when shape, layout, inputs and precision match."""
import argparse
import json
from pathlib import Path
import re


def compare(baseline, candidate, *, baseline_blas="default", candidate_blas="default",
            baseline_attention="torch-flash", candidate_attention="torch-flash"):
    schema = "freevideo-operator-parity-v1"
    if baseline.get("schema") != schema or candidate.get("schema") != schema:
        raise ValueError("Expected reports from benchmark_operators.py")

    def select(report, blas, attention):
        selected = {}
        for row in report["results"]:
            case = row["case"]
            variant = (attention if case["kind"] == "attention" else blas
                       if case["kind"] in ("gemm", "bmm") and case["dtype"] != "fp8" else "default")
            if row["variant"] == variant:
                if case["id"] in selected:
                    raise ValueError("Duplicate case/variant: " + case["id"])
                selected[case["id"]] = row
        return selected

    left = select(baseline, baseline_blas, baseline_attention)
    right = select(candidate, candidate_blas, candidate_attention)
    rows = []
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        reasons = []
        if a is None or b is None:
            reasons.append("Selected variant missing on one device")
        elif a["status"] != "measured" or b["status"] != "measured":
            reasons.append("Operator not measured successfully on both devices")
        else:
            for field in ("case", "inputs", "tensor_layouts", "flops", "minimum_io_bytes"):
                if a.get(field) != b.get(field):
                    reasons.append(field + " differs")
            if a["case"]["dtype"] == "fp8" and baseline["fp8_contract"] != candidate["fp8_contract"]:
                reasons.append("FP8 scale/accumulation contract differs")
            if a["case"]["dtype"] == "fp32" and baseline["tf32"] != candidate["tf32"]:
                reasons.append("FP32/TF32 arithmetic differs")
            if a.get("attention_arithmetic") != b.get("attention_arithmetic"):
                reasons.append("Attention arithmetic differs (quantized versus BF16)")
            if baseline["bf16_reduced_reduction"] != candidate["bf16_reduced_reduction"]:
                reasons.append("BF16 reduction flag differs")
            if not a.get("finite") or not b.get("finite"):
                reasons.append("Nonfinite output")
        row = dict(case=name, comparable=not reasons, reasons=reasons)
        if not reasons:
            row.update(baseline_variant=a["variant"], candidate_variant=b["variant"],
                       baseline_ms=a["median_ms"], candidate_ms=b["median_ms"],
                       candidate_over_baseline=b["median_ms"] / a["median_ms"],
                       baseline_tflops=a.get("effective_tflops"), candidate_tflops=b.get("effective_tflops"))
        rows.append(row)
    core_version = lambda value: re.match(r"\d+\.\d+", value).group(0)
    return dict(
        scope="Warm operator comparison; no full-model or quality equivalence implied",
        baseline_gpu=baseline["gpu"], candidate_gpu=candidate["gpu"],
        baseline_torch=baseline["torch"], candidate_torch=candidate["torch"],
        torch_major_minor_matched=core_version(baseline["torch"]) == core_version(candidate["torch"]),
        baseline_source_sha256=baseline["source_sha256"], candidate_source_sha256=candidate["source_sha256"],
        rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    for side in ("baseline", "candidate"):
        parser.add_argument("--" + side + "-blas", default="default")
        parser.add_argument("--" + side + "-attention", default="torch-flash")
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
                     baseline_blas=args.baseline_blas, candidate_blas=args.candidate_blas,
                     baseline_attention=args.baseline_attention, candidate_attention=args.candidate_attention)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(row["case"], f'{row["candidate_over_baseline"]:.3f}x candidate time' if row["comparable"] else "; ".join(row["reasons"]))


if __name__ == "__main__":
    main()
