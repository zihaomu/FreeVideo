"""Guard against publishing speed ratios for mismatched operator workloads."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_operators import compare


def report(dtype="bf16", kind="gemm"):
    return dict(schema="freevideo-operator-parity-v1", gpu="test device", torch="2.12.0",
                source_sha256="test", fp8_contract="unit scales", tf32=False,
                bf16_reduced_reduction=True, results=[dict(
                    case=dict(id="case", kind=kind, dtype=dtype, m=16, k=32, n=64),
                    variant="torch-flash" if kind == "attention" else "default",
                    status="measured", finite=True, median_ms=2., flops=65536,
                    minimum_io_bytes=7168, inputs=[dict(seed=1, int8_sha256="input")],
                    tensor_layouts=[dict(shape=[16, 32], stride=[32, 1], dtype=dtype)],
                    **({"attention_arithmetic": "bf16-flash"} if kind == "attention" else {}))])


class ComparisonContracts(unittest.TestCase):
    def test_same_workload_ratio(self):
        a, b = report(), report()
        b["results"][0]["median_ms"] = 6.
        self.assertEqual(compare(a, b)["rows"][0]["candidate_over_baseline"], 3.)

    def test_shape_input_and_stride_mismatches(self):
        for field, value in [("case", dict(id="case", kind="gemm", dtype="bf16", m=8, k=32, n=64)),
                             ("inputs", [dict(seed=1, int8_sha256="different")]),
                             ("tensor_layouts", [dict(shape=[16, 32], stride=[1, 16], dtype="bf16")])]:
            with self.subTest(field=field):
                a, b = report(), report()
                b["results"][0][field] = value
                row = compare(a, b)["rows"][0]
                self.assertFalse(row["comparable"])
                self.assertNotIn("candidate_over_baseline", row)

    def test_tf32_mismatch(self):
        a, b = report("fp32"), report("fp32")
        b["tf32"] = True
        self.assertFalse(compare(a, b)["rows"][0]["comparable"])

    def test_fp8_scale_mismatch(self):
        a, b = report("fp8"), report("fp8")
        b["fp8_contract"] = "rowwise scales"
        self.assertFalse(compare(a, b)["rows"][0]["comparable"])

    def test_quantized_attention_mismatch(self):
        a, b = report(kind="attention"), report(kind="attention")
        b["results"][0].update(variant="sage2", attention_arithmetic="quantized-sage2")
        self.assertFalse(compare(a, b, candidate_attention="sage2")["rows"][0]["comparable"])

    def test_failed_and_missing_results(self):
        a = report()
        for change in ("failed", "missing", "nonfinite"):
            b = copy.deepcopy(a)
            if change == "missing":
                b["results"] = []
            elif change == "failed":
                b["results"][0]["status"] = "failed"
            else:
                b["results"][0]["finite"] = False
            self.assertFalse(compare(a, b)["rows"][0]["comparable"])


if __name__ == "__main__":
    unittest.main()
