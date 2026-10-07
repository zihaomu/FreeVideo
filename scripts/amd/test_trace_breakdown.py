"""Guard asynchronous attribution, overlapping kernels and nested GPU annotations."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze_official_trace import analyze,trace_events


def event(cat,name,start,duration,external):
    return dict(ph='X',cat=cat,name=name,ts=start,dur=duration,pid=1,tid=1,
                args={'External id':external,'stream':1,'device':0})


class TraceAccounting(unittest.TestCase):
    def test_async_dispatch_counts_kernels_once(self):
        # Device execution occurs after CPU scopes have ended. Time enclosure
        # alone would miss the attention attribution, and summing annotation
        # ranges would count each kernel twice.
        values=[event('kernel','_flash.kd',1000,1500,3),
            event('gpu_user_annotation','fv::softmax::window_attention',1000,2500,2),
            event('gpu_memcpy','Memcpy HtoD',3000,500,5),
            event('cpu_op','aten::_scaled_mm',30,20,4),
            event('user_annotation','fv::softmax::window_attention',10,80,2),
            event('cpu_op','aten::attention',20,5,3),
            event('user_annotation','fv::phase::base_8_steps',0,100,1),
            event('kernel','Cijk_F8N_gemm.kd',1500,1500,4)]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'trace.json';p.write_text(json.dumps({'traceEvents':values}))
            r=analyze(p)
        self.assertEqual(r['kernel_count'],2)
        self.assertEqual(r['excluded_gpu_annotations'],1)
        self.assertAlmostEqual(r['gpu_kernel_sum_seconds'],.003)
        self.assertAlmostEqual(r['gpu_compute_busy_seconds'],.002)
        self.assertAlmostEqual(r['exposed_copy_seconds'],.0005)
        self.assertEqual(r['correlation_match_fraction'],1.)
        self.assertEqual({x['name'] for x in r['categories']},{'fp8_gemm','window_attention'})

    def test_nested_convolution_gemm_is_a_convolution(self):
        values=[event('cpu_op','aten::mm',40,10,4),
            event('cpu_op','aten::convolution',30,40,3),
            event('user_annotation','fv::phase::video_vae',0,100,1),
            event('kernel','Tensile_gemm.kd',200,1000,4)]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'trace.json';p.write_text(json.dumps({'traceEvents':values}))
            r=analyze(p)
        self.assertEqual(r['categories'][0]['name'],'convolution')

    def test_direct_hip_launch_recovers_missing_external_id(self):
        launch=event('cuda_runtime','hipModuleLaunchKernel',40,5,0)
        launch['args']['correlation']=77
        kernel=event('kernel','Tensile_gemm.kd',300,1000,999)
        kernel['args']['correlation']=77
        values=[event('user_annotation','fv::phase::base_8_steps',0,100,1),
            event('user_annotation','fv::attention::block_00',20,60,2),launch,kernel]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'trace.json';p.write_text(json.dumps({'traceEvents':values}))
            r=analyze(p)
        self.assertEqual(r['runtime_correlation_fallback_count'],1)
        self.assertEqual(r['correlation_match_fraction'],1.)
        self.assertEqual(r['categories'][0]['name'],'bf16_gate_other_projection')

    def test_large_event_and_truncated_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'trace.json';row={'payload':'x'*(2*1024*1024)}
            p.write_text(json.dumps({'traceEvents':[row,{'name':'last'}]}))
            self.assertEqual(len(list(trace_events(p))),2)
            p.write_text('{"traceEvents":[{"name":"incomplete"}')
            with self.assertRaises(ValueError):list(trace_events(p))


if __name__=='__main__':unittest.main()
