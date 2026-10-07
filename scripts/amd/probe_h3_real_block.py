#!/usr/bin/env python3
"""Cross-check H3-inspired routes on real first-step blocks of both canvas sizes.

The full canvas uses seeded first-step activations, not original refinement
latents. This is a local speed/numerical gate, not an end-to-end media gate.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--conditioning',type=Path,required=True)
    p.add_argument('--prototype-root',type=Path,required=True)
    p.add_argument('--device-uuid',required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    import triton
    from freevideo_engine.hardware import _detect_local
    from freevideo_engine.locking import runtime_lock,device_lock_path
    from freevideo_engine.geometry import geometry
    from freevideo_engine.runtime import Engine
    from freevideo_engine.attention import WindowAttention
    from freevideo_engine.rocm_attention import _flash
    torch.set_num_threads(8);torch.set_grad_enabled(False)
    assert _detect_local().gpu_uuid==args.device_uuid
    sys.path.insert(0,str(args.prototype_root))
    from indexed_attention import attention_indexed
    path=args.prototype_root/'merged_softmax_head_chunk.py'
    spec=importlib.util.spec_from_file_location('freevideo_engine.h3_merged_softmax_probe',path)
    merged=importlib.util.module_from_spec(spec);spec.loader.exec_module(merged)
    report=dict(state='running',scope=__doc__,hardware_uuid=args.device_uuid,
        torch=str(torch.__version__),hip=torch.version.hip,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        conditioning_sha256=hashlib.sha256(args.conditioning.read_bytes()).hexdigest(),
        prototype_sha256={name:hashlib.sha256((args.prototype_root/name).read_bytes()).hexdigest()
            for name in ('indexed_attention.py','merged_softmax_head_chunk.py')},phases=[])
    def save():
        (args.out/'real-block.json').write_text(json.dumps(report,indent=2)+'\n')
    def timed(fn):
        b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        b.record();value=fn();e.record();e.synchronize()
        return b.elapsed_time(e),value
    @contextmanager
    def record_failure():
        try:yield
        except BaseException as error:
            report.update(state='failed',error=repr(error));save();raise
    class Finished(Exception):pass
    with record_failure(), ExitStack() as lease:
        lease.enter_context(runtime_lock(shared=True))
        lease.enter_context(runtime_lock(device_lock_path(args.device_uuid),inherit=False))
        for width,height in ((672,384),(1344,768)):
            row=dict(width=width,height=height,state='running',results=[]);report['phases'].append(row);save()
            profile=json.loads(args.profile.read_text())
            engine=Engine('/data/models/prepared/edge-5e6ecc39f472f98c/cache',base='/data/models/vdn/h3-base',
                canvas=geometry(width,height,frames=243),steps=8,
                input_cache_dir=None,**profile['engine'])
            block=engine.transformer.transformer_blocks[0];original=block.forward
            policy_original=WindowAttention.__call__
            batched_original=engine.attention.kernels.batched
            def indexed(self,q,k,v,layout,bounds,scale,anchor_frames='none'):
                plan=self.prepare(layout,bounds,q.device,anchor_frames,batches=True)
                out=torch.empty_like(q)
                if len(plan.dense_q):out[plan.dense_q]=self.dense(q[plan.dense_q],k,v,scale)
                for queries,keys in self.batches:
                    out[queries]=attention_indexed(q,k,v,queries,keys,scale)
                return out
            def global_tuned(leg,q,k,v,scale,**kw):
                if kw.get('window') or q.shape[1]<2048:
                    return batched_original(leg,q,k,v,scale,**kw)
                batch,nq,heads,d=q.shape
                out=torch.empty_like(q)
                _flash[(triton.cdiv(nq,128),batch*heads)](q,k,v,out,nq,k.shape[1],heads,d,
                    *q.stride(),*k.stride(),*v.stride(),scale,
                    BM=128,BN=64,num_warps=8,num_stages=1,matrix_instr_nonkdim=16)
                return out
            def pipeline_tuned(leg,q,k,v,scale,**kw):
                if not kw.get('window') or q.shape[1]<2048:
                    return batched_original(leg,q,k,v,scale,**kw)
                batch,nq,heads,d=q.shape
                out=torch.empty_like(q)
                _flash[(triton.cdiv(nq,128),batch*heads)](q,k,v,out,nq,k.shape[1],heads,d,
                    *q.stride(),*k.stride(),*v.stride(),scale,
                    BM=128,BN=64,num_warps=8,num_stages=2,matrix_instr_nonkdim=16)
                return out
            def wrapped(*inputs,**kw):
                from src.models.hybrid_transform import iter_hybrids
                hybrids=list(iter_hybrids(engine.transformer))
                original_forwards=[h._hybrid_forward for h in hybrids]
                original(*inputs,**kw)
                reference=original(*inputs,**kw)
                def restore():
                    WindowAttention.__call__=policy_original
                    engine.attention.kernels.batched=batched_original
                    for h,f in zip(hybrids,original_forwards):h._hybrid_forward=f
                modes=['indexed','merged_softmax','indexed_merged_softmax','global_triton','window_pipeline2']
                if width==672:modes=modes[:3]
                for mode in modes:
                    def select():
                        restore()
                        if 'indexed' in mode:WindowAttention.__call__=indexed
                        if 'merged_softmax' in mode:
                            merged.install_head_chunks(engine.transformer,engine.attention,16,
                                projection_chunk=1024,parallelism=1)
                        if mode=='global_triton':engine.attention.kernels.batched=global_tuned
                        if mode=='window_pipeline2':engine.attention.kernels.batched=pipeline_tuned
                    select();actual=original(*inputs,**kw)
                    diff=actual.float()-reference.float()
                    item=dict(mode=mode,exact=torch.equal(actual,reference),finite=bool(actual.isfinite().all()),
                        relative_rmse=float((diff.square().mean()/reference.float().square().mean()).sqrt()),
                        max_abs=float(diff.abs().max()))
                    del actual,diff
                    bt,ct=[],[]
                    for cycle in range(2):
                        for arm in ('baseline','candidate','candidate','baseline'):
                            if arm=='baseline':restore()
                            else:select()
                            elapsed,value=timed(lambda:original(*inputs,**kw))
                            (bt if arm=='baseline' else ct).append(elapsed);del value
                    restore()
                    item.update(baseline_ms=bt,candidate_ms=ct,baseline_median_ms=statistics.median(bt),
                        candidate_median_ms=statistics.median(ct),speedup=statistics.median(bt)/statistics.median(ct))
                    row['results'].append(item);save();print(json.dumps(dict(width=width,**item)),flush=True)
                restore();raise Finished()
            block.forward=wrapped
            try:
                engine.sample(str(args.conditioning),2026090901,frames=243,width=width,height=height)
            except Finished:
                row['state']='complete';save()
            finally:
                WindowAttention.__call__=policy_original
                engine.attention.kernels.batched=batched_original
                engine.close()
                torch.cuda.empty_cache()
        report['state']='complete';save()
    return 0


if __name__=='__main__':raise SystemExit(main())
