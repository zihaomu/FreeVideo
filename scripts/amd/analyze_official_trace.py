#!/usr/bin/env python3
"""Account GPU kernels once, correlate CPU semantics, and retain overlap separately."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import sys
import time


def trace_events(path):
    """Stream Chrome's traceEvents array; large traces need no giant JSON tree."""
    opener=gzip.open if str(path).endswith('.gz') else open
    decoder=json.JSONDecoder()
    with opener(path,'rt',encoding='utf-8') as stream:
        buffer=''
        while '"traceEvents"' not in buffer:
            chunk=stream.read(1024*1024)
            if not chunk: raise ValueError('Missing traceEvents: '+str(path))
            buffer+=chunk
        buffer=buffer.split('"traceEvents"',1)[1]
        buffer=buffer[buffer.index('[')+1:]
        position=0
        while True:
            while position<len(buffer) and buffer[position] in ' \t\n\r,':position+=1
            if position<len(buffer) and buffer[position]==']':return
            try:
                value,end=decoder.raw_decode(buffer,position)
            except json.JSONDecodeError:
                buffer=buffer[position:]
                position=0
                chunk=stream.read(1024*1024)
                if not chunk:raise ValueError('Truncated traceEvents: '+str(path))
                buffer+=chunk
                continue
            yield value
            position=end
            if position>1024*1024:
                buffer=buffer[position:];position=0


def merge_intervals(values):
    merged=[]
    for start,end in sorted(values):
        if not merged or start>merged[-1][1]:merged.append([start,end])
        else:merged[-1][1]=max(end,merged[-1][1])
    return merged


def union_us(values):
    return sum(end-start for start,end in merge_intervals(values))


def kind(name,op,scope,flags):
    low=name.lower()
    attention=('flash' in low or 'fmha' in low or 'attn_fwd' in low or 'scaled_dot_product' in low or 'scaled_dot_product' in op)
    if name=='_flash.kd' or ('fv::softmax::window_attention' in scope and attention):return 'window_attention'
    if attention:return 'other_attention'
    if '_depthwise_5x5' in low:return 'spatial_depthwise_conv'
    if '_tconv_act_kernel' in low:return 'temporal_conv'
    if flags&1:
        return 'temporal_conv' if 'fv::linear_state' in scope else 'convolution'
    if flags&2 or any(x in low for x in ('potrf','trsm','trtri','cholesky')):return 'cholesky_triangular_solve'
    gemm=('_scaled_mm' in op or any(x in low for x in ('gemm','tensile','cijk_','matmul')))
    if gemm:
        if '_scaled_mm' in op or re.search(r'(?:^|_)f8(?:n|h|b|_|$)',low):return 'fp8_gemm'
        if 'fv::linear_state::frame_statistics' in scope:return 'state_statistics_gemm'
        if 'fv::linear_state::_run_scans_inference' in scope:return 'state_scan_gemm'
        if 'fv::linear_state::_readout_inference' in scope:return 'state_readout_other_gemm'
        if 'fv::attention::' in scope:return 'bf16_gate_other_projection'
        return 'other_gemm'
    if any(x in low for x in ('amax','quantiz','_absmax_kernel','_cast_scaled_kernel','_cast_with_scale','_swiglu_rowmax_kernel')):return 'quantization_reduction'
    if any(x in low for x in ('rms_norm','layer_norm','layernorm','rmsnorm')):return 'normalization_fused'
    if 'index' in low or 'gather' in low:return 'index_gather_scatter'
    if any(x in low for x in ('copy','cast','convert')):return 'copy_cast'
    if 'fv::linear_state::_run_scans_inference' in scope:return 'state_scan_other'
    if 'fv::linear_state::frame_statistics' in scope:return 'state_statistics_other'
    return 'other_elementwise_kernels'


def analyze(path):
    cpu=defaultdict(list);gpu=[];annotations=0
    for ev in trace_events(path):
        if ev.get('ph')!='X' or ev.get('dur',0)<=0:continue
        cat=ev.get('cat','');arg=ev.get('args',{});start=ev['ts'];end=start+ev['dur']
        name=sys.intern(ev.get('name',''));external=arg.get('External id')
        if cat in ('cpu_op','user_annotation','cuda_runtime','hip_runtime'):
            cpu[(ev.get('pid'),ev.get('tid'))].append((start,end,external,name,arg.get('correlation'),cat))
        elif cat in ('kernel','gpu_memcpy','gpu_memset'):
            gpu.append((start,end,name,external,cat,arg.get('stream'),arg.get('device'),arg.get('correlation')))
        elif cat=='gpu_user_annotation':annotations+=1
    # External dispatch IDs connect asynchronous device events to the CPU call.
    # GPU annotation ranges overlap kernels and are deliberately excluded.
    index={};runtime_index={};scope_counts=defaultdict(int)
    for rows in cpu.values():
        stack=[]
        for start,end,external,name,correlation,cat in sorted(rows,key=lambda x:(x[0],-x[1])):
            while stack and start>=stack[-1][0]:stack.pop()
            # An independent sibling can overlap a bookkeeping range; retain
            # only true containing parents for semantic attribution.
            parents=[x for x in stack if x[0]>=end]
            labels=[x[1] for x in parents if x[1].startswith('fv::')]
            flags=0
            for parent in parents:flags|=parent[2]
            if 'convolution' in name or name in ('aten::conv1d','aten::conv2d','aten::conv3d'):flags|=1
            if any(x in name for x in ('linalg_cholesky','linalg_solve_triangular','cholesky_solve')):flags|=2
            if name.startswith('fv::'):
                labels.append(name);scope_counts[name]+=1
            scope=sys.intern('/'.join(labels))
            if cat in ('cuda_runtime','hip_runtime'):
                if correlation is not None:
                    op=next((x[1] for x in reversed(parents) if not x[1].startswith('fv::')),name)
                    runtime_index[correlation]=(op,scope,flags)
            elif external is not None:index[external]=(name,scope,flags)
            stack.append((end,name,flags))
    del cpu
    groups=defaultdict(lambda:[0,0.]);ops=defaultdict(lambda:[0,0.]);kernels=defaultdict(lambda:[0,0.]);semantic=defaultdict(lambda:[0,0.]);fp8_sites=defaultdict(lambda:[0,0.])
    transfers=defaultdict(lambda:[0,0.]);kernel_intervals=[];copy_intervals=[];matched=0;runtime_matched=0;unmatched_us=0.;streams=defaultdict(list)
    for start,end,name,external,cat,stream,device,correlation in gpu:
        duration=end-start
        if cat=='kernel':
            metadata=index.get(external)
            if metadata is None:
                metadata=runtime_index.get(correlation)
                runtime_matched+=metadata is not None
            op,scope,flags=metadata or ('uncorrelated','',0)
            matched+=metadata is not None
            if metadata is None:unmatched_us+=duration
            category=kind(name,op,scope,flags)
            kernel_intervals.append((start,end));streams[(device,stream)].append((start,end))
            bucket=next((x for x in reversed(scope.split('/')) if x.startswith('fv::') and not x.startswith('fv::phase::')),'unscoped')
            for table,key in ((groups,category),(ops,op),(kernels,name),(semantic,bucket)):
                table[key][0]+=1;table[key][1]+=duration
            if category=='fp8_gemm':
                site=('feed_forward' if 'fv::feed_forward::' in scope else
                      'attention_projections' if 'fv::attention::' in scope else 'other')
                fp8_sites[site][0]+=1;fp8_sites[site][1]+=duration
        else:
            copy_intervals.append((start,end));transfers[name][0]+=1;transfers[name][1]+=duration
    def table(rows):
        return [dict(name=k,count=n,gpu_seconds=us/1e6) for k,(n,us) in sorted(rows.items(),key=lambda x:x[1][1],reverse=True)]
    kernel_us=sum(end-start for start,end in kernel_intervals)
    assert abs(sum(x[1] for x in groups.values())-kernel_us)<max(1.,kernel_us*1e-8)
    assert abs(sum(x[1] for x in semantic.values())-kernel_us)<max(1.,kernel_us*1e-8)
    compute_busy=union_us(kernel_intervals);copy_busy=union_us(copy_intervals)
    busy=union_us(kernel_intervals+copy_intervals)
    first=min((x[0] for x in gpu),default=0);last=max((x[1] for x in gpu),default=first)
    window=last-first
    return dict(path=str(path),trace_bytes=Path(path).stat().st_size,kernel_count=len(kernel_intervals),
        gpu_kernel_sum_seconds=kernel_us/1e6,gpu_compute_busy_seconds=compute_busy/1e6,
        gpu_copy_busy_seconds=copy_busy/1e6,gpu_busy_seconds=busy/1e6,
        exposed_copy_seconds=(busy-compute_busy)/1e6,device_window_seconds=window/1e6,
        idle_between_device_events_seconds=max(0.,window-busy)/1e6,
        overlap_kernel_seconds=(kernel_us-compute_busy)/1e6,
        correlation_match_fraction=matched/max(1,len(kernel_intervals)),uncorrelated_kernel_seconds=unmatched_us/1e6,
        runtime_correlation_fallback_count=runtime_matched,
        excluded_gpu_annotations=annotations,scope_counts=dict(scope_counts),
        categories=table(groups),cpu_dispatch_ops=table(ops),top_kernels=table(kernels)[:100],
        semantic_scopes=table(semantic),transfers=table(transfers),
        fp8_sites=table(fp8_sites),
        per_stream_busy_seconds={str(k):union_us(v)/1e6 for k,v in streams.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-dirs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    report=dict(scope='Actual GPU kernel duration sums from a same-configuration profiled rerun. Not the original unprofiled 385.77-second request; GPU busy union, copies and idle are separate.',phases=[])
    for directory in args.profile_dirs:
        meta=json.loads((directory/'profile.json').read_text())
        if meta['state']!='complete':raise ValueError('Incomplete profiler run: '+str(directory))
        for phase in meta['phases']:
            print(json.dumps(dict(event='analyzing',phase=phase['name'],path=str(directory/phase['trace']))),flush=True)
            tick=time.perf_counter()
            row=analyze(directory/phase['trace']);row.update(phase=phase['name'],profile_wall_seconds=phase['profile_wall_seconds'])
            report['phases'].append(row)
            print(json.dumps(dict(event='analyzed',phase=phase['name'],kernel_seconds=row['gpu_kernel_sum_seconds'],kernel_count=row['kernel_count'],seconds=time.perf_counter()-tick)),flush=True)
            args.out.with_suffix('.partial.json').write_text(json.dumps(report,indent=2)+'\n')
    totals=defaultdict(lambda:[0,0.])
    for phase in report['phases']:
        for row in phase['categories']:
            totals[row['name']][0]+=row['count'];totals[row['name']][1]+=row['gpu_seconds']
    report['categories']=[dict(name=name,count=n,gpu_seconds=s) for name,(n,s) in sorted(totals.items(),key=lambda x:x[1][1],reverse=True)]
    report['gpu_kernel_sum_seconds']=sum(x['gpu_kernel_sum_seconds'] for x in report['phases'])
    report['state']='complete'
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
