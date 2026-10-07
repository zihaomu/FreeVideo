#!/usr/bin/env python3
"""Plot original request latency and the independent GPU trace without mixing clocks."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phases',type=Path,required=True)
    p.add_argument('--gpu',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='Output filename stem for SVG and PNG')
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=json.loads(args.phases.read_text());gpu=json.loads(args.gpu.read_text())
    values=[x['seconds'] for x in original['stages'][:4]]
    labels=['Base: 8 steps, 672x384','Latent upscale','Refine: 3 steps, 1344x768','Video VAE']
    values.append(original['request_seconds']-sum(values));labels.append('All other request work')
    order=[2,0,3,4,1]
    fig,(a,b)=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1,1.25]})
    fig.suptitle('FreeVideo on R9700: 1344x768, 10s request, 8+3 steps',fontsize=16,y=.98)
    a.barh(range(5),[values[i] for i in order],color=['#dc5c4d','#4477aa','#44aa99','#aaaabb','#ddaa33'])
    a.set_yticks(range(5),[labels[i] for i in order]);a.invert_yaxis()
    a.set_xlim(0,235);a.set_xlabel('Seconds of original unprofiled request')
    a.set_title(f'Original cold request: {original["request_seconds"]:.2f}s',loc='left')
    for j,i in enumerate(order):a.text(values[i]+3,j,f'{values[i]:.2f}s ({values[i]/original["request_seconds"]*100:.1f}%)',va='center',fontsize=9)
    rows=[next(x for x in gpu['phases'] if x['phase']==phase) for phase in ('base_8_steps','refine_3_steps','video_vae')]
    buckets=[('FP8 GEMM',{'fp8_gemm'},'#4477aa'),('Window attention',{'window_attention'},'#dc5c4d'),
        ('Other attention',{'other_attention'},'#ee9955'),('Other GEMM',{'other_gemm','bf16_gate_other_projection'},'#44aa99'),
        ('Linear states / solves',{'cholesky_triangular_solve','state_scan_gemm','state_readout_other_gemm','state_statistics_gemm','state_statistics_other','state_scan_other'},'#aa77aa'),
        ('Conv',{'convolution','temporal_conv','spatial_depthwise_conv'},'#ddaa33'),
        ('Norm / quant / data movement',{'copy_cast','index_gather_scatter','quantization_reduction','normalization_fused'},'#9999aa'),
        ('Other kernels',{'other_elementwise_kernels'},'#cccccc')]
    left=[0.,0.,0.]
    for title,kinds,color in buckets:
        widths=[sum(x['gpu_seconds'] for x in row['categories'] if x['name'] in kinds) for row in rows]
        b.barh(range(3),widths,left=left,label=title,color=color)
        left=[x+y for x,y in zip(left,widths)]
    for i,row in enumerate(rows):
        assert abs(left[i]-row['gpu_kernel_sum_seconds'])<1e-6
        b.text(left[i]+2,i,f'{left[i]:.2f}s',va='center',fontsize=9)
    b.set_yticks(range(3),['Base, all 8 steps','Refine, all 3 steps','Video VAE']);b.invert_yaxis()
    b.set_xlim(0,195);b.set_xlabel('GPU kernel seconds from profiled replay')
    b.set_title('GPU kernel time: same inputs, identical outputs',loc='left')
    b.legend(loc='upper left',bbox_to_anchor=(0,-.16),ncol=2,frameon=False,fontsize=9)
    for ax in (a,b):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.015,'Separate measurements: GPU sums exclude original cold-start, loading, CPU work and serialization; overlapping GPU ranges are counted once per kernel.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.18,1,.92))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out.with_suffix('.svg'),bbox_inches='tight')
    svg=args.out.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(args.out.with_suffix('.png'),dpi=160,bbox_inches='tight')
    return 0


if __name__=='__main__':raise SystemExit(main())
