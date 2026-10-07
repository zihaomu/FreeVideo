#!/usr/bin/env python3
"""Decode every video/audio frame and retain objective checks and a contact sheet."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--width',type=int,required=True)
    parser.add_argument('--height',type=int,required=True)
    parser.add_argument('--frames',type=int,required=True)
    args=parser.parse_args()
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    report=dict(file=str(args.video),expected=dict(width=args.width,height=args.height,frames=args.frames,fps=24),
                frames=[],success=False,quality_scope='Objective decode checks; semantic/visual/audio review remains separate')
    sheet=Image.new('RGB',(4*336,3*212),'#222222')
    draw=ImageDraw.Draw(sheet)
    indices={round(i*(args.frames-1)/11):i for i in range(12)}
    previous=None
    with av.open(str(args.video)) as media:
        stream=media.streams.video[0]
        assert (stream.width,stream.height)==(args.width,args.height)
        assert float(stream.average_rate)==24
        report['video_codec']=stream.codec_context.name
        for number,frame in enumerate(media.decode(video=0)):
            rgb=frame.to_ndarray(format='rgb24')
            assert rgb.shape==(args.height,args.width,3)
            # Thumbnail motion statistics avoid allocating full float frames.
            thumb=np.asarray(frame.to_image().resize((168,96))).astype(np.float32)
            row=dict(index=number,mean=float(thumb.mean()),std=float(thumb.std()),
                     min=int(rgb.min()),max=int(rgb.max()),sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                     mean_abs_change=None if previous is None else float(np.abs(thumb-previous).mean()))
            report['frames'].append(row)
            previous=thumb
            if number in indices:
                i=indices[number]
                tile=frame.to_image();tile.thumbnail((336,189))
                sheet.paste(tile,((i%4)*336,(i//4)*212))
                draw.text(((i%4)*336+5,(i//4)*212+191),f'frame {number}',fill='white')
    assert len(report['frames'])==args.frames
    audio_samples=0
    audio_sum=0.
    audio_peak=0.
    with av.open(str(args.video)) as media:
        assert len(media.streams.audio)==1
        stream=media.streams.audio[0]
        rate=stream.codec_context.sample_rate
        channels=stream.codec_context.channels
        assert channels==2 and rate==32000
        for frame in media.decode(audio=0):
            value=frame.to_ndarray().astype(np.float64)
            assert np.isfinite(value).all()
            if np.issubdtype(frame.to_ndarray().dtype,np.integer):
                value/=np.iinfo(frame.to_ndarray().dtype).max
            audio_sum+=float((value*value).sum())
            audio_peak=max(audio_peak,float(np.abs(value).max()))
            audio_samples+=frame.samples
        audio_seconds=audio_samples/rate
        assert abs(audio_seconds-args.frames/24)<=1/24+.001
        report['audio']=dict(codec=stream.codec_context.name,channels=channels,sample_rate=rate,
                             samples=audio_samples,seconds=audio_seconds,finite=True,
                             rms=(audio_sum/max(1,audio_samples*channels))**.5,peak=audio_peak)
    report['black_frames']=[r['index'] for r in report['frames'] if r['mean']<1 and r['std']<1]
    report['exact_repeated_frames']=[r['index'] for r,p in zip(report['frames'][1:],report['frames']) if r['sha256']==p['sha256']]
    report['video_seconds']=args.frames/24
    import torch
    retained=args.video.with_suffix('.artifacts')
    raw_audio=np.load(retained/'audio.npy',allow_pickle=False)
    report['raw_audio']=dict(finite=bool(np.isfinite(raw_audio).all()),
        rms=float(np.sqrt(np.mean(raw_audio*raw_audio))),peak=float(np.abs(raw_audio).max()),
        saturation_fraction=float(np.mean(np.abs(raw_audio)>=.999)))
    latents=torch.load(retained/'latents.pt',map_location='cpu',weights_only=True)
    conditioning=torch.load(retained/'conditioning.pt',map_location='cpu',weights_only=True)
    def finite_tensors(value):
        if isinstance(value,torch.Tensor):
            assert bool(value.isfinite().all())
            return 1
        if isinstance(value,dict):
            return sum(finite_tensors(item) for item in value.values())
        if isinstance(value,(list,tuple)):
            return sum(finite_tensors(item) for item in value)
        return 0
    report['finite_latent_tensors']=finite_tensors(latents)
    report['finite_conditioning_tensors']=finite_tensors(conditioning)
    assert report['finite_latent_tensors']>=2 and report['finite_conditioning_tensors']>=1
    report['success']=report['raw_audio']['finite'] and report['raw_audio']['saturation_fraction']<.05
    report['audio_saturation_limit']=.05
    report['contact_sheet']=str(args.video.with_suffix('.contact.png'))
    sheet.save(report['contact_sheet'])
    args.video.with_suffix('.verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'}),flush=True)
    return not report['success']


if __name__=='__main__':
    raise SystemExit(main())
