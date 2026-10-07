#!/usr/bin/env python3
"""Supplement decoded-audio checks with pinned CLAP inference on CPU only."""
import argparse
import json
from pathlib import Path
import time

LABELS=[
    'The sound of a toy car rolling quietly on a wooden table.',
    'The sound of wheels rolling softly with a faint rattle.',
    'The sound of footsteps on a path with birds singing in a park.',
    'The sound of a tennis ball bouncing on a hard court and rolling.',
    'The sound of a crowd in an outdoor market and a bicycle bell.',
    'The sound of a loud car engine revving.',
    'The sound of a person speaking clearly.',
    'The sound of pop music playing.',
    'The sound of wind blowing.',
    'The sound of birds singing.',
    'The sound of footsteps.',
    'The sound of a ball bouncing.',
    'The sound of a bicycle bell ringing.',
    'The sound of a busy crowd talking.',
    'The sound of rain falling.',
    'The sound of running water.',
    'The sound of static noise.',
    'The sound of silence.',
    'The sound of glass breaking.',
    'The sound of a dog barking.',
    'The sound of an air conditioner humming.',
]
EXPECTED={'car':[0,1],'person-motion':[2,9,10],'object-motion':[3,11],
          'complex-scene':[4,12,13]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('/data/models/quality-audio/clap-htsat-unfused'))
    parser.add_argument('--video',type=Path,action='append')
    parser.add_argument('--scan',type=Path,help='Scan completed, objectively verified video outputs')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from transformers import ClapModel,ClapProcessor
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    started=time.perf_counter()
    # The pinned repository has a PyTorch state dict, not safetensors. Explicit
    # weights_only loading avoids evaluating checkpoint object constructors.
    model=ClapModel.from_pretrained(str(args.model),local_files_only=True,
        use_safetensors=False,weights_only=True,dtype=torch.float32).cpu().eval()
    processor=ClapProcessor.from_pretrained(str(args.model),local_files_only=True)
    assert all(p.device.type=='cpu' for p in model.parameters())
    text=processor(text=LABELS,return_tensors='pt',padding=True)
    report=dict(model='laion/clap-htsat-unfused',revision='8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a',
        device='cpu',threads=2,torch=str(torch.__version__),transformers=__import__('transformers').__version__,
        scope='Supplemental zero-shot audio/text similarity; scores depend on candidate labels and are not calibrated human quality ratings.',
        processor_seed_per_file=0,
        processor_audio_scope='Pinned CLAP extractor; seed reset per file so identical waveforms receive the same truncation when longer than the model limit.',
        labels=LABELS,model_load_seconds=time.perf_counter()-started,runs=[])
    videos=list(args.video or [])
    if args.scan:
        for path in sorted(args.scan.rglob('video.mp4')):
            verify=path.with_suffix('.verification.json')
            if not verify.exists():continue
            result=json.loads(verify.read_text())
            raw=result.get('raw_audio') or {}
            if result.get('success') and raw.get('finite') and raw.get('saturation_fraction',1)<.05:
                videos.append(path)
    for video in dict.fromkeys(videos):
        source=video.with_suffix('.artifacts')/'audio.npy'
        waveform=np.load(source)
        if waveform.ndim==2:
            if waveform.shape[0]==2:
                waveform=waveform.mean(axis=0)
            else:
                assert waveform.shape[1]==2
                waveform=waveform.mean(axis=1)
        assert waveform.ndim==1 and np.isfinite(waveform).all()
        audio=resample_poly(waveform,3,2).astype(np.float32)
        np.random.seed(0)
        features=processor(audio=audio,sampling_rate=48000,return_tensors='pt')
        tick=time.perf_counter()
        result=model(**text,**features)
        similarity=(result.audio_embeds @ result.text_embeds.T)[0].float().numpy()
        probabilities=result.logits_per_audio[0].float().softmax(-1).numpy()
        assert np.isfinite(similarity).all()
        order=np.argsort(-similarity)
        family=next((name for name in EXPECTED if name!='car' and name in str(video)),'car')
        expected=EXPECTED[family]
        row=dict(video=str(video),source=str(source),family=family,source_rate=32000,model_rate=48000,
            waveform_rms=float(np.sqrt(np.mean(waveform**2))),waveform_seconds=len(waveform)/32000,
            expected_indices=expected,expected_best_rank=min(int(np.where(order==i)[0][0])+1 for i in expected),
            scores=[dict(label=LABELS[int(i)],index=int(i),cosine=float(similarity[i]),
                         candidate_softmax=float(probabilities[i])) for i in order],
            seconds=time.perf_counter()-tick)
        report['runs'].append(row)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.out)
        print(json.dumps(dict(video=str(video),expected_best_rank=row['expected_best_rank'],top=row['scores'][:3])),flush=True)


if __name__=='__main__':
    main()
