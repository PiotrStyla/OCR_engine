"""GPU evaluation only: baseline vs first adapter-merged model, starts 2 and 0.

Public, pinned artifacts. No training or Hub upload. Requires explicit CUDA unless
--device cpu is requested. All variants share the same decoding policy.
"""
import argparse
import gc
import hashlib
import importlib.metadata
from pathlib import Path
import time

from .protocol import configure_generation, pair_manifest, write_json

FIRST_MODEL = 'PiotrSty/trocr-pl-base'
FIRST_REVISION = '0d42a14958063e4e0e94b02f6dce361337198c9f'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--base-revision', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = ap.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        ap.error('Positive batch size and limit required')
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, set_seed
    from jiwer import cer, wer
    from .dataset import load_pairs
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error('This evaluation requires a GPU; use the Kaggle reevaluation notebook')
    records = pair_manifest(args.data)
    samples = load_pairs(args.data)
    if args.limit:
        records, samples = records[:args.limit], samples[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    write_json(args.output/'inputs.json', records)
    write_json(args.output/'environment.json', {n:importlib.metadata.version(n) for n in ['torch','transformers','jiwer']})
    summaries = {}
    for model_id, revision, variants in [
        ('microsoft/trocr-base-printed', args.base_revision, [('baseline-start2',2)]),
        (FIRST_MODEL, FIRST_REVISION, [('first-start2',2),('first-start0',0)]),
    ]:
        processor = TrOCRProcessor.from_pretrained(model_id,revision=revision)
        model = VisionEncoderDecoderModel.from_pretrained(model_id,revision=revision).to(args.device).eval()
        for name, start in variants:
            generation = configure_generation(model,processor.tokenizer,start)
            rows = []
            began = time.perf_counter()
            for offset in range(0,len(samples),args.batch_size):
                batch = samples[offset:offset+args.batch_size]
                pixels = processor([s.image for s in batch],return_tensors='pt').pixel_values.to(args.device)
                with torch.inference_mode():
                    ids = model.generate(pixels)
                texts = processor.batch_decode(ids,skip_special_tokens=True)
                for index, (sample,text,tokens) in enumerate(zip(batch,texts,ids)):
                    record = records[offset+index]
                    # Keep incomplete text and flag it; never silently exclude it from CER.
                    truncated = processor.tokenizer.sep_token_id not in tokens[1:].tolist()
                    rows.append(dict(**record,reference=sample.text,text=text,truncated=truncated))
            write_json(args.output/f'{name}-predictions.json',rows)
            refs,hyps = [r['reference'] for r in rows],[r['text'] for r in rows]
            summaries[name] = dict(model=model_id,revision=revision,generation=generation,
                n_lines=len(rows),cer=cer(refs,hyps),wer=wer(refs,hyps),
                truncated=sum(r['truncated'] for r in rows),elapsed_seconds=time.perf_counter()-began)
            write_json(args.output/'summary.json',summaries)
            print(f'{name}: CER={summaries[name]["cer"]:.6f}, WER={summaries[name]["wer"]:.6f}',flush=True)
        del model,processor
        gc.collect()
        if args.device=='cuda':
            torch.cuda.empty_cache()
    write_json(args.output/'checksums.json', {f.name:hashlib.sha256(f.read_bytes()).hexdigest()
        for f in args.output.iterdir() if f.is_file()})


if __name__ == '__main__':
    main()
