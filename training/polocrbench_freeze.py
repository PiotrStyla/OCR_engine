"""Freeze the IMPACT-Polish v2 assets into the PolOCRBench data layout.

PolOCRBench (subtask A: full-page transcription) historical-print subset:
- test A pool  <- frozen IMPACT-Polish v2 test (36 pages, 52,291 GT chars,
                 split inherited from impact-psnc-polish-ocr v1, collection-level)
- train pool   <- IMPACT-Polish v2 train+val region records (2,531 regions
                 over 27 train/val collections; page images referenced, not copied)

This script only copies and relabels frozen artifacts; it never re-splits.
Outputs (under --outdir, e.g. benchmarks/polocrbench):
  history_testA_manifest.jsonl   transcription_eval format, subset=history_print
  history_train_pool.jsonl       region records, subset=history_print
  MANIFEST.sha256                checksums of the two jsonl files
  README.md                      provenance, license, metric reference

Usage:
  python -m training.polocrbench_freeze --bench-dir benchmarks/impact-print-v2 \
      --corpus-dir data/impact-corpus --outdir benchmarks/polocrbench
"""
import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSET = 'history_print'


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-dir', required=True, help='impact-print-v2 outputs (test_manifest.jsonl, ...)')
    parser.add_argument('--corpus-dir', required=True, help='impact-corpus with page images (resolved paths kept)')
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    bench = Path(args.bench_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # test A pool: verbatim copy of the frozen test manifest, relabeled
    src_test = bench / 'test_manifest.jsonl'
    assert src_test.exists(), f'missing {src_test} - run training.impact_split first'
    test_rows = [json.loads(l) for l in src_test.read_text(encoding='utf-8').splitlines() if l.strip()]
    with (outdir / 'history_testA_manifest.jsonl').open('w', encoding='utf-8') as fh:
        for row in test_rows:
            fh.write(json.dumps({**row, 'subset': SUBSET}, ensure_ascii=False) + '\n')

    # train pool: train + val region records
    pool = []
    for name in ('train_regions.jsonl', 'val_regions.jsonl'):
        src = bench / name
        assert src.exists(), f'missing {src}'
        split = 'train' if 'train' in name else 'val'
        for line in src.read_text(encoding='utf-8').splitlines():
            if line.strip():
                pool.append({**json.loads(line), 'subset': SUBSET, 'split': split})
    with (outdir / 'history_train_pool.jsonl').open('w', encoding='utf-8') as fh:
        for row in pool:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')

    # checksums + readme
    with (outdir / 'MANIFEST.sha256').open('w', encoding='utf-8') as fh:
        for name in ('history_testA_manifest.jsonl', 'history_train_pool.jsonl'):
            fh.write(f'{sha256_file(outdir / name)}  {name}\n')

    cols_test = sorted({r['id'].split('__')[0] for r in test_rows})
    cols_pool = sorted({r['id'].split('__')[0] for r in pool})
    (outdir / 'README.md').write_text(f"""# PolOCRBench — historyczno-drukowy podzbiór (subset: {SUBSET})

Zamrożony wkład IMPACT-Polish v2 do subtasku A (transkrypcja pełnostronicowa).

## Test A pool: {len(test_rows)} stron, {sum(len(r['text']) for r in test_rows)} znaków GT
Kolekcje: {', '.join(cols_test)}
Podział odziedziczony z IMPACT-Polish v1 (per kolekcja); strony testowe nigdy nie
występują w puli treningowej (weryfikacja: impact_split + impact_freeze).

## Train pool: {len(pool)} regionów z {len(cols_pool)} kolekcji
Rekordy regionów (tekst + bbox jeśli dostępny); obrazy stron w `impact-corpus`.

## Provenance
- Źródło: IMPACT ground truth (PSNC), github.com/impactcentre/groundtruth-pol
- Obrazy: dLibra ribes-80.man.poznan.pl (TLS certyfikat wygasł; integralność: SHA-256)
- Licencja: CC-BY-3.0
- Zamrożono: {date.today().isoformat()}

## Metryka
`training/transcription_eval.py` (CER/WER micro + struktura Markdown; NFC,
cudzysłowy typograficzne -> proste, białe znaki; wielkość liter i diakrytyki zachowane).
""", encoding='utf-8')

    print(f'testA: {len(test_rows)} stron | train pool: {len(pool)} regionow | -> {outdir}')
    print('Zamrozone. Nie zmieniaj plikow po tym kroku.')


if __name__ == '__main__':
    main()
