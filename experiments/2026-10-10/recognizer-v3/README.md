# Recognizer v3 — guarded training (2026-10-10)

Eksperyment treningu LoRA na TrOCR (`PiotrSty/trocr-pl-mixed-v3`) z pakietem
**226 par** replay, w kontrakcie identycznym z recognizer V2: pary wariantów,
chronione domeny rozwojowe, bramki selekcji z fallbackiem do niezmienionego
baseline'u. **Bez promocji produkcyjnej i bez twierdzenia SOTA.**

- Config (zamrożony): `config.json` — piny pakietów HF, warianty, receptura,
  polityka selekcji (bez zmian względem V2)
- Runner: `training/runpod_recognizer_v3.py` (importuje helpery V2: `evaluate`,
  `preflight`, `select`, `summarize`, `package_model`, `run_logged`)
- Dane: `corpus/train` (70 recenzowanych, ×4 powtórzenia) + `reviewed-v3`
  (226 par: 127 z minera + 99 z puli) + replay syntetyczny (500/2000); probe
  (65) trzymany poza treningiem
- Bramki: historyczne 9 linii — poprawa; zwykłe 75 — bez regresji; łączny CER
  — poprawa; zastępstwa — bez wzrostu; fallback: baseline
- Cel: historyczne <20% na liniach → test A <17% CER

## Uruchomienie (Runpod, GPU T4/4090)

Pod z szablonu `runpod-torch-v280` (torch w obrazie), start command:

```bash
git clone https://github.com/PiotrStyla/OCR_engine.git /workspace/OCR_engine &&
git -C /workspace/OCR_engine checkout --detach <PIN> &&
python3 -m pip install -q transformers==4.57.6 torchao==0.17.0 peft==0.19.1 \
  accelerate==1.13.0 jiwer==4.0.0 huggingface_hub==0.36.2 &&
mkdir -p /workspace/out && (cd /workspace/out && nohup python3 -m http.server 8080 &) &&
python3 -u /workspace/OCR_engine/training/runpod_recognizer_v3.py
```

Dowody (logi wariantów, metryki baseline i kandydatów, selection.json, ZIP
modelu) trafiają do `/workspace/recognizer-v3-*/`; `*-evidence.zip` i
`recognizer-v3-model.zip` pobierane przez `https://<pod>-8080.proxy.runpod.net/`.
Szacunek: 3 warianty × ~300 par × 3 epoki na 4090 ≈ 1–2 h ≈ $1–2.

## Otwarte bramki przed promocją

drugi recenzent kandydata referencji; 5 rzadkich PUA; U+FFFD; errata;
klasyfikacja treści; 10 flagowanych wycinków puli.
