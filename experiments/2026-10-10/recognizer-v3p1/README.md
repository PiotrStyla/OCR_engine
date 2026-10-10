# Recognizer v3.1 — guarded training (2026-10-10)

Wariacja miksu danych na harnessie v3 (identyczne piny, bramki, polityka
selekcji i fallback). Zmiany względem v3:

- `training.historical_repeats`: 4 → **8** (waga domenowa dla historii)
- replay syntetyczny: 500/2000 → **200** (podzbiór zagnieżdżony w kolejności
  v3 — porównywalny kontrolnie) + wariant diagnostyczny **replay 0**
  (`low-lr3e6-replay0`), który rozstrzyga, czy sama syntyka powodowała
  regresję zwykłego druku w v3
- warianty: `control-lr1e5-replay200`, `low-lr3e6-replay200`,
  `low-lr3e6-replay0`

Cel: utrzymać poprawę na historycznych (v3: 33,0% → 31,7%) bez regresji
zwykłych (v3: 5,33% → 5,53–5,86%). Bramki i fallback do niezmienionego
baseline'u bez zmian. Bez promocji i bez twierdzenia SOTA.

## Uruchomienie (Runpod, GPU T4/4090)

```bash
if [ -f /workspace/V3P1_DONE ]; then sleep infinity; fi
(cd /workspace && nohup python3 -m http.server 8080 >/tmp/http.log 2>&1 &) &&
for i in $(seq 1 20); do rm -rf /workspace/OCR_engine &&
  git clone https://github.com/PiotrStyla/OCR_engine.git /workspace/OCR_engine && break;
  echo BOOT_DNS_RETRY $i; sleep 6; done &&
git -C /workspace/OCR_engine checkout --detach <PIN> &&
python3 -m pip install -q hf_transfer transformers==4.57.6 torchao==0.17.0 peft==0.19.1 \
  accelerate==1.13.0 jiwer==4.0.0 huggingface_hub==0.36.2 &&
cd /workspace/OCR_engine &&
python3 -u -m training.runpod_recognizer_v3 \
  --config experiments/2026-10-10/recognizer-v3p1/config.json &&
touch /workspace/V3P1_DONE && sleep infinity
```

Uwagi z przebiegu v3 (uwzględnione wyżej): pętla retry musi czyścić niepusty
katalog po przerwanym clone; runner uruchamiany przez `-m`; szablon wymaga
pakietu `hf_transfer` (`HF_HUB_ENABLE_HF_TRANSFER=1`); marker `V3P1_DONE`
zapobiega ponowieniu treningu po restarcie kontenera.
