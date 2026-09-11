# Fine-tuning polskiego TrOCR — QLoRA

TrOCR (Microsoft) jest wytrenowany tylko na tekście angielskim. Aby silnik poprawnie
rozpoznawał polskie diakrytyki (ą, ć, ę, ł, ń, ó, ś, ź, ż), należy dotrenować
fine-tune modelu bazowego na polskich liniach tekstu.

## Metoda: QLoRA (wg wskazówek Slayer)

Zgodnie z [metodologią Slayer](https://slayer.fabryka.ai/trening) (SOTA 2026):

- **QLoRA** — 4-bit quantization bazowego modelu + trening tylko adapterów LoRA.
  Pozwala trenować 11B modeli na jednym GPU (8–16 GB VRAM) zamiast full-FT.
- **NEFTune** — dodaje szum do embeddingów, darmowy zysk jakości.
- **Wyższy LR** (2e-4) niż full-FT (5e-5) — standard dla LoRA.

## Wymagania

- **GPU (CUDA)** — QLoRA wymaga bitsandbytes (tylko CUDA). Brak GPU → korekta tekstu
  przez Fabryka API (patrz README głównego projektu).
- Zależności: `pip install -e ".[train]"` (peft, bitsandbytes, datasets, accelerate).

## Format danych

Katalog z parami plików — każdy obraz to **pojedyncza linia tekstu**:

```
data/pl_lines_train/
  0001.png
  0001.txt   -> "Ala ma kota, a kot ma Alę"
  0002.png
  0002.txt   -> "Żółw chodzi po łące"
  ...
```

Źródła danych:
- Własne skany dokumentów + transkrypcje (najlepsze).
- **Syntetyczne** — wbudowany generator linii z polskim tekstem:
  ```bash
  python -m training.generate_synthetic --output ./data/pl_lines_train --count 5000
  ```
  Renderuje linie różnymi czcionkami systemowymi (tylko te z polskimi glifami),
  z augmentacjami: pochylenie, blur, szum, gamma. Korpus: `training/corpus_pl.py`.
- Zbiory publiczne z polskimi dokumentami (np. fragmenty OCR-ów z korektą).

## Uruchomienie

```bash
python -m training.train_trocr_pl \
    --train-dir ./data/pl_lines_train \
    --val-dir   ./data/pl_lines_val \
    --base microsoft/trocr-base-printed \
    --output ./ocr/trocr-pl-base \
    --epochs 10 --batch-size 8 \
    --lora-rank 16 --lora-alpha 32
```

Opcje QLoRA:
- `--no-4bit` — wyłącz 4-bit (pełne LoRA, więcej VRAM)
- `--no-neftune` — wyłącz NEFTune

Po treningu model trafia do `./ocr/trocr-pl-base` (domyślna ścieżka w `OcrConfig.recognizer_pl`).
Silnik automatycznie go użyje, gdy wykryje język polski.

## Ewaluacja

CER/WER można policzyć z `evaluate` + `jiwer` (zależności `[train]`).

## Uwagi

- Im więcej linii (≥ kilka tysięcy) i różnorodniejsze czcionki/krój/pochylenie,
  tym lepszy fine-tune.
- `trocr-base` (334M) to dobry kompromis; dla trudnych przypadków rozważ `trocr-large`.
- Jeśli brak GPU, rozważ alternatywę: włącz korektę tekstu przez Fabryka API (Bielik)
  — patrz README głównego projektu. Albo PaddleOCR-VL + LoRA RysOCR.
