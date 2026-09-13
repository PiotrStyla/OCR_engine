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

  # z dodatkowymi zdaniami z polskiej Wikipedii (~30% próbek):
  python -m training.generate_synthetic --output ./data/pl_lines_train \
      --count 5000 --wiki-sentences 1000 --wiki-cache ./data/wiki_sentences.txt
  ```
  Renderuje linie różnymi czcionkami systemowymi (tylko te z polskimi glifami),
  z augmentacjami: pochylenie, blur, szum, gamma.
  Korpus (`training/corpus_pl.py`): zdania potoczne i urzędowe, faktury
  (`Faktura VAT nr ...`, kwoty, NIP/KRS/REGON), umowy i język prawniczy,
  formuły medyczne, daty, adresy, imiona i nazwiska, telefony, e-maile.
  Opcja `--wiki-sentences` pobiera losowe zdania z pl.wikipedia.org
  (`training/wiki_corpus.py`) — filtrowane pod kątem długości i jakości,
  z preferencją zdań z diakrytykami. `--wiki-cache` zapisuje pobrane zdania
  do pliku, więc kolejne generowania nie wymagają sieci.
- Zbiory publiczne z polskimi dokumentami (np. fragmenty OCR-ów z korektą).

## Hugging Face + Kaggle (darmowe GPU)

Gotowy zbiór i notebook:

- **Dataset:** [PiotrSty/ocr-pl-lines](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines)
  — 2000 par train + 200 val, wygenerowane `generate_synthetic` (seed 42/123).
  Notebook pobiera pojedyncze archiwum `ocr-pl-lines-v1.tar.gz`, aby nie przekraczać
  limitu żądań HF i nie wymagać sekretu `HF_TOKEN` dla publicznych danych.
- **Model:** [PiotrSty/trocr-pl-base](https://huggingface.co/PiotrSty/trocr-pl-base)
  — docelowy repo; po treningu `OcrConfig(recognizer_pl="PiotrSty/trocr-pl-base")`
  pobierze go automatycznie przez HF Hub.
- **Notebook:** `training/kaggle_trocr_pl.ipynb` — pełny pipeline na darmowym
  GPU Kaggle (klon repo → dataset z HF → LoRA → ewaluacja → lokalny eksport).
  Wymaga `HF_TOKEN` i GPU T4; bez automatycznej publikacji modelu.
  Gdy Kaggle Secrets nie działa, dostępna jest opisana w
  [instrukcji naprawy](../docs/TRAINING_RECOVERY.md) metoda interaktywna z `getpass`.

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
Zapisywane są: **scalony pełny model** (od razu ładowany przez silnik) oraz adaptery
LoRA w podkatalogu `adapter/` (do dalszego treningu / inspekcji).
Silnik automatycznie go użyje, gdy wykryje język polski.

Trening używa `AlignedSeq2SeqTrainer`: wejście dekodera jest przesuwane raz przez
`prepare_decoder_input_ids_from_labels`, natomiast strata porównuje logits z etykietami
na tych samych pozycjach. Zapobiega to podwójnemu przesunięciu etykiet przez domyślną
`ForCausalLMLoss` w Transformers 4.57.x. Adaptery obejmują `q_proj`, `k_proj`, `v_proj`,
`out_proj`, `fc1` i `fc2` dekodera TrOCR.

## Ewaluacja

CER/WER liczone przez `jiwer` (zależność `[train]`):

```bash
# baseline (EN, przed fine-tunem):
python -m training.evaluate --data ./data/pl_lines_val --model microsoft/trocr-base-printed

# po fine-tunie:
python -m training.evaluate --data ./data/pl_lines_val --model ./ocr/trocr-pl-base

# z korektą Bielik (wymaga FABRYKA_API_KEY):
python -m training.evaluate --data ./data/pl_lines_val --model microsoft/trocr-base-printed --correct
```

Wynik dla syntetycznych danych PL (12 próbek, CPU): EN baseline CER ~55%.
Korekta Bielik **pogarsza** CER przy tak zepsutym wejściu — działa dobrze tylko
na lekko zniekształconym tekście.

## Uwagi

- Im więcej linii (≥ kilka tysięcy) i różnorodniejsze czcionki/krój/pochylenie,
  tym lepszy fine-tune.
- `trocr-base` (334M) to dobry kompromis; dla trudnych przypadków rozważ `trocr-large`.
- Jeśli brak GPU, rozważ alternatywę: włącz korektę tekstu przez Fabryka API (Bielik)
  — patrz README głównego projektu. Albo PaddleOCR-VL + LoRA RysOCR.
# Naprawa i ponowna ewaluacja pierwszego treningu

Aktualna procedura: [TRAINING_RECOVERY.md](../docs/TRAINING_RECOVERY.md).
Najpierw uruchom `kaggle_reevaluate_first_run.ipynb` na GPU — porównuje
istniejący model bez treningu i bez publikacji. Poprawiony notebook treningowy
wybiera checkpoint według CER; dodatkowe warstwy MLP są opcjonalne.
