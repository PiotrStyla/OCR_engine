---
language: [pl]
license: cc-by-4.0
task_categories: [image-to-text]
tags: [ocr, kraken, polski, ehri, segmentation, recognition]
---

# EHRI Polish OCR dataset + modele Kraken

Strony archiwalne EHRI (zydowskie instytuty historyczne, polskie kolekcje)
z adnotacjami ALTO XML (baseline'y + transkrypcje). Modele wytrenowane
Kraken 7.1.1 (ketos) na Kaggle T4.

## Dane
- 15 stron .tif + ALTO XML: split 12 train / 3 validation (seed 42).

## Modele (models/)

| plik | co to | trening | score walidacyjny |
|---|---|---|---|
| polish_nfd_9313.mlmodel | recognizer bazowy | - | - |
| polish_seg_best.safetensors | segmenter (fine-tune blla.mlmodel, `--resize new`) | `ketos segtrain`, 50 epok, `--augment` | val_metric 0.5277 |
| polish_nfd_finetuned.safetensors | recognizer (fine-tune polish_nfd_9313, `--resize union`) | `ketos train`, 50 epok, `--augment` | val score 0.9567 |

## Wyniki e2e (3 strony walidacyjne, kraken.tasks API, CER/WER z jiwer)

| konfiguracja | CER | WER |
|---|---|---|
| default segmenter + polish_nfd_9313 | 13.45% | 47.40% |
| polish_seg_best + polish_nfd_9313 | 13.09% | 45.60% |
| polish_seg_best + polish_nfd_finetuned | **7.11%** | **33.30%** |

Fine-tuning rozpoznawania dal ~18x wiekszy przyrost CER niz tuning segmentacji
(-5.98 pp vs -0.36 pp) - segmentacja nie jest watkim gardlem.

## Uzycie (kraken 7.1.1)

```python
from kraken.tasks import SegmentationTaskModel, RecognitionTaskModel
from kraken.configs import SegmentationInferenceConfig, RecognitionInferenceConfig
from PIL import Image

seg = SegmentationTaskModel.load_model('polish_seg_best.safetensors')
rec = RecognitionTaskModel.load_model('polish_nfd_finetuned.safetensors')
img = Image.open('strona.tif').convert('L')
segmentation = seg.predict(img, SegmentationInferenceConfig(accelerator='cuda', device=[0]))
pred = rec.predict(img, segmentation, RecognitionInferenceConfig(accelerator='cuda', device=[0]))
for record in pred:
    print(record.prediction)
```

## Trening / notebook

https://github.com/PiotrStyla/OCR_engine/blob/main/training/kaggle_kraken_segtrain.ipynb
