"""Dataset do fine-tuningu TrOCR na polskich liniach tekstu.

Oczekuje katalogu z parami:
  - <id>.png  (obraz pojedynczej linii tekstu)
  - <id>.txt  (transkrypcja, jedna linia)

Albo pliku JSONL z polami {"image": "...", "text": "..."}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class LineSample:
    image: Image.Image
    text: str


def load_pairs(directory: str | Path) -> list[LineSample]:
    """Wczytuje pary .png/.txt z katalogu."""
    directory = Path(directory)
    samples: list[LineSample] = []
    for img_path in sorted(directory.glob("*.png")):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        samples.append(LineSample(Image.open(img_path).convert("RGB"), text))
    return samples


def load_jsonl(path: str | Path, image_root: str | Path | None = None) -> list[LineSample]:
    """Wczytuje JSONL: {"image": "rel/path.png", "text": "..."}."""
    path = Path(path)
    root = Path(image_root) if image_root else path.parent
    samples: list[LineSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        img = Image.open(root / obj["image"]).convert("RGB")
        samples.append(LineSample(img, obj["text"].strip()))
    return samples


class TrOCRLineDataset:
    """Dataset kompatybilny z transformers.Trainer."""

    def __init__(self, samples: list[LineSample], processor, max_target_length: int = 128) -> None:
        self.samples = samples
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        pixel_values = self.processor(sample.image, return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(
            sample.text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids[0]
        # zamień padding na -100 (ignorowane w loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}
