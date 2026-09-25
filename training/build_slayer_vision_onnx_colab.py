"""Build the pinned SLAYER Vision ONNX two-page Colab diagnostic."""

from __future__ import annotations

import json
from pathlib import Path


def build(target: Path) -> None:
    root = Path(__file__).resolve().parent
    runner = (root / "slayer_vision_onnx_smoke.py").read_text(encoding="utf-8")
    code = runner + "\n\nresult_archive = run(pages=2, max_new_tokens=128)\n"
    compile(code, "<slayer-vision-onnx-colab>", "exec")

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "colab": {"name": Path(target).name, "provenance": []},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "scope",
                "metadata": {},
                "source": [
                    "# SLAYER Vision ONNX: test OCR na dwoch stronach\n",
                    "Wybierz **T4 GPU** i kliknij **Uruchom wszystko**. Notebook pobiera "
                    "przypiete rewizje modelu, tokenizera, preprocessora i zamrozonego testu IMPACT. "
                    "Ostatnia komorka pobierze ZIP z predykcjami, metrykami i pochodzeniem.\n",
                    "To test diagnostyczny, nie dowod SOTA. Model byl opisany jako obraz do podpisu; "
                    "sprawdzamy, czy potrafi wykonac wierna transkrypcje calej strony. Pisownia "
                    "historyczna pozostaje bez zmian. Tekst jest dekodowany z calej sekwencji ID, "
                    "bez wadliwej konkatenacji `tokens_decoded.json`.\n",
                    "Kontekst LM ma 512 pozycji, a obraz zajmuje 196. Notebook ogranicza odpowiedz "
                    "do 128 tokenow i jawnie zapisuje EOS, bledy oraz znak U+FFFD.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "install",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "%pip uninstall -q -y onnxruntime onnxruntime-gpu\n",
                    "%pip install -q onnxruntime-gpu==1.23.0 transformers==4.57.6 "
                    "huggingface_hub==0.36.2 jiwer==4.0.0 Pillow==11.3.0 numpy==2.2.6\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "run",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [code],
            },
            {
                "cell_type": "code",
                "id": "download",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "from google.colab import files\n",
                    "files.download(str(result_archive))\n",
                ],
            },
        ],
    }
    Path(target).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    build(Path(__file__).with_name("colab_slayer_vision_onnx_smoke.ipynb"))
