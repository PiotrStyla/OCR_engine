from types import SimpleNamespace

from PIL import Image
import pytest

from training.dataset import LineSample, TrOCRLineDataset


class Tokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        values = list(range(len(text) + 2))
        if kwargs.get("padding") == "max_length":
            values = values[:kwargs["max_length"]]
            values += [0] * (kwargs["max_length"] - len(values))
        return SimpleNamespace(input_ids=values if not kwargs.get("return_tensors") else [values])


class Processor:
    tokenizer = Tokenizer()


def test_dataset_rejects_silent_target_truncation():
    samples = [LineSample(Image.new("RGB", (8, 8)), "123456789")]
    with pytest.raises(ValueError, match="Do not silently truncate"):
        TrOCRLineDataset(samples, Processor(), max_target_length=10)
