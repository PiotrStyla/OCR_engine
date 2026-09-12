import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ocr.config import OcrConfig
from ocr.detector import TextDetector
from ocr.pipeline import OcrEngine
from ocr.recognizer import _paddlevl_pipeline_version
from ocr.result import BBox, OcrResult, TextLine
from training.generate_synthetic import _augment


def test_craft_accepts_numpy_boxes():
    detector = TextDetector(OcrConfig(device="cpu"))
    detector._craft = SimpleNamespace(detect_text=lambda _: {
        "boxes":np.array([[[0,0],[30,0],[30,10],[0,10]]])})
    assert detector._detect_craft(np.zeros((40,40,3),dtype=np.uint8)) == [BBox(0,0,30,10)]


def test_custom_model_is_not_silently_replaced():
    with pytest.raises(ValueError, match="custom weights"):
        _paddlevl_pipeline_version("custom/weights-1.6")
    assert _paddlevl_pipeline_version("PaddlePaddle/PaddleOCR-VL-1.6") == "v1.6"


def test_correction_preserves_raw_and_drops_stale_confidence():
    engine = OcrEngine(OcrConfig(device="cpu",correct_text=True))
    engine.corrector = SimpleNamespace(enabled=True,correct=lambda _:"poprawiony")
    source = OcrResult([TextLine("surowy",BBox(0,0,30,10),"pl",0.9)])
    result = engine._correct_result(source)
    assert result.lines[0].raw_text == "surowy"
    assert math.isnan(result.lines[0].confidence)
    assert result.lines[0].raw_confidence == 0.9
    source.lines.append(TextLine("drugi",BBox(0,20,30,30),"pl",0.8))
    assert engine._correct_result(source) is source


def test_inverse_geometry_and_metadata():
    source = OcrResult([TextLine("x",BBox(20,30,40,40),"en",0.9)],(100,100))
    transform = np.array([[1.,0.,10.],[0.,1.,20.]])
    mapped = OcrEngine._source_coordinates(source,transform)
    assert mapped.lines[0].bbox == BBox(10,10,30,20)
    assert mapped.lines[0].source_polygon[0] == (10.,10.)
    assert mapped.to_dict()["coordinate_space"] == "source_image_pixels"


def test_seed_reproduces_pixels():
    image = Image.new("RGB",(100,30),"gray")
    assert np.array_equal(_augment(image,random.Random(0)),_augment(image,random.Random(0)))


def test_cli_environment_and_explicit_override(monkeypatch,tmp_path):
    from ocr.cli import main
    image = tmp_path / "image.png"
    image.touch()
    seen=[]
    class Engine:
        def __init__(self,config): seen.append(config)
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def recognize(self,*args): return OcrResult()
    monkeypatch.setattr("ocr.cli.OcrEngine",Engine)
    monkeypatch.setenv("OCR_RECOGNIZER_BACKEND","paddlevl")
    monkeypatch.setenv("OCR_RECOGNIZER_PL","custom/pl")
    main(["recognize",str(image)])
    assert seen[-1].recognizer_backend == "paddlevl"
    assert seen[-1].recognizer_pl == "custom/pl"
    main(["recognize",str(image),"--backend","trocr"])
    assert seen[-1].recognizer_backend == "trocr"


def test_confidence_per_sequence_eos_and_beams():
    torch = pytest.importorskip("torch")
    from ocr.recognizer import _token_confidences
    seen={}
    def transition(sequences,scores,**kwargs):
        seen.update(kwargs)
        return torch.log(torch.tensor([[.99,.8,.001],[.01,.2,.001]]))
    model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2,pad_token_id=1),
                          compute_transition_scores=transition)
    generated=SimpleNamespace(sequences=torch.tensor([[0,3,2,1],[0,3,2,1]]),
                              scores=(1,2,3),beam_indices="ancestry")
    assert _token_confidences(model,generated) == pytest.approx([.99,.01])
    assert seen == {"beam_indices":"ancestry","normalize_logits":True}
