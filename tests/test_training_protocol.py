from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.protocol import (
    AlignedSeq2SeqTrainer,
    aligned_token_loss,
    compute_ocr_metrics,
    configure_generation,
    pair_manifest,
)


class _PerfectModel:
    def __init__(self):
        self.call = None

    def prepare_decoder_input_ids_from_labels(self, labels):
        shifted = torch.zeros_like(labels)
        shifted[:, 1:] = labels[:, :-1].clamp_min(0)
        return shifted

    def __call__(self, **kwargs):
        self.call = kwargs
        logits = torch.full((1, 3, 6), -20.0)
        logits[0, 0, 2] = 20.0
        logits[0, 1, 3] = 20.0
        return SimpleNamespace(logits=logits)


def test_generation_uses_base_start_and_synchronizes():
    from transformers import GenerationConfig

    model = SimpleNamespace(
        config=SimpleNamespace(decoder_start_token_id=0),
        generation_config=GenerationConfig(decoder_start_token_id=2),
    )
    tokenizer = SimpleNamespace(pad_token_id=1, sep_token_id=2)
    configure_generation(model, tokenizer)
    assert model.config.decoder_start_token_id == 2
    assert model.generation_config.decoder_start_token_id == 2
    configure_generation(model, tokenizer, 0)
    assert model.config.decoder_start_token_id == 0
    assert model.generation_config.decoder_start_token_id == 0


def test_metrics_ignore_padding_without_mutating_labels():
    class Tokenizer:
        pad_token_id = 1

        def batch_decode(self, rows, skip_special_tokens):
            return [" ".join(str(x) for x in row if x != 1) for row in rows]

    labels = np.array([[4, 5, -100]])
    result = compute_ocr_metrics(Tokenizer())(
        SimpleNamespace(predictions=np.array([[4, 6, 1]]), label_ids=labels)
    )
    assert result["wer"] == 0.5
    assert labels[0, 2] == -100


def test_manifest_rejects_missing_labels(tmp_path):
    (tmp_path / "a.png").write_bytes(b"image")
    with pytest.raises(ValueError, match="label"):
        pair_manifest(tmp_path)


def test_aligned_loss_compares_same_token_positions():
    logits = torch.full((1, 3, 6), -20.0)
    logits[0, 0, 2] = 20.0
    logits[0, 1, 3] = 20.0
    labels = torch.tensor([[2, 3, -100]])
    assert aligned_token_loss(logits, labels).item() < 1e-6


def test_trainer_shifts_decoder_input_once_but_not_loss_labels():
    model = _PerfectModel()
    trainer = object.__new__(AlignedSeq2SeqTrainer)
    labels = torch.tensor([[2, 3, -100]])
    loss = trainer.compute_loss(
        model, {"pixel_values": torch.ones(1), "labels": labels}
    )
    assert loss.item() < 1e-6
    assert "labels" not in model.call
    assert model.call["decoder_input_ids"].tolist() == [[0, 2, 3]]
    assert model.call["use_cache"] is False


def test_aligned_loss_honors_gradient_accumulation_item_count():
    logits = torch.zeros((1, 2, 2))
    labels = torch.tensor([[0, 1]])
    loss = aligned_token_loss(logits, labels, torch.tensor(4))
    assert loss.item() == pytest.approx(2 * torch.log(torch.tensor(2.0)).item() / 4)


def test_tiny_lora_training_and_best_checkpoint(tmp_path):
    pytest.importorskip("peft")
    from transformers import (
        Seq2SeqTrainingArguments,
        TrOCRConfig,
        ViTConfig,
        VisionEncoderDecoderConfig,
        VisionEncoderDecoderModel,
    )
    from training.train_trocr_pl import _inject_lora

    torch.set_num_threads(1)
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        ViTConfig(
            image_size=16,
            patch_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        ),
        TrOCRConfig(
            vocab_size=20,
            d_model=16,
            decoder_layers=1,
            decoder_attention_heads=2,
            decoder_ffn_dim=32,
            max_position_embeddings=32,
        ),
    )
    model = VisionEncoderDecoderModel(config)
    model.generation_config.decoder_start_token_id = 2
    configure_generation(model, SimpleNamespace(pad_token_id=1, sep_token_id=2))
    model.generation_config.max_length = 5
    model = _inject_lora(model, 2, 4)
    assert all(not p.requires_grad for p in model.base_model.model.encoder.parameters())
    attention = model.base_model.model.decoder.model.decoder.layers[0].self_attn
    assert hasattr(attention.out_proj, "lora_A")
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    samples = [
        {"pixel_values": torch.zeros(3, 16, 16), "labels": torch.tensor([0, 7, 2, -100])}
    ]
    scores = iter([0.1, 0.8])
    trainer = AlignedSeq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(tmp_path),
            num_train_epochs=2,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            use_cpu=True,
            report_to="none",
            save_strategy="epoch",
            eval_strategy="epoch",
            predict_with_generate=True,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
        ),
        train_dataset=samples,
        eval_dataset=samples,
        compute_metrics=lambda _: dict(cer=next(scores)),
    )
    trainer.train()
    assert trainer.state.best_model_checkpoint.endswith("checkpoint-1")
    assert trainer.state.best_metric == 0.1
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in before
    )
