"""Exercise TrOCR LoRA forward/backward/merge before an expensive Colab run."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile


def check(model, pixels, labels, rank, alpha):
    import torch
    from training.protocol import AlignedSeq2SeqTrainer
    from training.train_trocr_pl import _inject_lora

    model = _inject_lora(model, rank, alpha).train()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any('lora_' not in name for name in trainable):
        raise RuntimeError('Preflight requires decoder LoRA parameters only')
    before = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer = torch.optim.SGD(trainable.values(), lr=1e-3)
    cuda = pixels.device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    with torch.autocast(pixels.device.type, dtype=torch.float16, enabled=cuda):
        loss = AlignedSeq2SeqTrainer.compute_loss(SimpleNamespace(accelerator=None), model,
            {'pixel_values': pixels, 'labels': labels})
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite preflight loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in trainable.values() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError('Missing or non-finite LoRA gradients')
    if not any(torch.count_nonzero(g) for g in gradients):
        raise RuntimeError('All LoRA gradients are zero')
    scaler.step(optimizer)
    scaler.update()
    changed = sum(not torch.equal(before[name], p) for name, p in trainable.items())
    if not changed:
        raise RuntimeError('Preflight optimizer did not update adapters')
    merged = model.merge_and_unload().eval()
    from transformers import Seq2SeqTrainingArguments
    with tempfile.TemporaryDirectory(prefix='trocr-loss-preflight-') as output:
        trainer = AlignedSeq2SeqTrainer(model=merged, args=Seq2SeqTrainingArguments(
            output_dir=output, use_cpu=not cuda, report_to='none', predict_with_generate=True))
        batch = {'pixel_values': pixels, 'labels': labels}
        with torch.no_grad():
            expected_loss = trainer.compute_loss(merged, batch).detach()
        evaluation_loss, _, evaluation_labels = trainer.prediction_step(
            merged, batch, False, max_new_tokens=2, num_beams=1, early_stopping=False)
        if not torch.allclose(evaluation_loss, expected_loss, rtol=1e-5, atol=1e-5):
            raise RuntimeError('Generation evaluation bypassed aligned token loss')
        if not torch.equal(evaluation_labels[:, :labels.shape[-1]], labels):
            raise RuntimeError('Evaluation changed reference token positions')
    with torch.inference_mode(), torch.autocast(pixels.device.type, dtype=torch.float16, enabled=cuda):
        generated = merged.generate(pixel_values=pixels, max_new_tokens=2, num_beams=1,
                                    early_stopping=False)
    if generated.ndim != 2 or generated.shape[0] != pixels.shape[0]:
        raise RuntimeError('Merged-model generation failed')
    return {'status': 'ok', 'loss': float(loss.detach()), 'updated_adapter_tensors': changed,
        'trainable_parameters': sum(p.numel() for p in trainable.values()),
        'finite_gradients': True, 'merge_and_generation': True,
        'generation_evaluation_aligned_loss': True,
        'aligned_evaluation_loss': float(evaluation_loss),
        'disposable_model': True, 'updates_used_in_training': False,
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'torchao', 'transformers', 'peft')}}


def tiny():
    import torch
    from transformers import TrOCRConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
    from training.protocol import configure_generation

    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        ViTConfig(image_size=16, patch_size=8, hidden_size=16, num_hidden_layers=1,
                  num_attention_heads=2, intermediate_size=32),
        TrOCRConfig(vocab_size=20, d_model=16, decoder_layers=1,
                    decoder_attention_heads=2, decoder_ffn_dim=32, max_position_embeddings=32))
    model = VisionEncoderDecoderModel(config)
    model.generation_config.decoder_start_token_id = 2
    configure_generation(model, SimpleNamespace(pad_token_id=1, sep_token_id=2))
    return check(model, torch.zeros(1, 3, 16, 16), torch.tensor([[0, 7, 2, -100]]), 2, 4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = tiny()
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('ADAPTER_PREFLIGHT_OK', json.dumps(report), flush=True)
