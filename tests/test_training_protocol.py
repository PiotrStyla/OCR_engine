import json
from types import SimpleNamespace
import numpy as np
import pytest

from training.protocol import configure_generation, compute_ocr_metrics, pair_manifest


def test_generation_uses_base_start_and_synchronizes():
    from transformers import GenerationConfig
    model=SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0),
                          generation_config=GenerationConfig(decoder_start_token_id=2))
    tokenizer=SimpleNamespace(pad_token_id=1,sep_token_id=2)
    configure_generation(model,tokenizer)
    assert model.config.decoder_start_token_id==model.generation_config.decoder_start_token_id==2
    configure_generation(model,tokenizer,0)
    assert model.config.decoder_start_token_id==model.generation_config.decoder_start_token_id==0


def test_metrics_ignore_padding_without_mutating_labels():
    class Tokenizer:
        pad_token_id=1
        def batch_decode(self,rows,skip_special_tokens):
            return [' '.join(str(x) for x in r if x!=1) for r in rows]
    labels=np.array([[4,5,-100]])
    result=compute_ocr_metrics(Tokenizer())(SimpleNamespace(predictions=np.array([[4,6,1]]),label_ids=labels))
    assert result['wer']==0.5 and labels[0,2]==-100


def test_manifest_rejects_missing_labels(tmp_path):
    (tmp_path/'a.png').write_bytes(b'image')
    with pytest.raises(ValueError,match='label'):
        pair_manifest(tmp_path)


def test_tiny_lora_training_and_best_checkpoint(tmp_path):
    import torch
    from transformers import (ViTConfig,TrOCRConfig,VisionEncoderDecoderConfig,
        VisionEncoderDecoderModel,Seq2SeqTrainer,Seq2SeqTrainingArguments)
    from training.train_trocr_pl import _inject_lora
    torch.set_num_threads(1)
    config=VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        ViTConfig(image_size=16,patch_size=8,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32),
        TrOCRConfig(vocab_size=20,d_model=16,decoder_layers=1,decoder_attention_heads=2,decoder_ffn_dim=32,max_position_embeddings=32))
    model=VisionEncoderDecoderModel(config)
    model.generation_config.decoder_start_token_id=2
    configure_generation(model,SimpleNamespace(pad_token_id=1,sep_token_id=2))
    model.generation_config.max_length=5
    model=_inject_lora(model,2,4)
    assert all(not p.requires_grad for p in model.base_model.model.encoder.parameters())
    assert hasattr(model.base_model.model.decoder.model.decoder.layers[0].self_attn.out_proj,'lora_A')
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    samples=[{'pixel_values':torch.zeros(3,16,16),'labels':torch.tensor([0,7,2,-100])}]
    scores=iter([0.1,0.8])
    trainer=Seq2SeqTrainer(model=model,args=Seq2SeqTrainingArguments(output_dir=str(tmp_path),
        num_train_epochs=2,per_device_train_batch_size=1,per_device_eval_batch_size=1,
        use_cpu=True,report_to='none',save_strategy='epoch',eval_strategy='epoch',
        predict_with_generate=True,load_best_model_at_end=True,metric_for_best_model='cer',greater_is_better=False),
        train_dataset=samples,eval_dataset=samples,compute_metrics=lambda _:dict(cer=next(scores)))
    trainer.train()
    assert trainer.state.best_model_checkpoint.endswith('checkpoint-1')
    assert trainer.state.best_metric==0.1
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n in before)
