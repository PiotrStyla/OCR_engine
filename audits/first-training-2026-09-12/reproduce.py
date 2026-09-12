"""Tiny random CPU model: prove decoder-start mismatch without pretrained weights."""
import json
from pathlib import Path
import torch
from transformers import ViTConfig, TrOCRConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, TaskType

torch.set_num_threads(1)
encoder = ViTConfig(image_size=16, patch_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
decoder = TrOCRConfig(vocab_size=20, d_model=16, decoder_layers=1, decoder_attention_heads=2, decoder_ffn_dim=32, max_position_embeddings=32)
model = VisionEncoderDecoderModel(VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder))
model.config.decoder_start_token_id = 0
model.config.pad_token_id = 1
model.generation_config.decoder_start_token_id = 2
model.generation_config.pad_token_id = 1
model.generation_config.eos_token_id = 2
shifted = model.prepare_decoder_input_ids_from_labels(torch.tensor([[0, 7, 2, -100]]))
with torch.no_grad():
    generated = model.generate(torch.zeros(1, 3, 16, 16), max_new_tokens=2)
old_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
names = [n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)]
result = {'training_first_token': shifted[0,0].item(), 'generation_first_token': generated[0,0].item(),
          'matched_modules': {t:sum(n.endswith('.'+t) for n in names) for t in old_targets+['out_proj','fc1','fc2']}}
peft = get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=old_targets,task_type=TaskType.SEQ_2_SEQ_LM))
result['old_config_trainable_parameters'] = sum(p.numel() for p in peft.parameters() if p.requires_grad)
assert result['training_first_token']==0 and result['generation_first_token']==2
assert result['old_config_trainable_parameters']>0
path=Path(__file__).with_name('reproduction-result.json')
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
