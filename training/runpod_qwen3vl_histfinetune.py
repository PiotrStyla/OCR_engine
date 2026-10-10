"""Qwen3-VL-4B: fine-tuninhg domenowy na IMPACT + pomiar test A (Runpod GPU).

Dane (zamrozone): ``PiotrSty/impact-psnc-polish-ocr`` — 65 stron ``train`` i
15 stron ``validation`` z GT na poziomie strony. Split ``test`` zbioru
(9 stron) jest wykluczony: pochodzi z tych samych kolekcji co test A i mógłby
się z nim pokrywać. Kontrola: żaden ``image_sha256`` zbioru treningowego nie
może wystąpić w manifeście test A.

Przebieg: LoRA (NF4, jak w pomiarze SOTA) na parach (prompt A -> tekst strony),
3 epoki z checkpointem po każdej; wybór epoki po CER na zbiorze ``validation``
(jedyny wybór), potem JEDEN pomiar test A wybranym adapterem tym samym
promptem i parametrami co pomiar zero-shot. Bez promocji i bez twierdzenia
SOTA; to rozszerzenie pomiaru o system wytrenowany.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE_REVISION = "f3eb05f5fe728767769398907345119077330bf7"
SOURCE_REPOSITORY = "PiotrSty/impact-psnc-polish-ocr"
SOURCE_REVISION = "c7cb156fb95d2880699c33725bbaf1fbc1008fea"
IMPACT_REPOSITORY = "PiotrSty/impact-print-v2"
IMPACT_REVISION = "a2480fde6f15284701458ff370b81cce50dc5c2d"
IMPACT_SHA256 = "0a9ffa126029703726fc5883a8279ddccf15763c4fdd483ed9b8cc034bdb42f0"
IMPACT_PATH = "impact-print-v2-test.tar.gz"
MODEL_SOURCE = "Qwen/Qwen3-VL-4B-Instruct"
EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
MAX_NEW_TOKENS = 4096
MAX_PIXELS = 4194304
MIN_PIXELS = 262144
EVIDENCE_REPOSITORY = "PiotrSty/slayer-ocr-experiment-evidence"
EVIDENCE_PATH = "experiments/2026-10-10/qwen3vl-hist-finetune"

WORK = Path("/workspace/qwen3vl-hist-finetune")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_finetune_rows(pages, test_image_hashes):
    """Split the corpus pages into train/dev, excluding the test split.

    Raises when a test-split row or an image shared with test A would enter
    training: the benchmark stays untouched.
    """
    train, dev = [], []
    for row in pages:
        if row.get("split") == "test":
            continue
        if row.get("image_sha256") in test_image_hashes:
            raise ValueError(f"Training page overlaps test A: {row['id']}")
        (train if row.get("split") == "train" else dev).append(row)
    if not train or not dev:
        raise ValueError("Empty train or dev split")
    ids = [row["id"] for row in train + dev]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate page ids")
    return train, dev


def chat_example(row, prompt):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": row["_image"]},
            {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": row["text"]}]}]}


def main():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    python = sys.executable
    subprocess.run([python, "-m", "pip", "install", "-q", "transformers==4.57.6", "peft==0.19.1",
                    "bitsandbytes", "accelerate==1.13.0", "jiwer==4.0.0", "huggingface_hub==0.36.2"], check=True)
    repo = Path("/workspace/OCR_engine-finetune")
    if not repo.exists():
        subprocess.run(["git", "clone", "https://github.com/PiotrStyla/OCR_engine.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "fetch", "origin", "main"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", CODE_REVISION], check=True)
    print("CODE_REVISION", subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
          flush=True)
    sys.path.insert(0, str(repo))
    import torch
    from PIL import Image
    from jiwer import cer
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration,
                             Trainer, TrainingArguments)
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from training.run_sota_benchmark import run as run_benchmark
    from training.run_vision_baseline import load_templates
    from training.transcription_eval import normalize

    assert torch.cuda.is_available(), "Pod bez GPU CUDA."
    WORK.mkdir(parents=True, exist_ok=True)
    prompt = load_templates()["A"]

    # --- data ------------------------------------------------------------
    corpus = Path(snapshot_download(SOURCE_REPOSITORY, repo_type="dataset", revision=SOURCE_REVISION,
                                    local_dir=WORK / "corpus"))
    pages = read_rows(corpus / "pages.jsonl")
    for row in pages:
        image = corpus / row["image"]
        if digest(image) != row["image_sha256"]:
            raise ValueError(f"Corpus image checksum mismatch: {row['id']}")
        row["_image"] = str(image)
    archive = Path(hf_hub_download(IMPACT_REPOSITORY, IMPACT_PATH, repo_type="dataset", revision=IMPACT_REVISION))
    assert digest(archive) == IMPACT_SHA256, "Checksum archiwum test A nie zgadza sie."
    staged = WORK / "benchmark"
    if not (staged / "manifest.jsonl").exists():
        subprocess.run([sys.executable, "-m", "training.stage_impact_benchmark", "--archive", str(archive),
                        "--output", str(staged)], cwd=repo, check=True)
    test_rows = read_rows(staged / "manifest.jsonl")
    assert len(test_rows) == 36, f"Oczekiwano 36 stron testu A, jest {len(test_rows)}."
    train_rows, dev_rows = build_finetune_rows(pages, {row["sha256"] for row in test_rows})
    print("DATA", len(train_rows), "train /", len(dev_rows), "dev | test split excluded", flush=True)

    # --- model -----------------------------------------------------------
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    engine = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_SOURCE, device_map={"": 0}, torch_dtype=torch.float16,
        attn_implementation="sdpa", quantization_config=quantization)
    processor = AutoProcessor.from_pretrained(MODEL_SOURCE, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    engine = get_peft_model(engine, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    engine.print_trainable_parameters()

    class PagesDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            messages = chat_example(row, prompt)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            with Image.open(row["_image"]) as source:
                image = source.convert("RGB")
            tokens = processor(text=[text], images=[image], return_tensors="pt",
                               truncation=False, padding=False)
            input_ids = tokens["input_ids"][0]
            labels = input_ids.clone()
            answer = processor.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
            prompt_len = len(processor.tokenizer(answer, add_special_tokens=False)["input_ids"])
            labels[:prompt_len] = -100
            return {"input_ids": input_ids, "labels": labels,
                    "attention_mask": tokens["attention_mask"][0],
                    "pixel_values": tokens["pixel_values"][0],
                    "image_grid_thw": tokens["image_grid_thw"][0]}

    def collate(batch):
        pad = processor.tokenizer.pad_token_id
        width = max(item["input_ids"].shape[0] for item in batch)
        ids, labels, masks = [], [], []
        for item in batch:
            pad_len = width - item["input_ids"].shape[0]
            ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            masks.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        return {"input_ids": torch.stack(ids), "labels": torch.stack(labels),
                "attention_mask": torch.stack(masks),
                "pixel_values": torch.cat([item["pixel_values"] for item in batch]),
                "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch])}

    def predict(path):
        with Image.open(path) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(engine.device)
        eos = engine.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        with torch.inference_mode():
            outputs = engine.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def dev_cer():
        hypotheses, references = [], []
        for row in dev_rows:
            hypotheses.append(normalize(predict(row["_image"])))
            references.append(normalize(row["text"]))
        return cer(references, hypotheses)

    # --- training --------------------------------------------------------
    args = TrainingArguments(
        output_dir=str(WORK / "checkpoints"), num_train_epochs=1, per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION, learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=5, save_strategy="no",
        fp16=True, remove_unused_columns=False, report_to=[])
    trainer = Trainer(model=engine, args=args, train_dataset=PagesDataset(train_rows), data_collator=collate)
    dev_scores = []
    best = (None, float("inf"))
    for epoch in range(1, EPOCHS + 1):
        trainer.train()
        adapter = WORK / f"adapter-epoch{epoch}"
        engine.save_pretrained(adapter)
        score = dev_cer()
        dev_scores.append({"epoch": epoch, "dev_cer": score})
        print("DEV_EPOCH", epoch, "cer", score, flush=True)
        if score < best[1]:
            best = (adapter, score)
    print("SELECTED_EPOCH", best[0].name, "dev_cer", best[1], flush=True)
    (WORK / "dev-metrics.json").write_text(json.dumps({
        "model": MODEL_SOURCE, "source_dataset": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
        "train_pages": len(train_rows), "dev_pages": len(dev_rows), "epochs": dev_scores,
        "selected": best[0].name, "selected_dev_cer": best[1],
        "prompt_version": "polocrbench-zero-shot-prompt-v1",
        "learning_rate": LEARNING_RATE, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "max_pixels": MAX_PIXELS, "max_new_tokens": MAX_NEW_TOKENS}, indent=2), encoding="utf-8")

    # --- test A: jeden pomiar wybranym adapterem -------------------------
    engine.load_adapter(str(best[0]), adapter_name="selected")
    engine.set_adapter("selected")
    out = WORK / "testA"
    shutil.rmtree(out, ignore_errors=True)
    run_benchmark("qwen3vl", staged, out, predictor=predict)
    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    print("TEST_A_RESULT", {key: score[key] for key in ("cer_micro", "wer_micro", "structure_similarity",
                                                        "errors_or_missing")}, flush=True)

    evidence = WORK / "evidence"
    shutil.rmtree(evidence, ignore_errors=True)
    evidence.mkdir()
    for name in ("predictions.jsonl", "score.json", "run.json"):
        shutil.copyfile(out / name, evidence / f"qwen3vl-finetuned-{name}")
    shutil.copyfile(WORK / "dev-metrics.json", evidence / "dev-metrics.json")
    shutil.copyfile(staged / "verification.json", evidence / "staged-verification.json")
    (evidence / "receipt.json").write_text(json.dumps({
        "protocol": "polocrbench-qwen3vl-hist-finetune-v1", "code_revision": CODE_REVISION,
        "model": MODEL_SOURCE, "source_dataset": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
        "impact_repository": IMPACT_REPOSITORY, "impact_revision": IMPACT_REVISION,
        "impact_sha256": IMPACT_SHA256, "training_performed": True,
        "benchmark_touched": False, "promotion": False, "sota_claim": False}, indent=2), encoding="utf-8")
    shutil.copytree(best[0], evidence / "adapter", dirs_exist_ok=True)
    (evidence / "checksums.json").write_text(json.dumps(
        {path.name: digest(path) for path in evidence.iterdir() if path.is_file()}, indent=2), encoding="utf-8")
    zip_path = WORK / "qwen3vl-hist-finetune-evidence.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", evidence)
    print("GOTOWY_ZIP", zip_path, zip_path.stat().st_size, flush=True)
    token = os.environ.get("HF_TOKEN", "")
    if token:
        api = HfApi(token=token)
        info = api.upload_folder(repo_id=EVIDENCE_REPOSITORY, repo_type="dataset", path_in_repo=EVIDENCE_PATH,
                                 folder_path=str(WORK),
                                 allow_patterns=["qwen3vl-hist-finetune-evidence.zip"],
                                 commit_message="Add Qwen3-VL historical fine-tune evidence")
        print("HF_UPLOAD", EVIDENCE_REPOSITORY, info.oid, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
