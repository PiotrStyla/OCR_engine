"""Image-expanded prompt masking and patch-preserving Qwen3-VL batches."""


def tokenize_page(row, processor, prompt):
    import torch
    from PIL import Image

    if not isinstance(row.get("text"), str) or not row["text"].strip():
        raise ValueError(f"Empty supervision: {row.get('id')}")
    with Image.open(row["_image"]) as source:
        image = source.convert("RGB")
    user = {"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt},
    ]}
    messages = [user, {"role": "assistant", "content": [
        {"type": "text", "text": row["text"]},
    ]}]
    tokens = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt", truncation=False, padding=False,
    )
    prefix = processor.apply_chat_template(
        [user], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", truncation=False, padding=False,
    )["input_ids"][0]
    input_ids = tokens["input_ids"][0]
    if len(prefix) >= len(input_ids):
        raise ValueError(f"No assistant tokens: {row.get('id')}")
    if not torch.equal(input_ids[:len(prefix)], prefix):
        raise ValueError(f"Chat prefix mismatch: {row.get('id')}")
    labels = input_ids.clone()
    labels[:len(prefix)] = -100
    pixels, grid = tokens["pixel_values"], tokens["image_grid_thw"]
    if pixels.ndim != 2 or grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(f"Invalid image tensor shapes: {row.get('id')}")
    return {
        "input_ids": input_ids, "labels": labels,
        "attention_mask": tokens["attention_mask"][0],
        "pixel_values": pixels, "image_grid_thw": grid,
    }


def collate_pages(batch, pad_token_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if not batch:
        raise ValueError("Empty training batch")
    if pad_token_id is None:
        raise ValueError("Missing tokenizer pad token")
    return {
        "input_ids": pad_sequence([row["input_ids"] for row in batch],
                                  batch_first=True, padding_value=pad_token_id),
        "labels": pad_sequence([row["labels"] for row in batch],
                               batch_first=True, padding_value=-100),
        "attention_mask": pad_sequence([row["attention_mask"] for row in batch],
                                       batch_first=True, padding_value=0),
        "pixel_values": torch.cat([row["pixel_values"] for row in batch], dim=0),
        "image_grid_thw": torch.cat([row["image_grid_thw"] for row in batch], dim=0),
    }
