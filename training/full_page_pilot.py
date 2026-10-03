"""Stage and diagnose full-page OCR without promoting source labels to gold."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import sys
import time
import unicodedata
from urllib.parse import quote
from urllib.request import urlopen
from xml.etree import ElementTree


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                          encoding="utf-8", newline="\n")


def safe_relative(value):
    path = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or path.is_absolute()
            or ".." in path.parts):
        raise ValueError(f"Unsafe relative path: {value}")
    return path.as_posix()


def fetch(config, relative, target, expected, opener=urlopen):
    target = Path(target)
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"Cached checksum mismatch: {target}")
        return target
    source = config["dataset"]
    url = (f'https://huggingface.co/datasets/{source["repo"]}/resolve/'
           f'{source["revision"]}/{quote(safe_relative(relative), safe="/")}')
    with opener(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Source checksum mismatch: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def select_pages(rows, count, salt):
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected unique nonempty page IDs")
    if not 1 <= count <= len(rows):
        raise ValueError("Page count outside available non-test pool")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["collection"])].append(row)
    key = lambda value: hashlib.sha256((salt + "\0" + value).encode()).hexdigest()
    for group in groups.values():
        group.sort(key=lambda row: key(row["id"]))
    # Validation comes first for a bounded smoke run; never select by OCR score.
    order = sorted(groups, key=lambda item: (item[0] != "validation", key(item[1])))
    selected = []
    while len(selected) < count:
        for name in order:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop(0))
    return selected


def coordinates(element, width, height):
    coords = element.find("./{*}Coords")
    if coords is None:
        return None
    points = [(int(p.attrib["x"]), int(p.attrib["y"]))
              for p in coords.findall("./{*}Point")]
    if not points and coords.get("points"):
        points = [tuple(map(int, value.split(","))) for value in coords.get("points").split()]
    if not points:
        return None
    xs, ys = zip(*points)
    if any(x < 0 or y < 0 or x > width or y > height for x, y in points):
        return None
    box = [min(xs), min(ys), min(width, max(xs) + 1), min(height, max(ys) + 1)]
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        return None
    return box


def inspect_reference(row, xml):
    root = ElementTree.fromstring(xml)
    page = root.find(".//{*}Page")
    if page is None or (int(page.get("imageWidth")), int(page.get("imageHeight"))) != (
            row["width"], row["height"]):
        raise ValueError(f'PAGE dimensions mismatch: {row["id"]}')
    order = {item.get("regionRef"): int(item.get("index"))
             for item in root.findall(".//{*}RegionRefIndexed")}
    regions = list(page.findall("./{*}TextRegion"))
    regions.sort(key=lambda element: order.get(element.get("id"), 1_000_000))
    lines, region_boxes, invalid_boxes, invalid_regions, content_regions = [], [], 0, 0, []
    for region in regions:
        direct = region.find("./{*}TextEquiv/{*}Unicode")
        content = (direct.text or "") if direct is not None else "\n".join(
            node.text or "" for node in region.findall("./{*}TextLine/{*}TextEquiv/{*}Unicode"))
        if not content.strip():
            continue
        content_regions.append(region)
        region_box = coordinates(region, row["width"], row["height"])
        if region_box is None:
            invalid_regions += 1
        else:
            region_boxes.append({"id": region.get("id"), "bbox_xyxy": region_box,
                                 "source_order_known": region.get("id") in order})
        for line in region.findall("./{*}TextLine"):
            box = coordinates(line, row["width"], row["height"])
            if box is None:
                invalid_boxes += 1
            else:
                lines.append({"id": line.get("id"), "bbox_xyxy": box,
                              "region_id": region.get("id")})
    text = row["text"]
    flags = []
    if "\ufffd" in text:
        flags.append("replacement-character")
    if any(unicodedata.category(char) == "Co" for char in text):
        flags.append("private-use-character")
    if len("".join(text.split())) < 20:
        flags.append("very-short-reference")
    if not lines:
        flags.append("no-source-line-geometry")
    if not region_boxes:
        flags.append("no-valid-region-geometry")
    if invalid_boxes:
        flags.append("invalid-line-geometry")
    if invalid_regions:
        flags.append("invalid-region-geometry")
    if content_regions and any(region.get("id") not in order for region in content_regions):
        flags.append("reading-order-incomplete")
    return {"id": row["id"], "source_split": row["split"], "collection": row["collection"],
            "reference_characters": len(text), "replacement_characters": text.count("\ufffd"),
            "private_use_characters": sum(unicodedata.category(c) == "Co" for c in text),
            "text_regions": len(content_regions), "valid_line_boxes": len(lines),
            "valid_region_boxes": len(region_boxes), "invalid_region_boxes": invalid_regions,
            "empty_source_regions": len(regions) - len(content_regions),
            "invalid_line_boxes": invalid_boxes, "flags": flags,
            "completeness": "unverified", "source_regions": region_boxes}


def stage(config, output, count=None, opener=urlopen):
    from PIL import Image
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output belongs to a different configuration")
    write_json(config_path, config)
    rows = []
    for split, spec in config["dataset"]["splits"].items():
        if split not in ("train", "validation"):
            raise ValueError("Final test must not be staged by this pilot")
        path = fetch(config, f"pages/{split}/metadata.jsonl", output / "sources" / f"{split}.jsonl",
                     spec["metadata_sha256"], opener)
        current = read_rows(path)
        if len(current) != spec["pages"] or any(row["split"] != split for row in current):
            raise ValueError("Source metadata split/count mismatch")
        rows.extend(current)
    if len(rows) != config["available_non_test_pages"]:
        raise ValueError("Non-test pool size changed")
    for row in rows:
        if (row["collection"] in config["dataset"]["excluded_collections"]
                or row["license"] != config["dataset"]["license"]):
            raise ValueError("Forbidden collection or license mismatch")
    inference_splits = config.get("inference_splits", ["train", "validation"])
    if (not inference_splits or len(set(inference_splits)) != len(inference_splits)
            or set(inference_splits) - {"train", "validation"}):
        raise ValueError("Invalid inference split restriction")
    eligible = [row for row in rows if row["split"] in inference_splits]
    selected = select_pages(eligible, len(eligible) if count is None else count, config["selection_salt"])
    selection_hash = hashlib.sha256(json.dumps([row["id"] for row in selected]).encode()).hexdigest()
    if (output / "selection.json").exists():
        old = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        if old["sha256"] != selection_hash:
            raise ValueError("Use a new output for a different page selection")
    write_json(output / "selection.json", {"ids": [row["id"] for row in selected], "sha256": selection_hash})
    manifests, audits, inference = [], [], []
    for index, row in enumerate(selected):
        source = fetch(config, f'pages/{row["split"]}/{safe_relative(row["file_name"])}',
                       output / "sources" / f"{index:04d}.jpg", row["image_sha256"], opener)
        xml = fetch(config, row["pagexml"], output / "pagexml" / f"{index:04d}.xml",
                    row["pagexml_sha256"], opener)
        png = output / "images" / f"{index:04d}.png"
        png.parent.mkdir(exist_ok=True)
        with Image.open(source) as image:
            image.seek(0)
            if image.size != (row["width"], row["height"]):
                raise ValueError(f'Image dimensions mismatch: {row["id"]}')
            image.save(png, format="PNG")
            with Image.open(png) as check:
                if check.mode != image.mode or check.tobytes() != image.tobytes():
                    raise ValueError("PNG pixel round-trip failed")
        audit = inspect_reference(row, xml.read_bytes())
        manifest = {**row, "image": png.relative_to(output).as_posix(), "sha256": digest(png),
                    "source_sha256": row["image_sha256"], "reference_status": "source-unreviewed",
                    "completeness_verified": False, "final_test": False,
                    "pagexml_local": xml.relative_to(output).as_posix()}
        manifests.append(manifest)
        inference.append({key: manifest[key] for key in ("id", "image", "sha256", "width", "height")} |
                         {"source_regions": audit.pop("source_regions")})
        audits.append(audit)
        print(f'Staged {index + 1}/{len(selected)}: {row["id"]}', flush=True)
    write_rows(output / "manifest.jsonl", manifests)
    write_rows(output / "inference-inputs.jsonl", inference)
    counts = Counter(flag for audit in audits for flag in audit["flags"])
    report = {"schema": "slayer-reference-audit-v1", "pages": len(selected),
              "available_non_test_pages": len(rows), "target_review_pages": config["target_review_pages"],
              "additional_pages_needed": max(0, config["target_review_pages"] - len(rows)),
              "manifest_sha256": digest(output / "manifest.jsonl"),
              "inference_inputs_sha256": digest(output / "inference-inputs.jsonl"),
              "source_split_counts": dict(Counter(row["split"] for row in selected)),
              "flag_counts": dict(counts), "results": audits, "gold_pages": 0,
              "claim_boundary": "Flags are triage only; their absence does not establish completeness. "
                                "Train pages may have been seen by the existing recognizer. No SOTA claim."}
    write_json(output / "reference-audit.json", report)
    return report


def validate_inputs(path):
    path = Path(path)
    rows = read_rows(path)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Invalid inference page IDs")
    for row in rows:
        if (set(row) != {"id", "image", "sha256", "width", "height", "source_regions"}
                or any(set(region) != {"id", "bbox_xyxy", "source_order_known"}
                       for region in row["source_regions"])):
            raise ValueError("Reference text is forbidden in model inputs")
        image = path.parent / safe_relative(row["image"])
        if digest(image) != row["sha256"]:
            raise ValueError(f'Input checksum mismatch: {row["id"]}')
    return rows


def load_mixed(spec):
    from huggingface_hub import snapshot_download
    from ocr.config import OcrConfig
    from ocr.detector import sort_reading_order, sort_reading_order_columns
    from ocr.opencv_detector import OpenCVDetector
    from ocr.preprocess import load_image
    from ocr.recognizer import Recognizer
    model_dir = Path(snapshot_download(spec["repo"], revision=spec["revision"]))
    if digest(model_dir / "model.safetensors") != spec["weights_sha256"]:
        raise ValueError("mixed-v3 weights checksum mismatch")
    config = OcrConfig(recognizer_pl=str(model_dir), recognizer_en=str(model_dir),
                       recognizer_fallback=str(model_dir), recognizer_backend="trocr",
                       use_fallback_if_pl_missing=False, device="cuda", force_language="pl",
                       deskew=False, correct_text=False, batch_size=spec["batch_size"],
                       line_padding=spec["line_padding"])
    detector, recognizer = OpenCVDetector(config), Recognizer(config)

    def infer(path, page):
        image = load_image(path)
        boxes = detector.detect(image)
        decoded = recognizer.recognize_lines(image, boxes, ["pl"] * len(boxes))
        if len(decoded) != len(boxes):
            raise ValueError("Recognizer output count mismatch")
        texts = {box.to_tuple(): text for box, (text, _) in zip(boxes, decoded)}
        row_major = sort_reading_order(boxes, image.shape[0])
        columns = sort_reading_order_columns(boxes, image.shape[1], image.shape[0])
        results = {"mixed-v3-row-major": {"text": "\n".join(texts[box.to_tuple()] for box in row_major)},
                   "mixed-v3-column-order": {"text": "\n".join(texts[box.to_tuple()] for box in columns)}}
        if page["source_regions"]:
            region_texts, detected_region_lines = [], 0
            for region in page["source_regions"]:
                x1, y1, x2, y2 = region["bbox_xyxy"]
                crop = image[y1:y2, x1:x2]
                local_boxes = sort_reading_order(detector.detect(crop), crop.shape[0])
                recognized = recognizer.recognize_lines(crop, local_boxes, ["pl"] * len(local_boxes))
                if len(recognized) != len(local_boxes):
                    raise ValueError("Region recognizer output count mismatch")
                detected_region_lines += len(local_boxes)
                region_texts.append("\n".join(text for text, _ in recognized))
            results["mixed-v3-source-regions"] = {"text": "\n".join(region_texts),
                                                   "region_detected_lines": detected_region_lines}
        else:
            results["mixed-v3-source-regions"] = {"status": "error", "text": "",
                                                   "error": "No valid source region geometry"}
        for result in results.values():
            result["detected_lines"] = len(boxes)
        return results

    return infer


OVIS_PROMPT = ("Extract all readable content from the image in natural human reading order "
               "and output a single Markdown document. Represent tables as HTML and formulas "
               "as LaTeX. Represent visual regions with an HTML image tag. Preserve original "
               "text without translation, paraphrasing, spelling correction or modernization. "
               "Preserve historical characters including \u00e1 and \u017f.")


def ovis_geometry(inputs, processor, spec, original_size):
    grids = inputs['image_grid_thw'].tolist()
    if len(grids) != 1 or len(grids[0]) != 3:
        raise ValueError('Expected one full-page image grid')
    temporal, height, width = (int(value) for value in grids[0])
    patch = int(processor.image_processor.patch_size)
    merge = int(processor.image_processor.merge_size)
    if min(temporal, height, width, patch, merge) <= 0:
        raise ValueError('Invalid visual geometry')
    pixels = height * width * patch * patch
    tokens = temporal * height * width // (merge * merge)
    if pixels > spec['max_pixels'] or tokens > spec.get('max_visual_tokens', tokens):
        raise ValueError(f'Visual budget exceeded before GPU transfer: {pixels} pixels, {tokens} tokens')
    return {'original_size_wh': list(original_size),
            'processed_size_wh': [width * patch, height * patch],
            'image_grid_thw': [temporal, height, width], 'processed_pixels': pixels,
            'visual_tokens': tokens, 'input_tokens': int(inputs['input_ids'].shape[1]),
            'requested_max_pixels': spec['max_pixels']}


def ovis_generation_settings(model, processor, spec):
    settings = {'max_new_tokens':spec['max_new_tokens'], 'do_sample':False}
    tokenizer = processor.tokenizer
    default_eos = getattr(model.generation_config, 'eos_token_id', None)
    trace = {'model_default_eos_token_id':default_eos,
             'tokenizer_eos_token_id':tokenizer.eos_token_id,
             'stop_override_applied':False}
    if expected := spec.get('stop_token_ids'):
        ids = []
        for token, expected_id in expected.items():
            actual = tokenizer.convert_tokens_to_ids(token)
            if actual != expected_id or tokenizer.convert_ids_to_tokens(actual) != token:
                raise ValueError(f'Unexpected stop-token identity: {token} -> {actual}')
            ids.append(actual)
        if tokenizer.eos_token_id not in ids or tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer EOS and padding must be explicitly supported')
        settings.update(eos_token_id=sorted(set(ids)), pad_token_id=tokenizer.pad_token_id)
        trace['stop_override_applied'] = True
    trace['applied_generation_settings'] = settings
    return settings, trace


def load_ovis(spec):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        spec["repo"], revision=spec["revision"], dtype=dtype,
        attn_implementation="sdpa").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(spec["repo"], revision=spec["revision"])
    generation_settings, generation_trace = ovis_generation_settings(model, processor, spec)
    print(json.dumps({'ovis_generation_settings':generation_trace}), flush=True)

    def infer(path, page):
        with Image.open(path) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": OVIS_PROMPT}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
            processor_kwargs={'images_kwargs': {'size': {
                'shortest_edge': spec['min_pixels'], 'longest_edge': spec['max_pixels']}}})
        geometry = ovis_geometry(inputs, processor, spec, image.size)
        print(json.dumps({'page': page['id'], 'ovis_input_geometry': geometry}), flush=True)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_settings)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        token_ids = generated.tolist() if hasattr(generated, 'tolist') else list(generated)
        eos = generation_settings.get('eos_token_id', generation_trace['model_default_eos_token_id'])
        eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
        ended_with_eos = bool(token_ids and token_ids[-1] in eos)
        hit_limit = len(token_ids) >= spec['max_new_tokens'] and not ended_with_eos
        raw = processor.decode(generated, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        trace = {**generation_trace, 'generated_token_ids':token_ids,
                 'eos_positions':[index for index,token in enumerate(token_ids) if token in eos],
                 'decoded_with_special_tokens':processor.decode(generated, skip_special_tokens=False,
                                                                clean_up_tokenization_spaces=False)}
        return {"ovis-ocr2-full-page": {"text": raw, "format": "markdown",
                "generated_tokens": len(generated),
                "input_geometry": geometry,
                'generation_trace':trace,
                'finish_reason':'eos' if ended_with_eos else ('length' if hit_limit else 'unknown'),
                "token_limit_reached": hit_limit}}

    return infer


def load_qwen(spec):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                      bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_use_double_quant=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        spec['repo'], revision=spec['revision'], device_map={'': 0},
        torch_dtype=torch.float16, attn_implementation='sdpa',
        quantization_config=quantization).eval()
    processor = AutoProcessor.from_pretrained(
        spec['repo'], revision=spec['revision'],
        min_pixels=spec['min_pixels'], max_pixels=spec['max_pixels'])
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(value is None for value in eos):
        raise ValueError('Qwen requires an explicit model EOS token')

    def infer(path, page):
        with Image.open(path) as source:
            image = source.convert('RGB')
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image}, {'type': 'text', 'text': spec['prompt']}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt')
        grid = inputs['image_grid_thw'].tolist()
        patch = processor.image_processor.patch_size
        merge = processor.image_processor.merge_size
        geometry = {'original_width': image.width, 'original_height': image.height,
                    'image_grid_thw': grid, 'patch_size': patch, 'merge_size': merge,
                    'processed_width': grid[0][2]*patch, 'processed_height': grid[0][1]*patch,
                    'processed_pixels': grid[0][1]*grid[0][2]*patch*patch,
                    'visual_tokens': sum(t*h*w//(merge*merge) for t,h,w in grid),
                    'min_pixels': spec['min_pixels'], 'max_pixels': spec['max_pixels']}
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=spec['max_new_tokens'],
                                     do_sample=False, eos_token_id=eos,
                                     pad_token_id=processor.tokenizer.pad_token_id)
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        ids = generated.tolist()
        ended = bool(ids and ids[-1] in eos)
        capped = len(ids) >= spec['max_new_tokens'] and not ended
        raw = processor.decode(generated, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        return {'qwen3-vl-4b-full-page': {
            'text': raw, 'format': 'plain', 'generated_tokens': len(ids),
            'finish_reason': 'eos' if ended else ('length' if capped else 'unknown'),
            'token_limit_reached': capped, 'input_geometry': geometry,
            'generation_trace': {'generated_token_ids': ids, 'eos_token_ids': eos,
                'decoded_with_special_tokens': processor.decode(
                    generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)}}}

    return infer


def run_worker(engine, config, inputs, output):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = validate_inputs(inputs)
    spec = config["models"][engine]
    identity = {"engine": engine, "spec": spec, "input_sha256": digest(inputs),
                "runner_sha256": digest(__file__)}
    identity_file = output / "identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text(encoding="utf-8")) != identity:
        raise ValueError("Existing predictions belong to different inputs/code/model")
    write_json(identity_file, identity)
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab")
    torch.manual_seed(42)
    variants = {'mixed-v3': ['mixed-v3-row-major', 'mixed-v3-column-order', 'mixed-v3-source-regions'],
                'ovis-ocr2': ['ovis-ocr2-full-page'],
                'qwen3-vl-4b': ['qwen3-vl-4b-full-page']}[engine]
    existing = {}
    for variant in variants:
        path = output / f"{variant}.jsonl"
        records = read_rows(path) if path.exists() else []
        if len({row["id"] for row in records}) != len(records):
            raise ValueError("Duplicate checkpoint prediction")
        if set(row["id"] for row in records) - set(row["id"] for row in rows):
            raise ValueError("Unknown checkpoint page")
        existing[variant] = {row["id"] for row in records}
    pending = [row for row in rows if any(row["id"] not in existing[v] for v in variants)]
    load_error = None
    started = time.perf_counter()
    try:
        loader = {'mixed-v3': load_mixed, 'ovis-ocr2': load_ovis, 'qwen3-vl-4b': load_qwen}[engine]
        infer = loader(spec) if pending else None
    except Exception as exc:
        infer, load_error = None, f"{type(exc).__name__}: {exc}"
    load_seconds = time.perf_counter() - started
    for index, page in enumerate(pending, 1):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            if load_error:
                raise RuntimeError(load_error)
            torch.cuda.synchronize()
            predictions = infer(Path(inputs).parent / page["image"], page)
            torch.cuda.synchronize()
        except Exception as exc:
            predictions = {variant: {"status": "error", "text": "",
                                     "error": f"{type(exc).__name__}: {exc}"} for variant in variants}
        elapsed = time.perf_counter() - started
        for variant in variants:
            if page["id"] in existing[variant]:
                continue
            record = {"id": page["id"], "status": "ok", "elapsed_seconds": elapsed,
                      "elapsed_scope": "all variants for this engine/page; not per-variant latency",
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(), **predictions[variant]}
            with (output / f"{variant}.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
        print(f'{engine} {index}/{len(pending)}: {page["id"]}', flush=True)
    environment = {"gpu": torch.cuda.get_device_name(0), "python": sys.version,
                   "model_load_seconds": load_seconds, "load_error": load_error,
                   "reference_text_sent_to_model": False, "packages": {}}
    for package in ("torch", "transformers", "huggingface_hub", "tokenizers", "Pillow"):
        environment["packages"][package] = importlib.metadata.version(package)
    if engine == 'qwen3-vl-4b':
        for name in ('bitsandbytes', 'accelerate'):
            environment['packages'][name] = importlib.metadata.version(name)
    write_json(output / "environment.json", environment)
    failed = {row['id'] for variant in variants for row in read_rows(output / f'{variant}.jsonl')
              if row.get('status') != 'ok'}
    return {'pages': len(rows), 'failed_pages': len(failed), 'successful_pages': len(rows) - len(failed)}


class TextHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag in ("p", "br", "td", "th", "tr", "div"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if tag in ("p", "td", "th", "tr", "div"):
            self.parts.append(" ")

    def handle_data(self, value):
        if not self.hidden:
            self.parts.append(value)


def markdown_text(value):
    from markdown_it import MarkdownIt
    # Parse presentation syntax only. Do not repair letters, words or OCR repeats.
    html = MarkdownIt("commonmark", {"html": True}).render(value)
    parser = TextHTMLParser()
    parser.feed(html)
    return "".join(parser.parts)


def score(manifest, prediction_root, output):
    from training.benchmark_pages import evaluate
    rows = read_rows(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for source in sorted(Path(prediction_root).glob("*/*.jsonl")):
        variant = source.stem
        projected = []
        for row in read_rows(source):
            projected.append({**row, "raw_text": row["text"],
                              "text": markdown_text(row["text"]) if row.get("format") == "markdown"
                              else row["text"]})
        path = output / f"{variant}.jsonl"
        write_rows(path, projected)
        report = evaluate(manifest, path)
        report["cer_macro"] = sum(item["cer"] for item in report["results"]) / report["pages"]
        report["token_limit_pages"] = sum(item.get("token_limit_reached", False) for item in projected)
        report["oracle_geometry"] = "source-regions" in variant
        # This stage has no human completeness attestations; never silently promote it.
        report["reference_status"] = "source-unreviewed"
        report["eligible_for_model_promotion"] = False
        report["groups"] = {}
        for key in ("split", "collection"):
            for group in sorted({row[key] for row in rows}):
                group_hash = hashlib.sha256(f"{variant}:{key}:{group}".encode()).hexdigest()[:16]
                subset = Path(manifest).parent / f".pilot-score-{group_hash}.jsonl"
                supplied = {item["id"]: item for item in projected}
                selected = [row for row in rows if row[key] == group]
                predictions = output / f".subset-{variant}.jsonl"
                try:
                    write_rows(subset, selected)
                    write_rows(predictions, [supplied[row["id"]] for row in selected if row["id"] in supplied])
                    group_report = evaluate(subset, predictions)
                    report["groups"][f"{key}:{group}"] = {field: group_report[field]
                                                         for field in ("pages", "cer_micro", "wer_micro", "errors_or_missing")}
                finally:
                    subset.unlink(missing_ok=True)
                    predictions.unlink(missing_ok=True)
        reports[variant] = report
    if not reports:
        raise ValueError("No prediction artifacts found")
    summary = {"schema": "slayer-full-page-pilot-result-v1", "pages": len(rows),
               "reports": reports, "model_promotion": False, "sota_claim": False,
               "normalization": "Unicode NFC + whitespace only",
               "projection": "CommonMark to HTML to visible text; raw outputs retained",
               "claim_boundary": "Unreviewed source references, possible training exposure, "
                                 "small development sample. Oracle and automatic results are separate."}
    write_json(output / "metrics.json", summary)
    return summary


def package(work):
    work = Path(work)
    files = [path for path in work.rglob("*") if path.is_file()
             and not any(part.startswith(".") for part in path.relative_to(work).parts)
             and path.suffix in (".json", ".jsonl", ".log")
             and "model" not in path.relative_to(work).parts]
    import zipfile
    config_path = work / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    archive_name = safe_relative(config.get('evidence_archive_name', 'full-page-pilot-v1-evidence.zip'))
    if '/' in archive_name or not archive_name.endswith('.zip'):
        raise ValueError('Expected an evidence ZIP basename')
    archive = work / archive_name
    checksums = {path.relative_to(work).as_posix(): digest(path) for path in files}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            stream.write(path, path.relative_to(work).as_posix())
        stream.writestr("checksums.json", json.dumps(checksums, indent=2))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "worker"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pages", type=int)
    parser.add_argument("--engine", choices=("mixed-v3", "ovis-ocr2", "qwen3-vl-4b"))
    parser.add_argument("--inputs")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.command == "stage":
        report = stage(config, args.output, args.pages)
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    else:
        if not args.engine or not args.inputs:
            parser.error("worker requires --engine and --inputs")
        result = run_worker(args.engine, config, args.inputs, args.output)
        print(json.dumps({'worker_result': result}), flush=True)
        if result['failed_pages']:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
