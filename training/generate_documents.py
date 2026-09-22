"""Generate synthetic Polish document pages for PolOCRBench with full ground truth.

One run produces page images plus ground truth for all three subtasks:
``manifest-A.jsonl`` (Markdown transcription), ``manifest-B.jsonl`` (per-table
HTML) and ``manifest-C.jsonl`` (KIE fields) — every sample scores in
``training.transcription_eval``, ``training.table_eval`` and
``training.kie_eval``. Content comes from ``training.document_templates``
(fields, tables and visible text are always consistent).

Degradation recipes (``--degradations``, Test B can hold out any subset):

- ``clean`` — control, plain render;
- ``scan`` — skew, sensor noise, soft blur, brightness/contrast jitter;
- ``photo`` — perspective, shadow, blur, low resolution, JPEG artifacts;
- ``print_scan`` — contrast crush, noise, skew, JPEG artifacts;
- ``compress`` — downscale plus heavy JPEG artifacts.

Deterministic: sample ``i`` is built from ``sha256(seed:index)``; the same seed
reproduces identical files with the same Pillow version. Output layout::

    images/{id}.png
    manifest-A.jsonl / manifest-B.jsonl / manifest-C.jsonl
    generation.json   (per-sample recipes, versions, manifest hashes)

Usage: python -m training.generate_documents --output data/polocrbench-synth-v1 \
    --count 1200 --seed 20260922 --split train
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from training.document_templates import (
    DOC_TYPES,
    PROTOCOL_VERSION as TEMPLATES_VERSION,
    build_document,
    document_text,
)
from training.generate_synthetic import _find_fonts, _font_supports_pl

PROTOCOL_VERSION = 'polocrbench-synth-pages-v1'
PAGE_SIZE = (1240, 1754)  # A4 at 150 dpi
MARGIN = 90
DEGRADATION_NAMES = ('clean', 'scan', 'photo', 'print_scan', 'compress')


# --- fonts and text measurement ----------------------------------------------


def load_fonts(fonts_dir=None):
    """Candidate .ttf/.otf files; [] falls back to the Pillow default font."""
    if fonts_dir is None:
        for candidate in (Path(r'C:\Windows\Fonts'), Path('/usr/share/fonts')):
            if candidate.is_dir():
                fonts_dir = candidate
                break
    if fonts_dir is None:
        return []
    fonts = _find_fonts(Path(fonts_dir))
    try:
        supported = [font for font in fonts if _font_supports_pl(font)]
    except ImportError:  # fontTools unavailable: keep every readable candidate
        return fonts
    return supported or fonts


def _font(fonts, cache, size):
    if size not in cache:
        if fonts:
            cache[size] = ImageFont.truetype(str(fonts[0]), size)
        else:
            try:
                cache[size] = ImageFont.load_default(size=size)
            except TypeError:  # Pillow < 10.1
                cache[size] = ImageFont.load_default()
    return cache[size]


def _text_width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _wrap(draw, text, font, width):
    lines, current = [], ''
    for word in text.split():
        candidate = f'{current} {word}'.strip()
        if current and _text_width(draw, candidate, font) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or ['']


# --- rendering ----------------------------------------------------------------


def render_page(doc, rng, fonts):
    """Rasterize one document model into an A4 page (pure white background)."""
    image = Image.new('RGB', PAGE_SIZE, 'white')
    draw = ImageDraw.Draw(image)
    cache, y = {}, MARGIN
    width = PAGE_SIZE[0] - 2 * MARGIN

    def put(text, size, x, indent=0, spacing=6):
        nonlocal y
        font = _font(fonts, cache, size)
        line_h = size + spacing
        for line in _wrap(draw, text, font, width - indent):
            draw.text((x, y), line, fill='black', font=font)
            y += line_h
        return line_h

    for block in doc['blocks']:
        kind = block['kind']
        if kind == 'heading':
            y += 10
            font = _font(fonts, cache, 42 if block['level'] == 1 else 34)
            for line in _wrap(draw, block['text'], font, width):
                text_w = _text_width(draw, line, font)
                x = MARGIN if block['level'] > 1 else (PAGE_SIZE[0] - text_w) // 2
                draw.text((x, y), line, fill='black', font=font)
                y += (42 if block['level'] == 1 else 34) + 8
            y += 8
        elif kind == 'kv':
            put(f"{block['label']}: {block['value']}", 30, MARGIN)
            y += 4
        elif kind == 'paragraph':
            put(block['text'], 30, MARGIN, spacing=10)
            y += 10
        elif kind == 'list':
            for index, item in enumerate(block['items'], 1):
                prefix = f'{index}. ' if block['ordered'] else '- '
                font = _font(fonts, cache, 30)
                lines = _wrap(draw, item, font, width - 40)
                for offset, line in enumerate(lines):
                    text = prefix + line if offset == 0 else line
                    draw.text((MARGIN + (0 if offset == 0 else 40), y), text,
                              fill='black', font=font)
                    y += 40
            y += 6
        elif kind == 'table':
            y = _draw_table(draw, doc['tables'][block['index']], fonts, cache, y, width) + 16
        elif kind == 'signatures':
            y += 30
            font = _font(fonts, cache, 28)
            column_w = width // 2 - 30
            for left, right in zip(block['left'], block['right']):
                left_lines = _wrap(draw, left, font, column_w)
                right_lines = _wrap(draw, right, font, column_w)
                for offset in range(max(len(left_lines), len(right_lines))):
                    if offset < len(left_lines):
                        draw.text((MARGIN, y), left_lines[offset], fill='black', font=font)
                    if offset < len(right_lines):
                        draw.text((MARGIN + width // 2 + 30, y), right_lines[offset],
                                  fill='black', font=font)
                    y += 38
    return image


def _draw_table(draw, table, fonts, cache, y, width):
    rows = table['cells']
    font = _font(fonts, cache, 26)
    columns = max(sum(span for _, span in row) for row in rows)
    natural = [0] * columns
    for row in rows:
        column = 0
        for text, span in row:
            if span == 1:
                natural[column] = max(natural[column], _text_width(draw, text, font) + 16)
            column += span
    natural = [value or 80 for value in natural]
    total = sum(natural)
    widths = [max(60, int(value * width / total)) for value in natural]
    scale = width / sum(widths)
    widths = [int(value * scale) for value in widths]
    x_positions = [MARGIN + sum(widths[:i]) for i in range(columns + 1)]
    for row in rows:
        column, wrapped = 0, []
        for text, span in row:
            cell_width = sum(widths[column:column + span]) - 12
            wrapped.append((_wrap(draw, text, font, cell_width), span))
            column += span
        row_h = max(len(lines) for lines, _ in wrapped) * 32 + 12
        column = 0
        for lines, span in wrapped:
            x0, x1 = x_positions[column], x_positions[column + span]
            draw.rectangle([x0, y, x1, y + row_h], outline='black', width=1)
            for offset, line in enumerate(lines):
                draw.text((x0 + 6, y + 6 + offset * 32), line, fill='black', font=font)
            column += span
        y += row_h
    return y


# --- degradations -------------------------------------------------------------


def _noise(image, rng, sigma):
    generator = np.random.default_rng(rng.getrandbits(64))
    array = np.asarray(image).astype(np.int16)
    array = np.clip(array + generator.normal(0, sigma, array.shape), 0, 255)
    return Image.fromarray(array.astype(np.uint8), 'RGB')


def _jpeg(image, rng, low, high):
    quality = rng.randint(low, high)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB'), quality


def _degrade_clean(image, rng):
    return image.copy(), {'kind': 'clean'}


def _degrade_scan(image, rng):
    angle = rng.uniform(-1.5, 1.5)
    sigma = rng.uniform(1.5, 4.0)
    radius = rng.uniform(0.3, 0.8)
    result = image.rotate(angle, resample=Image.Resampling.BICUBIC,
                          expand=False, fillcolor='white')
    result = _noise(result, rng, sigma)
    result = result.filter(ImageFilter.GaussianBlur(radius))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.9, 1.1))
    result = ImageEnhance.Brightness(result).enhance(rng.uniform(0.95, 1.05))
    return result, {'kind': 'scan', 'angle': round(angle, 3), 'noise': round(sigma, 3),
                    'blur': round(radius, 3)}


def _degrade_photo(image, rng):
    width, height = image.size
    spread = 0.06

    def jitter(x, y):
        return (x + rng.uniform(-spread, spread) * width,
                y + rng.uniform(-spread, spread) * height)

    quad = (*jitter(0, 0), *jitter(0, height), *jitter(width, height), *jitter(width, 0))
    result = image.transform((width, height), Image.Transform.QUAD, quad,
                             resample=Image.Resampling.BICUBIC, fillcolor='white')
    generator = np.random.default_rng(rng.getrandbits(64))
    x = np.linspace(0, 1, width)[None, :]
    y = np.linspace(0, 1, height)[:, None]
    ramp = generator.uniform(0.4, 1.0) * x + generator.uniform(0.4, 1.0) * y
    ramp = ramp / ramp.max()
    depth = rng.uniform(0.35, 0.6)
    factor = (1.0 - depth * (1.0 - ramp))[..., None]
    array = np.clip(np.asarray(result).astype(np.float32) * factor, 0, 255).astype(np.uint8)
    result = Image.fromarray(array, 'RGB')
    radius = rng.uniform(0.8, 2.0)
    result = result.filter(ImageFilter.GaussianBlur(radius))
    scale = rng.uniform(0.5, 0.8)
    small = result.resize((int(width * scale), int(height * scale)),
                          Image.Resampling.BICUBIC)
    result = small.resize((width, height), Image.Resampling.BICUBIC)
    result = _noise(result, rng, rng.uniform(2.0, 5.0))
    result, quality = _jpeg(result, rng, 35, 65)
    return result, {'kind': 'photo', 'quad': [round(v, 1) for v in quad],
                    'shadow_depth': round(depth, 3), 'blur': round(radius, 3),
                    'scale': round(scale, 3), 'jpeg_quality': quality}


def _degrade_print_scan(image, rng):
    result = ImageOps.autocontrast(image)
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(1.5, 2.0))
    result = _noise(result, rng, rng.uniform(3.0, 6.0))
    angle = rng.uniform(-1.0, 1.0)
    result = result.rotate(angle, resample=Image.Resampling.BICUBIC,
                           expand=False, fillcolor='white')
    result = result.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))
    result, quality = _jpeg(result, rng, 50, 75)
    return result, {'kind': 'print_scan', 'angle': round(angle, 3),
                    'jpeg_quality': quality}


def _degrade_compress(image, rng):
    width, height = image.size
    scale = rng.uniform(0.6, 0.9)
    result = image.resize((int(width * scale), int(height * scale)),
                          Image.Resampling.BICUBIC)
    result = result.resize((width, height), Image.Resampling.BICUBIC)
    result, quality = _jpeg(result, rng, 15, 35)
    return result, {'kind': 'compress', 'scale': round(scale, 3),
                    'jpeg_quality': quality}


DEGRADATIONS = {'clean': _degrade_clean, 'scan': _degrade_scan, 'photo': _degrade_photo,
                'print_scan': _degrade_print_scan, 'compress': _degrade_compress}


# --- generation ---------------------------------------------------------------


def _sample_rng(seed, index):
    digest = hashlib.sha256(f'{seed}:{index}'.encode('utf-8')).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_jsonl(path, rows):
    Path(path).write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n',
                          encoding='utf-8', newline='\n')


def generate(output, count, seed, split='train', types=DOC_TYPES,
             degradations=DEGRADATION_NAMES, fonts_dir=None):
    """Write images, A/B/C manifests and generation.json; refuses existing output."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    (output / 'images').mkdir(parents=True, exist_ok=True)
    fonts = load_fonts(fonts_dir)
    rows_a, rows_b, rows_c, samples = [], [], [], []
    for index in range(count):
        rng = _sample_rng(seed, index)
        doc_type = types[index % len(types)]
        degradation = degradations[(index // len(types)) % len(degradations)]
        doc = build_document(rng, doc_type)
        image = render_page(doc, rng, fonts)
        image, recipe = DEGRADATIONS[degradation](image, rng)
        sample_id = f'{doc_type}-{seed}-{index:05d}'
        image_path = output / 'images' / f'{sample_id}.png'
        image.save(image_path)
        relative = f'images/{sample_id}.png'
        sha = _digest(image_path)
        rows_a.append({'id': sample_id, 'image': relative, 'sha256': sha,
                       'text': document_text(doc)})
        for table_index, table in enumerate(doc['tables']):
            rows_b.append({'id': f'{sample_id}__t{table_index}', 'image': relative,
                           'sha256': sha, 'html': table['html'],
                           'table_index': table_index})
        rows_c.append({'id': sample_id, 'image': relative, 'sha256': sha,
                       'doc_type': doc_type, 'fields': doc['fields']})
        samples.append({'id': sample_id, 'doc_type': doc_type,
                        'degradation': degradation, 'recipe': recipe,
                        'font': fonts[0].name if fonts else 'pil-default'})
    _write_jsonl(output / 'manifest-A.jsonl', rows_a)
    _write_jsonl(output / 'manifest-B.jsonl', rows_b)
    _write_jsonl(output / 'manifest-C.jsonl', rows_c)
    report = {'schema': PROTOCOL_VERSION, 'templates': TEMPLATES_VERSION,
              'seed': seed, 'count': count, 'split': split,
              'types': list(types), 'degradations': list(degradations),
              'fonts': len(fonts), 'python': sys.version.split()[0],
              'pillow': PIL.__version__, 'samples': samples,
              'manifest_sha256': {name: _digest(output / name) for name in
                                  ('manifest-A.jsonl', 'manifest-B.jsonl', 'manifest-C.jsonl')}}
    (output / 'generation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding='utf-8', newline='\n')
    return report


def _csv_list(value, allowed):
    items = [item.strip() for item in value.split(',') if item.strip()]
    unknown = set(items) - set(allowed)
    if not items or unknown:
        raise argparse.ArgumentTypeError(f'Expected a subset of {sorted(allowed)}, got {value!r}')
    return tuple(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--split', default='train')
    parser.add_argument('--types', default=','.join(DOC_TYPES),
                        help='comma-separated subset of document types')
    parser.add_argument('--degradations', default=','.join(DEGRADATION_NAMES),
                        help='comma-separated subset of degradation recipes')
    parser.add_argument('--fonts-dir')
    args = parser.parse_args()
    report = generate(args.output, args.count, args.seed, split=args.split,
                      types=_csv_list(args.types, DOC_TYPES),
                      degradations=_csv_list(args.degradations, DEGRADATION_NAMES),
                      fonts_dir=args.fonts_dir)
    print(json.dumps({key: report[key] for key in
                      ('seed', 'count', 'split', 'types', 'degradations', 'fonts')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
