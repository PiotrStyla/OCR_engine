import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from training import run_sota_benchmark as sota
from training.run_vision_baseline import PROMPT_VERSION
from training.validate_submission import ZERO_SHOT_PROMPTS


def staged(tmp_path, texts):
    (tmp_path / "images").mkdir(parents=True)
    rows = []
    for index, text in enumerate(texts):
        image = tmp_path / "images" / f"page{index}.png"
        Image.new("RGB", (8, 8), "white").save(image)
        rows.append({"id": f"page{index}", "image": f"images/page{index}.png", "text": text,
                     "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def by_path(rows):
    return {row["image"]: row["text"] for row in rows}


TEXTS = ["# Naglowek\nPierwszy akapit strony.", "Druga strona bez naglowka.",
         "### Sekcja\n- punkt pierwszy\n- punkt drugi"]


def score_of(directory):
    return json.loads((directory / "score.json").read_text(encoding="utf-8"))


def test_perfect_predictor_scores_zero_error(tmp_path):
    rows = staged(tmp_path, TEXTS)
    lookup = by_path(rows)
    report = sota.run("paddlevl", tmp_path, tmp_path / "out", predictor=lambda p: lookup[p.relative_to(tmp_path).as_posix()])
    predictions = [json.loads(line) for line in (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()]
    assert [row["status"] for row in predictions] == ["ok"] * 3
    score = score_of(tmp_path / "out")
    assert score["cer_micro"] == 0.0 and score["wer_micro"] == 0.0
    assert score["structure_similarity"] == 1.0 and score["errors_or_missing"] == 0
    assert report["measurement_only"] is True and report["training_performed"] is False
    assert report["prompt_used"] is False and report["prompt_sha256"] is None


def test_failed_page_is_scored_not_fatal(tmp_path):
    rows = staged(tmp_path, TEXTS)
    lookup = by_path(rows)

    def predict(path):
        if path.name == "page1.png":
            raise RuntimeError("model crashed")
        return lookup[path.relative_to(tmp_path).as_posix()]

    sota.run("paddlevl", tmp_path, tmp_path / "out", predictor=predict)
    score = score_of(tmp_path / "out")
    assert score["errors_or_missing"] == 1
    failed = [row for row in score["results"] if row["id"] == "page1"][0]
    assert failed["status"] == "error" and failed["cer"] == 1.0
    assert failed["structure"]["structure_similarity"] == 0.0
    saved = json.loads((tmp_path / "out" / "predictions.jsonl").read_text().splitlines()[1])
    assert saved["text"] == "" and "model crashed" in saved["error"]


def test_smoke_limit_keeps_full_manifest_score(tmp_path):
    rows = staged(tmp_path, TEXTS)
    lookup = by_path(rows)
    report = sota.run("paddlevl", tmp_path, tmp_path / "out",
                      predictor=lambda p: lookup[p.relative_to(tmp_path).as_posix()], limit=1)
    assert report["pages_total"] == 3 and report["pages_predicted"] == 1
    assert score_of(tmp_path / "out")["errors_or_missing"] == 2
    assert len((tmp_path / "out" / "predictions.jsonl").read_text().splitlines()) == 1


def test_vlm_run_records_frozen_prompt(tmp_path):
    rows = staged(tmp_path, TEXTS)
    lookup = by_path(rows)
    report = sota.run("qwen3vl", tmp_path, tmp_path / "out",
                      predictor=lambda p: lookup[p.relative_to(tmp_path).as_posix()])
    assert report["prompt_used"] is True and report["prompt_version"] == PROMPT_VERSION
    assert report["prompt_sha256"] == ZERO_SHOT_PROMPTS[PROMPT_VERSION]
    assert report["decoding"] == "greedy"
    assert report["max_pixels"] == sota.DEFAULT_MAX_PIXELS and report["attn_implementation"] == "sdpa"
    prompt = Path(__file__).resolve().parents[1] / "benchmarks/polocrbench/prompts/zero_shot_prompt_v1.md"
    assert report["prompt_sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()


@pytest.mark.parametrize("result, expected", [
    (type("R", (), {"markdown": {"markdown_texts": "  # Tytul  "}})(), "# Tytul"),
    ({"markdown": {"markdown_texts": "tekst"}}, "tekst"),
    ({"markdown_text": "wprost"}, "wprost"),
    (type("R", (), {"markdown": "surowy markdown"})(), "surowy markdown"),
])
def test_extract_markdown_documented_shapes(result, expected):
    assert sota.extract_markdown(result) == expected


@pytest.mark.parametrize("result", [{"markdown": {}}, {"markdown": {"markdown_texts": "  "}}, object()])
def test_extract_markdown_rejects_unusable_results(result):
    with pytest.raises(ValueError, match="markdown"):
        sota.extract_markdown(result)


def test_model_image_transcodes_tiff_and_passes_png(tmp_path):
    from PIL import Image
    source = tmp_path / "page.jpg"
    pixels = Image.new("L", (20, 10))
    pixels.putpixel((4, 5), 77)
    pixels.save(source, format="TIFF")
    converted = sota.model_image(source)
    assert converted.suffix == ".png" and converted != source
    with Image.open(converted) as image:
        assert image.format == "PNG" and image.size == (20, 10)
        assert image.getpixel((4, 5)) == 77
    png = tmp_path / "plain.png"
    pixels.save(png, format="PNG")
    assert sota.model_image(png) == png


def test_runner_passes_transcoded_image_to_predictor(tmp_path):
    from PIL import Image
    rows = staged(tmp_path, TEXTS)
    Image.new("L", (8, 8)).save(tmp_path / rows[0]["image"], format="TIFF")
    rows[0]["sha256"] = hashlib.sha256((tmp_path / rows[0]["image"]).read_bytes()).hexdigest()
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    seen = []

    def predict(path):
        seen.append(path)
        return TEXTS[0]

    sota.run("qwen3vl", tmp_path, tmp_path / "out", predictor=predict, limit=1)
    assert seen[0].suffix == ".png" and seen[0] != tmp_path / rows[0]["image"]


def test_runner_rejects_unknown_model_and_existing_output(tmp_path):
    with pytest.raises(ValueError, match="Unknown model"):
        sota.run("gpt5", tmp_path, tmp_path / "out", predictor=lambda p: "")
    rows = staged(tmp_path, TEXTS)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        sota.run("paddlevl", tmp_path, tmp_path / "out", predictor=lambda p: "")


def test_evidence_is_deterministic_for_identical_predictions(tmp_path):
    rows = staged(tmp_path, TEXTS)
    lookup = by_path(rows)
    first = sota.run("paddlevl", tmp_path, tmp_path / "a",
                     predictor=lambda p: lookup[p.relative_to(tmp_path).as_posix()])
    second = sota.run("paddlevl", tmp_path, tmp_path / "a2",
                      predictor=lambda p: lookup[p.relative_to(tmp_path).as_posix()])
    for key in ("cer_micro", "wer_micro", "structure_similarity", "benchmark_manifest_sha256"):
        assert first[key] == second[key]

    def payloads(directory):
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        return [{k: v for k, v in row.items() if k != "elapsed_seconds"} for row in rows]

    def score(directory):
        data = json.loads((directory / "score.json").read_text())
        data.pop("predictions_sha256")
        return data

    assert payloads(tmp_path / "a") == payloads(tmp_path / "a2")
    assert score(tmp_path / "a") == score(tmp_path / "a2")
