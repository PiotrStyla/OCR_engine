import hashlib
import importlib.util
from contextlib import nullcontext
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
from PIL import Image

from training.full_page_pilot import (
    digest, inspect_reference, markdown_text, package, safe_relative, score,
    run_worker, select_pages, stage, validate_inputs, write_json, write_rows, ovis_geometry, load_ovis,
)
from training.build_full_page_pilot_colab import ENVIRONMENT_SETUP_SOURCE, build


def xml(width=20, height=30):
    return f'''<PcGts><Page imageWidth="{width}" imageHeight="{height}">
    <ReadingOrder><OrderedGroup><RegionRefIndexed index="0" regionRef="r"/></OrderedGroup></ReadingOrder>
    <TextRegion id="r"><Coords points="1,2 18,2 18,10 1,10"/>
    <TextLine id="l"><Coords points="1,2 18,2 18,10 1,10"/>
    <TextEquiv><Unicode>Do not send this reference to the model</Unicode></TextEquiv>
    </TextLine></TextRegion></Page></PcGts>'''.encode()


def fixture(tmp_path):
    stream = io.BytesIO()
    Image.new("RGB", (20, 30), "white").save(stream, format="JPEG")
    image = stream.getvalue()
    row = {"id": "p", "split": "validation", "collection": "C", "width": 20, "height": 30,
           "file_name": "images/p.jpg", "pagexml": "pages/validation/pagexml/p.xml",
           "image_sha256": hashlib.sha256(image).hexdigest(),
           "pagexml_sha256": hashlib.sha256(xml()).hexdigest(),
           "text": "W świetne błáwaty ſtara pisownia", "license": "CC-BY-3.0"}
    metadata = (json.dumps(row) + "\n").encode()
    data = {"pages/validation/metadata.jsonl": metadata,
            "pages/validation/images/p.jpg": image, "pages/validation/pagexml/p.xml": xml()}
    config = {"dataset": {"repo": "Owner/data", "revision": "a" * 40, "license": "CC-BY-3.0",
                          "splits": {"validation": {"pages": 1, "metadata_sha256": hashlib.sha256(metadata).hexdigest()}},
                          "excluded_collections": ["Forbidden"]},
              "selection_salt": "v1", "target_review_pages": 100, "available_non_test_pages": 1}
    def opener(url, **kwargs):
        return io.BytesIO(data[url.split("/resolve/" + "a" * 40 + "/")[1]])
    output = tmp_path / "dataset"
    stage(config, output, opener=opener)
    return config, output, opener


def test_stage_is_repeatable_and_never_gold(tmp_path):
    config, output, opener = fixture(tmp_path)
    original = digest(output / "manifest.jsonl")
    report = stage(config, output, opener=opener)
    assert digest(output / "manifest.jsonl") == original
    assert report["gold_pages"] == 0
    assert report["additional_pages_needed"] == 99
    rows = validate_inputs(output / "inference-inputs.jsonl")
    assert "text" not in rows[0]
    assert "Unicode" not in json.dumps(rows)
    assert rows[0]["source_regions"][0]["bbox_xyxy"] == [1, 2, 19, 11]


def test_zero_page_scope_is_rejected(tmp_path):
    config, output, opener = fixture(tmp_path)
    with pytest.raises(ValueError, match="Page count"):
        stage(config, output, count=0, opener=opener)


@pytest.mark.parametrize("path", ["../x", "/x", "C:/x", "x\\y", ""])
def test_paths_cannot_escape(path):
    with pytest.raises(ValueError):
        safe_relative(path)


def test_source_flags_are_not_completeness_proof():
    row = {"id": "p", "width": 20, "height": 30, "split": "validation",
           "collection": "C", "text": "á ſ \ufffd \ue000"}
    audit = inspect_reference(row, xml())
    assert audit["completeness"] == "unverified"
    assert audit["replacement_characters"] == 1
    assert audit["private_use_characters"] == 1
    with pytest.raises(ValueError, match="dimensions"):
        inspect_reference(row, xml(21))


def test_selection_is_independent_of_ocr_and_prefers_validation():
    rows = [{"id": str(i), "split": "validation" if i < 2 else "train",
             "collection": str(i % 2)} for i in range(8)]
    assert select_pages(rows, 2, "v1") == select_pages(list(reversed(rows)), 2, "v1")
    assert all(row["split"] == "validation" for row in select_pages(rows, 2, "v1"))
    with pytest.raises(ValueError):
        select_pages(rows, 9, "v1")


def test_changed_config_is_rejected(tmp_path):
    config, output, opener = fixture(tmp_path)
    config["selection_salt"] = "changed"
    with pytest.raises(ValueError, match="configuration"):
        stage(config, output, opener=opener)


def test_reference_cannot_enter_inference_inputs(tmp_path):
    _, output, _ = fixture(tmp_path)
    path = output / "inference-inputs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["text"] = "SECRET"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="Reference"):
        validate_inputs(path)


def test_nested_reference_is_rejected(tmp_path):
    _, output, _ = fixture(tmp_path)
    path = output / "inference-inputs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["source_regions"][0]["text"] = "SECRET"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="Reference"):
        validate_inputs(path)


def test_markdown_projection_preserves_historical_letters_and_tables():
    text = markdown_text("# W świetne **błáwaty**\n\nſłowo <table><tr><td>á</td><td>ſ</td></tr></table>")
    assert "błáwaty" in text and "ſłowo" in text
    assert "á ſ" in " ".join(text.split())


def test_scoring_counts_errors_and_never_promotes(tmp_path):
    _, output, _ = fixture(tmp_path)
    predictions = tmp_path / "predictions" / "test"
    predictions.mkdir(parents=True)
    write_rows(predictions / "candidate.jsonl", [{"id": "p", "status": "error", "text": "ignored"}])
    report = score(output / "manifest.jsonl", predictions.parent, tmp_path / "scores")
    assert report["reports"]["candidate"]["cer_micro"] == 1
    assert report["reports"]["candidate"]["errors_or_missing"] == 1
    assert report["model_promotion"] is False
    assert report["sota_claim"] is False


def test_historical_text_survives_scoring(tmp_path):
    _, output, _ = fixture(tmp_path)
    predictions = tmp_path / "predictions" / "test"
    predictions.mkdir(parents=True)
    write_rows(predictions / "candidate.jsonl", [{"id": "p", "status": "ok",
               "text": "# W świetne **błáwaty** ſtara pisownia", "format": "markdown"}])
    report = score(output / "manifest.jsonl", predictions.parent, tmp_path / "scores")
    assert report["reports"]["candidate"]["cer_micro"] == 0
    assert report["reports"]["candidate"]["eligible_for_model_promotion"] is False


def test_worker_load_failure_is_recorded_and_resume_does_not_erase_it(tmp_path, monkeypatch):
    from training import full_page_pilot as pilot
    _, dataset, _ = fixture(tmp_path)
    fake_torch = SimpleNamespace(manual_seed=lambda seed: None, cuda=SimpleNamespace(
        is_available=lambda: True, reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None, max_memory_allocated=lambda: 123,
        get_device_name=lambda index: "fixture"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(pilot.importlib.metadata, "version", lambda name: "fixture")
    calls = []
    def fail(spec):
        calls.append(spec)
        raise ImportError("fixture model load failed")
    monkeypatch.setattr(pilot, "load_mixed", fail)
    config = {"models": {"mixed-v3": {"revision": "a" * 40}}}
    output = tmp_path / "predictions"
    run_worker("mixed-v3", config, dataset / "inference-inputs.jsonl", output)
    before = (output / "mixed-v3-row-major.jsonl").read_bytes()
    run_worker("mixed-v3", config, dataset / "inference-inputs.jsonl", output)
    assert (output / "mixed-v3-row-major.jsonl").read_bytes() == before
    assert len(calls) == 1
    row = json.loads(before)
    assert row["status"] == "error" and "fixture model load failed" in row["error"]
    config["models"]["mixed-v3"]["revision"] = "b" * 40
    with pytest.raises(ValueError, match="different"):
        run_worker("mixed-v3", config, dataset / "inference-inputs.jsonl", output)


def test_evidence_reruns_and_excludes_scans_weights(tmp_path):
    _, output, _ = fixture(tmp_path)
    write_json(output / "metrics.json", {"cer": 1})
    for _ in range(2):
        archive = package(output)
        with zipfile.ZipFile(archive) as stream:
            names = stream.namelist()
            assert "metrics.json" in names and "checksums.json" in names
            assert not any(name.endswith((".jpg", ".png", ".safetensors", ".zip")) for name in names)
            checksums = json.loads(stream.read("checksums.json"))
            assert all(hashlib.sha256(stream.read(name)).hexdigest() == value for name, value in checksums.items())


def test_notebook_embeds_matching_runner_and_all_python_cells_compile(tmp_path):
    path = tmp_path / "pilot.ipynb"
    build(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "files.upload" not in combined
    assert "system_site_packages=True" in combined
    assert "with_pip=False" in combined
    assert "with_pip=True" not in combined
    assert "'--python', str(python)" in combined
    assert "PARTIAL_EVIDENCE_READY" in combined
    assert "files.download(str(evidence_zip))" in combined
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code" and not cell["source"][0].startswith("%"):
            compile("".join(cell["source"]), cell["id"], "exec")


@pytest.mark.parametrize("partial_environment", [False, True])
def test_environment_install_without_ensurepip_and_rerun(tmp_path, monkeypatch, partial_environment):
    import venv
    namespace = {}
    exec(ENVIRONMENT_SETUP_SOURCE, namespace)
    environment = tmp_path / "model-env"
    marker = environment / "preserve.txt"
    if partial_environment:
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
        marker.write_text("preserve existing files")
    def ensurepip_must_not_run(*args, **kwargs):
        raise AssertionError("ensurepip is unavailable in the simulated runtime")
    monkeypatch.setattr(venv.EnvBuilder, "_setup_pip", ensurepip_must_not_run)
    python = namespace["prepare_pilot_environment"](environment)
    wheel = tmp_path / "pilot_env_probe-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as stream:
        stream.writestr("pilot_env_probe.py", "VALUE = 'isolated'\n")
        stream.writestr("pilot_env_probe-1.0.dist-info/METADATA",
                        "Metadata-Version: 2.1\nName: pilot-env-probe\nVersion: 1.0\n")
        stream.writestr("pilot_env_probe-1.0.dist-info/WHEEL",
                        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        stream.writestr("pilot_env_probe-1.0.dist-info/RECORD", "")
    command = namespace["pilot_pip_command"](
        python, "install", "--no-index", "--no-deps", "--disable-pip-version-check", str(wheel))
    installation = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert installation.returncode == 0, installation.stdout + installation.stderr
    namespace["prepare_pilot_environment"](environment)
    probe = subprocess.check_output([str(python), "-c",
        "import json,pilot_env_probe; print(json.dumps([pilot_env_probe.VALUE,pilot_env_probe.__file__]))"], text=True)
    value, location = json.loads(probe)
    assert value == "isolated"
    assert Path(location).is_relative_to(environment)
    if partial_environment:
        assert marker.read_text() == "preserve existing files"
    assert not importlib.util.find_spec("pilot_env_probe")


def test_environment_rejects_wrong_interpreter_prefix(tmp_path, monkeypatch):
    import venv
    namespace = {}
    exec(ENVIRONMENT_SETUP_SOURCE, namespace)
    monkeypatch.setattr(venv.EnvBuilder, "create", lambda *args: None)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: str(tmp_path))
    with pytest.raises(RuntimeError, match="outside"):
        namespace["prepare_pilot_environment"](tmp_path / "target")


def test_publication_rejects_short_sha_and_incomplete_pool(tmp_path):
    from tools.publish_full_page_pilot_inputs import prepare
    _, dataset, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="Git commit"):
        prepare(dataset, tmp_path / "publication", "abcdef")
    with pytest.raises(ValueError, match="80-page"):
        prepare(dataset, tmp_path / "publication", "a" * 40)


@pytest.mark.parametrize('grid,limit,valid', [([1,64,64],1024,True),
    ([1,65,64],1024,False), ([1,64,64],1000,False), ([0,64,64],1024,False)])
def test_ovis_processed_budget_guard(grid, limit, valid):
    inputs = {'image_grid_thw': SimpleNamespace(tolist=lambda: [grid]),
              'input_ids': SimpleNamespace(shape=(1, 1100))}
    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16, merge_size=2))
    spec = {'max_pixels':1048576, 'max_visual_tokens':limit}
    if valid:
        geometry = ovis_geometry(inputs, processor, spec, (2409,3042))
        assert geometry['visual_tokens'] == 1024
        assert geometry['processed_size_wh'] == [1024,1024]
    else:
        with pytest.raises(ValueError):
            ovis_geometry(inputs, processor, spec, (2409,3042))


@pytest.mark.parametrize('over_budget', [False, True])
def test_ovis_processor_kwargs_and_budget_before_gpu(tmp_path, monkeypatch, over_budget):
    calls = {}
    class Inputs(dict):
        def to(self, device):
            calls['transferred'] = device
            return self
    class Model:
        device = 'cuda'
        def to(self, device):
            return self
        def eval(self):
            return self
        def generate(self, **kwargs):
            calls['generation'] = kwargs
            return [[1, 2, 3, 4]]
    class Processor:
        image_processor = SimpleNamespace(patch_size=16, merge_size=2)
        def apply_chat_template(self, messages, **kwargs):
            calls['processor'] = kwargs
            return Inputs(image_grid_thw=SimpleNamespace(tolist=lambda: [[1, 65 if over_budget else 64, 64]]),
                          input_ids=SimpleNamespace(shape=(1,2)))
        def decode(self, output, **kwargs):
            return 'błáwaty ſtara'
    def model_factory(*args, **kwargs):
        calls['loader'] = kwargs
        return Model()
    fake_transformers = SimpleNamespace(
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=model_factory),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Processor()))
    fake_torch = SimpleNamespace(bfloat16='bf16', float16='fp16',
        cuda=SimpleNamespace(is_bf16_supported=lambda:False), inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    spec = {'repo':'fixture','revision':'a'*40,'min_pixels':200704,
            'max_pixels':1048576,'max_visual_tokens':1024,'max_new_tokens':4096}
    image = tmp_path/'page.png'
    Image.new('RGB',(40,50)).save(image)
    infer = load_ovis(spec)
    if over_budget:
        with pytest.raises(ValueError, match='Visual budget'):
            infer(image, {'id':'p'})
        assert 'transferred' not in calls and 'generation' not in calls
    else:
        result = infer(image, {'id':'p'})['ovis-ocr2-full-page']
        assert result['text'] == 'błáwaty ſtara'
        assert result['input_geometry']['visual_tokens'] == 1024
        assert calls['generation']['do_sample'] is False
    assert calls['loader']['dtype'] == 'fp16'
    assert calls['processor']['processor_kwargs']['images_kwargs']['size']['longest_edge'] == 1048576
    assert 'images_kwargs' not in calls['processor']


def test_t4_notebook_has_independent_output_and_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = root/'experiments/2026-10-03/full-page-pilot-t4-v2/config.json'
    target = tmp_path/'colab_full_page_pilot_t4_v2.ipynb'
    build(target, config)
    notebook = json.loads(target.read_text(encoding='utf-8'))
    sources = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'slayer-full-page-pilot-t4-v2' in sources
    assert 'full-page-pilot-t4-v2-evidence.zip' in sources
    assert notebook['metadata']['colab']['name'] == target.name
    v1 = json.loads((root/'experiments/2026-10-02/full-page-pilot-v1/config.json').read_text())
    v2 = json.loads(config.read_text())
    assert v1['models']['ovis-ocr2']['max_pixels'] == 8294400
    assert v2['models']['ovis-ocr2']['max_pixels'] == 1048576
    assert v1['dataset'] == v2['dataset'] and v1['selection_salt'] == v2['selection_salt']


@pytest.mark.parametrize('unsafe', ['../escape.json', 'script.py', 'C:/escape.json'])
def test_evidence_import_rejects_unsafe_members(tmp_path, unsafe):
    from tools.audit_full_page_pilot import verify_archive
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr(unsafe, '{}')
    with pytest.raises(ValueError):
        verify_archive(archive)


def test_evidence_import_verifies_all_checksums(tmp_path):
    from tools.audit_full_page_pilot import verify_archive
    _, output, _ = fixture(tmp_path)
    archive = package(output)
    verified = verify_archive(archive)
    assert 'manifest.jsonl' in verified
    corrupted = tmp_path/'corrupted.zip'
    with zipfile.ZipFile(corrupted, 'w') as stream:
        for name, data in verified.items():
            stream.writestr(name, b'{}' if name == 'manifest.jsonl' else data)
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_archive(corrupted)


def test_v2_package_name_and_path_guard(tmp_path):
    work = tmp_path/'work'
    work.mkdir()
    write_json(work/'config.json', {'evidence_archive_name':'full-page-pilot-t4-v2-evidence.zip'})
    assert package(work).name == 'full-page-pilot-t4-v2-evidence.zip'
    write_json(work/'config.json', {'evidence_archive_name':'../escape.zip'})
    with pytest.raises(ValueError):
        package(work)
