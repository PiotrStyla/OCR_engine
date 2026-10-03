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
    ovis_generation_settings,
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


@pytest.mark.parametrize('over_budget,explicit_eos', [(False,False),(True,False),(False,True)])
def test_ovis_processor_kwargs_and_budget_before_gpu(tmp_path, monkeypatch, over_budget, explicit_eos):
    calls = {}
    class Inputs(dict):
        def to(self, device):
            calls['transferred'] = device
            return self
    class Model:
        device = 'cuda'
        generation_config = SimpleNamespace(eos_token_id=248044)
        def to(self, device):
            return self
        def eval(self):
            return self
        def generate(self, **kwargs):
            calls['generation'] = kwargs
            return [[1, 2, 3, 248046 if explicit_eos else 4]]
    class Processor:
        image_processor = SimpleNamespace(patch_size=16, merge_size=2)
        tokenizer = SimpleNamespace(eos_token_id=248046, pad_token_id=248044,
            convert_tokens_to_ids=lambda token: {'<|im_end|>':248046,'<|endoftext|>':248044}[token],
            convert_ids_to_tokens=lambda token_id: {248046:'<|im_end|>',248044:'<|endoftext|>'}[token_id])
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
    if explicit_eos:
        spec.update(stop_token_ids={'<|im_end|>':248046,'<|endoftext|>':248044}, max_new_tokens=2)
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
        if explicit_eos:
            assert calls['generation']['eos_token_id'] == [248044,248046]
            assert result['finish_reason'] == 'eos' and result['token_limit_reached'] is False
            assert result['generation_trace']['generated_token_ids'] == [3,248046]
            assert result['generation_trace']['eos_positions'] == [1]
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


@pytest.mark.parametrize('actual,roundtrip', [(0,'<|im_end|>'),(248046,'wrong-token')])
def test_eos_identity_failures_are_not_silently_accepted(actual, roundtrip):
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044))
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=248046,pad_token_id=248044,
        convert_tokens_to_ids=lambda token:actual, convert_ids_to_tokens=lambda token:roundtrip))
    spec = {'max_new_tokens':4096,'stop_token_ids':{'<|im_end|>':248046}}
    with pytest.raises(ValueError,match='stop-token'):
        ovis_generation_settings(model,processor,spec)


def test_eos_v3_preserves_v2_prompt_scope_and_memory_profile(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root/'experiments/2026-10-03/full-page-pilot-eos-v3/config.json'
    v2 = json.loads((root/'experiments/2026-10-03/full-page-pilot-t4-v2/config.json').read_text())
    v3 = json.loads(path.read_text())
    assert v2['dataset']==v3['dataset'] and v2['selection_salt']==v3['selection_salt']
    for key in ('repo','revision','packages','max_new_tokens','min_pixels','max_pixels','max_visual_tokens'):
        assert v2['models']['ovis-ocr2'][key] == v3['models']['ovis-ocr2'][key]
    assert set(v3['models']) == {'ovis-ocr2'}
    target = tmp_path/'colab_full_page_pilot_eos_v3.ipynb'
    build(target,path)
    notebook = json.loads(target.read_text(encoding='utf-8'))
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'Only OvisOCR2 runs' in source
    assert 'slayer-full-page-pilot-eos-v3' in source
    assert 'no_repeat_ngram_size' not in source and 'repetition_penalty' not in source


@pytest.mark.parametrize('truncated', [False,True])
def test_archive_audit_distinguishes_runtime_success_and_token_cap(tmp_path, truncated):
    from tools.audit_full_page_pilot import audit
    config, dataset, _ = fixture(tmp_path)
    spec = {'repo':'fixture','revision':'a'*40,'max_new_tokens':2,
            'max_pixels':1024,'max_visual_tokens':4}
    config['models'] = {'ovis-ocr2':spec}
    config_path = tmp_path/'config.json'
    write_json(config_path,config)
    write_json(tmp_path/'code-provenance.json',{'embedded_runner_sha256':'b'*64})
    predictions = tmp_path/'predictions'/'ovis-ocr2'
    predictions.mkdir(parents=True)
    write_json(predictions/'identity.json',{'spec':spec,'runner_sha256':'b'*64,
        'input_sha256':digest(dataset/'inference-inputs.jsonl')})
    write_rows(predictions/'ovis-ocr2-full-page.jsonl',[{
        'id':'p','status':'ok','text':json.loads((dataset/'manifest.jsonl').read_text())['text'],
        'generated_tokens':2,'token_limit_reached':truncated,
        'input_geometry':{'requested_max_pixels':1024,'processed_pixels':1024,'visual_tokens':1}}])
    score(dataset/'manifest.jsonl',predictions.parent,tmp_path/'scores')
    archive = package(tmp_path)
    report = audit(archive,tmp_path/'verified',config_path)
    engine = report['reports']['ovis-ocr2-full-page']
    assert engine['cer_micro']==0 and engine['execution_complete'] is True
    assert engine['token_limit_pages']==int(truncated)
    assert engine['quality_comparison_available'] is not truncated
    assert report['head_to_head_quality_comparison_available'] is not truncated


def test_stage_validation_scope_cannot_backfill_training_pages(tmp_path):
    config, original, opener = fixture(tmp_path)
    train = json.loads((original / 'manifest.jsonl').read_text(encoding='utf-8'))
    train.update(id='training-page', split='train')
    metadata = (json.dumps(train) + '\n').encode()
    config['dataset']['splits']['train'] = {'pages':1, 'metadata_sha256':hashlib.sha256(metadata).hexdigest()}
    config.update(available_non_test_pages=2, inference_splits=['validation'])
    def with_train(url, **kwargs):
        return io.BytesIO(metadata) if url.endswith('/train/metadata.jsonl') else opener(url, **kwargs)
    report = stage(config, tmp_path/'validation-only', opener=with_train)
    assert report['source_split_counts'] == {'validation':1}
    with pytest.raises(ValueError, match='Page count'):
        stage(config, tmp_path/'too-many', count=2, opener=with_train)


def test_validation_v4_frozen_scope_and_notebook(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root/'experiments/2026-10-03/full-page-validation-v4/config.json'
    previous = json.loads((root/'experiments/2026-10-03/full-page-pilot-eos-v3/config.json').read_text())
    config = json.loads(path.read_text())
    assert config['dataset'] == previous['dataset']
    assert config['inference_splits'] == ['validation']
    assert config['default_inference_pages'] == 15
    assert config['policy'] == previous['policy']
    assert config['selection_salt'] == previous['selection_salt']
    for key, value in previous['models']['ovis-ocr2'].items():
        if key != 'adapter_status':
            assert config['models']['ovis-ocr2'][key] == value
    target = tmp_path/'v4.ipynb'
    build(target,path)
    notebook = json.loads(target.read_text(encoding='utf-8'))
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'Default: 15 historical development pages' in source
    assert 'Training pages must not enter validation' in source
    assert 'full-page-validation-v4-evidence.zip' in source
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and not ''.join(cell['source']).startswith('%'):
            compile(''.join(cell['source']), '<notebook>', 'exec')


@pytest.mark.parametrize('tokens,reason,capped', [([12,248046],'eos',False),
    ([12,248044],'eos',False),([12,13],'length',True),([12],'unknown',False)])
def test_termination_audit_uses_raw_tokens_including_eos_at_cap(tokens, reason, capped):
    from tools.audit_full_page_pilot import verify_ovis_termination
    spec = {'max_new_tokens':2,'stop_token_ids':{'<|im_end|>':248046,'<|endoftext|>':248044}}
    trace = {'generated_token_ids':tokens, 'tokenizer_eos_token_id':248046,
             'stop_override_applied':True,
             'applied_generation_settings':{'eos_token_id':[248044,248046],
                 'max_new_tokens':2,'do_sample':False,'pad_token_id':248044},
             'eos_positions':[index for index,token in enumerate(tokens) if token in [248044,248046]]}
    record = {'generated_tokens':len(tokens),'generation_trace':trace,
              'finish_reason':reason,'token_limit_reached':capped}
    verify_ovis_termination(record,spec)
    record['finish_reason'] = 'invented'
    with pytest.raises(ValueError,match='termination reason'):
        verify_ovis_termination(record,spec)


@pytest.mark.parametrize('change', ['count','settings','positions','early-eos','missing'])
def test_termination_audit_rejects_inconsistent_traces(change):
    from tools.audit_full_page_pilot import verify_ovis_termination
    spec = {'max_new_tokens':4096,'stop_token_ids':{'<|im_end|>':248046,'<|endoftext|>':248044}}
    trace = {'generated_token_ids':[12,248046], 'tokenizer_eos_token_id':248046,
             'stop_override_applied':True,'eos_positions':[1],
             'applied_generation_settings':{'eos_token_id':[248044,248046],
                 'max_new_tokens':4096,'do_sample':False,'pad_token_id':248044}}
    record = {'generated_tokens':2,'generation_trace':trace,
              'finish_reason':'eos','token_limit_reached':False}
    if change == 'count':
        record['generated_tokens'] = 3
    elif change == 'settings':
        trace['applied_generation_settings']['eos_token_id'] = [248044]
    elif change == 'positions':
        trace['eos_positions'] = []
    elif change == 'early-eos':
        trace['generated_token_ids'] = [248046,12]
        trace['eos_positions'] = [0]
        record['finish_reason'] = 'unknown'
    else:
        del record['generation_trace']
    with pytest.raises(ValueError,match='termination'):
        verify_ovis_termination(record,spec)


@pytest.mark.parametrize('tokens,reason,capped', [([12,248046],'eos',False),
    ([12,13],'length',True),([12],'unknown',False)])
def test_explicit_eos_audit_blocks_incomplete_termination(tmp_path, tokens, reason, capped):
    from tools.audit_full_page_pilot import audit
    config, dataset, _ = fixture(tmp_path)
    spec = {'repo':'fixture','revision':'a'*40,'max_new_tokens':2,
            'max_pixels':1024,'max_visual_tokens':4,
            'stop_token_ids':{'<|im_end|>':248046,'<|endoftext|>':248044}}
    config['models'] = {'ovis-ocr2':spec}
    config_path = tmp_path/'config.json'
    write_json(config_path,config)
    write_json(tmp_path/'code-provenance.json',{'embedded_runner_sha256':'b'*64})
    destination = tmp_path/'predictions/ovis-ocr2'
    destination.mkdir(parents=True)
    write_json(destination/'identity.json',{'spec':spec,'runner_sha256':'b'*64,
        'input_sha256':digest(dataset/'inference-inputs.jsonl')})
    trace = {'generated_token_ids':tokens, 'tokenizer_eos_token_id':248046,
             'stop_override_applied':True,
             'applied_generation_settings':{'eos_token_id':[248044,248046],
                 'max_new_tokens':2,'do_sample':False,'pad_token_id':248044},
             'eos_positions':[index for index,token in enumerate(tokens) if token in [248044,248046]]}
    write_rows(destination/'ovis-ocr2-full-page.jsonl',[{
        'id':'p','status':'ok','text':json.loads((dataset/'manifest.jsonl').read_text(encoding='utf-8'))['text'],
        'generated_tokens':len(tokens),'generation_trace':trace,
        'finish_reason':reason,'token_limit_reached':capped,
        'input_geometry':{'requested_max_pixels':1024,'processed_pixels':1024,'visual_tokens':1}}])
    score(dataset/'manifest.jsonl',destination.parent,tmp_path/'scores')
    result = audit(package(tmp_path),tmp_path/'verified',config_path)
    report = result['reports']['ovis-ocr2-full-page']
    assert report['cer_micro'] == 0
    assert report['execution_complete'] is True
    assert report['termination_trace_verified'] is True
    assert report['complete_generation_pages'] == int(reason == 'eos')
    assert report['quality_comparison_available'] is (reason == 'eos')


@pytest.mark.parametrize('corruption', ['source','reference','scope'])
def test_archive_audit_checks_pinned_reference_and_complete_scope(tmp_path, corruption):
    from tools.audit_full_page_pilot import audit
    config, dataset, _ = fixture(tmp_path)
    if corruption == 'source':
        path = dataset/'sources/validation.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
        message = 'source metadata checksum'
    elif corruption == 'reference':
        rows = [json.loads((dataset/'manifest.jsonl').read_text(encoding='utf-8'))]
        rows[0]['text'] = 'silently replaced reference'
        write_rows(dataset/'manifest.jsonl',rows)
        message = 'reference differs'
    else:
        config['default_inference_pages'] = 2
        message = 'Page count'
    path = tmp_path/'config.json'
    write_json(path,config)
    with pytest.raises(ValueError,match=message):
        audit(package(tmp_path),tmp_path/'verified',path)
