from types import SimpleNamespace
import pytest
from training import colab_environment as env


def setup_versions(monkeypatch):
    versions = {**env.REQUIRED_PACKAGES, 'tokenizers': '0.22.2'}
    monkeypatch.setattr(env.importlib.metadata, 'version', lambda name: versions[name])
    for name in versions:
        monkeypatch.delitem(env.sys.modules, name, raising=False)
    return versions


def test_pinned_environment(monkeypatch):
    setup_versions(monkeypatch)
    env.check_colab_environment()


def test_wrong_transformers_rejected(monkeypatch):
    versions = setup_versions(monkeypatch)
    versions['transformers'] = '5.16.1'
    with pytest.raises(RuntimeError, match='required=4.57.6'):
        env.check_colab_environment()


def test_stale_import_requires_restart(monkeypatch):
    setup_versions(monkeypatch)
    monkeypatch.setitem(env.sys.modules, 'transformers', SimpleNamespace(__version__='5.16.1'))
    with pytest.raises(RuntimeError, match='restart required'):
        env.check_colab_environment()


def test_missing_package_rejected(monkeypatch):
    setup_versions(monkeypatch)
    def missing(name):
        raise env.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(env.importlib.metadata, 'version', missing)
    with pytest.raises(RuntimeError, match='installed=missing'):
        env.check_colab_environment()


def test_extra_package_pin(monkeypatch):
    versions = setup_versions(monkeypatch)
    versions['opencv-python-headless'] = '4.12.0.88'
    env.check_colab_environment({'opencv-python-headless': '4.12.0.88'})
