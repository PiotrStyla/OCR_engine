"""Reject missing, drifted or stale notebook dependencies before inference."""
import importlib.metadata
import sys

REQUIRED_PACKAGES = {
    'transformers': '4.57.6', 'huggingface_hub': '0.36.0',
    'jiwer': '4.0.0', 'sentencepiece': '0.2.1',
}


def check_colab_environment():
    problems = []
    for name, expected in REQUIRED_PACKAGES.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = 'missing'
        if installed != expected:
            problems.append(f'{name}: installed={installed}, required={expected}')
    for name in [*REQUIRED_PACKAGES, 'tokenizers']:
        loaded = sys.modules.get(name)
        version = getattr(loaded, '__version__', None)
        if version is not None:
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed = 'missing'
            if version != installed:
                problems.append(f'{name}: loaded={version}, installed={installed}; restart required')
    if problems:
        raise RuntimeError('Environment check failed. Run the install cell, restart the Colab session, '
                           'then Run all.\n' + '\n'.join(problems))
    print('Pinned dependencies verified; no stale package versions detected.')
