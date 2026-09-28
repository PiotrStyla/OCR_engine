"""Generate the locked 60-page Qwen retry notebook."""
from pathlib import Path

from training.build_slayer_layout_teacher_colab import build


QWEN_RETRY_CODE_REVISION = "d42ae1eed9a076e3508329482978eb3dfe61e1b8"
TARGET = "colab_slayer_layout_qwen3_vl_4b_full_v3.ipynb"


def build_qwen_retry(directory: str | Path) -> Path:
    target = Path(directory) / TARGET
    build(
        target,
        teacher_id="qwen3-vl-4b",
        pages=60,
        code_revision=QWEN_RETRY_CODE_REVISION,
        record_code_revision=True,
        output_name="qwen3-vl-4b-evidence-v3.zip",
    )
    return target


if __name__ == "__main__":
    build_qwen_retry(Path(__file__).parent)
