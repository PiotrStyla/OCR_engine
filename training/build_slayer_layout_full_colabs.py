"""Generate three locked 60-page SLAYER layout teacher notebooks."""
from pathlib import Path

from training.build_slayer_layout_teacher_colab import build


FULL_RUN_CODE_REVISION = "c5740e8fb83172106eacc6d6870612baf12a6afd"
TARGETS = {
    "qwen3-vl-4b": "colab_slayer_layout_qwen3_vl_4b_full_v1.ipynb",
    "doclayout-yolo": "colab_slayer_layout_doclayout_yolo_full_v1.ipynb",
    "surya-layout2": "colab_slayer_layout_surya_full_v1.ipynb",
}


def build_all(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    outputs = []
    for teacher_id, name in TARGETS.items():
        target = directory / name
        build(
            target,
            teacher_id=teacher_id,
            pages=60,
            code_revision=FULL_RUN_CODE_REVISION,
            record_code_revision=True,
        )
        outputs.append(target)
    return outputs


if __name__ == "__main__":
    build_all(Path(__file__).parent)
