"""Generate the pinned full-run consensus v3 Colab notebook."""
from pathlib import Path

from training.build_slayer_layout_consensus_colab import build


CONSENSUS_V3_CODE_REVISION = "91128ab311eeac38dda20b4de9ef9add6f92a198"
POLICY_V3_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-28/slayer-layout-consensus-policy-v3.json"
)
TARGET = "colab_slayer_layout_consensus_v3.ipynb"


def build_v3(directory: str | Path) -> Path:
    target = Path(directory) / TARGET
    build(
        target,
        code_revision=CONSENSUS_V3_CODE_REVISION,
        policy_path=POLICY_V3_PATH,
    )
    return target


if __name__ == "__main__":
    build_v3(Path(__file__).parent)
