# Pomiar SOTA GPU na Runpodie (2026-10-10)

Qwen3-VL-4B (4-bit) i PaddleOCR-VL-1.6 zmierzone na zamrożonym teście A
(36 stron, IMPACT `history_print`) na podzie **RTX 4090** (`9z5bgz2idmq83h`,
EU-RO-1, $0,89/h, 1,2 h ≈ $1,06, pod już zgaszony). Pomiar zero-shot: bez
treningu, adapterów i korekt; nic nie jest promowane.

**Wynik:** Qwen3-VL-4B **CER 20,03%** (17,06% bez strony z uszkodzonym GT),
PaddleOCR-VL-1.6 **CER 33,80%** — poziom baseline'u Tesseracta mimo #1 na
OmniDocBench. Pełna tabela szesciu systemów i wnioski:
`docs/POLLOCR_SOTA_GPU_RESULT_20261010.md`.

Zawartość: `predictions.jsonl`, `score.json`, `run.json` dla obu modelów,
`receipt.json` (piny wejścia i rewizji `607ae333…`), `audit.json` (metryki
niezależnie przeliczone z predykcji), oryginalny `polocrbench-sota-measurement-v1-evidence.zip`
(SHA-256 `dac9ef12216f79c28797b35407a62a04fe1e3c5d5b6a0a5ebeed75e6071b306a`)
oraz `checksums.sha256`.
