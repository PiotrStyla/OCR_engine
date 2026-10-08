# API zero-shot na zamrożonym teście A (2026-10-08)

Trzy modele vision zmierzone na 36 stronach testu A ze zmrożonym promptem
`polocrbench-zero-shot-prompt-v1` (szablon A dosłownie), temperatura 0,
`max_tokens` 8192. Strony IMPACT to TIFF-y pod nazwami `.jpg`; transport do
endpointów czatowych to bezstratna transkodowanie do PNG (piksele bez zmian,
zamrożone pliki nietknięte).

- Wynik i wnioski: `docs/POLLOCR_SOTA_API_RESULT_20261008.md`
- Protokół pomiaru: `docs/POLLOCR_SOTA_MEASUREMENT_V1_20261008.md`

Zawartość: `predictions.jsonl` i `run.json` dla każdego modelu (per strona:
tekst, status, czas, użycie tokenów), `receipt.json` (piny wejścia, rewizja
kodu) oraz `checksums.sha256`. To pomiar zero-shot/API: bez treningu, bez
korekt i bez promocji jakichkolwiek artefaktów.
