# PolOCRBench — historyczno-drukowy podzbiór (subset: history_print)

Zamrożony wkład IMPACT-Polish v2 do subtasku A (transkrypcja pełnostronicowa).

## Test A pool: 36 stron, 52291 znaków GT
Kolekcje: NA2_FT, Nowiny_z_Rakuz_FT, Powodzenia_FT
Podział odziedziczony z IMPACT-Polish v1 (per kolekcja); strony testowe nigdy nie
występują w puli treningowej (weryfikacja: impact_split + impact_freeze).

## Train pool: 2531 regionów z 27 kolekcji
Rekordy regionów (tekst + bbox jeśli dostępny); obrazy stron w `impact-corpus`.

## Provenance
- Źródło: IMPACT ground truth (PSNC), github.com/impactcentre/groundtruth-pol
- Obrazy: dLibra ribes-80.man.poznan.pl (TLS certyfikat wygasł; integralność: SHA-256)
- Licencja: CC-BY-3.0
- Zamrożono: 2026-09-18

## Metryka
`training/transcription_eval.py` (CER/WER micro + struktura Markdown; NFC,
cudzysłowy typograficzne -> proste, białe znaki; wielkość liter i diakrytyki zachowane).
