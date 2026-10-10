# Materiał recenzyjny referencji (2026-10-10)

Dowody maszynowe do decyzji o 34 pozycjach wad referencji testu A, zebrane
z predykcji **6 systemów** (Tesseract.js, gpt-4o-mini, Qwen3-VL-235B, gpt-5.4,
Qwen3-VL-4B, PaddleOCR-VL-1.6) przez wyrównanie tekstu GT z predykcjami.
Żadna zamrożona referencja nie została zmieniona — model nie naprawia GT,
dostarcza wskazówki do ludzkiej decyzji.

## PUA: 10 kodpunktów, 624 wystąpienia — glify zidentyfikowane konsensem

| Kodpunkt | × | co drukują systemy | propozycja |
|---|---:|---|---|
| U+EADA | 416 | `st` 688×, `ſt` 230× (gpt-5.4: 190×), `ft` 299× (Tesseract: długie „s" jako f) | **`ſt`** |
| U+EBA2 | 113 | `si` 162×, `ſi` 55× | **`ſi`** |
| U+F51E | 51 | `sł` 46×, `ſł` 19× | **`ſł`** |
| U+EBA6 | 16 | `ff` 36×, `ss` 22×, `ſſ` 4× | **`ſſ`/`ff`** |
| U+EEC5 | 13 | `ct` 37× | **`ct`** |
| U+EBA7 | 8 | `ssi` 7×, `ſſi` 4× | **`ſsi`** |
| U+E5DC | 3 | `n`/`ń`/`ni` | do wglądu |
| U+F516 | 2 | `z` (4 systemy) | do wglądu |
| U+F50E / U+EBA3 | 2 | rozbieżne | do wglądu |

Pełne liczniki per system: `pua-evidence.json`.

## U+FFFD: 86 wystąpień

Przeważnie puste u wszystkich modeli (`<empty>` 44–55×) — nie do odzyskania
z GT; wymaga transkrypcji ze skanu albo akceptacji jako szum.
`uffd-evidence.json`: 40 próbek z kontekstem i podmianami per system.

## Errata `NA2_FT__434735`

`errata-draft.md`: transkrypcje 6 systemów obok siebie + propozycja draftu
konsensusowego do weryfikacji dwu-recenzyjnej ze skanem.

## Decyzje (ludzkie)

1. Polityka PUA: mapować do **właściwych glifów historycznych** (propozycja
   wyżej — zachowuje zasadę nie-normalizowania pisowni) / do odpowiedników
   nowoczesnych / zostawić.
2. U+FFFD: transkrypcja ze skanu albo akceptacja.
3. Errata: zatwierdzić draft ze skanem albo wykluczyć stronę.

`experiments/2026-10-10/reference-defects/decisions.json` — rejestr decyzji.
