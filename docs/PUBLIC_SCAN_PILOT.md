# Pilot rzeczywistych skanów — 2026-09-12

Tesseract.js 7.0.0, pol+eng, OEM 1, PSM 3, jeden worker CPU. Dwie strony
elementarza z 1930 r. zawierają drukowane wzory pisma odręcznego i ilustracje.
To celowo trudna próba diagnostyczna jednej książki, nie reprezentatywny
benchmark polskich dokumentów ani dowód jakości całego OCR_engine.

| Strona | CER | WER | Wynik |
|---|---:|---:|---|
| 15 | 100% | 100% | pusty odczyt |
| 51 | 86,57% | 96,30% | błędny odczyt |
| Łącznie (micro) | 89,94% | 97,50% | 0/2 poprawnych stron |

Odczyt obu obrazów trwał około 1,95 s, start workera z lokalnym cache 0,32 s
na Ryzen 5 5500U. Nie uruchamiano GPU, treningu ani płatnego API.
Wyniki nie są bezpośrednio porównywalne z wcześniejszymi próbkami syntetycznymi.

Silnik zwrócił confidence 95 dla pustej strony. Runner zapisuje teraz
`EmptyRecognition`, `status=error` i `confidence=null`, zachowując oryginalną
wartość w `raw_engine_confidence`. Niepusty wynik oznacza wykonanie odczytu,
nie poprawność tekstu. Ewaluator liczy pusty odczyt jako usunięcie całej referencji.

## Źródła i referencje

- [Elementarz, strona 15](https://commons.wikimedia.org/wiki/File:Elementarz_1930_(104452362).jpg)
- [Elementarz, strona 51](https://commons.wikimedia.org/wiki/File:Elementarz_1930_(104452465).jpg)

Autor: Franciszka Arnoldowa. Commons deklaruje Public domain; pełne metadane
są zapisane w `benchmarks/public-pilot-v1/source-metadata.json`. Obrazy
sprawdzono względem SHA1 Commons oraz zapisano własne SHA256. Oryginały
są lokalnie w ignorowanym `data/`; pobranie można odtworzyć z manifestu.

Referencje sporządził asystent wzrokowo przed pierwszym OCR; wymagają niezależnej
weryfikacji człowieka przed użyciem w rankingu. Zachowano wielkość liter,
interpunkcję, numer strony, znaki obok numeru i stopkę `4*`. Ciasne odstępy
po przecinkach zapisano jako pojedynczą spację. Kreski przy numerach przyjęto
jako em dash, łącznik w `motyle-tam` jako hyphen. Pominięto ilustracje i drobne
sygnatury rysownika. Normalizacja oceny: NFC i białe znaki, bez korekty językowej.

Pobrano też [okładkę Czerwonej rakiety](https://commons.wikimedia.org/wiki/File:Jerzy_Bandrowski_-_Czerwona_rakieta_(page_1_crop).jpg).
Wyłączono ją **przed OCR**: napis jest przycięty i występuje dopisek przy brzegu.
Nie uwzględniono jej w liczbie stron ani metrykach.

## Odtworzenie

Z katalogu repozytorium, z Pythonem i Node na PATH:

```powershell
python -m training.download_public_pilot
pnpm --dir tools/cpu-baseline install --frozen-lockfile
New-Item -ItemType Directory -Force validation | Out-Null
node tools/cpu-baseline/run.mjs benchmarks/public-pilot-v1/manifest.jsonl validation/public-reproduction
uv run --no-project --python 3.11 --with jiwer python -m training.benchmark_pages --manifest benchmarks/public-pilot-v1/manifest.jsonl --predictions validation/public-reproduction/predictions.jsonl --output validation/public-reproduction/summary.json
```

Katalog wyniku musi być nowy. Pierwsze uruchomienie pobiera wagi Tesseract;
kolejne korzystają z cache. `training/prepare_public_pilot.py` służy do
odtworzenia referencji z oryginalnego lokalnego pobrania i metadanych.
Zamrożone wyniki: `experiments/2026-09-12/public-scans/`.

Następny krok: osobna, większa próbka współczesnego druku z kilku dokumentów
oraz porównanie modelu wizyjnego na identycznych obrazach. Obecna próba
uzasadnia potrzebę sprawdzenia rozpoznawania kursywy; sama nie uzasadnia treningu.
