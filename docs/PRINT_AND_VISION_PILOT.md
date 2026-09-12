# Zwykły druk i przygotowanie vision API — 2026-09-12

## Wynik CPU

Tesseract.js 7.0.0, pol+eng, OEM 1, PSM 3, jeden worker; te same ustawienia
co w pilocie kursywy. Dwie oryginalne strony historycznego druku z dwóch książek:

| Dokument | CER | WER |
|---|---:|---:|
| Odezwa do matek, strona 12 | 1,06% | 7,29% |
| Toruński elementarz polski, strona 74 | 0,60% | 2,65% |
| Łącznie (micro) | **0,79%** | **4,48%** |

Odczyt: około 5,89 s łącznie, bez GPU i bez korektora językowego.
Wynik kursywy (CER 89,94%) pozostaje osobnym pomiarem. Różnica opisuje
te próbki; nie jest estymacją skuteczności na wszystkich dokumentach.
Żadna strona nie była idealna. Błędy obejmowały numer strony, polskie znaki,
łączenie słów i interpunkcję. Na przykład `sił` stało się `sit`.

Nie zmieniano referencji na podstawie odpowiedzi OCR. Zachowano historyczną
pisownię (np. `najlepszem`, `prawdziwem`, `poprostu`), łączniki na końcach
wierszy, numerację i wielkość liter. Różnice spacji między słowami są błędami;
NFC i ciągi białych znaków normalizuje wspólny ewaluator. Ozdobne linie są
pomijane, długie kreski tekstowe zapisano jako U+2014, wielokropek jako trzy
kropki. Przepisał je asystent wzrokowo przed odczytem — brak drugiego recenzenta.

## Źródła i ograniczenia

- [Antoni Cyryl Królicki, Odezwa do matek, str. 12](https://commons.wikimedia.org/wiki/File:PL_Kr%C3%B3licki_Odezwa_do_matek_012.png)
- [Toruński elementarz polski, 1910, str. 74](https://commons.wikimedia.org/wiki/File:Torunski_elementarz_polski_1910_(55498679).jpg)

Commons oznacza oba materiały jako Public domain. Manifest przechowuje URL,
SHA1 źródła, SHA256 pobranego obrazu, grupę dokumentu i metodę transkrypcji.
Surowe metadane zachowano w `benchmarks/print-pilot-v1/source-metadata.json`.
Trzeci pobrany [Elementarz śląski](https://commons.wikimedia.org/wiki/File:Elementarz_slaski_1930_(57216193).jpg)
wyłączono przed OCR: zawiera kursywę, a ta próba dotyczy zwykłego druku.

To dwie strony historycznych książek, nie współczesne formularze, faktury,
tabele ani dokumenty wielokolumnowe. Dane są publiczne; ich obecności w danych
treningowych modeli nie można wykluczyć. Żadnego modelu nie trenowano na pilocie.
Wyniki i ich sumy kontrolne są w `experiments/2026-09-12/print-scans/`.

Odtworzenie z katalogu repozytorium (Python, Node i pnpm na PATH):

```powershell
python -m training.download_public_pilot --pilot print-pilot-v1
pnpm --dir tools/cpu-baseline install --frozen-lockfile
New-Item -ItemType Directory -Force validation | Out-Null
node tools/cpu-baseline/run.mjs benchmarks/print-pilot-v1/manifest.jsonl validation/print-reproduction
uv run --no-project --python 3.11 --with jiwer python -m training.benchmark_pages --manifest benchmarks/print-pilot-v1/manifest.jsonl --predictions validation/print-reproduction/predictions.jsonl --output validation/print-reproduction/summary.json
```

Katalog wynikowy musi być nowy. `prepare_print_pilot.py` odtwarza manifest z
referencji i oryginalnego pobrania, nie jest wymagany do powtórzenia odczytu.

## Zdalny model wizyjny

Ponowny odczyt [schematu Fabryki](https://fabryka.ai/openapi.json) potwierdza
`Message.content: string | null`; brak udokumentowanego pola obrazu i endpointu
treningowego. Nie wykonywano nowych zapytań z kluczem Fabryki.

[Gemini dokumentuje wejście obrazu](https://ai.google.dev/gemini-api/docs/image-understanding)
i [warstwę zgodną z klientem OpenAI](https://ai.google.dev/gemini-api/docs/openai).
[Cennik](https://ai.google.dev/gemini-api/docs/pricing) w dniu sprawdzenia
wymienia darmową warstwę m.in. dla `gemini-3.8-flash`; dostęp i limity projektu
trzeba sprawdzić na koncie. Cennik wskazuje wykorzystywanie danych darmowej
warstwy do ulepszania produktów, z warunkami opisanymi przez dostawcę.
Ten pilot zawiera wyłącznie materiały publiczne. Brak skonfigurowanego
`GEMINI_API_KEY` w sprawdzonym środowisku, dlatego **nie wykonano testu Gemini**.
Zgodność potwierdzono w dokumentacji i testach z atrapą, nie w rzeczywistym API.

Przygotowano `training.probe_vision`:

- domyślnie lokalny dry-run, bez sieci, klucza i zależności ML;
- po `--execute`: jedna strona domyślnie, maksymalnie pięć, bez retry,
  timeout 60 s na żądanie, jawny model i limit tokenów;
- wysyła obraz i stały prompt, **nie wysyła transkrypcji wzorcowej**;
- kontroluje SHA256, zapisuje model zwrócony, usage i czas;
- odrzuca ucięte odpowiedzi; przerywa serię po pierwszym błędzie;
- nie zapisuje treści wyjątków, nagłówków ani klucza;
- pusta odpowiedź pozostaje pusta: ocenę względem referencji wykonuje scorer.

```powershell
# Bez API: sprawdza obrazy i pokazuje dokładny zakres pierwszego żądania.
python -m training.probe_vision --manifest benchmarks/print-pilot-v1/manifest.jsonl --model gemini-3.8-flash --output validation/gemini-print-smoke

# Dopiero po ustawieniu GEMINI_API_KEY w środowisku procesu i sprawdzeniu planu konta:
uv run --no-project --python 3.11 --with openai python -m training.probe_vision --manifest benchmarks/print-pilot-v1/manifest.jsonl --model gemini-3.8-flash --output validation/gemini-print-smoke --execute
```

Nie wklejać klucza do repozytorium ani argumentów polecenia. Runner nie potrafi
potwierdzić bezpłatności konta i nie ma automatycznego budżetu dolarowego.
Pierwszy smoke obejmuje tylko `odezwa-12`; do porównania całej próbki użyć nowego
katalogu i `--limit 2`. Pełny manifest z brakującymi odpowiedziami policzy je
jako usunięcia — nie należy przedstawiać wyniku jednej strony jako całego pilota.

Przygotowanie klienta zdalnego nie ładuje już `ocr.pipeline`: lokalny silnik
jest importowany dopiero przy użyciu `OcrEngine` lub `recognize`. Zachowano
dotychczasowe publiczne importy. Test subprocess z `python -S` sprawdza, że
tryb przygotowawczy nie wymaga OpenCV, NumPy, torch ani SDK API.

Weryfikacja: 107 testów z `tests/` przeszło na CPU, sprawdzono SHA1/SHA256
obu oryginałów i sumy kontrolne ośmiu artefaktów, a dry-run wykonano także
poza środowiskiem testów przy wyłączonych site-packages. Nie wykonywano
rzeczywistych zapytań Gemini.

Następne porównanie powinno obejmować osobno zwykły druk i kursywę, przy
identycznych obrazach i zamrożonych referencjach. Nadal brak podstaw do
ogłoszenia SOTA lub zamawiania treningu.
