# OCR bez GPU na laptopie

## Stan 2026-09-12

**Aktualizacja po eksperymentach:** poniższe punkty 1–4 opisują stan sprzed
prób. Wykonano już testy korekty Fabryki, Tesseract CPU i dwa osobne piloty
skanów. Aktualne wyniki: [korekta i syntetyki](EXPERIMENTS_2026-09-12.md),
[kursywa](PUBLIC_SCAN_PILOT.md), [zwykły druk i przygotowanie vision](PRINT_AND_VISION_PILOT.md).
Zdalne OCR obrazów i trening nadal nie były uruchamiane.

1. Poprawki audytu: lokalna gałąź `codex/ocr-correctness`. Testy CPU, bez wag OCR. Własne nazwy checkpointów Paddle są teraz odrzucane zamiast ignorowane; obsługa adapterów wymaga osobnej implementacji. Bboxy są mapowane do oryginalnego rastra (dla PDF przy wybranym DPI). Confidence poprawionego tekstu jest nieznane; surowy tekst i score są zachowane. Nie jest to jeszcze kalibracja score.
2. Znaleziono `C:/Users/Hipek/OneDrive/Pulpit/OCR/test_document.png` i 12 par w `data/pl_lines_sample`. Strona została obejrzana i ręcznie przepisana do `benchmarks/smoke-v1`. To test funkcjonalny, nie reprezentatywny benchmark polskich dokumentów ani test SOTA. Widoczne ASCII trzeba przepisywać dosłownie, bez odtwarzania polskich znaków. Pary 12 linii pozostają poza testem jakości do czasu kontroli etykiet i niezależności od treningu.
3. Fabryka: publiczna dokumentacja opisuje chat/completions, models i credits, z tekstowym przykładem `qwen3.6-35b-a3b`. Nie znaleziono w niej potwierdzenia image input ani fine-tuningu. Strona główna oznacza aliasy smart router jako projekt MVP. Nie wysłano żadnych danych ani płatnych zapytań. Źródła: https://router.fabryka.ai/docs oraz https://router.fabryka.ai/ (odczyt 2026-09-12).
4. Wybór modelu i trening pozostają zależne od rzeczywistych pomiarów. Na razie nie ma podstaw do zamawiania treningu TrOCR lub konkretnego VLM.

## Podział pracy

- Laptop: testy, adnotacje, checksums, ewaluacja gotowych predykcji, małe obrazy, przygotowanie runów.
- Zdalne inference: serwer przyjmujący obrazy, konkretny model i wersja, niewielki limit stron i tokenów. Najpierw pojedynczy smoke-test, dopiero później benchmark. Tekstowa korekta to osobna kolumna wyników, nie OCR.
- Trening: zdalny proces na GPU z kontrolą danych, adapterów, checkpointów i logów. Hugging Face Jobs udostępnia uruchamianie własnego skryptu przez API/CLI; nie jest automatycznie darmowe. Dokumentacja: https://huggingface.co/docs/huggingface_hub/guides/jobs. Bez potwierdzonego bezpłatnego limitu lub budżetu nie uruchamiamy joba.

## Ewaluacja offline

Predykcje zapisuje się jako JSONL z `id`, `status` (`ok` lub `error`) i `text`. Brak strony jest liczony jako pusta predykcja, nie usuwany z mianownika. Skrypt nie wykonuje inferencji:

```powershell
python -m training.benchmark_pages --manifest benchmarks/smoke-v1/manifest.jsonl --predictions predictions.jsonl --output validation/page-report.json
```

Właściwy benchmark powinien zawierać rzeczywiste skany, kolumny, tabele, diakrytyki, liczby i puste strony, z oddzielnymi dokumentami train/dev/test. Aktualny scorer obejmuje CER/WER i exact match tekstu; TEDS, reading-order i przedziały ufności wymagają dalszego rozszerzenia oraz adnotacji.

## Następna bramka

Potwierdzić dostępny endpoint przyjmujący obraz i jego model. Do tego czasu skończyć lekkie poprawki oraz fixture i scorer. Nie przedstawiać tego jako zakończonego porównania modeli lub treningu. Fabrykę można zbadać jako korektor po ustaleniu bezpłatnego limitu/budżetu i bez wklejania klucza do repozytorium.

Przygotowano `ocr.page_parser.RemotePageParser`: niezależny od detektora linii, wysyła całą stronę PNG/JPEG do jawnie wskazanego klienta API, zapisuje żądany/zwrócony model, hash obrazu, zużycie i czas. Odrzuca ucięte odpowiedzi. Przetestowany atrapą endpointu; nie potwierdza kompatybilności z Fabryką. Nie tworzy fikcyjnych bboxów z tekstowej odpowiedzi. Podłączanie następuje dopiero po potwierdzeniu vision, limitów oraz ceny.

Bezpłatny kandydat do późniejszego interaktywnego eksperymentu to Google Colab: oferuje zasoby GPU, ale ich dostępność i limity nie są gwarantowane. To notebook, nie bezpłatny stały serwer API. Źródło: https://research.google.com/colaboratory/faq.html (2026-09-12). Nie uruchomiono sesji Colab ani jobów HF.
