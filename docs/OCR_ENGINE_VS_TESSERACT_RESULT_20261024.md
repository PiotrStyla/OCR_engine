# Porównanie silników OCR na materiale ewaluacyjnym — wynik i metoda (2026-10-24)

Pomiar porównawczy własnego pipeline'u (CRAFT + TrOCR) z klasycznym OCR-em
(Tesseract) na **niejawnym materiale ewaluacyjnym** (held-out, bez treningu na nim).
Ten dokument opisuje metodę i wnioski; sam materiał pozostaje niejawny do czasu
jego publikacji, dlatego pominięto jego skład, rozmiar i wartości bezwzględne.

## Metoda

- **Materiał:** wewnętrzny zbiór ewaluacyjny polskich stron dokumentów — syntetyczne
  renderowania dokumentów z kontrolowanymi degradacjami obrazu (m.in. zdjęcie,
  druk-skan, kompresja). Żadna strona nie wystąpiła w danych treningowych; zbiór
  nie jest publicznie dostępny.
- **Próba:** 48 stron dobranych równomiernie (co n-ta strona z każdego wariantu
  materiału), aby wynik nie zależał od kolejności generowania ani od jednego typu
  dokumentu.
- **Silniki:** (1) własny pipeline — `ocr` (detektor CRAFT + rozpoznanie
  `PiotrSty/trocr-pl-mixed-v3`, backend trocr, bez korekty API, wymuszony język pl);
  (2) Tesseract 5 (tessdata_fast, `pol+eng`, PSM 3).
- **Ewaluacja:** `training.transcription_eval` (protokół `polocrbench-transcription-v1.1`,
  publiczny w tym repo): CER (błędne znaki) i WER (błędne słowa) po normalizacji NFC,
  cudzysłowów i białych znaków; wielkość liter i diakrytyki liczone.
- **Sprawiedliwość porównania:** oba silniki oceniano na **tych samych stronach**,
  tym samym skryptem, z tymi samymi wzorcami; oba wypisują czysty tekst, więc
  kara za brak znaczników struktury (wzorzec jest w Markdownie) dotyczy ich po równo.
- **Powtarzalność:** skrypty pomiarowe i surowe predykcje zapisane w prywatnym
  archiwum wyników; pomiar CPU (~224 s/strona własnym silnikiem) wymusił próbę
  zamiast pełnego zbioru.

## Wyniki (wartości względne)

- Własny pipeline popełnia **o 10 punktów procentowych mniej błędnych znaków**
  na stronach mieszanych i **o 23 punkty mniej** na stronach z degradacjami
  niż Tesseract — na tej samej próbie, tym samym ewaluatorem.
- **Odporność na degradacje to przewaga własnego silnika:** Tesseract traci na
  zdegradowanych stronach (wzrost błędów o ok. jedną trzecią wobec stron
  mieszanych), podczas gdy model trenowany z augmentacjami radzi sobie na nich
  **lepiej niż na stronach mieszanych**. Hipoteza: degradacje były reprezentowane
  w treningu, a strony mieszane zawierają gęstszy układ i więcej linii do poskładania
  (do potwierdzenia pełnym pomiarem).
- Mały model wizyjno-językowy bez dostrajania (SmolVLM-256M, zero-shot, jednolity
  prompt) **nie rozwiązuje zadania** — zamiast transkrypcji generuje niezwiązany
  z dokumentem tekst. Wniosek ogólny: pełnostronicowa transkrypcja polskich
  dokumentów wymaga modelu dostrajanego, nie gotowego malucha.

## Ograniczenia

1. Próba 48 stron — wyniki kierunkowe; pomiar pełnym zbiorem planowany na GPU.
2. Wzorzec ewaluacyjny jest w Markdownie (nagłówki, tabele jako HTML), a silniki
  wypisują czysty tekst — absolutne wartości CER są przez to zawyżone dla obu
  silników. Porównanie między silnikami pozostaje ważne.
3. Wymuszony język `pl` (obejście błędu ładowania domyślnego modelu angielskiego
  w transformers 5.x — patrz zgłoszenie w Beads `yf2`).

## Decyzja

Własny pipeline jest domyślnym silnikiem rozpoznawania w pomiarach; Tesseract
pozostaje baseline'em referencyjnym. Absolutne wyniki i materiał ewaluacyjny
zostaną opublikowane razem z nim — po jego ujawnieniu.
