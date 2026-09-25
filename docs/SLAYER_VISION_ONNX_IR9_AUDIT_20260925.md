# SLAYER Vision ONNX IR9 - audit paczki

Data audytu: 2026-09-25

## Zakres

Artefakt lokalny: `SLAYER-Vision-ONNX-IR9.zip`

- rozmiar ZIP: 832 049 622 B;
- SHA-256 ZIP: `3b1570ac4956ca0ca0e7b9150283462ee467d6d8265e39673b895f7f5e199592`;
- 9 wpisow, 897 538 341 B po rozpakowaniu;
- brak sciezek wychodzacych poza katalog, duplikatow nazw i symptomow bomby ZIP;
- audyt statyczny, bez uruchamiania inferencji.

## Wynik statycznej walidacji

Wszystkie trzy grafy przechodza `onnx.checker.check_model(..., full_check=True)`
z ONNX 1.17.0. Kazdy graf ma IR 9, opset 18 i nie korzysta z domen
niestandardowych.

| Graf | Wejscie | Wyjscie | Wezly | Wagi zewnetrzne |
| --- | --- | --- | ---: | ---: |
| `vision_projector.onnx` | `pixel_values`: float `[batch,3,224,224]` | `image_embeds`: float `[1,196,768]` | 567 | 355 794 944 B |
| `embed_tokens.onnx` | `ids`: int64 `[batch,seq]` | `embeds`: float `[batch,seq,768]` | 1 | 98 304 000 B |
| `lm_embeds.onnx` | `inputs_embeds`: float `[batch,seq,768]`, `attention_mask`: int64 `[batch,seq]` | `logits`: float `[batch,seq,32000]` | 593 | 440 139 776 B |

Wszystkie odwolania do plikow danych ONNX sa obecne. Wejscie enkodera obrazu
jest dynamiczne po wymiarze batch, ale jego wyjscie ma batch zapisany jako `1`.
Do czasu testu wykonawczego nalezy traktowac eksport jako model batch-size 1.

## Ustalenia krytyczne

1. Paczka opisuje model `obraz -> podpis po polsku`, a nie silnik wiernej
   transkrypcji OCR. Nie ma jeszcze dowodu CER/WER ani testu zachowania pisowni
   historycznej.
2. Dekoder opisany w `manifest.json` jest wadliwy dla czesci slownika.
   `tokens_decoded.json` ma 32 000 pozycji, ale 167 pozycji zawiera znak
   zastepczy U+FFFD, a 156 wartosci jest zduplikowanych. Konkatenacja osobno
   zdekodowanych tokenow moze niszczyc znaki wielobajtowe, w tym polskie znaki.
   Nalezy dolaczyc oryginalny tokenizer i dekodowac cala sekwencje identyfikatorow.
3. Brakuje kompletnej umowy wykonawczej: normalizacji obrazu SigLIP,
   kolejnosci kanalow, promptu/prefiksu, tokenu start/stop, limitu generacji,
   obslugi EOS oraz kryteriow zatrzymania.
4. Manifest nie przypina rewizji modeli bazowych ani checkpointu `out\\run-final`,
   nie zawiera sum kontrolnych plikow i nie dokumentuje zbioru treningowego.
5. Graf LM nie ma wejsc/wyjsc KV-cache. Generowanie autoregresyjne bedzie
   przeliczalo cala sekwencje przy kazdym tokenie i moze byc wolne na CPU.

## Pochodzenie i licencje

Manifest deklaruje:

- LM: `SlayerLab/goLLeM-110M-PL-SFT-merged`, CC-BY-SA-4.0;
- enkoder: `google/siglip-base-patch16-224`, Apache-2.0;
- projector i dostrojenie: Piotr Styla / PiotrSty;
- licencja MIT kodu aplikacji nie obejmuje wag.

Te informacje sa deklaracja paczki. Przed publikacja trzeba przypiac rewizje
zrodel, zachowac teksty licencji i ustalic licencje calego modelu pochodnego.

## Sumy kontrolne payloadu

| Plik | SHA-256 |
| --- | --- |
| `embed_tokens.onnx` | `191338b81038c8f9d209441f3dc90ebd7484926037784312f6479aa73897735b` |
| `embed_tokens.onnx.data` | `ac2b33679389d26b90508779f2aa518eeabb3d9d9e2d00966c392c863ed7c78e` |
| `lm_embeds.onnx` | `84310fc4aecc1cbf114da515a6e19ff202ea710642612833c71a168e351e3da5` |
| `lm_embeds.onnx.data` | `593f4b01165010201d4f6a6c76fcb24c308a5cbdc24c74861bac16aecbfe3265` |
| `vision_projector.onnx` | `88fbdec4f6aec429c3c2630e14e828ddf9a9e1e85b96569071b828f90e41d31a` |
| `vision_projector.onnx.data` | `0582b7f59cabae62838085f0cc41bd29adfa192347a280be65a5b65424c6d973` |
| `tokens_decoded.json` | `1b9c2bb8ab9acc28c990c5bc72e14478e75cd59c146a671c1b22634b44915afd` |
| `manifest.json` | `996fc9517af8b6118a8694004d7b0134a72e82bc96d0a5dcd6b0773c72f9f3da` |
| `MODEL-ATTRIBUTION.txt` | `849c9964bc8ac2feaac69796dfd421400c9d646db7fc0c61bd247d035d7ea71d` |

## Bramka przed integracja z OCR_engine

1. Wyeksportowac lub dolaczyc `tokenizer.json` i `tokenizer_config.json` z
   przypietej rewizji LM; usunac dekodowanie przez `tokens_decoded.json`.
2. Dodac jawny `preprocessor_config.json` oraz test zgodnosci obrazu i logitow
   PyTorch versus ONNX na co najmniej trzech rzeczywistych skanach.
3. Dodac wykonywalny runner ONNX Runtime z EOS, limitem tokenow i raportem czasu.
4. Zmierzyc CER/WER na zamrozonym holdoucie pelnych polskich stron oraz osobno
   na materialach historycznych. Ground truth moze miec tylko normalizacje
   NFC i bialych znakow; bez modernizacji pisowni.
5. Dopiero po tych testach klasyfikowac model jako kandydat OCR. Obecny status:
   poprawny technicznie eksport eksperymentalnego image-captioning VLM.
