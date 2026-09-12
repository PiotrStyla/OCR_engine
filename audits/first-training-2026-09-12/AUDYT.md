# Audyt pierwszego treningu TrOCR PL

Sprawdzono model HF `PiotrSty/trocr-pl-base` w rewizji
`0d42a14958063e4e0e94b02f6dce361337198c9f`, GitHub `origin/main` przy `dc8161d`
oraz niezacommitowaną poprawkę w lokalnym `Pulpit/OCR`. Nie uruchomiono
polecenia użytkownika, kolejnego treningu, publikacji ani pobierania pełnych wag.

## Ustalenia

1. **Poprawka nazwy LoRA jest trafna, ale diagnoza przyczyny regresji niepełna.**
   W `adapter/adapter_config.json` figuruje `o_proj`. Nagłówek safetensors
   zawiera 144 tensory LoRA: po 48 dla q_proj, k_proj i v_proj; zero dla
   o_proj/out_proj/fc1/fc2. Pierwszy run miał zatem adaptery dla 72 modułów
   (A i B dla każdego), a nie zerowy trening adapterów. `out_proj` jest nazwą
   projekcji TrOCR. Dodanie fc1/fc2 to oddzielne rozszerzenie pojemności,
   nie naprawa literówki.

2. **Niezgodny token startowy między treningiem a generowaniem.**
   Opublikowany `config.json` ma decoder_start_token_id=0, natomiast
   `generation_config.json` ma decoder_start_token_id=2. Skrypt ustawia
   model.config na cls_token_id; nie synchronizuje generation_config.
   Trening przesuwa etykiety według model.config, a generate korzysta
   z generation_config. To realny czynnik zakłócający ewaluację; nie wyznaczono
   jego wpływu na CER bez ponownego odczytu tych samych przykładów.

3. **Brak dowodu, że samo zwiększenie epok naprawi wynik.**
   Checkpoint-375: 3 epoki, 375 kroków, train_batch_size=16.
   Eval loss maleje 4.2539 → 3.4695 → 3.2953, lecz nie zapisano CER/WER
   w log_history. Brak best_model_checkpoint i best_metric. Kod nie ma
   compute_metrics ani load_best_model_at_end; eksportuje model końcowy.
   CER 71.85% vs 69.32% pochodzi z informacji użytkownika; w odczytanych
   plikach HF nie znaleziono predykcji potwierdzających te liczby.

4. **Eksperyment nie jest jeszcze dostatecznie odtwarzalny.**
   Karta datasetu deklaruje 2000 train + 200 val, a nowy komentarz notebooka
   mówi o ~1100. Liczba kroków i efektywny batch są zgodne z 2000 train,
   ale faktyczną liczebność trzeba zapisać z uruchomionego loadera.
   Notebook pobiera nieprzypięty snapshot datasetu i robi git pull przed
   treningiem. Nie ma jednoznacznego manifestu runu z rewizjami i pakietami.

5. **Publikacja nie zależy od poprawy jakości.**
   Notebook wywołuje upload_folder po ewaluacji, bez porównania wyniku
   z baseline i bez warunku akceptacji. Pierwsze artefakty wskazują zwykłe
   LoRA (`--no-4bit`, brak quantization_config), mimo opisu QLoRA w karcie HF.
   Karta nadal deklaruje „w trakcie treningu”.

## Kolejność przed drugim treningiem

1. Zachować pierwszą rewizję modelu i dane. Na tej samej walidacji zapisać
   baseline i pierwszy model jako per-line predictions, z identyfikatorami
   próbek, hashami obrazów, referencjami oraz ustawieniami generate.
2. Dla już wytrenowanego modelu porównać token startowy 2 z 0 (z treningu).
   Nie przedstawiać żadnego wariantu jako poprawy, zanim zostanie zmierzony.
   W następnym treningu ustawić jawnie i zgodnie tokeny model.config oraz
   generation_config, świadomie wybierając konwencję bazowego checkpointu.
3. Naprawić out_proj i dodać kontrolę, że każdy oczekiwany typ modułu został
   faktycznie objęty LoRA. Osobno badać dodatkowe fc1/fc2.
4. Mierzyć CER/WER co epokę, wybierać najlepszy checkpoint według val CER
   (greater_is_better=False), zapisać loss i próbki. 10 epok może być limitem
   eksperymentu, ale nie gwarancją poprawy ani kryterium wyboru modelu.
5. Publikować run do osobnego repo/rewizji. Promować do modelu domyślnego
   dopiero po porównaniu. Trzymać niezależny test poza doborem hiperparametrów.

## Źródła

- https://huggingface.co/PiotrSty/trocr-pl-base/tree/0d42a14958063e4e0e94b02f6dce361337198c9f
- https://huggingface.co/datasets/PiotrSty/ocr-pl-lines/tree/773832ea94643b630f63e8c2aa2002c634a7ade5
- https://github.com/PiotrStyla/OCR_engine/tree/dc8161d

Kopie konfiguracji i logów oraz wynik inspekcji nagłówka adaptera są obok
raportu. `reproduce.py` sprawdza semantykę na losowym, malutkim modelu CPU,
bez pretrained weights; nie jest ewaluacją jakości opublikowanego modelu.

Test CPU przeszedł przy transformers 4.57.6 i peft 0.19.1: pierwszy token
wejściowy treningu = 0, pierwszy token wygenerowanej sekwencji = 2.
Stara lista target_modules z nieistniejącym o_proj nadal tworzy trenowalne
adaptery q/k/v (384 parametry w modelu testowym). Bazowy model Microsoft
w odczytanym generation_config.json ma decoder_start_token_id=2.
