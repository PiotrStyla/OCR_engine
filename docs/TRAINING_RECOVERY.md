# Naprawa pierwszego treningu

## Co zmieniono

- Trening i generowanie mają identyczne tokeny start/pad/eos, wybrane jawnie
  z generation_config bazowego checkpointu (dla przypiętego TrOCR start=2).
- LoRA obejmuje q/k/v/**out_proj** w dekoderze. Każdy typ modułu jest
  sprawdzany przed treningiem, a obecność adaptera po wstrzyknięciu.
  Encoder pozostaje zamrożony. Opcjonalne `--include-mlp` włącza fc1/fc2
  jako oddzielny eksperyment; domyślnie wyłączone.
- CER i WER są mierzone co epokę. Eksportowany model pochodzi z najlepszego
  checkpointu według walidacyjnego CER, nie automatycznie z ostatniej epoki.
- Zapisywane są hashe obrazów/etykiet, wersje bibliotek, rewizja modelu,
  commit kodu, parametry treningu i dekodowania oraz kryterium wyboru.
- Pusta walidacja, brakujące etykiety, identyczne obrazy train/val i używany
  katalog wyjściowy powodują błąd. To nie jest pełna kontrola podobieństwa tekstów.
- NEFTune jest domyślnie wyłączone. Notebook używa zwykłego LoRA bez 4-bit.
- Notebook nie drukuje fragmentów tokenu HF, nie robi niekontrolowanego
  `git pull` i nie publikuje bezwarunkowo wyniku na HF.

## Najpierw ponowna ewaluacja — bez treningu

Otwórz na Kaggle `training/kaggle_reevaluate_first_run.ipynb` z GPU.
Notebook nie potrzebuje HF tokenu: dane i modele są publiczne.
Kaggle API token służy wyłącznie do zewnętrznego uruchomienia notebooka;
przy uruchomieniu w interfejsie Kaggle nie trzeba go wklejać do notebooka.

Porównanie na tych samych 200 przykładach obejmuje:

1. bazowy Microsoft TrOCR, start=2;
2. pierwszy polski model, start=2;
3. ten sam pierwszy polski model, start=0, zgodnie z jego treningiem.

Wszystkie trzy używają wspólnej polityki: beam=4, length_penalty=1,
brak zakazu powtarzania n-gramów, max_length=128. To kontrolowane nowe
porównanie, **nie odtworzenie historycznych liczb 69,32%/71,85%** przy
niezapisanych dawnych ustawieniach. Referencje nie trafiają do modelu.
Niekompletne odpowiedzi są oznaczane i pozostają w mianowniku metryk.

Przypięte wejścia:

- pierwszy model: `0d42a14958063e4e0e94b02f6dce361337198c9f`;
- Microsoft: `93450be3f1ed40a930690d951ef3932687cc1892`;
- dataset: `773832ea94643b630f63e8c2aa2002c634a7ade5`.

Wyniki powstaną w `/kaggle/working/first-run-reevaluation`: `summary.json`,
predykcje każdego wariantu, hashe wejść, wersje bibliotek i checksums.
Pobrać ten katalog przed końcem sesji. Nie wykonywać nowego treningu,
żeby zastąpić brakujące wyniki tego porównania.

## Potem trening kandydata

`training/kaggle_trocr_pl.ipynb` zaczyna od przypiętego modelu bazowego,
trenuje 3 epoki attention-only LoRA i zapisuje nowy katalog `trocr-pl-run2`.
Nie kontynuuje wadliwego checkpointu ani nie nadpisuje pierwszego modelu.
`selection.json`, `best_metrics.json`, `trainer_state.json` oraz `run.json`
pozwalają sprawdzić wybór najlepszego checkpointu. Liczba epok to budżet
eksperymentu, nie obietnica jakości. Brak automatycznej publikacji jest celowy:
najpierw porównanie, później decyzja o promocji modelu.

Walidacja służy wyborowi modelu. Niezależny zbiór testowy, realne linie druku,
kontrola długości etykiet i podobieństwa tekstów pozostają niezbędne przed
wnioskami o generalizacji. Poprawki infrastruktury nie dowodzą poprawy CER.

Weryfikacja lokalna: 111 testów przeszło. Test na losowym małym modelu CPU potwierdził gradienty LoRA oraz wybór pierwszego checkpointu po pogorszeniu CER w drugiej epoce. Nie jest to pomiar jakości pełnego modelu.

Kaggle: wybierz GPU T4 (API: --accelerator NvidiaTeslaT4). Zaobserwowany PyTorch 2.10.0+cu128 nie obsługuje P100 sm_60. Notebook wykonuje teraz rzeczywisty test CUDA przed pobraniem danych; samo cuda.is_available() nie wystarcza.
