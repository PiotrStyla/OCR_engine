# Kandydat referencji v2 (2026-10-10)

Decyzja (użytkownik, rozmowa): mapować znaki PUA do **właściwych glifów
historycznych**. Zbudowany kandydat manifestu — zamrożony plik nietknięty.

- `history_testA_manifest.candidate-v2.jsonl` — kandydat (36 wierszy)
- `mapping.json` — mapa 5 kodpunktów (601 miejsc) + 5 mieszanych do wglądu
- `impact.json` — CER 6 systemów na zamrożonych vs kandydackich referencjach

**Efekt:** każdy system poprawia się o 0,55–1,78 pkt CER (gpt-5.4: 16,87% →
15,09%; Qwen3-VL-4B: 20,03% → 18,96%). Potwierdza to zmierzony wcześniej próg
szumu (≥1,36%); zysk rośnie z siłą systemu, bo najsilniejsze czytają `ſt`
poprawnie.

Zasada utrzymana: mapowanie idzie do **glifu historycznego** (`ſt`, `ſi`, `ſł`,
`ct`, `ſsi`), nie do odpowiednika nowoczesnego — pisownia historyczna pozostaje.

Status: **kandydat do wydania**, nie publikacja. Wymaga zgodności drugiego
recenzenta (workflow `tools/annotation-review`) przed podmianą w benchmarku.
Nierozstrzygnięte: 5 rzadkich kodpunktów (23 miejsca), U+FFFD (86), errata.
