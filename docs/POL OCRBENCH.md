# PolOCRBench: Polish Document Understanding

Transkrypcja, ekstrakcja tabel i informacji kluczowych z trudnych dokumentów
w języku polskim (obraz → tekst strukturalny). Szkic opisu zadania.

- Podzadanie A: transkrypcja pełnostronicowa (Markdown) — metryka CER (główna), WER, edit-distance struktury
- Podzadanie B: ekstrakcja tabel (HTML) — TEDS
- Podzadanie C: KIE (JSON per schemat) — field-level F1 po normalizacji
- Tracki: constrained / open / zero-shot-API
- Ewaluacja: deterministyczna, bez sędziów LLM; referencyjna implementacja A:
  `training/transcription_eval.py`

## 4. Źródła danych (wyłącznie licencje pozwalające na publikację i użycie komercyjne)

1. **Polona / Biblioteka Narodowa** — domena publiczna.

2. **IMPACT ground truth (PSNC)** — CC-BY-3.0 — **już zamrożone**:
   - subset `history_print`: 36 stron testu A (52 291 znaków GT) + 2 531 regionów
     treningowych z 27 kolekcji (XVII–XVIII w.),
   - provenance: `benchmarks/polocrbench/` (MANIFEST.sha256, README),
   - pipeline: `training/impact_index.py`, `training/impact_split.py`,
     `training/polocrbench_freeze.py`.

3. **Dokumenty publiczne z BIP, ISAP, KRS, wzory formularzy urzędowych** —
   informacja publiczna. Uwaga licencyjna: informacja publiczna ≠ automatyczna
   zgoda na redystrybucję komercyjną — status per źródło zapisywany w NOTICE
   datasetu (wzór: NOTICE.md w impact-psnc-polish-ocr).

4. **Dokumenty rzeczywiste od darczyńców (faktury, umowy, pisma) z
   kryptograficznym poświadczeniem redakcji — SealZero.**
   Firmy nie udostępnią dokumentów z PII (NIP, PESEL, kwoty, podpisy).
   Procedura przyjęcia dokumentu:
   - darczyńca (lub przyjmujący) maskuje regiony wrażliwe w przeglądarce
     (sealzero.dev, w pełni client-side, oryginał nie opuszcza maszyny),
   - SealZero wystawia certyfikat: SHA-256 obrazu zamaskowanego, współrzędne
     zadeklarowanych regionów redakcji, metryki zgodności poza regionami
     (odporne na re-kompresję JPEG), tożsamość weryfikującego, timestamp,
   - do benchmarku publikujemy: zamaskowany obraz + certyfikat.
   Oryginał nigdy nie opuszcza darczyńcy; społeczność otrzymuje
   kryptograficzny dowód, że zmieniono wyłącznie zadeklarowane regiony.
   Zapis w NOTICE subsetu: "obrazy rzeczywiste z certyfikatem autentyczności
   redakcji (SealZero); oryginały nie są publikowane". Karta certyfikatów
   per strona: JSONL obok MANIFEST.sha256 (konwencja jak w impact-print-v2).

5. **Dokumenty syntetyczne** — generator faktur/umów/pism z losową treścią
   (`training/generate_synthetic.py` + `corpus_pl.py`) + pipeline degradacji
   (skan, zdjęcie, druk-skan, kompresja). GT powstaje w generatorze
   (koszt anotacji zerowy); dla podzadania C generator emituje od razu
   do schematów JSON KIE.

6. **Pismo ręczne** — od wolontariuszy (społeczność Slayer AI Lab, ~15 osób
   zadeklarowanych) z pisemną zgodą, treść bez danych osobowych, licencja
   CC BY 4.0. Zbierane dokumenty nie wymagają certyfikatów SealZero
   (powstają wprost na potrzeby benchmarku).

## Anotacja GT
Test: ręczna transkrypcja z podwójną weryfikacją. Train: weryfikowana
pre-anotacja modelami + ręczna korekta. Rekordy pochodzenia anotacji
(annotator, data, SHA-256 obrazu, status weryfikacji) — analogia do certyfikatów
SealZero, JSONL w repozytorium datasetu.

## Rozmiary docelowe
Train ~2 000 stron (~1 200 syntetycznych), test A ~500 stron, test B ~500 stron
(typy dokumentów i degradacje nieobecne w train/test A).
Licencja: CC BY 4.0 (dane), MIT (skrypty). Publikacja: huggingface.co/SlayerLab.
