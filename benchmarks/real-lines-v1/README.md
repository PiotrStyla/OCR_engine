# Real Polish print lines v1

Benchmark samego rozpoznawacza linii, bez wpływu detektora dokumentu.
Zawiera 75 cropów z dwóch public-domain historycznych stron używanych w
`print-pilot-v1`:

- `odezwa-12`: 33 linie,
- `torun-74`: 42 linie.

Każdy crop ma odpowiadający plik `.txt`. `manifest.jsonl` przechowuje tekst,
SHA-256 cropa i obrazu źródłowego, indeks linii, współrzędne źródłowe oraz
metodę wycięcia.

Cropy wyznaczono ręcznie na podstawie projekcji poziomej i granic akapitów,
a następnie sprawdzono wizualnie na arkuszach kontaktowych. Referencje są
przeniesione bez zmian z `benchmarks/print-pilot-v1/manifest.jsonl`; mają
jednego autora i nie przeszły niezależnej drugiej weryfikacji.

## Ewaluacja

```bash
python -m training.evaluate \
  --data benchmarks/real-lines-v1/pairs \
  --model /path/to/trocr-pl-model
```

Nie należy łączyć tego wyniku z wynikiem pełnych stron: benchmark mierzy tylko
recognizer na prawidłowo wyciętych liniach, a pełna strona mierzy także detekcję,
segmentację i kolejność czytania.
