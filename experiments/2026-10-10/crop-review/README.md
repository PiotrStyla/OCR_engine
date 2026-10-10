# Pakiet recenzyjny granic wycinków (2026-10-10)

Materiał do bramki 2: 6 przypadków, w których wycinek zawiera atrament sąsiada.
Każdy ma arkusz z czerwonymi pikselami śladu, pomiar głębokości i mechaniczną
rekomendację. Nic nie zostało przycięte, przyjęte ani odrzucone — decyzja jest
ludzka.

| Przypadek | pikseli | głębokość | rekomendacja |
|---|---:|---:|---|
| andersen-basnie-1929-0225-line-015 | 30 | 2 px | trim-2px |
| may-…-0523-line-014 | 1 | 1 px | trim-2px |
| may-…-0523-line-026 | 1 | 0 px | trim-2px |
| may-…-0523-line-002 | 8 | 3 px | trim-to-depth-or-accept |
| may-…-0523-line-031 | 9 | 3 px | trim-to-depth-or-accept |
| may-…-0523-line-003 | 15 | 8 px | trim-to-depth-or-accept |

Żaden przypadek nie wymaga „review-closely" (najgłębszy ślad: 8 px w kadrze
zapasowanym 1–2 px). Arkusze: `sheets/*.png`. `review.json` — pomiary i
rekomendacje; status decyzji: `decided: false`.

Pochodne: [kontrola atramentu](../../docs/PRINTED_REPLAY_INK_CHECK_20261010.md),
[kolejka recenzyjna](../reference-defects/README.md).
