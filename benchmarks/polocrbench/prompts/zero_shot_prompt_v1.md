# PolOCRBench — jednolity prompt zero-shot v1

Wersja promptu: `polocrbench-zero-shot-prompt-v1`. Plik jest niezmienny; jego
SHA-256 jest przypięty w `training/validate_submission.py` i wymagany w
zgłoszeniu tracku zero-shot / API (`prompt_version` + `prompt_sha256`).

Zasady tracku zero-shot / API:

- szablony poniżej stosuje się dosłownie, bez dopisków, few-shot, filtrów
  systemowych ani dodatkowych instrukcji;
- `{image}` oznacza obraz wejściowy (strony lub wycinka tabeli) przesłany do
  modelu bez opisu treści;
- dozwolone są dowolne, niezależne wywołania z tym samym promptem oraz
  mechaniczne wyciągnięcie odpowiedzi (np. usunięcie plotek kodu), bez
  poprawek treści;
- niedozwolone jest strojenie wag, adaptery, pamięć między wywołaniami oraz
  korekta wyników modelem lub człowiekiem.

## A — transkrypcja pełnostronicowa

```text
Przepisz dokładnie całą widoczną treść tej strony dokumentu jako Markdown.
Zachowaj kolejność czytania, nagłówki, listy, akapity i wcięcia. Nie dodawaj
komentarzy, tytułów ani tłumaczeń. Odpowiedz wyłącznie treścią Markdown.
```

## B — ekstrakcja tabel

```text
Wyodrębnij tabelę z tego obrazu jako jedną tabelę HTML. Zachowaj strukturę
wierszy i kolumn, użycie th/td oraz atrybuty colspan i rowspan, a w komórkach
ich dokładną treść. Nie dodawaj atrybutów stylów, klas ani komentarzy.
Odpowiedz wyłącznie kodem HTML tabeli.
```

## C — ekstrakcja informacji kluczowych (KIE)

`{doc_type}` to typ dokumentu z zamrożonego schematu (faktura, umowa,
pismo_urzedowe, formularz), a `{fields}` to lista pól tego schematu wraz z
typami wartości (text, name, date, money, nip, pesel, address).

```text
Wyodrębnij z tego dokumentu pola zgodnie ze schematem. Typ dokumentu:
{doc_type}. Pola: {fields}. Zwróć wyłącznie obiekt JSON z kluczami ze schematu
i odczytanymi wartościami; pól brakujących nie umieszczaj. Nie zmyślaj
wartości. Nie dodawaj komentarzy ani tekstu spoza JSON-a.
```
