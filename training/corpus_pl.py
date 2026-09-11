"""Polski korpus tekstu do generatora syntetycznych danych treningowych.

Zdania i frazy z pełnym pokryciem polskich diakrytyków (ą, ć, ę, ł, ń, ó, ś, ź, ż).
Mieszanka: zdania potoczne, nazwy, formuły urzędowe, fragmenty znakowe.
"""

from __future__ import annotations

# Zdania z bogatym pokryciem diakrytyków
SENTENCES: list[str] = [
    "Żółw chodzi po łące i je trawę.",
    "Zażółć gęślą jaźń.",
    "Ala ma kota, a kot ma Alę.",
    "Cześć, jak się masz dzisiaj?",
    "Pies gonił zająca przez łąkę.",
    "Słowo klucz: mężczyzna ćwiczył ćwiczenia.",
    "Zażółcił on gęślą jaźń szybko.",
    "W źródle wytrysnęła struga wody.",
    "Książę Janusz wszedł do pałacu.",
    "Królowa Jadwiga władała królestwem.",
    "Mąż dobroci, czyni dobroć wszystkim.",
    "Rzeka płynie przez wioskę Białowieża.",
    "Gęś wędruje po podwórku w Gdańsku.",
    "Łódź płynie Wisłą w stronę Warszawy.",
    "Oświadczenie złożył dnia 15.03.2024.",
    "Urząd Miasta Kraków, ul. Wielicka 28.",
    "Wniosek o wydanie dowodu osobistego.",
    "Święty Marcin obchodzimy 11 listopada.",
    "Zażółć gęślą jaźń — test pangramu.",
    "Równość i braterstwo to nasze hasło.",
    "Śląsk Cieszyński leży na południu.",
    "Ulica Łąkowa 12, 80-001 Gdańsk.",
    "Pociąg do Poznania odjeżdża o 14:30.",
    "Zażółć gęślą jaźń w czerwcu 1999 r.",
    "Książka \"Pan Tadeusz\" Adama Mickiewicza.",
    "Zażółć gęślą jaźń — ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ.",
    "Imię i nazwisko: Jan Kowalski-Żyła.",
    "Nr rej. WA 12345, wydany w Łodzi.",
    "Data urodzenia: 01.05.1980 r.",
    "Miejscowość: Zamość, woj. lubelskie.",
    "Złożono w dniu 22 czerwca 2023 roku.",
    "Pieczęć: Urząd Skarbowy w Rzeszowie.",
    "Kwota: 1 234,56 zł (słownie: tysiąc...).",
    "Dziękuję za złożenie wniosku.",
    "Proszę o kontakt pod nr tel. 601-234-567.",
    "Załącznik nr 1 — kopia dowodu osobistego.",
    "Niniejszym oświadczam, iż powyższe dane są prawdziwe.",
    "W związku z powyższym wnoszę jak w sentencji.",
    "Gdańsk, dnia 14 września 2024 roku.",
    "Sygnatura akt: I ACa 123/24.",
    "Województwo mazowieckie, powiat piaseczyński.",
    "Ślub cywilny zawarty w USC w Krakowie.",
    "Źródło: Narodowy Bank Polski, tabela nr 123.",
    "Wskaźnik inflacji wyniósł 4,2% r/r.",
    "Ćwicz regularnie, a osiągniesz cel.",
    "Szczęście sprzyja lepiej przygotowanym.",
    "Późną jesienią liście żółkną i opadają.",
    "Młody żołnierz ćwiczył musztrę na placu.",
]

# Pojedyncze wyrazy z diakrytykami (do losowych kompozycji)
WORDS: list[str] = [
    "żółw", "łąka", "gęś", "jaźń", "mąż", "źródło", "kręgosłup", "ząb",
    "ślub", "węzeł", "pióro", "zamek", "oświetlenie", "kciuk", "wąż",
    "pęcherz", "ściana", "łódź", "królik", "żołądek", "śródmieście",
    "gąbka", "łącznik", "piętro", "ośmiokąt", "żyrafa", "wątroba",
    "łeb", "światło", "zięć", "pączek", "rękaw", "gęsia", "łazienka",
    "ćma", "źrebak", "ósmoklasista", "śruba", "lód", "kędzierzawy",
    "żubr", "mężczyzna", "łabędź", "pstrąg", "żelazo", "ślizgawka",
    "wąż", "oświetlenie", "zaułek", "łza", "kęs", "żagiel", "śpiew",
]

# Wielkie litery / nagłówki / formuły urzędowe
FORMULAS: list[str] = [
    "WNIOSEK O WYDANIE PASZPORTU",
    "Załącznik nr 2 do wniosku",
    "Potwierdzam odbiór pisma.",
    "Urząd Stanu Cywilnego w Krakowie",
    "Podpis: ...........................",
    "Dowód osobisty seria ABC nr 123456",
    "Zamieszkały: ul. Słoneczna 5/12",
    "NIP: 123-456-78-90, PESEL: 80010112345",
    "REGON: 123456789, KRS: 0000123456",
    "Warszawa, 01.06.2024 r.",
    "Do wiadomości: Sekretariat US",
    "Nr sprawy: 123/2024/XI",
]
