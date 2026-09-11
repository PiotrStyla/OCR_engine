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

# --- Domeny specjalistyczne ---

# Faktury / finanse
INVOICE: list[str] = [
    "Faktura VAT nr FV/2024/0345",
    "Termin płatności: 14 dni od daty wystawienia",
    "Forma płatności: przelew na rachunek bankowy",
    "Nr konta: PL 12 3456 7890 1234 5678 9012 3456",
    "Kwota netto: 1 234,56 zł, VAT 23%: 283,95 zł",
    "Kwota brutto: 1 518,51 zł (słownie: tysiąc pięćset osiemnaście zł 51/100)",
    "Sprzedawca: Firma Sp. z o.o., ul. Przemysłowa 7, 00-001 Warszawa",
    "Nabywca: Piotr Kowalski, ul. Kwiatowa 15/3, 30-002 Kraków",
    "Data wystawienia: 15.03.2024, data sprzedaży: 14.03.2024",
    "Miejsce wystawienia: Warszawa",
    "Poz. 1 — Usługa konsultingowa, 8 h × 150,00 zł = 1 200,00 zł",
    "Suma: 1 200,00 zł netto / 1 476,00 zł brutto",
    "Faktura korygująca do FV/2024/0299 z dnia 28.02.2024",
    "Rachunek nr R/2024/0112 — paragon fiskalny",
]

# Umowy / prawnicze
LEGAL: list[str] = [
    "Niniejsza umowa została zawarta w dniu ... pomiędzy:",
    "na podstawie art. 415 § 2 Kodeksu cywilnego",
    "Strony ustaliły, co następuje:",
    "§ 1. Przedmiot umowy",
    "§ 2. Wynagrodzenie i warunki płatności",
    "Umowa wchodzi w życie z dniem podpisania.",
    "W sprawach nieuregulowanych stosuje się przepisy k.c.",
    "Sporne kwestie rozstrzyga sąd właściwy dla siedziby.",
    "Niniejszym zawiadamiam, że odstępuję od umowy.",
    "Wypowiedzenie z zachowaniem 3-miesięcznego okresu.",
    "Pełnomocnictwo ogólne nr 45/2024",
    "Akt notarialny Rep. A nr 1234/2024",
    "Wierzyciel: ..., dłużnik: ..., kwota: ...",
    "Sąd Rejonowy w Krakowie, I Wydział Cywilny",
]

# Medyczne / recepty
MEDICAL: list[str] = [
    "Pacjent: 65 lat, waga 78 kg",
    "Dawka: 1 tabletka 2× dziennie rano i wieczorem",
    "Rp. Amoxicillin 500 mg — D.t.d. 20 tabl.",
    "Rozpoznanie: nadciśnienie tętnicze I10",
    "Zalecenia: dieta ubogosodowa, kontrola za 3 mies.",
    "Ciśnienie: 145/92 mmHg, tętno: 76/min",
    "Badanie krwi: morfologia + OB + CRP",
    "Upoważnienie do odbioru wyników: żona",
]

# Daty / liczby / kontakt — generowane losowo
_MONTHS_PL = [
    "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
    "lipca", "sierpnia", "września", "października", "listopada", "grudnia",
]
_CITIES = [
    "Warszawa", "Kraków", "Gdańsk", "Wrocław", "Poznań", "Łódź",
    "Szczecin", "Lublin", "Katowice", "Białystok", "Rzeszów", "Toruń",
]
_STREETS = [
    "Słoneczna", "Kwiatowa", "Długa", "Krótka", "Mickiewicza", "Słowackiego",
    "Piłsudskiego", "Kościuszki", "Wielicka", "Grunwaldzka", "Leśna", "Polna",
]
_SURNAMES = [
    "Kowalski", "Nowak", "Wiśniewski", "Wójcik", "Kowalczyk", "Kamiński",
    "Lewandowski", "Zieliński", "Szymański", "Woźniak", "Dąbrowski", "Mazur",
]


def random_date(rng) -> str:
    """Losowa data po polsku lub liczbowa."""
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1990, 2025)
    if rng.random() < 0.5:
        return f"{d} {_MONTHS_PL[m - 1]} {y} r."
    return f"{d:02d}.{m:02d}.{y}"


def random_amount(rng) -> str:
    """Losowa kwota w zł."""
    whole = rng.randint(1, 99999)
    cents = rng.randint(0, 99)
    fmt = f"{whole:,}".replace(",", " ")
    if rng.random() < 0.5:
        return f"{fmt},{cents:02d} zł"
    return f"{fmt},{cents:02d} PLN"


def random_address(rng) -> str:
    """Losowy adres."""
    n, m = rng.randint(1, 200), rng.randint(1, 50)
    code = f"{rng.randint(10, 99)}-{rng.randint(100, 999)}"
    street = f"ul. {rng.choice(_STREETS)} {n}"
    if rng.random() < 0.5:
        street += f"/{m}"
    return f"{street}, {code} {rng.choice(_CITIES)}"


def random_person(rng) -> str:
    """Losowe imię i nazwisko."""
    names = ["Jan", "Anna", "Piotr", "Maria", "Krzysztof", "Katarzyna",
             "Andrzej", "Małgorzata", "Tomasz", "Agnieszka"]
    return f"{rng.choice(names)} {rng.choice(_SURNAMES)}"


def random_contact(rng) -> str:
    """Losowy telefon lub e-mail."""
    if rng.random() < 0.5:
        return f"tel. +48 {rng.randint(500, 899)} {rng.randint(100, 999)} {rng.randint(100, 999)}"
    name = rng.choice(["jan", "anna", "biuro", "kontakt", "sekretariat"])
    domain = rng.choice(["firma.pl", "urzad.gov.pl", "gmail.com", "poczta.pl"])
    return f"{name}@{domain}"


def random_numbers_id(rng) -> str:
    """Losowy identyfikator urzędowy."""
    kind = rng.choice(["NIP", "PESEL", "REGON", "KRS"])
    if kind == "NIP":
        return f"NIP: {rng.randint(100, 999)}-{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}"
    if kind == "PESEL":
        return f"PESEL: {rng.randint(10**10, 10**11 - 1)}"
    if kind == "REGON":
        return f"REGON: {rng.randint(10**8, 10**9 - 1)}"
    return f"KRS: {rng.randint(10**9, 10**10 - 1):010d}"


DOMAIN_GENERATORS = [
    random_date, random_amount, random_address, random_person,
    random_contact, random_numbers_id,
]
