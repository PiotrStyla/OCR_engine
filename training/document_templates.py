"""Synthetic Polish document content for PolOCRBench (pure and deterministic).

Builds document models for the four KIE types from ``training.kie_eval.SCHEMAS``
(faktura, umowa, pismo_urzedowe, formularz): text blocks (subtask A ground
truth), tables with HTML (subtask B) and KIE fields (subtask C), always
mutually consistent — every field value appears verbatim in the block text and
every table cell appears in its rows.

No rendering and no ML/network dependencies (page rasterization and
degradations live in ``training.generate_documents``). Each builder takes a
seeded ``random.Random`` and returns::

    {'doc_type': str,
     'blocks': [ {'kind': 'heading'|'kv'|'paragraph'|'list'|'table'|'signatures', ...} ],
     'fields': {name: displayed value},
     'tables': [ {'html': str, 'rows': [[str, ...], ...]} ]}

Subtask A convention for tables: one line per row, cells joined with single
spaces, no Markdown/HTML markup — transcription scoring must not be dominated
by markup choices. Money is rendered in Polish notation (``1 234,56 zł``),
dates in one of three displayed formats, NIP/PESEL carry valid checksums.
"""
from __future__ import annotations

import html as _html
import random
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP

from training.kie_eval import SCHEMAS

PROTOCOL_VERSION = 'polocrbench-synth-docs-v1'
DOC_TYPES = tuple(SCHEMAS)

_CITIES = ['Warszawa', 'Kraków', 'Łódź', 'Wrocław', 'Poznań', 'Gdańsk', 'Szczecin',
           'Bydgoszcz', 'Lublin', 'Katowice', 'Białystok', 'Rzeszów', 'Toruń', 'Opole']
_STREETS = ['Marszałkowska', 'Długa', 'Piłsudskiego', 'Słoneczna', 'Krótka', 'Kwiatowa',
            'Polna', 'Lipowa', 'Nowowiejska', 'Graniczna', 'Fabryczna', 'Szkolna',
            'Kościuszki', '1 Maja', 'Warszawska', 'Leśna', 'Ogrodowa', 'Kolejowa']
_FIRST_F = ['Anna', 'Maria', 'Katarzyna', 'Małgorzata', 'Agnieszka', 'Joanna', 'Barbara',
            'Ewa', 'Magdalena', 'Halina']
_FIRST_M = ['Jan', 'Piotr', 'Andrzej', 'Tomasz', 'Krzysztof', 'Marek', 'Jerzy', 'Adam',
            'Stefan', 'Zbigniew']
_LAST_STEM = ['Kowal', 'Nowak', 'Wiśniew', 'Wójcik', 'Kowalczyk', 'Kamiń', 'Lewandow',
              'Zieliń', 'Szymań', 'Woźniak', 'Dąbrow', 'Kozłow', 'Jankow', 'Mazur',
              'Krawczyk', 'Piotrow', 'Grabow', 'Pawłow', 'Michal', 'Wróbel']
_COMPANY_PATTERNS = [
    'Zakład Handlowo-Usługowy {last} Sp. z o.o.',
    'Przedsiębiorstwo Wielobranżowe {stem} i Wspólnicy Sp. z o.o.',
    'PHU {person}',
    'Invest-Bud {stem} S.A.',
    '{city}-Tech Sp. z o.o.',
    'Centrum Usługowe {person} Sp. z o.o.',
    'Hurtownia Spożywcza {stem} Sp. jawna',
    'Biuro Projektowe {person}',
]
_INSTITUTIONS = [
    'Urząd Miasta {city}', 'Starostwo Powiatowe w {city}', 'Urząd Gminy {city}',
    'Urząd Skarbowy w {city}', 'Zakład Ubezpieczeń Społecznych Oddział w {city}',
    'Wojewódzki Urząd Pracy w {city}', 'Powiatowy Inspektorat Nadzoru Budowlanego w {city}',
    'Miejski Ośrodek Pomocy Społecznej w {city}',
]
_ITEMS = [('Usługa doradcza', 'godz.'), ('Materiały biurowe', 'szt.'), ('Serwis urządzenia', 'szt.'),
          ('Transport towaru', 'km'), ('Projekt techniczny', 'szt.'), ('Naprawa instalacji', 'szt.'),
          ('Szkolenie personelu', 'os.'), ('Dostawa komponentów', 'kpl.'), ('Opłata licencyjna', 'mies.'),
          ('Robocizna ogólnobudowlana', 'godz.'), ('Ekspertyza techniczna', 'szt.'),
          ('Obsługa informatyczna', 'mies.')]
_SUBJECTS_UMOWA = ['świadczenie usług doradczych', 'dostawę materiałów biurowych',
                   'prace remontowe lokalu', 'obsługę informatyczną', 'organizację szkolenia',
                   'wykonanie projektu technicznego', 'transport i montaż urządzeń']
_SUBJECTS_PISMO = ['wydania zaświadczenia o niezaleganiu', 'udzielenia pozwolenia na budowę',
                   'przyznania dodatku mieszkaniowego', 'umorzenia zaległości podatkowej',
                   'wpisu do rejestru działalności regulowanej', 'przyjęcia zgłoszenia prac',
                   'zmiany danych w ewidencji']
_FORMS = [('ZUS Z-3', 'Zaświadczenie płatnika składek'),
          ('WD-1', 'Wniosek o dokonanie wpisu do ewidencji'),
          ('PT-1', 'Podanie o wydanie duplikatu dokumentu'),
          ('US-24', 'Formularz zgłoszenia zmiany danych'),
          ('IN-1', 'Zgłoszenie zamiaru rozpoczęcia działalności'),
          ('MR-6', 'Wniosek o przyznanie świadczenia')]
_OBLIGATIONS = [
    'Wykonawca rozpocznie realizację przedmiotu umowy w terminie 7 dni od dnia podpisania.',
    'Zamawiający udostępnia pomieszczenia niezbędne do realizacji przedmiotu umowy.',
    'Strony zobowiązują się do niezwłocznego informowania o zmianie danych kontaktowych.',
    'Rozliczenie następuje na podstawie faktury VAT z terminem płatności 14 dni.',
    'Wykonawca ponosi odpowiedzialność za jakość wykonanych prac.',
    'Korespondencja dotycząca umowy doręczana jest listem poleconym za potwierdzeniem odbioru.',
    'Zmiana umowy wymaga formy pisemnej pod rygorem nieważności.',
]
_PISMO_BODY = [
    'W odpowiedzi na pismo z dnia {date} uprzejmie informuję, że sprawa została rozpatrzona.',
    'Postępowanie w niniejszej sprawie toczy się przed organem pierwszej instancji.',
    'Na podstawie zebranego materiału dowodowego organ ustalił następujący stan faktyczny.',
    'Zawiadamiam o wszczęciu z urzędu postępowania administracyjnego w przedmiotowej sprawie.',
    'Wzywam do uzupełnienia braków formalnych w terminie 14 dni od dnia doręczenia wezwania.',
    'Rozstrzygnięcie następuje w drodze decyzji administracyjnej.',
]
_POU = [
    'Stronie służy prawo odwołania do organu drugiej instancji w terminie 14 dni od dnia doręczenia.',
    'Pismo niniejsze sporządzono w dwóch jednobrzmiących egzemplarzach.',
    'Załączniki stanowią integralną część pisma.',
    'Brak odpowiedzi w wyznaczonym terminie skutkuje wydaniem decyzji na podstawie akt sprawy.',
]


def cell(text, span=1):
    return (str(text), int(span))


def make_table(rows, header=True):
    """rows: list of rows; each row is a list of cell(text, span) tuples."""
    parts = ['<table>']
    for index, cells in enumerate(rows):
        tag = 'th' if header and index == 0 else 'td'
        parts.append('<tr>')
        for text, span in cells:
            colspan = '' if span == 1 else f' colspan="{span}"'
            parts.append(f'<{tag}{colspan}>{_html.escape(text, quote=False)}</{tag}>')
        parts.append('</tr>')
    parts.append('</table>')
    return {'html': ''.join(parts),
            'rows': [[text for text, _ in cells] for cells in rows],
            'cells': [[(text, span) for text, span in cells] for cells in rows]}


def document_text(doc):
    """Subtask A ground truth: Markdown skeleton, table rows as plain lines."""
    lines = []
    for block in doc['blocks']:
        kind = block['kind']
        if kind == 'heading':
            lines.append('#' * block['level'] + ' ' + block['text'])
        elif kind == 'kv':
            lines.append(f"{block['label']}: {block['value']}")
        elif kind == 'paragraph':
            lines.append(block['text'])
        elif kind == 'list':
            for index, item in enumerate(block['items'], 1):
                lines.append(f'{index}. {item}' if block['ordered'] else f'- {item}')
        elif kind == 'table':
            for row in doc['tables'][block['index']]['rows']:
                lines.append(' '.join(row))
        elif kind == 'signatures':
            for left, right in zip(block['left'], block['right']):
                lines.append(f'{left} {right}'.strip())
    return '\n'.join(lines)


def verify_document(doc):
    """Ground truth consistency: every field value must appear in the text."""
    schema = SCHEMAS[doc['doc_type']]
    if set(doc['fields']) - set(schema):
        raise ValueError('Fields outside the schema')
    haystack = document_text(doc).casefold()
    for name, value in doc['fields'].items():
        if not str(value).strip() or str(value).casefold() not in haystack:
            raise ValueError(f'Field {name} not visible in document text: {value!r}')
    if len(doc['tables']) != sum(1 for block in doc['blocks'] if block['kind'] == 'table'):
        raise ValueError('Table blocks and table records disagree')
    return doc


# --- deterministic fake data -------------------------------------------------


def format_money(amount):
    amount = Decimal(amount).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    sign = '-' if amount < 0 else ''
    amount = abs(amount)
    whole, _, fraction = f'{amount:f}'.partition('.')
    grouped = f'{int(whole):,}'.replace(',', ' ')
    return f'{sign}{grouped},{fraction} zł'


def format_money_plain(amount):
    amount = Decimal(amount).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    whole, _, fraction = f'{amount:f}'.partition('.')
    grouped = f'{int(whole):,}'.replace(',', ' ')
    return f'{grouped},{fraction}'


def format_amount(rng, amount):
    return rng.choice((format_money, lambda value: format_money(value).replace(' zł', ' PLN'),
                       lambda value: format_money_plain(value) + ' zł'))(amount)


def random_date(rng):
    start = date(2024, 1, 1)
    return start + timedelta(days=rng.randint(0, 900))


def format_date(rng, day):
    month_names = {1: 'stycznia', 2: 'lutego', 3: 'marca', 4: 'kwietnia', 5: 'maja',
                   6: 'czerwca', 7: 'lipca', 8: 'sierpnia', 9: 'września', 10: 'października',
                   11: 'listopada', 12: 'grudnia'}
    style = rng.choice(('dotted', 'iso', 'named'))
    if style == 'dotted':
        return day.strftime('%d.%m.%Y')
    if style == 'iso':
        return day.isoformat()
    return f'{day.day} {month_names[day.month]} {day.year} r.'


def make_nip(rng):
    while True:
        digits = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(8)]
        check = sum(w * d for w, d in zip((6, 5, 7, 2, 3, 4, 5, 6, 7), digits)) % 11
        if check != 10:
            return ''.join(str(d) for d in digits + [check])


def make_pesel(rng):
    year, month, day = rng.randint(1950, 2010), rng.randint(1, 12), rng.randint(1, 28)
    encoded_month = month + (20 if year >= 2000 else 0)
    digits = [year % 100 // 10, year % 10, encoded_month // 10, encoded_month % 10,
              day // 10, day % 10] + [rng.randint(0, 9) for _ in range(4)]
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    check = (10 - sum(w * d for w, d in zip(weights, digits)) % 10) % 10
    return ''.join(str(d) for d in digits + [check])


def make_account(rng):
    digits = ''.join(str(rng.randint(0, 9)) for _ in range(26))
    return 'PL' + digits[:2] + ' ' + digits[2:6] + ' ' + digits[6:10] + ' ' + \
        digits[10:14] + ' ' + digits[14:18] + ' ' + digits[18:22] + ' ' + digits[22:26]


def format_nip(rng, nip):
    style = rng.choice(('dashes', 'spaces', 'plain'))
    if style == 'dashes':
        return f'{nip[:3]}-{nip[3:6]}-{nip[6:8]}-{nip[8:]}'
    if style == 'spaces':
        return f'{nip[:3]} {nip[3:6]} {nip[6:]}'
    return nip


def person(rng):
    female = rng.random() < 0.5
    first = rng.choice(_FIRST_F if female else _FIRST_M)
    last = rng.choice(_LAST_STEM) + ('ska' if female else 'ski')
    return f'{first} {last}'


def company(rng):
    return rng.choice(_COMPANY_PATTERNS).format(
        last=rng.choice(_LAST_STEM) + 'ski',
        stem=rng.choice(_LAST_STEM),
        person=person(rng),
        city=rng.choice(_CITIES))


def institution(rng):
    return rng.choice(_INSTITUTIONS).format(city=rng.choice(_CITIES))


def street_address(rng):
    city = rng.choice(_CITIES)
    postcode = f'{rng.randint(0, 99):02d}-{rng.randint(0, 999):03d}'
    number = rng.randint(1, 120)
    apartment = f'/{rng.randint(1, 40)}' if rng.random() < 0.5 else ''
    street = rng.choice(_STREETS)
    prefix = rng.choice(('ul.', 'al.', 'ul.'))
    if prefix == 'al.':
        street = rng.choice(('Jana Pawła II', 'Niepodległości', 'Solidarności', 'Legionów'))
    return f'{prefix} {street} {number}{apartment}, {postcode} {city}'


# --- document templates ------------------------------------------------------


def build_faktura(rng):
    day = random_date(rng)
    seller, buyer = company(rng), (company(rng) if rng.random() < 0.7 else person(rng))
    seller_address, buyer_address = street_address(rng), street_address(rng)
    invoice_number = f'FV/{day.year}/{day.month:02d}/{rng.randint(1, 999):03d}'
    net_total, vat_total = Decimal('0.00'), Decimal('0.00')
    rows = [[cell('Lp.'), cell('Nazwa towaru lub usługi'), cell('Ilość'), cell('J.m.'),
             cell('Cena netto'), cell('Wartość netto')]]
    for index in range(1, rng.randint(2, 5) + 1):
        name, unit = rng.choice(_ITEMS)
        quantity = Decimal(rng.randint(1, 40)) / Decimal(rng.choice((1, 2, 4)))
        price = Decimal(rng.randint(500, 250000)) / Decimal(100)
        value = (quantity * price).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        rate = rng.choice((23, 8, 5))
        vat_total += (value * Decimal(rate) / Decimal(100)).quantize(Decimal('0.01'),
                                                                    rounding=ROUND_HALF_UP)
        net_total += value
        rows.append([cell(index), cell(name), cell(f'{quantity:f}'.replace('.', ',')),
                     cell(unit), cell(format_money_plain(price)), cell(format_money_plain(value))])
    rows.append([cell('Razem', 5), cell(format_money_plain(net_total))])
    table = make_table(rows)
    gross_total = net_total + vat_total
    fields = {'invoice_number': invoice_number, 'issue_date': format_date(rng, day),
              'seller_name': seller, 'seller_nip': format_nip(rng, make_nip(rng)),
              'seller_address': seller_address, 'buyer_name': buyer,
              'net_total': format_amount(rng, net_total),
              'vat_amount': format_amount(rng, vat_total),
              'gross_total': format_amount(rng, gross_total),
              'currency': 'PLN'}
    blocks = [{'kind': 'heading', 'level': 1, 'text': f'FAKTURA VAT {invoice_number}'},
              {'kind': 'kv', 'label': 'Data wystawienia', 'value': fields['issue_date']},
              {'kind': 'kv', 'label': 'Miejsce wystawienia', 'value': rng.choice(_CITIES)}]
    if rng.random() < 0.8:
        due = day + timedelta(days=rng.choice((7, 14, 30)))
        fields['due_date'] = format_date(rng, due)
        blocks.append({'kind': 'kv', 'label': 'Termin płatności', 'value': fields['due_date']})
    blocks += [{'kind': 'kv', 'label': 'Sprzedawca', 'value': seller},
               {'kind': 'kv', 'label': 'Adres sprzedawcy', 'value': seller_address},
               {'kind': 'kv', 'label': 'NIP sprzedawcy', 'value': fields['seller_nip']},
               {'kind': 'kv', 'label': 'Nabywca', 'value': buyer}]
    if rng.random() < 0.7:
        fields['buyer_nip'] = format_nip(rng, make_nip(rng))
        blocks.append({'kind': 'kv', 'label': 'NIP nabywcy', 'value': fields['buyer_nip']})
    fields['buyer_address'] = buyer_address
    blocks += [{'kind': 'kv', 'label': 'Adres nabywcy', 'value': buyer_address},
               {'kind': 'heading', 'level': 2, 'text': 'Pozycje faktury'},
               {'kind': 'table', 'index': 0},
               {'kind': 'kv', 'label': 'Suma netto', 'value': fields['net_total']},
               {'kind': 'kv', 'label': 'Kwota VAT', 'value': fields['vat_amount']},
               {'kind': 'kv', 'label': 'Wartość brutto',
                'value': fields['gross_total']},
               {'kind': 'paragraph', 'text': f'Płatność przelewem w walucie {fields["currency"]} '
                                             f'na konto {make_account(rng)} '
                                             f'lub gotówką w kasie sprzedawcy.'},
               {'kind': 'signatures', 'left': ['................................',
                                               'Podpis wystawcy'],
                              'right': ['................................',
                                        'Podpis odbiorcy']}]
    return verify_document({'doc_type': 'faktura', 'blocks': blocks,
                            'fields': fields, 'tables': [table]})


def build_umowa(rng):
    day = random_date(rng)
    party_1, party_2 = company(rng), (company(rng) if rng.random() < 0.6 else person(rng))
    address_1, address_2 = street_address(rng), street_address(rng)
    subject_kind = rng.choice(_SUBJECTS_UMOWA)
    subject = f'{subject_kind.capitalize()} na rzecz {company(rng)} w okresie od ' \
              f'{format_date(rng, day)} do {format_date(rng, day + timedelta(days=rng.randint(30, 300)))}'
    contract_number = f'UM/{day.year}/{day.month:02d}/{rng.randint(1, 200):03d}'
    value = Decimal(rng.randint(50000, 5000000)) / Decimal(100)
    fields = {'contract_number': contract_number, 'contract_date': format_date(rng, day),
              'party_1': party_1, 'party_1_address': address_1,
              'party_2': party_2, 'party_2_address': address_2,
              'subject': subject, 'value': format_amount(rng, value), 'currency': 'PLN'}
    section = 0

    def next_section():
        nonlocal section
        section += 1
        return {'kind': 'heading', 'level': 2, 'text': f'§ {section}'}

    subject_sentence = f'Przedmiotem umowy jest {subject}'
    if not subject_sentence.endswith('.'):
        subject_sentence += '.'
    blocks = [{'kind': 'heading', 'level': 1,
               'text': f'UMOWA O {subject_kind.upper()} NR {contract_number}'},
              {'kind': 'paragraph',
               'text': f'Zawarta w dniu {fields["contract_date"]} pomiędzy '
                       f'{party_1}, {address_1}, zwanym dalej Wykonawcą, a '
                       f'{party_2}, {address_2}, zwanym dalej Zamawiającym.'},
              next_section(),
              {'kind': 'paragraph', 'text': subject_sentence},
              next_section(),
              {'kind': 'list', 'ordered': False,
               'items': rng.sample(_OBLIGATIONS, rng.randint(3, 5))}]
    tables = []
    if rng.random() < 0.5:
        rows = [[cell('Etap'), cell('Termin'), cell('Kwota')]]
        remaining = value
        for index in range(1, rng.randint(2, 3) + 1):
            part = (value / Decimal(3)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP) \
                if index < 3 else remaining
            remaining -= part
            rows.append([cell(f'Etap {index}'), cell(format_date(rng, day + timedelta(days=30 * index))),
                         cell(format_money_plain(part))])
        rows.append([cell('Razem', 2), cell(format_money_plain(value))])
        blocks.append(next_section())
        blocks.append({'kind': 'table', 'index': 0})
        tables.append(make_table(rows))
    blocks += [next_section(),
               {'kind': 'paragraph',
                'text': f'Wartość umowy wynosi {fields["value"]} brutto '
                        f'i jest płatna w walucie {fields["currency"]}.'},
               {'kind': 'signatures', 'left': ['................................',
                                               f'Wykonawca: {party_1}'],
                              'right': ['................................',
                                        f'Zamawiający: {party_2}']}]
    return verify_document({'doc_type': 'umowa', 'blocks': blocks,
                            'fields': fields, 'tables': tables})


def build_pismo_urzedowe(rng):
    day = random_date(rng)
    issuer = institution(rng)
    addressee = person(rng) if rng.random() < 0.6 else company(rng)
    subject = f'W sprawie {rng.choice(_SUBJECTS_PISMO)}'
    reference = f'{rng.choice(("WI", "ZP", "AB", "GK"))}-{rng.choice(("AA", "ZZ", "GK"))}.' \
                f'{rng.randint(1000, 9999)}.{rng.randint(1, 9)}.{day.year}/{rng.randint(1, 999)}'
    fields = {'reference_number': reference, 'issue_date': format_date(rng, day),
              'issuer': issuer, 'addressee': addressee,
              'addressee_address': street_address(rng), 'subject': subject}
    body = [line.format(date=format_date(rng, day - timedelta(days=rng.randint(5, 60))))
            for line in rng.sample(_PISMO_BODY, rng.randint(2, 4))]
    blocks = [{'kind': 'heading', 'level': 1, 'text': issuer},
              {'kind': 'kv', 'label': 'Adresat', 'value': addressee},
              {'kind': 'kv', 'label': 'Adres', 'value': fields['addressee_address']},
              {'kind': 'kv', 'label': 'Sygnatura', 'value': reference},
              {'kind': 'kv', 'label': 'Data', 'value': fields['issue_date']},
              {'kind': 'kv', 'label': 'Dotyczy', 'value': subject},
              {'kind': 'paragraph', 'text': body[0]}]
    if rng.random() < 0.6:
        blocks.append({'kind': 'paragraph', 'text': body[1] if len(body) > 1 else body[0]})
    if rng.random() < 0.4:
        rows = [[cell('Lp.'), cell('Załącznik'), cell('Stron')],
                [cell(1), cell('Wykaz dokumentów'), cell(rng.randint(1, 6))],
                [cell(2), cell('Dowód uiszczenia opłaty'), cell(1)]]
        blocks.append({'kind': 'table', 'index': 0})
        tables = [make_table(rows)]
    else:
        tables = []
    blocks += [{'kind': 'heading', 'level': 2, 'text': 'Pouczenie'},
               {'kind': 'list', 'ordered': False, 'items': rng.sample(_POU, rng.randint(1, 2))},
               {'kind': 'signatures', 'left': ['................................',
                                               'Imię i nazwisko pracownika'],
                              'right': ['................................',
                                        'Podpis kierownika']}]
    return verify_document({'doc_type': 'pismo_urzedowe', 'blocks': blocks,
                            'fields': fields, 'tables': tables})


def build_formularz(rng):
    day = random_date(rng)
    code, name = rng.choice(_FORMS)
    applicant = person(rng) if rng.random() < 0.7 else company(rng)
    fields = {'form_code': code, 'form_name': name,
              'submission_date': format_date(rng, day),
              'applicant_name': applicant, 'applicant_pesel': make_pesel(rng),
              'applicant_address': street_address(rng)}
    rows = [[cell('Rubryka'), cell('Wartość')],
            [cell('Imię i nazwisko *'), cell(fields['applicant_name'])],
            [cell('PESEL *'), cell(fields['applicant_pesel'])],
            [cell('Adres zamieszkania *'), cell(fields['applicant_address'])],
            [cell('Data złożenia *'), cell(fields['submission_date'])]]
    blocks = [{'kind': 'heading', 'level': 1, 'text': f'{name} ({code})'},
              {'kind': 'kv', 'label': 'Symbol formularza', 'value': code},
              {'kind': 'paragraph', 'text': 'Pola oznaczone gwiazdką są obowiązkowe. '
                                            'Dane wypełnia się drukowanymi literami.'},
              {'kind': 'table', 'index': 0},
              {'kind': 'list', 'ordered': False,
               'items': ['[ ] Wyrażam zgodę na kontakt telefoniczny.',
                         '[ ] Wyrażam zgodę na otrzymywanie korespondencji elektronicznej.']},
              {'kind': 'kv', 'label': 'Miejscowość', 'value': rng.choice(_CITIES)},
              {'kind': 'signatures', 'left': ['................................',
                                              'Podpis wnioskodawcy'],
                             'right': ['................................',
                                       'Przyjęto: pieczęć organu']}]
    return verify_document({'doc_type': 'formularz', 'blocks': blocks,
                            'fields': fields, 'tables': [make_table(rows)]})


BUILDERS = {'faktura': build_faktura, 'umowa': build_umowa,
            'pismo_urzedowe': build_pismo_urzedowe, 'formularz': build_formularz}


def build_document(rng, doc_type):
    """One verified document model of the given type."""
    if doc_type not in BUILDERS:
        raise ValueError(f'Unknown document type: {doc_type}')
    return BUILDERS[doc_type](rng)
