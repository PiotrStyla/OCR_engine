"""Testy korpusu PL: generatory domenowe i filtrowanie zdań wiki (bez sieci)."""

import random

from training.corpus_pl import (
    DOMAIN_GENERATORS,
    INVOICE,
    LEGAL,
    MEDICAL,
    SENTENCES,
    random_address,
    random_amount,
    random_date,
    random_person,
)
from training.wiki_corpus import (
    clean_text,
    has_diacritics,
    is_good_sentence,
    load_sentences,
    save_sentences,
    split_sentences,
)


class TestCorpusDomains:
    def test_corpora_nonempty(self):
        assert len(INVOICE) >= 10 and len(LEGAL) >= 10 and len(MEDICAL) >= 5

    def test_domains_have_diacritics(self):
        # przynajmniej część przykładów domenowych powinna mieć diakrytyki
        for corpus in (INVOICE, LEGAL, MEDICAL):
            assert any(has_diacritics(s) for s in corpus)

    def test_random_date(self):
        rng = random.Random(0)
        for _ in range(20):
            d = random_date(rng)
            assert any(str(y) in d for y in range(1990, 2026))

    def test_random_amount(self):
        rng = random.Random(0)
        for _ in range(20):
            a = random_amount(rng)
            assert a.endswith(" zł") or a.endswith(" PLN")

    def test_random_address(self):
        rng = random.Random(0)
        for _ in range(20):
            addr = random_address(rng)
            assert addr.startswith("ul.") and "-" in addr  # kod pocztowy

    def test_random_person(self):
        rng = random.Random(0)
        p = random_person(rng)
        assert len(p.split()) == 2

    def test_all_generators_return_str(self):
        rng = random.Random(1)
        for gen in DOMAIN_GENERATORS:
            s = gen(rng)
            assert isinstance(s, str) and len(s) > 3


class TestWikiFiltering:
    def test_split_sentences(self):
        text = "Ala ma kota. Kot ma Alę! Czy to prawda?"
        assert split_sentences(text) == ["Ala ma kota.", "Kot ma Alę!", "Czy to prawda?"]

    def test_clean_text_removes_refs(self):
        assert clean_text("Warszawa[1] jest  stolicą[2].") == "Warszawa jest stolicą."

    def test_is_good_sentence(self):
        assert is_good_sentence("Zażółć gęślą jaźń to znane zdanie testowe.")
        assert not is_good_sentence("Krótkie.")
        assert not is_good_sentence("x" * 200)
        assert not is_good_sentence("12345 67890 12345 67890 12345")  # mało liter
        assert not is_good_sentence("Tekst (a) (b) (c) za dużo nawiasów.")

    def test_has_diacritics(self):
        assert has_diacritics("Żółw")
        assert not has_diacritics("Zolw")

    def test_save_load_roundtrip(self, tmp_path):
        sents = ["Zdanie pierwsze.", "Żółw drugi."]
        p = tmp_path / "wiki.txt"
        save_sentences(sents, p)
        assert load_sentences(p) == sents

    def test_sentences_constant(self):
        assert len(SENTENCES) >= 40
