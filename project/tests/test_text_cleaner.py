"""Tests for text_cleaner module."""
import pytest
from project.utils.text_cleaner import (
    clean_model_text, clean_index_text, turkish_lowercase,
    unicode_normalize, normalize_whitespace, remove_html,
    clean_brand, clean_gender, clean_age_group,
)


class TestTurkishLowercase:
    def test_basic(self):
        assert turkish_lowercase("ABC") == "abc"

    def test_turkish_i(self):
        assert turkish_lowercase("İSTANBUL") == "istanbul"
        assert turkish_lowercase("I") == "ı"

    def test_turkish_special(self):
        result = turkish_lowercase("ÇĞŞÜÖİ")
        assert result == "çğşüöi"

    def test_already_lower(self):
        assert turkish_lowercase("test") == "test"


class TestCleanModelText:
    def test_basic(self):
        result = clean_model_text("  İPHONE 15 PRO KILIF  ")
        assert result == "iphone 15 pro kılıf"

    def test_preserves_turkish(self):
        result = clean_model_text("Türkçe Karakterler ÇŞĞÜÖİ")
        assert "ç" in result
        assert "ş" in result
        assert "ğ" in result

    def test_multiple_spaces(self):
        result = clean_model_text("  siyah   kadın   bot  ")
        assert result == "siyah kadın bot"

    def test_html_removal(self):
        result = clean_model_text("test <br/> product <b>bold</b>")
        assert "<" not in result
        assert ">" not in result

    def test_none_input(self):
        assert clean_model_text(None) == ""
        assert clean_model_text("") == ""

    def test_preserves_numbers(self):
        result = clean_model_text("iPhone 15 Pro Max 256GB")
        assert "15" in result
        assert "256" in result


class TestCleanIndexText:
    def test_ascii_conversion(self):
        result = clean_index_text("İPHONE 15 PRO KILIF")
        assert result == "iphone 15 pro kilif"

    def test_turkish_to_ascii(self):
        result = clean_index_text("çğışöü")
        assert result == "cgisou"

    def test_none(self):
        assert clean_index_text(None) == ""


class TestCleanBrand:
    def test_basic(self):
        assert clean_brand("Nike") == "nike"
        assert clean_brand("NIKE") == "nıke"
        assert clean_brand("  Nike  ") == "nike"

    def test_none(self):
        assert clean_brand(None) == ""
        assert clean_brand("") == ""


class TestCleanGender:
    def test_valid(self):
        assert clean_gender("Erkek") == "erkek"
        assert clean_gender("KADIN") == "kadın"
        assert clean_gender("Unisex") == "unisex"

    def test_unknown(self):
        assert clean_gender("unknown") == "unknown"
        assert clean_gender(None) == "unknown"
        assert clean_gender("") == "unknown"

    def test_invalid(self):
        assert clean_gender("ev & mobilya/ev/tablo") == "unknown"


class TestCleanAgeGroup:
    def test_valid(self):
        assert clean_age_group("Yetişkin") == "yetişkin"
        assert clean_age_group("Çocuk") == "çocuk"

    def test_unknown(self):
        assert clean_age_group("unknown") == "unknown"
        assert clean_age_group(None) == "unknown"
