"""Tests for attribute_parser module."""
import pytest
from project.utils.attribute_parser import (
    parse_attributes, extract_target_fields, get_color,
    get_material, get_pattern, get_season, attributes_to_text,
)


class TestParseAttributes:
    def test_basic_parsing(self):
        attr = "renk: siyah, materyal: pamuklu, desen: düz"
        result = parse_attributes(attr)
        assert result["color"] == "siyah"
        assert result["material"] == "pamuklu"
        assert result["pattern"] == "düz"

    def test_real_data(self):
        attr = 'materyal: tekstil, deri kalitesi: parça mevcut değil, renk: gri, ortam: casual/günlük, desen: düz, kumaş tipi: dokuma, koleksiyon: basic, sezon: tüm sezonlar, color detail: siyah - beyaz'
        result = parse_attributes(attr)
        assert result.get("material") == "tekstil"
        assert result.get("color") == "gri"
        assert result.get("pattern") == "düz"
        assert result.get("fabric") == "dokuma"
        assert result.get("collection") == "basic"
        assert result.get("season") == "tüm sezonlar"
        assert result.get("color_detail") == "siyah - beyaz"

    def test_empty_string(self):
        assert parse_attributes("") == {}
        assert parse_attributes(None) == {}

    def test_duplicate_keys(self):
        # Last value wins
        attr = "renk: siyah, renk: beyaz"
        result = parse_attributes(attr)
        assert result["color"] == "beyaz"

    def test_colon_in_value(self):
        attr = "bakım talimatları (gıda temas): ürünün kullanım ömrünü korumak için ürüne özel bakım talimatlarını takip ediniz."
        result = parse_attributes(attr)
        # Should not crash
        assert isinstance(result, dict)


class TestExtractTargetFields:
    def test_extraction(self):
        attrs = {"color": "siyah", "material": "pamuklu", "unknown_key": "value"}
        result = extract_target_fields(attrs)
        assert result["color"] == "siyah"
        assert result["material"] == "pamuklu"
        assert result.get("pattern") is None

    def test_color_detail_fallback(self):
        attrs = {"color_detail": "siyah - beyaz"}
        result = extract_target_fields(attrs)
        assert result["color"] == "siyah - beyaz"


class TestGetters:
    def test_get_color(self):
        assert get_color({"color": "siyah"}) == "siyah"
        assert get_color({"color_detail": "mavi"}) == "mavi"
        assert get_color({}) is None

    def test_get_material(self):
        assert get_material({"material": "pamuklu"}) == "pamuklu"
        assert get_material({}) is None


class TestAttributesToText:
    def test_basic(self):
        attrs = {"color": "siyah", "material": "pamuklu"}
        result = attributes_to_text(attrs)
        assert "color: siyah" in result
        assert "material: pamuklu" in result

    def test_empty(self):
        assert attributes_to_text({}) == ""
