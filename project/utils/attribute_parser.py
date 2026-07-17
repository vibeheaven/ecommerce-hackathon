"""
Attribute Parser — Parses attribute strings into structured dictionaries.
Extracts 9 specific fields: color, material, pattern, collection, season,
size, fabric, capacity, origin.
"""
import re
from typing import Any

from project.utils.logging_utils import setup_logger
from project.utils.text_cleaner import turkish_lowercase, normalize_whitespace

logger = setup_logger("attribute_parser")

# Known attribute key mappings (Turkish → canonical)
_KEY_ALIASES = {
    "renk": "color",
    "color": "color",
    "color detail": "color_detail",
    "materyal": "material",
    "material": "material",
    "materyal bileşeni": "material_composition",
    "desen": "pattern",
    "pattern": "pattern",
    "koleksiyon": "collection",
    "collection": "collection",
    "sezon": "season",
    "season": "season",
    "beden": "size",
    "size": "size",
    "kumaş tipi": "fabric",
    "fabric": "fabric",
    "kumaş": "fabric",
    "dokuma tipi": "fabric_type",
    "kapasite": "capacity",
    "capacity": "capacity",
    "menşei": "origin",
    "origin": "origin",
    "marka": "brand",
    "brand": "brand",
    "cinsiyet": "gender",
    "gender": "gender",
    "yaş": "age",
    "age": "age",
    "yaka tipi": "collar_type",
    "kol boyu": "sleeve_length",
    "kol tipi": "sleeve_type",
    "kalıp": "fit",
    "boy": "length",
    "ortam": "occasion",
    "siluet": "silhouette",
    "cep": "pocket",
    "sürdürülebilirlik detayı": "sustainability",
    "yıkama talimatı": "wash_instruction",
    "paket içeriği": "package_content",
    "parça sayısı": "piece_count",
    "garanti süresi": "warranty",
    "taş cinsi": "stone_type",
}

# Target fields for structured extraction
TARGET_FIELDS = [
    "color", "color_detail", "material", "material_composition",
    "pattern", "collection", "season", "size", "fabric", "fabric_type",
    "capacity", "origin",
]


def parse_attributes(attr_string: str | None) -> dict[str, str]:
    """
    Parse attribute string into a dictionary.

    Input format: "key1: value1, key2: value2, ..."
    Handles edge cases: colons in values, empty strings, duplicate keys.

    Returns: dict mapping canonical key → value
    """
    if not attr_string or not isinstance(attr_string, str):
        return {}

    result: dict[str, str] = {}
    raw_pairs: list[str] = []

    # Split by comma, but be careful with values containing commas
    # The format is "key: value, key: value"
    # Strategy: split by ", " followed by a word and colon
    parts = re.split(r",\s*(?=\w[\w\s]*:)", attr_string.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Find the first colon to split key:value
        colon_idx = part.find(":")
        if colon_idx == -1:
            continue

        key = part[:colon_idx].strip()
        value = part[colon_idx + 1:].strip()

        if not key or not value:
            continue

        # Normalize key
        key_lower = turkish_lowercase(key)
        canonical = _KEY_ALIASES.get(key_lower, key_lower)

        # Normalize value
        value = normalize_whitespace(turkish_lowercase(value))

        # Store (last wins for duplicates)
        result[canonical] = value

    return result


def extract_target_fields(attributes: dict[str, str]) -> dict[str, str | None]:
    """
    Extract the 9 target fields from parsed attributes.
    Returns dict with keys from TARGET_FIELDS, values are strings or None.
    """
    extracted: dict[str, str | None] = {}
    for field in TARGET_FIELDS:
        extracted[field] = attributes.get(field)

    # If color_detail exists but color doesn't, use color_detail
    if not extracted.get("color") and extracted.get("color_detail"):
        extracted["color"] = extracted["color_detail"]

    return extracted


def get_color(attributes: dict[str, str]) -> str | None:
    """Extract color from parsed attributes."""
    return attributes.get("color") or attributes.get("color_detail")


def get_material(attributes: dict[str, str]) -> str | None:
    """Extract material from parsed attributes."""
    return attributes.get("material")


def get_pattern(attributes: dict[str, str]) -> str | None:
    """Extract pattern from parsed attributes."""
    return attributes.get("pattern")


def get_season(attributes: dict[str, str]) -> str | None:
    """Extract season from parsed attributes."""
    return attributes.get("season")


def get_origin(attributes: dict[str, str]) -> str | None:
    """Extract origin from parsed attributes."""
    return attributes.get("origin")


def attributes_to_text(attributes: dict[str, str]) -> str:
    """Convert parsed attributes dict back to clean text representation."""
    if not attributes:
        return ""
    parts = [f"{k}: {v}" for k, v in sorted(attributes.items()) if v]
    return ", ".join(parts)
