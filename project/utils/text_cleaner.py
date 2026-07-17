"""
Text Cleaner — Unicode normalization, Turkish lowercase, HTML cleanup.
Produces two text variants:
  - model_text: preserves Turkish characters (for cross encoder input)
  - index_text: ASCII-normalized (for BM25, fuzzy matching, Levenshtein)
"""
import re
import unicodedata
from functools import lru_cache

from project.utils.logging_utils import setup_logger

logger = setup_logger("text_cleaner")

# Turkish lowercase mapping (İ→i, I→ı is Turkish-specific)
_TURKISH_LOWER_MAP = str.maketrans(
    "ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ",
    "abcçdefgğhıijklmnoöpqrsştuüvwxyz",
)

# ASCII transliteration for index_text
_TURKISH_ASCII_MAP = str.maketrans(
    "çğıöşüÇĞİÖŞÜ",
    "cgiosuCGIOSU",
)

# HTML tag pattern
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Multiple whitespace
_MULTI_SPACE_RE = re.compile(r"\s+")
# Non-printable characters (excluding standard whitespace)
_NON_PRINTABLE_RE = re.compile(r"[^\S\n\r\t ]+|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def unicode_normalize(text: str) -> str:
    """Apply NFC unicode normalization."""
    return unicodedata.normalize("NFC", text)


def turkish_lowercase(text: str) -> str:
    """Turkish-aware lowercase conversion."""
    return text.translate(_TURKISH_LOWER_MAP)


def ascii_normalize(text: str) -> str:
    """Convert Turkish characters to ASCII equivalents."""
    return text.translate(_TURKISH_ASCII_MAP)


def remove_html(text: str) -> str:
    """Remove HTML tags."""
    return _HTML_TAG_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space and strip."""
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def remove_non_printable(text: str) -> str:
    """Remove non-printable characters."""
    return _NON_PRINTABLE_RE.sub("", text)


def clean_model_text(text: str | None) -> str:
    """
    Clean text for model input (cross encoder).
    Preserves Turkish characters.
    """
    if not text or not isinstance(text, str):
        return ""
    text = unicode_normalize(text)
    text = remove_html(text)
    text = remove_non_printable(text)
    text = turkish_lowercase(text)
    text = normalize_whitespace(text)
    return text


def clean_index_text(text: str | None) -> str:
    """
    Clean text for indexing (BM25, fuzzy matching, Levenshtein).
    Converts Turkish characters to ASCII.
    """
    if not text or not isinstance(text, str):
        return ""
    text = unicode_normalize(text)
    text = remove_html(text)
    text = remove_non_printable(text)
    text = turkish_lowercase(text)
    text = ascii_normalize(text)
    text = normalize_whitespace(text)
    return text


def clean_query_model(query: str | None) -> str:
    """Clean query for model input."""
    return clean_model_text(query)


def clean_query_index(query: str | None) -> str:
    """Clean query for indexing."""
    return clean_index_text(query)


def clean_title_model(title: str | None) -> str:
    """Clean product title for model input."""
    return clean_model_text(title)


def clean_title_index(title: str | None) -> str:
    """Clean product title for indexing."""
    return clean_index_text(title)


def clean_category(category: str | None) -> str:
    """Clean and normalize category string."""
    if not category or not isinstance(category, str):
        return ""
    category = unicode_normalize(category)
    category = turkish_lowercase(category)
    category = normalize_whitespace(category)
    return category


def clean_brand(brand: str | None) -> str:
    """Clean and normalize brand string."""
    if not brand or not isinstance(brand, str):
        return ""
    brand = unicode_normalize(brand)
    brand = turkish_lowercase(brand)
    brand = normalize_whitespace(brand)
    return brand


def clean_gender(gender: str | None) -> str:
    """Clean and normalize gender string."""
    if not gender or not isinstance(gender, str):
        return "unknown"
    gender = turkish_lowercase(gender.strip())
    if gender in ("erkek", "kadın", "unisex"):
        return gender
    return "unknown"


def clean_age_group(age_group: str | None) -> str:
    """Clean and normalize age group string."""
    if not age_group or not isinstance(age_group, str):
        return "unknown"
    age_group = turkish_lowercase(age_group.strip())
    valid = {"yetişkin", "çocuk", "genç", "bebek", "bebek & çocuk"}
    if age_group in valid:
        return age_group
    return "unknown"
