"""
Pair Text Builder — Single source of truth for the cross-encoder input format.

Both training (cross_encoder_trainer) and inference (batch_builder) MUST build
their model inputs through this module, so there is no train/serve skew.

The model receives a text pair:
    text_a = cleaned query
    text_b = product text = title | marka: .. | kategori: .. | cinsiyet: .. |
             yas: .. | renk: .. | beden: .. | kapasite: .. | materyal: .. | ...

Only a whitelist of informative attributes is included; noisy placeholder
values ("parça mevcut değil", "belirtilmemiş", ...) are dropped.
"""
from project.utils.text_cleaner import (
    clean_model_text,
    clean_brand,
    clean_category,
    clean_gender,
    clean_age_group,
)
from project.utils.attribute_parser import parse_attributes

# Bump when the input format changes; stored in inference_meta.json so that a
# checkpoint is never served with a different input format than it was trained on.
PAIR_TEXT_VERSION = "v2-enriched"

# (canonical attribute key, Turkish label) — order defines output order
_ATTR_WHITELIST: list[tuple[str, str]] = [
    ("color", "renk"),
    ("size", "beden"),
    ("capacity", "kapasite"),
    ("material", "materyal"),
    ("pattern", "desen"),
    ("fabric", "kumaş"),
    ("piece_count", "adet"),
]

# Placeholder / non-informative values that must not reach the model
_NOISE_VALUES = {
    "parça mevcut değil", "parca mevcut degil", "belirtilmemiş", "belirtilmemis",
    "yok", "hayır", "hayir", "bilinmiyor", "diğer", "diger", "tüm yaş grupları",
    "tum yas gruplari", "standart", "tek ebat", "none", "nan", "-", "0",
}

_MAX_ATTR_VALUE_LEN = 40


def _clean_attr_value(value: str | None) -> str | None:
    """Return a usable attribute value or None if it is noise."""
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > _MAX_ATTR_VALUE_LEN:
        return None
    if value in _NOISE_VALUES:
        return None
    return value


def build_product_text(
    title: str | None,
    brand: str | None = None,
    category: str | None = None,
    gender: str | None = None,
    age_group: str | None = None,
    attributes: str | None = None,
) -> str:
    """Build the enriched product-side text for the cross encoder."""
    parts = [clean_model_text(title)]

    brand_clean = clean_brand(brand)
    if brand_clean:
        parts.append(f"marka: {brand_clean}")

    cat_clean = clean_category(category)
    if cat_clean:
        # Keep the last 3 levels of the category path — the leaf levels carry
        # the product-type signal; the root levels are near-constant noise.
        segments = [s.strip() for s in cat_clean.replace(">", "/").split("/") if s.strip()]
        parts.append(f"kategori: {' > '.join(segments[-3:])}")

    gender_clean = clean_gender(gender)
    if gender_clean != "unknown":
        parts.append(f"cinsiyet: {gender_clean}")

    age_clean = clean_age_group(age_group)
    if age_clean != "unknown":
        parts.append(f"yaş: {age_clean}")

    if attributes and isinstance(attributes, str):
        parsed = parse_attributes(attributes)
        for key, label in _ATTR_WHITELIST:
            value = _clean_attr_value(parsed.get(key))
            if value:
                parts.append(f"{label}: {value}")

    return " | ".join(parts)


def build_query_text(query: str | None) -> str:
    """Build the query-side text for the cross encoder."""
    return clean_model_text(query)


def build_pair_texts_from_frame(df) -> tuple[list[str], list[str]]:
    """
    Vectorized helper: build (queries, product_texts) lists from a DataFrame
    with columns: query, title, brand, category, gender, age_group, attributes.
    Missing columns are treated as empty.
    """
    def col(name):
        if name in df.columns:
            return df[name].tolist()
        return [None] * len(df)

    queries_raw = col("query")
    titles = col("title")
    brands = col("brand")
    categories = col("category")
    genders = col("gender")
    ages = col("age_group")
    attrs = col("attributes")

    queries = [build_query_text(q) for q in queries_raw]
    products = [
        build_product_text(t, b, c, g, a, at)
        for t, b, c, g, a, at in zip(titles, brands, categories, genders, ages, attrs)
    ]
    return queries, products
