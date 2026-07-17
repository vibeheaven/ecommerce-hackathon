"""
Product Normalizer — Converts products into standardized text format.
"""
from project.utils.text_cleaner import (
    clean_model_text, clean_category, clean_brand,
    clean_gender, clean_age_group,
)
from project.utils.attribute_parser import parse_attributes, attributes_to_text


def normalize_product(
    title: str | None,
    category: str | None,
    brand: str | None,
    gender: str | None,
    age_group: str | None,
    attributes: str | None,
) -> str:
    """
    Convert product fields into a single normalized text for cross encoder input.

    Format:
        title: <title>
        category: <category>
        brand: <brand>
        gender: <gender>
        age: <age_group>
        attributes: <key: value, ...>
    """
    parts = []

    t = clean_model_text(title)
    if t:
        parts.append(f"title: {t}")

    c = clean_category(category)
    if c:
        parts.append(f"category: {c}")

    b = clean_brand(brand)
    if b:
        parts.append(f"brand: {b}")

    g = clean_gender(gender)
    if g and g != "unknown":
        parts.append(f"gender: {g}")

    a = clean_age_group(age_group)
    if a and a != "unknown":
        parts.append(f"age: {a}")

    if attributes and isinstance(attributes, str):
        parsed = parse_attributes(attributes)
        attr_text = attributes_to_text(parsed)
        if attr_text:
            parts.append(f"attributes: {attr_text}")

    return " | ".join(parts)


def normalize_product_row(row) -> str:
    """Normalize a product from a DataFrame row."""
    return normalize_product(
        title=row.get("title"),
        category=row.get("category"),
        brand=row.get("brand"),
        gender=row.get("gender"),
        age_group=row.get("age_group"),
        attributes=row.get("attributes"),
    )
