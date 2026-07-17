"""
Product Analyzer — Structural analysis of products.
Detects: Category Hierarchy, Brand, Attribute count, Missing fields, Normalized product text, Attribute statistics.
"""
from typing import Any

from project.utils.text_cleaner import clean_category, clean_brand, clean_gender, clean_age_group
from project.utils.attribute_parser import parse_attributes
from project.features.product_normalizer import normalize_product

class ProductAnalyzer:
    """Analyzes product features and meta-information."""

    def __init__(self):
        pass

    def analyze(self, row: dict[str, Any] | pd.Series if 'pd' in globals() else Any) -> dict[str, Any]:
        """
        Analyze a single product row (containing title, category, brand, gender, age_group, attributes).
        Returns a dict of structured product properties and statistics.
        """
        # Convert row/Series to dictionary
        data = row.to_dict() if hasattr(row, "to_dict") else dict(row)

        title = data.get("title")
        category = data.get("category")
        brand = data.get("brand")
        gender = data.get("gender")
        age_group = data.get("age_group")
        attributes_str = data.get("attributes")

        # 1. Category Hierarchy Split
        category_clean = clean_category(category)
        category_levels = [c.strip() for c in category_clean.split(">") if c.strip()] if category_clean else []
        category_depth = len(category_levels)

        # 2. Brand normalization
        brand_clean = clean_brand(brand)
        has_brand = bool(brand_clean)

        # 3. Attribute parsing & count
        parsed_attrs = parse_attributes(attributes_str)
        attribute_count = len(parsed_attrs)

        # 4. Missing fields detection
        missing_fields = []
        if not title or not isinstance(title, str) or not title.strip():
            missing_fields.append("title")
        if not category_clean:
            missing_fields.append("category")
        if not brand_clean:
            missing_fields.append("brand")
        if not gender or clean_gender(gender) == "unknown":
            missing_fields.append("gender")
        if not age_group or clean_age_group(age_group) == "unknown":
            missing_fields.append("age_group")
        if not attribute_count:
            missing_fields.append("attributes")

        # 5. Normalized product text (clean representation)
        normalized_text = normalize_product(
            title=title,
            category=category,
            brand=brand,
            gender=gender,
            age_group=age_group,
            attributes=attributes_str
        )

        # 6. Extract levels safely
        level_1 = category_levels[0] if category_depth > 0 else None
        level_2 = category_levels[1] if category_depth > 1 else None
        level_3 = category_levels[2] if category_depth > 2 else None

        return {
            "product_category_depth": category_depth,
            "product_category_levels": category_levels,
            "product_category_level_1": level_1,
            "product_category_level_2": level_2,
            "product_category_level_3": level_3,
            "product_has_brand": has_brand,
            "product_brand_clean": brand_clean,
            "product_attribute_count": attribute_count,
            "product_parsed_attributes": parsed_attrs,
            "product_missing_fields": missing_fields,
            "product_missing_count": len(missing_fields),
            "product_normalized_text": normalized_text,
        }
