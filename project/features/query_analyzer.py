"""
Query Analyzer — Analyzes queries for features.
Detects: Length, Brand, Color, Material, Numeric, Gender, Intent, Token Stats.
"""
import re
from typing import Any

from project.utils.text_cleaner import clean_index_text, clean_gender, clean_age_group
from project.utils.logging_utils import setup_logger

logger = setup_logger("query_analyzer")

# Turkish color list (ASCII normalized)
COLORS_ASCII = {
    "siyah", "beyaz", "kirmizi", "mavi", "yesil", "sari", "pembe",
    "mor", "turuncu", "gri", "kahverengi", "lacivert", "bej",
    "krem", "bordo", "haki", "ekru", "antrasit", "altin", "gumus",
    "lila", "turkuaz", "fuşya", "fusya", "leopar",
}

# Material list (ASCII normalized)
MATERIALS_ASCII = {
    "deri", "pamuk", "pamuklu", "polyester", "ipek", "koton", "keten",
    "yun", "yün", "yünlü", "tekstil", "seramik", "metal", "plastik",
    "cam", "ahsap", "ahşap", "celik", "çelik", "gumus", "gümüş",
    "altin", "altın", "bakir", "bakır", "porselen", "kadife", "kot",
    "jean", "saten", "tül", "tul", "dantel", "parlak", "mat",
}

# Common Turkish stop words
STOP_WORDS = {
    "ve", "veya", "ile", "için", "icin", "da", "de", "ki", "en", "daha",
    "bir", "cok", "çok", "her", "bazı", "bazi", "gibi", "kadar", "olan",
    "olarak", "ise", "ise", "mi", "mı", "mu", "mü", "bu", "şu", "o",
}


class QueryAnalyzer:
    """Analyzes search queries to extract semantic and lexical features."""

    def __init__(self, known_brands: set[str] | None = None):
        """
        Args:
            known_brands: Optional set of clean brand names (ASCII normalized) for brand detection.
        """
        self.known_brands = known_brands if known_brands else set()

    def set_known_brands(self, brands: set[str]):
        """Dynamically update known brands from data loading."""
        self.known_brands = {clean_index_text(b) for b in brands if b and isinstance(b, str)}
        # Remove empty string if present
        self.known_brands.discard("")
        logger.info(f"QueryAnalyzer loaded {len(self.known_brands):,} unique brands.")

    def analyze(self, query: str | None) -> dict[str, Any]:
        """
        Analyze a raw query.
        Returns a dict of features.
        """
        if not query or not isinstance(query, str):
            return {
                "query_char_len": 0,
                "query_token_len": 0,
                "query_has_brand": False,
                "query_detected_brand": None,
                "query_has_color": False,
                "query_detected_color": None,
                "query_has_material": False,
                "query_detected_material": None,
                "query_has_numeric": False,
                "query_detected_numeric": None,
                "query_has_gender": False,
                "query_detected_gender": "unknown",
                "query_intent": "other",
                "query_stopword_ratio": 0.0,
                "query_unique_token_count": 0,
            }

        # ASCII normalized version for matching
        q_clean = clean_index_text(query)
        tokens = q_clean.split()
        char_len = len(query)
        token_len = len(tokens)

        # 1. Brand Detection (find longest matching brand substring)
        detected_brand = None
        has_brand = False
        if self.known_brands:
            # Sort brands by length descending to match longest first (e.g. "newish polo" instead of "polo")
            matched_brands = []
            for brand in self.known_brands:
                # Use word boundaries or check if brand is exactly the query or a token prefix/suffix
                # To be fast and robust, check if brand is in the query index with word boundaries
                pattern = r"\b" + re.escape(brand) + r"\b"
                if re.search(pattern, q_clean):
                    matched_brands.append(brand)

            if matched_brands:
                detected_brand = max(matched_brands, key=len)
                has_brand = True

        # 2. Color Detection
        detected_colors = [c for c in COLORS_ASCII if c in tokens]
        has_color = len(detected_colors) > 0
        detected_color = detected_colors[0] if has_color else None

        # 3. Material Detection
        detected_materials = [m for m in MATERIALS_ASCII if m in tokens]
        has_material = len(detected_materials) > 0
        detected_material = detected_materials[0] if has_material else None

        # 4. Numeric Detection
        # Match standalone numbers or numbers attached to letters (e.g. "15", "s24", "42", "256gb")
        numeric_matches = re.findall(r"\b\d+\w*\b|\b\w*\d+\b", q_clean)
        has_numeric = len(numeric_matches) > 0
        detected_numeric = numeric_matches[0] if has_numeric else None

        # 5. Gender Detection
        detected_gender = "unknown"
        has_gender = False
        if "erkek" in tokens:
            detected_gender = "erkek"
            has_gender = True
        elif "kadin" in tokens or "kız" in query or "kiz" in tokens:
            detected_gender = "kadın"
            has_gender = True
        elif "unisex" in tokens:
            detected_gender = "unisex"
            has_gender = True

        # 6. Intent Detection (Rule-based simple categorization)
        intent = "other"
        # Categorize by common target categories
        if any(w in tokens for w in ["kilif", "kılıf", "sarj", "şarj", "kablo", "telefon", "kulaklik", "kulaklık"]):
            intent = "electronic_accessory"
        elif any(w in tokens for w in ["ayakkabi", "ayakkabı", "bot", "cizme", "çizme", "terlik", "spor"]):
            intent = "footwear"
        elif any(w in tokens for w in ["elbise", "tisort", "tişört", "gomlek", "gömlek", "pantolon", "ceket", "mont"]):
            intent = "apparel"
        elif any(w in tokens for w in ["canta", "çanta", "cuzdan", "cüzdan", "kemer", "saat", "gozluk", "gözlük"]):
            intent = "accessory"
        elif any(w in tokens for w in ["bardak", "kupa", "tabak", "tava", "bıçak", "bicak", "sofra", "hali", "halı"]):
            intent = "home_living"
        elif any(w in tokens for w in ["ruj", "krem", "parfum", "parfüm", "sampuan", "şampuan", "makyaj"]):
            intent = "beauty"

        # 7. Token Statistics
        stopword_count = sum(1 for t in tokens if t in STOP_WORDS)
        stopword_ratio = stopword_count / token_len if token_len > 0 else 0.0
        unique_tokens = set(tokens)
        unique_token_count = len(unique_tokens)

        return {
            "query_char_len": char_len,
            "query_token_len": token_len,
            "query_has_brand": has_brand,
            "query_detected_brand": detected_brand,
            "query_has_color": has_color,
            "query_detected_color": detected_color,
            "query_has_material": has_material,
            "query_detected_material": detected_material,
            "query_has_numeric": has_numeric,
            "query_detected_numeric": detected_numeric,
            "query_has_gender": has_gender,
            "query_detected_gender": detected_gender,
            "query_intent": intent,
            "query_stopword_ratio": stopword_ratio,
            "query_unique_token_count": unique_token_count,
        }
