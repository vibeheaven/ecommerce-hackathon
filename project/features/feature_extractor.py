"""
Feature Extractor — Extracts advanced tabular, lexical, and structural features for query-product pairs.
Optimized for high-speed execution to handle millions of test pairs in minutes.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz

from project.utils.text_cleaner import clean_index_text, clean_brand, clean_category, clean_gender
from project.utils.attribute_parser import parse_attributes
from project.utils.logging_utils import setup_logger

logger = setup_logger("feature_extractor")

# Turkish colors in ASCII form for query conflict detection
_COLORS_ASCII = {
    "siyah", "beyaz", "kirmizi", "mavi", "yesil", "sari", "pembe",
    "mor", "turuncu", "gri", "kahverengi", "lacivert", "bej",
    "krem", "bordo", "haki", "ekru", "antrasit", "altin", "gumus",
    "lila", "turkuaz", "fusya",
}

_GENDER_MAP = {
    "erkek": "erkek", "bay": "erkek",
    "kadin": "kadin", "bayan": "kadin", "kiz": "kadin",
}


class FeatureExtractor:
    """Extracts lexical, semantic, and structural features from query-product pairs."""

    def __init__(self, items_df: pd.DataFrame):
        logger.info("Initializing FeatureExtractor...")
        self.items_df = items_df.copy()
        
        # Clean titles, brands, and categories for matching
        self.items_df["clean_title"] = self.items_df["title"].fillna("").apply(clean_index_text)
        self.items_df["clean_brand"] = self.items_df["brand"].fillna("").apply(clean_brand)
        self.items_df["clean_category"] = self.items_df["category"].fillna("").apply(clean_category)
        
        # Build lookup dicts
        self.item_lookup = self.items_df.set_index("item_id").to_dict(orient="index")
        
        # Pre-parse attributes for fast color conflict checks
        logger.info("Pre-parsing colors and genders for fast attribute matching...")
        for item_id, item in self.item_lookup.items():
            # Extract color from attributes
            attrs = parse_attributes(item.get("attributes"))
            color_val = attrs.get("color")
            item["parsed_color"] = clean_index_text(color_val) if color_val else ""
            
            # Extract gender
            item["parsed_gender"] = clean_gender(item.get("gender"))

    def extract_features(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for each row in pairs_df (must contain 'query' and 'item_id').
        Returns a DataFrame of numeric features.
        """
        logger.info(f"Extracting features for {len(pairs_df):,} pairs...")
        
        features = []
        queries = pairs_df["query"].fillna("").tolist()
        item_ids = pairs_df["item_id"].tolist()
        
        for q_raw, item_id in tqdm(zip(queries, item_ids), total=len(pairs_df), desc="Extracting Features"):
            q_clean = clean_index_text(q_raw)
            
            # Get item metadata
            item = self.item_lookup.get(item_id, {})
            title = item.get("clean_title", "")
            brand = item.get("clean_brand", "")
            category = item.get("clean_category", "")
            item_color = item.get("parsed_color", "")
            item_gender = item.get("parsed_gender", "unknown")
            
            # 1. Word counts & overlap
            q_words = q_clean.split()
            t_words = title.split()
            q_len = len(q_words)
            t_len = len(t_words)
            
            q_set = set(q_words)
            t_set = set(t_words)
            common_words = q_set & t_set
            common_count = len(common_words)
            common_ratio = common_count / max(1, q_len)
            jaccard_sim = common_count / max(1, len(q_set | t_set))
            
            # 2. RapidFuzz Lexical Similarity
            ratio = fuzz.ratio(q_clean, title)
            
            # 3. Substring match
            query_in_title = 1.0 if q_clean and q_clean in title else 0.0
            
            # 4. Brand Match & Conflict
            brand_in_query = None
            brand_match = 0.5  # Neutral default
            brand_conflict = 0.0
            
            if brand:
                if brand in q_clean:
                    brand_in_query = brand
                    brand_match = 1.0
                else:
                    for qw in q_words:
                        if qw == brand:
                            brand_in_query = brand
                            brand_match = 1.0
                            break
            
            # 5. Color Match & Conflict
            query_colors = _COLORS_ASCII & q_set
            item_colors = _COLORS_ASCII & set(item_color.split())
            if not item_colors and title:
                item_colors = _COLORS_ASCII & t_set
                
            color_match = 0.5
            color_conflict = 0.0
            if query_colors:
                if item_colors:
                    if query_colors & item_colors:
                        color_match = 1.0
                    else:
                        color_conflict = 1.0
                        color_match = 0.0
                else:
                    # Query specifies a color, but product doesn't have it
                    color_match = 0.0
            
            # 6. Gender Match & Conflict
            query_genders = {g_canonical for tok, g_canonical in _GENDER_MAP.items() if tok in q_set}
            gender_match = 0.5
            gender_conflict = 0.0
            
            if query_genders:
                if item_gender != "unknown" and item_gender != "unisex":
                    if item_gender in query_genders:
                        gender_match = 1.0
                    else:
                        gender_conflict = 1.0
                        gender_match = 0.0
            
            # 7. Category Overlap
            cat_words = set(category.replace(">", " ").split())
            cat_overlap_count = len(q_set & cat_words)
            cat_overlap_ratio = cat_overlap_count / max(1, q_len)
            
            # 8. Query position matching
            starts_with = 1.0 if (q_words and t_words and q_words[0] == t_words[0]) else 0.0
            
            features.append({
                "q_len": q_len,
                "t_len": t_len,
                "common_count": common_count,
                "common_ratio": common_ratio,
                "jaccard_sim": jaccard_sim,
                "fuzz_ratio": ratio / 100.0,
                "query_in_title": query_in_title,
                "brand_match": brand_match,
                "brand_conflict": brand_conflict,
                "color_match": color_match,
                "color_conflict": color_conflict,
                "gender_match": gender_match,
                "gender_conflict": gender_conflict,
                "cat_overlap_count": cat_overlap_count,
                "cat_overlap_ratio": cat_overlap_ratio,
                "starts_with": starts_with,
            })
            
        return pd.DataFrame(features)
