"""
Feature Extractor — Extracts tabular, lexical, and structural features for query-product pairs.
Optimized for high-speed execution to handle millions of test pairs in minutes.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz

from project.utils.text_cleaner import clean_index_text, clean_brand, clean_category
from project.utils.logging_utils import setup_logger

logger = setup_logger("feature_extractor")


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
            
            # 1. Word counts & overlap
            q_words = q_clean.split()
            t_words = title.split()
            q_len = len(q_words)
            t_len = len(t_words)
            
            common_words = set(q_words) & set(t_words)
            common_count = len(common_words)
            common_ratio = common_count / max(1, q_len)
            
            # 2. RapidFuzz Lexical Similarities (Using only fast fuzz.ratio)
            ratio = fuzz.ratio(q_clean, title)
            
            # 3. Brand Match
            brand_in_query = False
            brand_match = 0.5  # Neutral default
            if brand:
                if brand in q_clean:
                    brand_in_query = True
                    brand_match = 1.0
                else:
                    for qw in q_words:
                        if qw == brand:
                            brand_in_query = True
                            brand_match = 1.0
                            break
            
            # 4. Category Overlap
            cat_words = set(category.replace(">", " ").split())
            cat_overlap_count = len(set(q_words) & cat_words)
            cat_overlap_ratio = cat_overlap_count / max(1, q_len)
            
            # 5. Query position matching
            starts_with = 1.0 if (q_words and t_words and q_words[0] == t_words[0]) else 0.0
            
            features.append({
                "q_len": q_len,
                "t_len": t_len,
                "common_count": common_count,
                "common_ratio": common_ratio,
                "fuzz_ratio": ratio / 100.0,
                "brand_match": brand_match,
                "cat_overlap_count": cat_overlap_count,
                "cat_overlap_ratio": cat_overlap_ratio,
                "starts_with": starts_with,
            })
            
        return pd.DataFrame(features)
