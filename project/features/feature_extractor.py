"""
Feature Extractor — Extracts tabular, lexical, and structural features for query-product pairs.
These features are fused with Cross-Encoder scores to train a LightGBM Meta-Classifier.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi

from project.utils.text_cleaner import clean_index_text, clean_brand, clean_category
from project.utils.attribute_parser import parse_attributes
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
        
        # Initialize BM25 index on all item titles
        logger.info("Building BM25 index on item titles...")
        corpus = [row["clean_title"].split() for row in self.item_lookup.values()]
        self.bm25 = BM25Okapi(corpus)
        
        # Map item_id to its index in BM25 corpus
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.items_df["item_id"])}

    def extract_features(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for each row in pairs_df (must contain 'query' and 'item_id').
        Returns a DataFrame of numeric features.
        """
        logger.info(f"Extracting features for {len(pairs_df):,} pairs...")
        
        features = []
        for row in tqdm(pairs_df.itertuples(index=False), total=len(pairs_df), desc="Extracting Features"):
            q_raw = row.query if isinstance(row.query, str) else ""
            q_clean = clean_index_text(q_raw)
            item_id = row.item_id
            
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
            
            # 2. RapidFuzz Lexical Similarities
            ratio = fuzz.ratio(q_clean, title)
            partial_ratio = fuzz.partial_ratio(q_clean, title)
            token_sort_ratio = fuzz.token_sort_ratio(q_clean, title)
            
            # 3. Brand Match
            # If query has a brand, check if it matches product brand
            brand_in_query = False
            brand_match = 0.5  # Neutral default
            if brand:
                if brand in q_clean:
                    brand_in_query = True
                    brand_match = 1.0
                else:
                    # Check if any query word matches product brand
                    for qw in q_words:
                        if qw == brand:
                            brand_in_query = True
                            brand_match = 1.0
                            break
            
            # If we detect a brand in the query, but the product brand is different, it's a conflict
            if not brand_in_query:
                # Check if query contains any other known brands (simple heuristic)
                # If product has no brand, or different brand
                pass
            
            # 4. Category Overlap
            # Check if query terms appear in the category hierarchy
            cat_words = set(category.replace(">", " ").split())
            cat_overlap_count = len(set(q_words) & cat_words)
            cat_overlap_ratio = cat_overlap_count / max(1, q_len)
            
            # 5. BM25 Score
            bm25_score = 0.0
            idx = self.item_id_to_idx.get(item_id)
            if idx is not None and q_len > 0:
                # Score only the specific document index
                bm25_score = self.bm25.get_batch_scores(q_words, [idx])[0]
            
            # 6. Query position matching
            # Does query start with the first word of the title?
            starts_with = 1.0 if (q_words and t_words and q_words[0] == t_words[0]) else 0.0
            
            features.append({
                "q_len": q_len,
                "t_len": t_len,
                "common_count": common_count,
                "common_ratio": common_ratio,
                "fuzz_ratio": ratio / 100.0,
                "fuzz_partial_ratio": partial_ratio / 100.0,
                "fuzz_token_sort_ratio": token_sort_ratio / 100.0,
                "brand_match": brand_match,
                "cat_overlap_count": cat_overlap_count,
                "cat_overlap_ratio": cat_overlap_ratio,
                "bm25_score": bm25_score,
                "starts_with": starts_with,
            })
            
        return pd.DataFrame(features)
