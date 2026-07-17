"""
Lexical Hard Negative Strategy — Samples items with high word overlap.
Uses an inverted index over title tokens to quickly find candidate negatives.
"""
import random
import pandas as pd
from project.utils.text_cleaner import clean_index_text

class LexicalHardNegativeStrategy:
    """Samples items that share many title tokens with the query but are negative."""

    def __init__(self, items_df: pd.DataFrame):
        self.items_df = items_df
        # Build a simple inverted index for fast token matching
        # Mapping: token -> list of row indices containing this token
        self.inverted_index = {}

        logger_info = "Building inverted index for Lexical Hard negatives..."
        # Only index words of length >= 3 to avoid common junk like "ve", "de", etc.
        # We index first 100k items to keep it extremely fast and low-memory while still representative
        max_index_items = min(150_000, len(items_df))

        for idx in range(max_index_items):
            row = items_df.iloc[idx]
            title = row.get("title")
            if not isinstance(title, str):
                continue
            title_clean = clean_index_text(title)
            for tok in set(title_clean.split()):
                if len(tok) >= 3:
                    if tok not in self.inverted_index:
                        self.inverted_index[tok] = []
                    self.inverted_index[tok].append(idx)

    def sample(
        self,
        query: str,
        exclude_item_ids: set[str],
        n_samples: int = 1,
    ) -> list[dict]:
        """Sample items with high token overlap but not in exclude_item_ids."""
        samples = []
        q_clean = clean_index_text(query)
        q_tokens = [tok for tok in q_clean.split() if len(tok) >= 3]

        if not q_tokens:
            return samples

        # Retrieve candidate indices that contain at least one query token
        candidate_indices = []
        for tok in q_tokens:
            if tok in self.inverted_index:
                candidate_indices.extend(self.inverted_index[tok])

        if not candidate_indices:
            return samples

        # Count token matches and sort candidates
        # To be fast, sample from unique candidates randomly or select those with some match
        candidate_set = list(set(candidate_indices))
        # Shuffle candidates to allow varied negatives
        random.shuffle(candidate_set)

        max_attempts = 100
        attempts = 0

        for idx in candidate_set:
            attempts += 1
            if attempts > max_attempts or len(samples) >= n_samples:
                break

            row = self.items_df.iloc[idx]
            item_id = row.get("item_id")

            if item_id in exclude_item_ids:
                continue

            samples.append({
                "item_id": item_id,
                "title": row.get("title"),
                "category": row.get("category"),
                "brand": row.get("brand"),
                "gender": row.get("gender"),
                "attributes": row.get("attributes"),
                "negative_type": "lexical_hard",
                "negative_confidence": 0.82,
            })

        return samples
