"""
Cache Manager — 5-layer caching system.
Caches:
  1. Embedding Cache (.npy)
  2. Token Cache (.json)
  3. Feature Cache (.parquet)
  4. Parsed Attribute Cache (.json)
  5. Normalized Product Cache (.json)

Each cache is stored in its own sub-folder under config['data']['cache_dir']
and can be invalidated independently.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
from typing import Any

from project.utils.logging_utils import setup_logger

logger = setup_logger("cache_manager")


class CacheManager:
    """Manages 5 separate caches under project/data/cache."""

    def __init__(self, base_dir: str = ".", cache_root: str = "project/data/cache"):
        self.root = Path(base_dir) / cache_root
        self.dirs = {
            "embedding": self.root / "embedding",
            "token": self.root / "token",
            "feature": self.root / "feature",
            "attribute": self.root / "attribute",
            "product": self.root / "product",
        }
        self._create_directories()

    def _create_directories(self):
        """Create separate cache subdirectories."""
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)

    def _get_key_hash(self, key: str) -> str:
        """Helper to generate a secure filename hash for string keys."""
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    # --- Invalidation ---
    def invalidate(self, cache_name: str):
        """Clear all files in a specific cache directory."""
        if cache_name not in self.dirs:
            raise ValueError(f"Unknown cache: {cache_name}. Valid caches: {list(self.dirs.keys())}")

        path = self.dirs[cache_name]
        logger.info(f"Invalidating cache: {cache_name} at {path}")
        count = 0
        for f in path.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        logger.info(f"  Removed {count} files from {cache_name} cache.")

    def invalidate_all(self):
        """Clear all caches."""
        for name in self.dirs.keys():
            self.invalidate(name)

    # --- 1. Embedding Cache (.npy) ---
    def get_embedding(self, key: str) -> np.ndarray | None:
        """Retrieve cached embedding array."""
        h = self._get_key_hash(key)
        path = self.dirs["embedding"] / f"{h}.npy"
        if path.exists():
            return np.load(path)
        return None

    def set_embedding(self, key: str, value: np.ndarray):
        """Cache an embedding array."""
        h = self._get_key_hash(key)
        path = self.dirs["embedding"] / f"{h}.npy"
        np.save(path, value)

    # --- 2. Token Cache (.json) ---
    def get_tokens(self, key: str) -> list[int] | list[str] | None:
        """Retrieve token list from cache."""
        h = self._get_key_hash(key)
        path = self.dirs["token"] / f"{h}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set_tokens(self, key: str, value: list[int] | list[str]):
        """Cache a list of tokens."""
        h = self._get_key_hash(key)
        path = self.dirs["token"] / f"{h}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)

    # --- 3. Feature Cache (.parquet) ---
    def get_features(self, key: str) -> pd.DataFrame | None:
        """Retrieve features DataFrame."""
        h = self._get_key_hash(key)
        path = self.dirs["feature"] / f"{h}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def set_features(self, key: str, df: pd.DataFrame):
        """Cache features DataFrame."""
        h = self._get_key_hash(key)
        path = self.dirs["feature"] / f"{h}.parquet"
        df.to_parquet(path, index=False)

    # --- 4. Parsed Attribute Cache (.json) ---
    def get_attributes(self, key: str) -> dict[str, str] | None:
        """Retrieve parsed attributes dict."""
        h = self._get_key_hash(key)
        path = self.dirs["attribute"] / f"{h}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set_attributes(self, key: str, value: dict[str, str]):
        """Cache parsed attributes dict."""
        h = self._get_key_hash(key)
        path = self.dirs["attribute"] / f"{h}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)

    # --- 5. Normalized Product Cache (.json) ---
    def get_normalized_product(self, item_id: str) -> str | None:
        """Retrieve normalized product text by item_id."""
        path = self.dirs["product"] / f"{item_id}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set_normalized_product(self, item_id: str, value: str):
        """Cache normalized product text by item_id."""
        path = self.dirs["product"] / f"{item_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
