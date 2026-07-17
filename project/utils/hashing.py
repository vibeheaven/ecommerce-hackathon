"""
Hashing Utilities — Normalized query hash for fold isolation.
"""
import hashlib

from project.utils.text_cleaner import clean_index_text


def normalized_query_hash(query: str | None) -> str:
    """
    Generate a hash from the normalized query text.
    Used as group key for fold splitting to prevent duplicate query leakage.

    Two queries that normalize to the same text get the same hash,
    ensuring they end up in the same fold.
    """
    if not query:
        return hashlib.md5(b"").hexdigest()[:16]

    normalized = clean_index_text(query)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]
