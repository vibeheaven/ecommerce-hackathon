"""
Negative Sampler — Orchestrates all negative sampling strategies.

Rewritten (v5) to fix the systemic issues of the first implementation:
  * Strategy quotas are allocated with the largest-remainder method over the
    configured ratios. The old code gave every strategy `max(1, ...)` slots and
    then truncated the merged list to the first N — which silently dropped the
    strategies that ran last (lexical_hard, attribute_conflict, cross_query).
  * Every strategy reports produced / filtered / shortfall counts; nothing
    fails silently. A production report is logged after each build.
  * Candidate generation is vectorized through ItemCatalog (no per-candidate
    `.iloc` calls, no per-strategy full-catalog scans).
  * Sampling is deterministic: every (query, anchor item) pair gets its own
    seeded RNG derived from the global seed.
  * Lexical similarity is actually computed and passed to the confidence
    scorer / probable-positive filter (the old code never passed it).
"""
import hashlib
import random
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from project.utils.logging_utils import setup_logger
from project.utils.text_cleaner import clean_index_text, clean_brand
from project.negative_samples.item_catalog import ItemCatalog
from project.negative_samples.probable_positive_filter import ProbablePositiveFilter
from project.negative_samples.confidence_scorer import ConfidenceScorer

logger = setup_logger("negative_sampler")

# Colors used for query-attribute conflict detection (ASCII normalized)
_COLORS_ASCII = {
    "siyah", "beyaz", "kirmizi", "mavi", "yesil", "sari", "pembe",
    "mor", "turuncu", "gri", "kahverengi", "lacivert", "bej",
    "krem", "bordo", "haki", "ekru", "antrasit", "altin", "gumus",
    "lila", "turkuaz", "fusya",
}

_GENDER_TOKENS = {
    "erkek": "erkek", "bay": "erkek",
    "kadin": "kadın", "bayan": "kadın", "kiz": "kadın",
}

_STOP_TOKENS = {"ve", "ile", "icin", "the", "set", "adet", "cm", "mm", "lt"}

# How many extra candidates a strategy inspects per requested slot, to survive
# the probable-positive filter and dedup without under-producing.
_OVERSAMPLE = 4

_IMPLEMENTED_STRATEGIES = [
    "random", "cross_category", "same_category", "same_brand",
    "lexical_hard", "attribute_conflict", "cross_query", "embedding_hard",
]

# When a strategy cannot fill its quota (small category, 1-token query, ...),
# the deficit is refilled from these, hardest first.
_BACKFILL_CHAIN = ["same_category", "lexical_hard", "cross_query", "same_brand", "random"]


class NegativeSampler:
    """Orchestrates negative sampling for train and validation splits."""

    def __init__(
        self,
        config: dict,
        neg_config: dict,
        items_df: pd.DataFrame,
        train_df: pd.DataFrame,
        embedding_neighbors: dict[str, list[tuple[int, float]]] | None = None,
    ):
        self.config = config
        self.neg_config = neg_config
        self.seed = int(config.get("project", {}).get("seed", 42))

        self.catalog = ItemCatalog(items_df, seed=self.seed)
        self.filter = ProbablePositiveFilter(neg_config)
        self.scorer = ConfidenceScorer(neg_config)

        self.ratio = int(neg_config.get("positive_negative_ratio", 3))
        raw_ratios = neg_config.get("strategy_ratios", {})
        unknown = set(raw_ratios) - set(_IMPLEMENTED_STRATEGIES)
        if unknown:
            raise ValueError(
                f"strategy_ratios contains unimplemented strategies: {sorted(unknown)}. "
                f"Implemented: {_IMPLEMENTED_STRATEGIES}"
            )

        self.embedding_neighbors = embedding_neighbors
        if raw_ratios.get("embedding_hard", 0.0) > 0 and embedding_neighbors is None:
            raise ValueError(
                "strategy_ratios.embedding_hard > 0 but no embedding neighbors were provided. "
                "Run project/embeddings/build_embedding_neighbors.py first, or set the ratio to 0. "
                "(This is intentionally a hard error — the old pipeline silently produced 0 "
                "embedding_hard samples.)"
            )

        active = {k: v for k, v in raw_ratios.items() if v > 0}
        total = sum(active.values())
        if total <= 0:
            raise ValueError("strategy_ratios must contain at least one positive ratio.")
        self.strategy_ratios = {k: v / total for k, v in active.items()}
        logger.info(f"Active strategy ratios (normalized): { {k: round(v, 3) for k, v in self.strategy_ratios.items()} }")

        # --- Query-side structures ---
        logger.info("Building known-positives lookup and cross-query token map...")
        self.known_positive_ids: dict[str, set] = {}
        self.known_positive_pos: dict[str, set[int]] = {}
        cross_query_token_map: dict[str, list[int]] = {}
        all_positive_pos: list[int] = []

        for query, item_id in zip(train_df["query"].tolist(), train_df["item_id"].tolist()):
            if not isinstance(query, str) or not query:
                continue
            norm_q = clean_index_text(query)
            self.known_positive_ids.setdefault(norm_q, set()).add(item_id)
            pos = self.catalog.pos_by_item_id.get(item_id)
            if pos is None:
                continue
            self.known_positive_pos.setdefault(norm_q, set()).add(pos)
            all_positive_pos.append(pos)
            for tok in set(norm_q.split()):
                if len(tok) >= 3 and tok not in _STOP_TOKENS:
                    cross_query_token_map.setdefault(tok, []).append(pos)

        self._cross_query_token_map = {
            tok: np.unique(np.asarray(v, dtype=np.int32))
            for tok, v in cross_query_token_map.items()
        }
        self._all_positive_pos = np.unique(np.asarray(all_positive_pos, dtype=np.int32)) if all_positive_pos else np.asarray([], dtype=np.int32)

        # Lazy per-query caches
        self._lexical_cache: dict[str, list[tuple[int, float]]] = {}
        self._cross_query_cache: dict[str, np.ndarray] = {}
        self._used_negatives: dict[str, set[int]] = {}

        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ #
    # Per-query candidate pools
    # ------------------------------------------------------------------ #

    def _query_tokens(self, norm_q: str) -> list[str]:
        return [t for t in norm_q.split() if len(t) >= 3 and t not in _STOP_TOKENS]

    def _lexical_candidates(self, norm_q: str) -> list[tuple[int, float]]:
        """Ranked (position, overlap_ratio) candidates sharing tokens with the query."""
        if norm_q in self._lexical_cache:
            return self._lexical_cache[norm_q]

        toks = self._query_tokens(norm_q)
        postings = [self.catalog.inverted_index[t] for t in toks if t in self.catalog.inverted_index]
        if not postings:
            self._lexical_cache[norm_q] = []
            return []

        merged = np.concatenate(postings)
        positions, counts = np.unique(merged, return_counts=True)
        ratios = counts / max(len(toks), 1)

        # Keep candidates matching at least half the query tokens
        keep = ratios >= 0.5
        positions, ratios, counts = positions[keep], ratios[keep], counts[keep]

        # Deterministic shuffle then stable sort by match count desc,
        # so ties are broken randomly but reproducibly.
        rng = np.random.default_rng(self._seed_for(norm_q, "lex"))
        perm = rng.permutation(len(positions))
        positions, ratios, counts = positions[perm], ratios[perm], counts[perm]
        order = np.argsort(-counts, kind="stable")[:400]

        result = [(int(positions[i]), float(ratios[i])) for i in order]
        self._lexical_cache[norm_q] = result
        return result

    def _cross_query_pool(self, norm_q: str) -> np.ndarray:
        """Items that are positives of *other* queries sharing a token with this one."""
        if norm_q in self._cross_query_cache:
            return self._cross_query_cache[norm_q]

        toks = self._query_tokens(norm_q)
        arrays = [self._cross_query_token_map[t] for t in toks if t in self._cross_query_token_map]
        pool = np.unique(np.concatenate(arrays)) if arrays else self._all_positive_pos

        own = self.known_positive_pos.get(norm_q, set())
        if own:
            pool = pool[~np.isin(pool, np.fromiter(own, dtype=np.int32, count=len(own)))]
        if len(pool) > 2000:
            rng = np.random.default_rng(self._seed_for(norm_q, "xq"))
            pool = rng.choice(pool, size=2000, replace=False)

        self._cross_query_cache[norm_q] = pool
        return pool

    def _query_attributes(self, norm_q: str) -> dict:
        toks = set(norm_q.split())
        genders = {g for t, g in _GENDER_TOKENS.items() if t in toks}
        colors = _COLORS_ASCII & toks
        return {"genders": genders, "colors": colors}

    def _seed_for(self, norm_q: str, salt: str) -> int:
        digest = hashlib.md5(f"{self.seed}:{salt}:{norm_q}".encode()).hexdigest()
        return int(digest[:12], 16)

    # ------------------------------------------------------------------ #
    # Strategy candidate generators — each yields catalog positions
    # ------------------------------------------------------------------ #

    def _gen_random(self, ctx: dict, k: int, rng: random.Random):
        n = len(self.catalog.item_ids)
        for _ in range(k * _OVERSAMPLE):
            pos = rng.randrange(n)
            if ctx["anchor_parent"] and self.catalog.parents[pos] == ctx["anchor_parent"]:
                continue
            yield pos, {}

    def _gen_cross_category(self, ctx: dict, k: int, rng: random.Random):
        parents = [p for p in self.catalog.all_parents if p != ctx["anchor_parent"]]
        if not parents:
            return
        for _ in range(k * _OVERSAMPLE):
            group = self.catalog.parent_groups[rng.choice(parents)]
            yield int(group[rng.randrange(len(group))]), {}

    def _gen_same_category(self, ctx: dict, k: int, rng: random.Random):
        group = self.catalog.cat_groups.get(ctx["anchor_category"], None)
        if group is None or len(group) == 0:
            return
        n_try = min(len(group), k * _OVERSAMPLE)
        for i in rng.sample(range(len(group)), n_try):
            yield int(group[i]), {}

    def _gen_same_brand(self, ctx: dict, k: int, rng: random.Random):
        group = self.catalog.brand_groups.get(ctx["anchor_brand"], None)
        if group is None or len(group) == 0:
            return
        n_try = min(len(group), k * _OVERSAMPLE)
        for i in rng.sample(range(len(group)), n_try):
            yield int(group[i]), {}

    def _gen_lexical_hard(self, ctx: dict, k: int, rng: random.Random):
        for pos, ratio in self._lexical_candidates(ctx["norm_q"]):
            yield pos, {"lexical_similarity": ratio}

    def _gen_cross_query(self, ctx: dict, k: int, rng: random.Random):
        pool = self._cross_query_pool(ctx["norm_q"])
        if len(pool) == 0:
            return
        n_try = min(len(pool), k * _OVERSAMPLE)
        for i in rng.sample(range(len(pool)), n_try):
            yield int(pool[i]), {}

    def _gen_attribute_conflict(self, ctx: dict, k: int, rng: random.Random):
        q_attrs = ctx["q_attrs"]
        if not q_attrs["genders"] and not q_attrs["colors"]:
            return

        # Prefer lexically similar candidates (hardest), then same-category ones.
        candidates: list[tuple[int, float]] = list(self._lexical_candidates(ctx["norm_q"]))
        group = self.catalog.cat_groups.get(ctx["anchor_category"], None)
        if group is not None and len(group) > 0:
            n_extra = min(len(group), k * _OVERSAMPLE * 2)
            candidates += [(int(group[i]), 0.0) for i in rng.sample(range(len(group)), n_extra)]

        for pos, ratio in candidates:
            conflict = False
            if q_attrs["genders"]:
                item_gender = self.catalog.clean_genders[pos]
                if item_gender in ("erkek", "kadın") and item_gender not in q_attrs["genders"]:
                    conflict = True
            if not conflict and q_attrs["colors"]:
                title_toks = set(self.catalog.clean_titles[pos].split())
                title_colors = _COLORS_ASCII & title_toks
                item_color = self.catalog.color_of(pos)
                colors_of_item = set(title_colors)
                if item_color:
                    colors_of_item |= _COLORS_ASCII & set(item_color.split())
                if colors_of_item and not (colors_of_item & q_attrs["colors"]):
                    conflict = True
            if conflict:
                yield pos, {"lexical_similarity": ratio, "has_attribute_conflict": True}

    def _gen_embedding_hard(self, ctx: dict, k: int, rng: random.Random):
        neighbors = self.embedding_neighbors.get(ctx["norm_q"], []) if self.embedding_neighbors else []
        for pos, sim in neighbors:
            yield pos, {"semantic_similarity": sim}

    # ------------------------------------------------------------------ #
    # Core sampling
    # ------------------------------------------------------------------ #

    def _allocate_quotas(self, total: int, rng: random.Random) -> dict[str, int]:
        """Largest-remainder allocation of `total` slots over strategy ratios."""
        quotas: dict[str, int] = {}
        remainders: list[tuple[str, float]] = []
        assigned = 0
        for name, ratio in self.strategy_ratios.items():
            exact = ratio * total
            base = int(exact)
            quotas[name] = base
            assigned += base
            remainders.append((name, exact - base))

        remaining = total - assigned
        if remaining > 0:
            names = [n for n, _ in remainders]
            weights = [max(r, 1e-9) for _, r in remainders]
            for name in rng.choices(names, weights=weights, k=remaining):
                quotas[name] += 1
        return quotas

    def sample_negatives_for_query(
        self,
        query: str,
        query_category: str | None = None,
        query_brand: str | None = None,
        anchor_item_id: str | None = None,
        rng: random.Random | None = None,
    ) -> list[dict]:
        """Sample a set of negative items for a single (query, positive item) row."""
        if not isinstance(query, str) or not query:
            return []

        norm_q = clean_index_text(query)
        if rng is None:
            rng = random.Random(self._seed_for(norm_q, str(anchor_item_id)))

        ctx = {
            "norm_q": norm_q,
            "q_tokens": norm_q.split(),
            "q_attrs": self._query_attributes(norm_q),
            "anchor_category": query_category.strip() if isinstance(query_category, str) else "",
            "anchor_parent": (
                query_category.split("/")[0].split(">")[0].strip()
                if isinstance(query_category, str) and query_category else ""
            ),
            "anchor_brand": clean_brand(query_brand),
        }

        exclude_ids = self.known_positive_ids.get(norm_q, set())
        exclude_pos = set(self.known_positive_pos.get(norm_q, set()))
        used = self._used_negatives.setdefault(norm_q, set())

        generators = {
            "random": self._gen_random,
            "cross_category": self._gen_cross_category,
            "same_category": self._gen_same_category,
            "same_brand": self._gen_same_brand,
            "lexical_hard": self._gen_lexical_hard,
            "attribute_conflict": self._gen_attribute_conflict,
            "cross_query": self._gen_cross_query,
            "embedding_hard": self._gen_embedding_hard,
        }

        total_target = self.ratio
        quotas = self._allocate_quotas(total_target, rng)
        negatives: list[dict] = []

        def try_fill(name: str, want: int) -> int:
            """Run one strategy until `want` accepted samples; returns accepted count."""
            accepted = 0
            self.stats[f"{name}_requested"] += want
            for pos, extra in generators[name](ctx, want, rng):
                if accepted >= want:
                    break
                item_id = self.catalog.item_ids[pos]
                if pos in used or pos in exclude_pos or item_id in exclude_ids:
                    continue
                if anchor_item_id is not None and item_id == anchor_item_id:
                    continue

                title_tokens = self.catalog.clean_titles[pos].split()
                category_match = (
                    self.catalog.cat_clean[pos] == ctx["anchor_category"]
                    if ctx["anchor_category"] else None
                )
                # Attribute conflicts override the overlap-based PP heuristics:
                # "kırmızı elbise" vs a blue dress is a certain negative no
                # matter how similar the titles are. The verbatim-substring
                # rule still applies inside the filter.
                if not extra.get("has_attribute_conflict"):
                    if self.filter.is_probable_positive(
                        query=norm_q,
                        product_title=self.catalog.clean_titles[pos],
                        category_match=category_match,
                        query_tokens=ctx["q_tokens"],
                        title_tokens=title_tokens,
                    ):
                        self.stats[f"{name}_filtered_pp"] += 1
                        continue
                elif ctx["norm_q"] in " ".join(title_tokens):
                    # even for conflicts, don't use titles containing the full query
                    self.stats[f"{name}_filtered_pp"] += 1
                    continue

                lex_sim = extra.get("lexical_similarity")
                if lex_sim is None:
                    t_set = set(title_tokens)
                    q_toks = ctx["q_tokens"]
                    lex_sim = sum(1 for t in q_toks if t in t_set) / max(len(q_toks), 1)

                confidence = self.scorer.score_negative(
                    negative_type=name,
                    query=norm_q,
                    title=self.catalog.clean_titles[pos],
                    lexical_similarity=lex_sim,
                    semantic_similarity=extra.get("semantic_similarity"),
                    has_attribute_conflict=extra.get("has_attribute_conflict", False),
                )
                weight = self.scorer.get_sample_weight(confidence, negative_type=name)
                if weight == 0.0:
                    self.stats[f"{name}_discarded_lowconf"] += 1
                    continue

                used.add(pos)
                record = self.catalog.record(pos, name, confidence)
                record["label"] = 0
                record["sample_weight"] = weight
                record["query"] = query
                negatives.append(record)
                accepted += 1

            self.stats[f"{name}_produced"] += accepted
            self.stats[f"{name}_shortfall"] += max(0, want - accepted)
            return accepted

        for name, want in quotas.items():
            if want > 0:
                try_fill(name, want)

        # Backfill deficits from the hardest available sources
        deficit = total_target - len(negatives)
        for name in _BACKFILL_CHAIN:
            if deficit <= 0:
                break
            got = try_fill(name, deficit)
            if got:
                self.stats[f"{name}_backfilled"] += got
            deficit = total_target - len(negatives)

        if deficit > 0:
            self.stats["unfilled_slots"] += deficit

        return negatives

    # ------------------------------------------------------------------ #
    # Dataset construction
    # ------------------------------------------------------------------ #

    def build_dataset(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Combine positive pairs with sampled negative pairs (fold-aware)."""
        logger.info(f"Building dataset on {len(split_df):,} positive pairs...")
        rows = []
        columns = split_df.columns

        for row in tqdm(split_df.itertuples(index=False), total=len(split_df)):
            row_d = dict(zip(columns, row))
            fold = row_d.get("fold", -1)
            rows.append({
                "term_id": row_d.get("term_id"),
                "item_id": row_d.get("item_id"),
                "query": row_d.get("query"),
                "title": row_d.get("title"),
                "category": row_d.get("category"),
                "brand": row_d.get("brand"),
                "gender": row_d.get("gender"),
                "age_group": row_d.get("age_group"),
                "attributes": row_d.get("attributes"),
                "label": 1,
                "sample_weight": 1.0,
                "negative_type": "positive",
                "fold": fold,
            })

            for neg in self.sample_negatives_for_query(
                query=row_d.get("query"),
                query_category=row_d.get("category"),
                query_brand=row_d.get("brand"),
                anchor_item_id=row_d.get("item_id"),
            ):
                neg_row = {
                    "term_id": row_d.get("term_id"),
                    "fold": fold,
                    **{k: neg.get(k) for k in (
                        "item_id", "query", "title", "category", "brand",
                        "gender", "age_group", "attributes", "label",
                        "sample_weight", "negative_type",
                    )},
                }
                rows.append(neg_row)

        dataset = pd.DataFrame(rows)
        self.log_production_report(dataset)
        return dataset

    def log_production_report(self, dataset: pd.DataFrame | None = None):
        """Log per-strategy production statistics — the anti-silent-failure report."""
        logger.info("=" * 72)
        logger.info("Negative Sampling Production Report")
        logger.info("=" * 72)
        header = f"{'strategy':<20} {'requested':>9} {'produced':>9} {'pp_filt':>8} {'lowconf':>8} {'backfill':>8} {'shortfall':>9}"
        logger.info(header)
        for name in _IMPLEMENTED_STRATEGIES:
            requested = self.stats.get(f"{name}_requested", 0)
            if requested == 0:
                continue
            logger.info(
                f"{name:<20} {requested:>9,} {self.stats.get(f'{name}_produced', 0):>9,} "
                f"{self.stats.get(f'{name}_filtered_pp', 0):>8,} "
                f"{self.stats.get(f'{name}_discarded_lowconf', 0):>8,} "
                f"{self.stats.get(f'{name}_backfilled', 0):>8,} "
                f"{self.stats.get(f'{name}_shortfall', 0):>9,}"
            )
        if self.stats.get("unfilled_slots", 0):
            logger.warning(f"Unfilled negative slots (after backfill): {self.stats['unfilled_slots']:,}")
        if dataset is not None and len(dataset):
            dist = dataset[dataset["label"] == 0]["negative_type"].value_counts()
            logger.info("Final negative distribution:")
            for neg_type, count in dist.items():
                logger.info(f"  {neg_type:<20} {count:>9,} ({count / dist.sum() * 100:5.1f}%)")
        logger.info("=" * 72)
