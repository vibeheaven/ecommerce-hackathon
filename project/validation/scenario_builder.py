"""
Scenario Builder — Builds distinct validation sets (Easy, Structural,
Lexical Hard, Semantic Hard, Candidate Set Simulation) for granular testing.

v5 changes:
  * Scenarios are class-balanced: positives are subsampled (seeded) to match
    the negative count. The old unbalanced scenarios (50k positives vs 741
    negatives) produced Macro F1 values dominated by the class imbalance —
    lexical_hard "F1 0.50" was an artifact, not a measurement.
  * semantic_hard no longer silently falls back to lexical_hard. It is only
    built when embedding_hard / mined_hard negatives exist; otherwise it is
    reported as skipped.
  * candidate_set_sim intentionally keeps the natural (unbalanced) mix as a
    submission-distribution simulation.
"""
import numpy as np
import pandas as pd

from project.utils.logging_utils import setup_logger

logger = setup_logger("scenario_builder")

_SCENARIO_NEG_TYPES = {
    "easy_mix": ["random", "cross_category"],
    "structural": ["same_category", "same_brand", "attribute_conflict"],
    "lexical_hard": ["lexical_hard"],
    "semantic_hard": ["embedding_hard", "mined_hard"],
}


class ScenarioBuilder:
    """Builds multi-scenario validation datasets from sampled fold data."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def build_scenarios(self, val_dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Slice val_dataset containing mixed negatives into balanced test scenarios.

        Args:
            val_dataset: DataFrame with 'label', 'negative_type' and features,
                         containing positives (label=1) and negatives (label=0).
        """
        positives = val_dataset[val_dataset["label"] == 1]
        negatives = val_dataset[val_dataset["label"] == 0]
        rng = np.random.default_rng(self.seed)

        scenarios: dict[str, pd.DataFrame] = {}

        for name, neg_types in _SCENARIO_NEG_TYPES.items():
            negs = negatives[negatives["negative_type"].isin(neg_types)]
            if len(negs) == 0:
                logger.info(f"  Scenario '{name}': skipped (no negatives of type {neg_types})")
                continue

            if len(positives) > len(negs):
                pos_idx = rng.choice(len(positives), size=len(negs), replace=False)
                pos_sample = positives.iloc[np.sort(pos_idx)]
            else:
                pos_sample = positives

            scenarios[name] = pd.concat([pos_sample, negs]).reset_index(drop=True)

        # Submission-distribution simulation: everything, natural imbalance.
        scenarios["candidate_set_sim"] = val_dataset.copy().reset_index(drop=True)

        for name, df in scenarios.items():
            pos_cnt = int((df["label"] == 1).sum())
            neg_cnt = int((df["label"] == 0).sum())
            logger.info(
                f"  Scenario '{name}': {len(df):,} rows "
                f"(positives: {pos_cnt:,}, negatives: {neg_cnt:,})"
            )

        return scenarios
