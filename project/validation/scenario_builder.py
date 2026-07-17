"""
Scenario Builder — Builds 5 distinct validation sets (Easy, Structural,
Lexical Hard, Semantic Hard, Candidate Set Simulation) for granular testing.
"""
import pandas as pd
from typing import Any

from project.utils.logging_utils import setup_logger

logger = setup_logger("scenario_builder")


class ScenarioBuilder:
    """Builds multi-scenario validation datasets from sampled fold data."""

    def __init__(self):
        pass

    def build_scenarios(self, val_dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Slice val_dataset containing mixed negatives into 5 specific test scenarios.

        Args:
            val_dataset: DataFrame containing 'label', 'negative_type', and features.
                         Contains both positive (label=1) and negatives (label=0).

        Returns:
            dict of scenario_name -> DataFrame
        """
        # Positives are included in all scenarios to compute F1
        positives = val_dataset[val_dataset["label"] == 1]
        negatives = val_dataset[val_dataset["label"] == 0]

        scenarios = {}

        # Scenario A: Easy Mix (random + cross-category negatives)
        easy_negs = negatives[negatives["negative_type"].isin(["random", "cross_category"])]
        scenarios["easy_mix"] = pd.concat([positives, easy_negs]).reset_index(drop=True)

        # Scenario B: Structural Mix (same-category + same-brand + attribute conflict negatives)
        struct_types = ["same_category", "same_brand", "attribute_conflict"]
        struct_negs = negatives[negatives["negative_type"].isin(struct_types)]
        scenarios["structural"] = pd.concat([positives, struct_negs]).reset_index(drop=True)

        # Scenario C: Lexical Hard (lexical hard negatives only)
        lex_negs = negatives[negatives["negative_type"] == "lexical_hard"]
        scenarios["lexical_hard"] = pd.concat([positives, lex_negs]).reset_index(drop=True)

        # Scenario D: Semantic Hard (embedding hard negatives only)
        # Fallback to lexical_hard if embedding_hard negatives are not present (V1 stage)
        sem_negs = negatives[negatives["negative_type"] == "embedding_hard"]
        if len(sem_negs) == 0:
            sem_negs = lex_negs
        scenarios["semantic_hard"] = pd.concat([positives, sem_negs]).reset_index(drop=True)

        # Scenario E: Candidate Set Simulation
        # Simulate query-level ranking by combining a positive with all sampled negatives
        # This includes everything
        scenarios["candidate_set_sim"] = val_dataset.copy().reset_index(drop=True)

        for name, df in scenarios.items():
            pos_cnt = len(df[df["label"] == 1])
            neg_cnt = len(df[df["label"] == 0])
            logger.info(
                f"  Scenario '{name}': {len(df):,} rows "
                f"(positives: {pos_cnt:,}, negatives: {neg_cnt:,})"
            )

        return scenarios
