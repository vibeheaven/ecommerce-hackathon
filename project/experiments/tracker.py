"""
Experiment Tracker — Logs experiment metadata and validation scores.
Stores results in a structured registry for experiment comparison.
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any

from project.utils.logging_utils import setup_logger

logger = setup_logger("tracker")


class ExperimentTracker:
    """Tracks training runs, hyperparameters, and resulting metrics."""

    def __init__(self, config: dict, registry_path: str = "project/experiments/experiment_registry.json"):
        self.config = config
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_git_commit(self) -> str:
        """Helper to get current git commit hash if git is initialized."""
        try:
            import subprocess
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            return commit[:8]
        except Exception:
            return "unknown"

    def log_experiment(
        self,
        model_name: str,
        metrics: dict[str, Any],
        hyperparams: dict[str, Any] | None = None,
        notes: str = "",
    ) -> str:
        """
        Record a complete experiment run to the registry.
        Returns a generated Experiment ID.
        """
        # Load existing registry
        registry = []
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read experiment registry: {e}. Starting fresh.")

        # Generate Experiment ID
        exp_id = f"EXP_{len(registry) + 1:03d}"

        # Combine data
        experiment_entry = {
            "experiment_id": exp_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "commit": self._get_git_commit(),
            "hyperparameters": hyperparams or {},
            "metrics": metrics,
            "notes": notes,
        }

        registry.append(experiment_entry)

        # Save registry back
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4, ensure_ascii=False)

        logger.info(f"✓ Experiment {exp_id} logged to registry at {self.registry_path}")
        logger.info(f"  Macro F1: {metrics.get('overall_macro_f1', 0.0):.4f} | Optimal Threshold: {metrics.get('threshold', 0.50):.4f}")

        return exp_id
