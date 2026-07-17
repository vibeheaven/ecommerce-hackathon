"""
Leaderboard Tracker — Logs submission IDs, public scores, and maps them to Experiment IDs.
Allows tracking of CV (cross-validation) vs LB (leaderboard) correlation.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Any

from project.utils.logging_utils import setup_logger

logger = setup_logger("leaderboard_tracker")


class LeaderboardTracker:
    """Manages leaderboard score records linked to local experiments."""

    def __init__(self, file_path: str = "project/experiments/leaderboard_log.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log_submission(
        self,
        submission_id: str,
        experiment_id: str,
        public_score: float,
        private_estimate: float | None = None,
        commit: str = "unknown",
        notes: str = "",
    ) -> dict[str, Any]:
        """Record a submission and its score."""
        records = []
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read leaderboard log: {e}. Starting fresh.")

        entry = {
            "submission_id": submission_id,
            "experiment_id": experiment_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "public_score": public_score,
            "private_estimate": private_estimate,
            "commit": commit,
            "notes": notes,
        }

        records.append(entry)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)

        logger.info(f"✓ Submission {submission_id} logged. Public LB Score: {public_score:.4f} (Experiment: {experiment_id})")
        return entry
