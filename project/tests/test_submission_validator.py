"""Tests for submission_validator module."""
import pytest
import pandas as pd
from pathlib import Path
from project.submission.submission_validator import validate_submission


@pytest.fixture
def create_mock_submissions(tmp_path):
    """Create various mock submission files for testing validation rules."""
    # 1. Correct submission
    valid_df = pd.DataFrame({
        "id": ["TST_1", "TST_2", "TST_3"],
        "prediction": [1, 0, 1]
    })
    valid_path = tmp_path / "valid.csv"
    valid_df.to_csv(valid_path, index=False)

    # 2. Duplicate IDs
    dup_df = pd.DataFrame({
        "id": ["TST_1", "TST_1", "TST_3"],
        "prediction": [1, 0, 1]
    })
    dup_path = tmp_path / "dup.csv"
    dup_df.to_csv(dup_path, index=False)

    # 3. Wrong columns
    col_df = pd.DataFrame({
        "id": ["TST_1", "TST_2", "TST_3"],
        "wrong_pred": [1, 0, 1]
    })
    col_path = tmp_path / "wrong_cols.csv"
    col_df.to_csv(col_path, index=False)

    # 4. Invalid predictions (not 0/1)
    val_df = pd.DataFrame({
        "id": ["TST_1", "TST_2", "TST_3"],
        "prediction": [1, 2, 0]
    })
    val_path = tmp_path / "invalid_vals.csv"
    val_df.to_csv(val_path, index=False)

    # 5. Sample submission for ordering/matching check
    sample_df = pd.DataFrame({
        "id": ["TST_1", "TST_2", "TST_3"],
        "prediction": [0, 0, 0]
    })
    sample_path = tmp_path / "sample.csv"
    sample_df.to_csv(sample_path, index=False)

    # 6. Wrong order submission
    order_df = pd.DataFrame({
        "id": ["TST_2", "TST_1", "TST_3"],
        "prediction": [1, 0, 1]
    })
    order_path = tmp_path / "wrong_order.csv"
    order_df.to_csv(order_path, index=False)

    return {
        "valid": valid_path,
        "dup": dup_path,
        "wrong_cols": col_path,
        "invalid_vals": val_path,
        "sample": sample_path,
        "wrong_order": order_path,
    }


def test_valid_submission(create_mock_submissions):
    """Test that a valid submission passes all checks."""
    res = validate_submission(
        submission_path=create_mock_submissions["valid"],
        sample_submission_path=create_mock_submissions["sample"],
        expected_row_count=3,
    )
    assert res["valid"] is True
    assert len(res["issues"]) == 0


def test_invalid_row_count(create_mock_submissions):
    """Test that wrong row count fails validation."""
    res = validate_submission(
        submission_path=create_mock_submissions["valid"],
        sample_submission_path=create_mock_submissions["sample"],
        expected_row_count=100,  # mismatch
    )
    assert res["valid"] is False
    assert any("row count" in issue for issue in res["issues"])


def test_duplicate_ids(create_mock_submissions):
    """Test that duplicate IDs fail validation."""
    res = validate_submission(
        submission_path=create_mock_submissions["dup"],
        sample_submission_path=create_mock_submissions["sample"],
        expected_row_count=3,
    )
    assert res["valid"] is False
    assert any("Duplicate IDs" in issue for issue in res["issues"])


def test_wrong_columns(create_mock_submissions):
    """Test that wrong column names fail validation."""
    res = validate_submission(
        submission_path=create_mock_submissions["wrong_cols"],
        sample_submission_path=create_mock_submissions["sample"],
        expected_row_count=3,
    )
    assert res["valid"] is False
    assert any("Wrong columns" in issue for issue in res["issues"])


def test_invalid_values(create_mock_submissions):
    """Test that predictions outside {0, 1} fail validation."""
    res = validate_submission(
        submission_path=create_mock_submissions["invalid_vals"],
        sample_submission_path=create_mock_submissions["sample"],
        expected_row_count=3,
    )
    assert res["valid"] is False
    assert any("Invalid prediction values" in issue for issue in res["issues"])


def test_wrong_order(create_mock_submissions):
    """Test that wrong order compared to sample submission fails validation."""
    res = validate_submission(
        submission_path=create_mock_submissions["wrong_order"],
        sample_submission_path=create_mock_submissions["sample"],
        expected_row_count=3,
    )
    assert res["valid"] is False
    assert any("order does not match" in issue for issue in res["issues"])
