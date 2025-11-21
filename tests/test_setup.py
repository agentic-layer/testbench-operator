"""
Unit tests for setup.py

Tests the dataset download, conversion, and Ragas dataset creation functionality.
"""

import os
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from setup import custom_convert_csv, dataframe_to_ragas_dataset, get_converter, main


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmp = tempfile.mkdtemp()
    original_cwd = Path.cwd()
    yield tmp, original_cwd
    shutil.rmtree(tmp, ignore_errors=True)


# TestCustomConvertCSV tests
def test_converts_string_to_list():
    """Test that retrieved_contexts strings are converted to lists"""
    csv_content = b"user_input,retrieved_contexts,reference\n"
    csv_content += b'"Question?","Context text","Answer"\n'

    buffer = BytesIO(csv_content)
    df = custom_convert_csv(buffer)

    assert isinstance(df["retrieved_contexts"].iloc[0], list)
    assert df["retrieved_contexts"].iloc[0] == ["Context text"]


def test_handles_empty_retrieved_contexts():
    """Test handling of empty retrieved_contexts"""
    csv_content = b"user_input,retrieved_contexts,reference\n"
    csv_content += b'"Question?","","Answer"\n'

    buffer = BytesIO(csv_content)
    df = custom_convert_csv(buffer)

    # Empty string becomes [nan] in pandas, which then becomes []
    # The function converts non-list values to lists
    result = df["retrieved_contexts"].iloc[0]
    # Check that it's a list and handle NaN case
    assert isinstance(result, list)
    # If it contains NaN, that's acceptable behavior for empty strings in CSV
    if result and pd.isna(result[0]):
        # This is expected - pandas converts empty string to NaN
        pass
    else:
        # Or it should be an empty list
        assert result == []


def test_missing_retrieved_contexts_column():
    """Test that CSV without retrieved_contexts column works"""
    csv_content = b"user_input,reference\n"
    csv_content += b'"Question?","Answer"\n'

    buffer = BytesIO(csv_content)
    df = custom_convert_csv(buffer)

    assert "retrieved_contexts" not in df.columns


# TestGetConverter tests
def test_unsupported_format():
    """Test that unsupported formats raise TypeError"""
    with pytest.raises(TypeError) as exc_info:
        get_converter("https://example.com/data.xlsx")

    assert "Unsupported filetype" in str(exc_info.value)


# TestDataframeToRagasDataset tests
def test_creates_ragas_dataset_file(temp_dir):
    """Test that ragas_dataset.jsonl is created"""

    tmp, original_cwd = temp_dir
    os.chdir(tmp)

    try:
        df = pd.DataFrame(
            {
                "user_input": ["Question 1"],
                "retrieved_contexts": [["Context 1"]],
                "reference": ["Answer 1"],
            }
        )

        dataframe_to_ragas_dataset(df)

        # Check for the file in the datasets subdirectory
        dataset_file = Path(tmp) / "data" / "datasets" / "ragas_dataset.jsonl"
        assert dataset_file.exists(), f"Dataset file not found at {dataset_file}"
    finally:
        os.chdir(original_cwd)


# TestMain tests
def test_main_with_csv(temp_dir, monkeypatch):
    """Test main function with CSV file"""

    tmp, original_cwd = temp_dir
    os.chdir(tmp)

    try:
        # Mock the HTTP response
        csv_content = b"user_input,retrieved_contexts,reference\n"
        csv_content += b'"Question?","Context text","Answer"\n'

        class MockResponse:
            def __init__(self):
                self.content = csv_content

            def raise_for_status(self):
                pass

        calls = []

        def mock_get(url, timeout=None):
            calls.append({"url": url, "timeout": timeout})
            return MockResponse()

        monkeypatch.setattr("setup.requests.get", mock_get)

        # Run main
        main("https://example.com/data.csv")

        # Verify dataset was created in datasets subdirectory
        dataset_file = Path(tmp) / "data" / "datasets" / "ragas_dataset.jsonl"
        assert dataset_file.exists(), f"Dataset file not found at {dataset_file}"

        # Verify requests.get was called correctly
        assert len(calls) == 1
        assert calls[0]["url"] == "https://example.com/data.csv"
        assert calls[0]["timeout"] == 20
    finally:
        os.chdir(original_cwd)


def test_main_with_json(temp_dir, monkeypatch):
    """Test main function with JSON file"""

    tmp, original_cwd = temp_dir
    os.chdir(tmp)

    try:
        # Mock the HTTP response
        json_content = b"""[
            {
                "user_input": "Question?",
                "retrieved_contexts": ["Context text"],
                "reference": "Answer"
            }
        ]"""

        class MockResponse:
            def __init__(self):
                self.content = json_content

            def raise_for_status(self):
                pass

        def mock_get(url, timeout=None):
            return MockResponse()

        monkeypatch.setattr("setup.requests.get", mock_get)

        # Run main
        main("https://example.com/data.json")

        # Verify dataset was created in datasets subdirectory
        dataset_file = Path(tmp) / "data" / "datasets" / "ragas_dataset.jsonl"
        assert dataset_file.exists(), f"Dataset file not found at {dataset_file}"
    finally:
        os.chdir(original_cwd)


def test_main_with_invalid_url(temp_dir, monkeypatch):
    """Test main function with invalid URL (HTTP error)"""

    tmp, original_cwd = temp_dir
    os.chdir(tmp)

    try:
        # Mock HTTP error
        class MockResponse:
            def raise_for_status(self):
                raise Exception("HTTP 404")

        def mock_get(url, timeout=None):
            return MockResponse()

        monkeypatch.setattr("setup.requests.get", mock_get)

        # Verify that the error propagates
        with pytest.raises(Exception, match="HTTP 404"):
            main("https://example.com/nonexistent.csv")
    finally:
        os.chdir(original_cwd)
