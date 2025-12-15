"""
Unit tests for evaluate.py

Tests the RAGAS evaluation functionality.
"""

import json
import os
import shutil
import sys
import tempfile
from argparse import ArgumentError
from pathlib import Path

import pandas as pd
import pytest
from ragas.metrics import Metric

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate import AVAILABLE_METRICS, convert_metrics, format_evaluation_scores, main


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmp = tempfile.mkdtemp()
    original_cwd = Path.cwd()
    yield tmp, original_cwd
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def experiment_data(temp_dir):
    """Create test experiment JSONL file"""
    tmp, original_cwd = temp_dir

    # Create test experiment JSONL
    experiment_dir = Path(tmp) / "data" / "experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    experiment_file = experiment_dir / "ragas_experiment.jsonl"
    test_data = [
        {
            "user_input": "What is the weather?",
            "retrieved_contexts": ["Context about weather"],
            "reference": "Expected answer",
            "response": "The weather is sunny.",
            "trace_id": "a1b2c3d4e5f6789012345678901234ab",
        }
    ]

    with open(experiment_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    return tmp, original_cwd, experiment_file


# TestFormatEvaluationScores tests
def test_overall_scores_calculation(tmp_path):
    """Test that overall scores are calculated correctly"""

    # Create temporary experiment file with trace_ids
    experiment_file = tmp_path / "experiment.jsonl"
    test_data = [
        {"trace_id": "trace1"},
        {"trace_id": "trace2"},
        {"trace_id": "trace3"},
    ]
    with open(experiment_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # Create mock result
    class MockTokenUsage:
        input_tokens = 100
        output_tokens = 50

    class MockResult:
        _repr_dict = {"faithfulness": 0.8, "answer_relevancy": 0.75}
        cost_cb = None

        def to_pandas(self):
            return pd.DataFrame({"faithfulness": [0.9, 0.8, 0.7], "answer_relevancy": [0.85, 0.75, 0.65]})

        def total_tokens(self):
            return MockTokenUsage()

        def total_cost(self):
            return 0.001

    mock_result = MockResult()

    formatted = format_evaluation_scores(mock_result, 5.0 / 1e6, 15.0 / 1e6, str(experiment_file))

    # Verify overall scores are correct
    assert abs(formatted.overall_scores["faithfulness"] - 0.8) < 0.01
    assert abs(formatted.overall_scores["answer_relevancy"] - 0.75) < 0.01


def test_individual_results_present(tmp_path):
    """Test that individual results are included"""

    # Create temporary experiment file with trace_ids
    experiment_file = tmp_path / "experiment.jsonl"
    test_data = [
        {"trace_id": "trace1"},
        {"trace_id": "trace2"},
    ]
    with open(experiment_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # Create mock result
    class MockTokenUsage:
        input_tokens = 100
        output_tokens = 50

    class MockResult:
        _repr_dict = {"faithfulness": 0.85}
        cost_cb = None

        def to_pandas(self):
            return pd.DataFrame({"user_input": ["Q1", "Q2"], "faithfulness": [0.9, 0.8]})

        def total_tokens(self):
            return MockTokenUsage()

        def total_cost(self):
            return 0.001

    mock_result = MockResult()

    formatted = format_evaluation_scores(mock_result, 5.0 / 1e6, 15.0 / 1e6, str(experiment_file))

    # Verify individual results
    assert len(formatted.individual_results) == 2


def test_token_usage_placeholders(tmp_path):
    """Test that token usage placeholders are returned when cost_cb is None"""

    # Create temporary experiment file with trace_id
    experiment_file = tmp_path / "experiment.jsonl"
    test_data = [{"trace_id": "trace1"}]
    with open(experiment_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # Create mock result
    class MockTokenUsage:
        input_tokens = 150
        output_tokens = 75

    class MockResult:
        _repr_dict = {"faithfulness": 0.9}
        cost_cb = None  # This causes placeholders to be used

        def to_pandas(self):
            return pd.DataFrame({"faithfulness": [0.9]})

        def total_tokens(self):
            return MockTokenUsage()

        def total_cost(self, **kwargs):
            return 0.002

    mock_result = MockResult()

    formatted = format_evaluation_scores(mock_result, 5.0 / 1e6, 15.0 / 1e6, str(experiment_file))

    # Verify placeholder token usage is returned (0 when cost_cb is None)
    assert formatted.total_tokens["input_tokens"] == 0
    assert formatted.total_tokens["output_tokens"] == 0
    assert formatted.total_cost == 0.0


def test_trace_id_preservation(tmp_path):
    """Test that trace_ids from experiment file are preserved in individual_results"""

    # Create temporary experiment file with trace_ids
    experiment_file = tmp_path / "experiment.jsonl"
    test_data = [
        {"trace_id": "a1b2c3d4e5f6789012345678901234ab"},
        {"trace_id": "b2c3d4e5f6789012345678901234abc2"},
        {"trace_id": "c3d4e5f6789012345678901234abc34"},
    ]
    with open(experiment_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # Create mock result
    class MockResult:
        _repr_dict = {"faithfulness": 0.85}
        cost_cb = None

        def to_pandas(self):
            return pd.DataFrame(
                {
                    "user_input": ["Q1", "Q2", "Q3"],
                    "faithfulness": [0.9, 0.8, 0.85],
                }
            )

        def total_tokens(self):
            class MockTokenUsage:
                input_tokens = 100
                output_tokens = 50

            return MockTokenUsage()

        def total_cost(self, **kwargs):
            return 0.001

    mock_result = MockResult()

    formatted = format_evaluation_scores(mock_result, 5.0 / 1e6, 15.0 / 1e6, str(experiment_file))

    # Verify trace_ids are preserved in individual results
    assert len(formatted.individual_results) == 3
    assert formatted.individual_results[0]["trace_id"] == "a1b2c3d4e5f6789012345678901234ab"
    assert formatted.individual_results[1]["trace_id"] == "b2c3d4e5f6789012345678901234abc2"
    assert formatted.individual_results[2]["trace_id"] == "c3d4e5f6789012345678901234abc34"


# TestMain tests
def test_main_no_metrics(experiment_data):
    """Test main function with no metrics provided"""

    tmp, original_cwd, experiment_file = experiment_data
    os.chdir(tmp)

    try:
        # When metrics is None, the function should raise an error
        # The actual error type depends on implementation
        with pytest.raises(ArgumentError):
            main(
                output_file="results/evaluation_scores.json",
                model="gemini-flash-latest",
                metrics=None,
            )
    finally:
        os.chdir(original_cwd)


def test_main_successful_execution(experiment_data, monkeypatch):
    """Test main function successful execution"""
    from pathlib import Path

    from ragas.dataset_schema import EvaluationResult

    tmp, original_cwd, experiment_file = experiment_data
    os.chdir(tmp)

    try:
        # Mock EvaluationDataset.from_jsonl
        class MockEvaluationDataset:
            pass

        mock_dataset = MockEvaluationDataset()

        def mock_from_jsonl(path):
            return mock_dataset

        # Mock the evaluate function
        class MockTokenUsage:
            input_tokens = 100
            output_tokens = 50

        class MockEvaluationResult(EvaluationResult):
            _repr_dict = {"faithfulness": 0.85}
            cost_cb = None

            def __init__(self):
                # Don't call super().__init__ to avoid initialization requirements
                pass

            def to_pandas(self):
                return pd.DataFrame({"user_input": ["Q1"], "faithfulness": [0.85]})

            def total_tokens(self):
                return MockTokenUsage()

            def total_cost(self, **kwargs):
                return 0.001

        mock_result = MockEvaluationResult()

        def mock_evaluate(dataset, metrics, llm, token_usage_parser):
            return mock_result

        # Mock ChatOpenAI and LangchainLLMWrapper
        class MockChatOpenAI:
            pass

        def mock_chat_openai_init(model, api_key):
            return MockChatOpenAI()

        class MockLLMWrapper:
            def __init__(self, llm):
                pass

        monkeypatch.setattr("evaluate.EvaluationDataset.from_jsonl", mock_from_jsonl)
        monkeypatch.setattr("evaluate.evaluate", mock_evaluate)
        monkeypatch.setattr("evaluate.ChatOpenAI", mock_chat_openai_init)
        monkeypatch.setattr("evaluate.LangchainLLMWrapper", MockLLMWrapper)

        # Get a valid metric name
        if not AVAILABLE_METRICS:
            pytest.skip("No metrics available")

        valid_metric = list(AVAILABLE_METRICS.keys())[0]

        # Run main
        output_file = "results/evaluation_scores.json"
        main(
            output_file=output_file,
            model="gemini-flash-latest",
            metrics=[valid_metric],
        )

        # Verify output file was created
        output_path = Path(tmp) / output_file
        assert output_path.exists(), f"Output file not found at {output_path}"

        # Verify output file has correct structure
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "overall_scores" in data
        assert "individual_results" in data
        assert "total_tokens" in data
        assert "total_cost" in data
    finally:
        os.chdir(original_cwd)


# TestAvailableMetrics tests
def test_available_metrics_loaded():
    """Test that AVAILABLE_METRICS is populated correctly"""
    # Should be a non-empty dictionary
    assert isinstance(AVAILABLE_METRICS, dict)
    assert len(AVAILABLE_METRICS) > 0

    # All keys should be strings
    for key in AVAILABLE_METRICS.keys():
        assert isinstance(key, str)

    # All values should be Metric instances
    for value in AVAILABLE_METRICS.values():
        assert isinstance(value, Metric)


# TestConvertMetrics tests
def test_convert_metrics_with_valid_metrics():
    """Test that convert_metrics correctly converts valid metric names to objects"""

    # Use metrics that are commonly available in RAGAS
    metric_names = ["faithfulness", "answer_relevancy"]

    # Only test with metrics that actually exist in AVAILABLE_METRICS
    available_names = [name for name in metric_names if name in AVAILABLE_METRICS]

    if not available_names:
        pytest.skip("Required metrics not available in this RAGAS version")

    metric_objects = convert_metrics(available_names)

    # Verify we got the right number of metrics
    assert len(metric_objects) == len(available_names)

    # Verify all returned objects are Metric instances
    for obj in metric_objects:
        assert isinstance(obj, Metric)


def test_convert_metrics_with_invalid_metrics():
    """Test that convert_metrics handles invalid metric names"""

    # Test with only invalid metrics - should raise ValueError
    with pytest.raises(ValueError, match="No valid metrics provided"):
        convert_metrics(["nonexistent_metric", "fake_metric"])


def test_convert_metrics_mixed_valid_invalid():
    """Test convert_metrics with mixed valid and invalid metric names"""

    # Get one valid metric name from AVAILABLE_METRICS
    if not AVAILABLE_METRICS:
        pytest.skip("No metrics available")

    valid_metric = list(AVAILABLE_METRICS.keys())[0]
    metric_names = [valid_metric, "nonexistent_metric", "fake_metric"]

    metric_objects = convert_metrics(metric_names)

    # Should only return the valid metric
    assert len(metric_objects) == 1

    assert isinstance(metric_objects[0], Metric)
