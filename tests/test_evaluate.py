"""
Unit tests for evaluate.py

Tests the RAGAS evaluation functionality.
"""

import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from ragas.metrics import Metric

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate import (
    AVAILABLE_METRIC_CLASSES,
    AVAILABLE_METRIC_INSTANCES,
    format_evaluation_scores,
    get_metric_by_name,
    instantiate_metric_from_class,
    load_metrics_config,
    main,
)


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
def test_main_no_config(experiment_data):
    """Test main function with missing metrics config file"""

    tmp, original_cwd, experiment_file = experiment_data
    os.chdir(tmp)

    try:
        # When config file doesn't exist, should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            main(
                output_file="results/evaluation_scores.json",
                model="gemini-flash-latest",
                metrics_config="nonexistent_config.json",
            )
    finally:
        os.chdir(original_cwd)


def test_main_successful_execution(experiment_data, monkeypatch, tmp_path):
    """Test main function successful execution with config file"""
    from pathlib import Path

    from ragas.dataset_schema import EvaluationResult

    tmp, original_cwd, experiment_file = experiment_data
    os.chdir(tmp)

    try:
        # Create a test config file
        config_file = tmp_path / "test_metrics.json"
        if not AVAILABLE_METRIC_INSTANCES:
            pytest.skip("No metric instances available")

        valid_metric = list(AVAILABLE_METRIC_INSTANCES.keys())[0]
        config = {"version": "1.0", "metrics": [{"type": "instance", "name": valid_metric}]}

        with open(config_file, 'w') as f:
            json.dump(config, f)

        # Mock EvaluationDataset.from_jsonl
        class MockEvaluationDataset:
            samples = []  # Add samples attribute for dataset type detection

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

        # Run main with config file
        output_file = "results/evaluation_scores.json"
        main(
            output_file=output_file,
            model="gemini-flash-latest",
            metrics_config=str(config_file),
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


# TestMetricDiscovery tests
def test_metric_discovery():
    """Test that both metric instances and classes are discovered"""
    # Test instances
    assert isinstance(AVAILABLE_METRIC_INSTANCES, dict)
    assert len(AVAILABLE_METRIC_INSTANCES) > 0
    for name, instance in AVAILABLE_METRIC_INSTANCES.items():
        assert isinstance(name, str)
        assert isinstance(instance, Metric)

    # Test classes
    assert isinstance(AVAILABLE_METRIC_CLASSES, dict)
    assert len(AVAILABLE_METRIC_CLASSES) > 0
    for name, cls in AVAILABLE_METRIC_CLASSES.items():
        assert isinstance(name, str)
        assert inspect.isclass(cls)
        assert issubclass(cls, Metric)


# Test get_metric_by_name
def test_get_metric_by_name_instance():
    """Test getting pre-configured metric instance"""
    if not AVAILABLE_METRIC_INSTANCES:
        pytest.skip("No metric instances available")

    # Get first available instance
    metric_name = list(AVAILABLE_METRIC_INSTANCES.keys())[0]
    metric = get_metric_by_name(metric_name)
    assert isinstance(metric, Metric)
    assert metric.name == metric_name


def test_get_metric_by_name_unknown():
    """Test error handling for unknown metric"""
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric_by_name('nonexistent_metric_xyz')


# Test instantiate_metric_from_class
def test_instantiate_metric_from_class_success():
    """Test successful class instantiation without parameters"""
    if not AVAILABLE_METRIC_CLASSES:
        pytest.skip("No metric classes available")

    # Find a class that can be instantiated without parameters
    for class_name, metric_class in AVAILABLE_METRIC_CLASSES.items():
        try:
            metric = instantiate_metric_from_class(class_name, {})
            assert isinstance(metric, Metric)
            return  # Success!
        except (TypeError, ValueError):
            continue  # Try next class
    pytest.skip("No metric classes can be instantiated without parameters")


def test_instantiate_metric_from_class_unknown():
    """Test error for unknown class"""
    with pytest.raises(ValueError, match="Unknown metric class"):
        instantiate_metric_from_class('NonexistentClass', {})


def test_instantiate_metric_from_class_invalid_params():
    """Test error for invalid parameters"""
    if not AVAILABLE_METRIC_CLASSES:
        pytest.skip("No metric classes available")

    # Use first available class with clearly invalid parameters
    class_name = list(AVAILABLE_METRIC_CLASSES.keys())[0]
    with pytest.raises(ValueError, match="Invalid parameters"):
        instantiate_metric_from_class(class_name, {'completely_invalid_param_name_xyz': 'value'})


# Test load_metrics_config
def test_load_metrics_config_json(tmp_path):
    """Test loading metrics from JSON config file"""
    if not AVAILABLE_METRIC_INSTANCES:
        pytest.skip("No metric instances available")

    config_file = tmp_path / "metrics.json"
    metric_name = list(AVAILABLE_METRIC_INSTANCES.keys())[0]

    config = {"version": "1.0", "metrics": [{"type": "instance", "name": metric_name}]}

    with open(config_file, 'w') as f:
        json.dump(config, f)

    metrics = load_metrics_config(str(config_file))
    assert len(metrics) == 1
    assert isinstance(metrics[0], Metric)
    assert metrics[0].name == metric_name


def test_load_metrics_config_with_class(tmp_path):
    """Test loading metrics with class instantiation"""
    if not AVAILABLE_METRIC_CLASSES:
        pytest.skip("No metric classes available")

    # Find a class that can be instantiated without parameters
    for class_name in AVAILABLE_METRIC_CLASSES.keys():
        try:
            # Test if this class can be instantiated
            instantiate_metric_from_class(class_name, {})

            config_file = tmp_path / "metrics.json"
            config = {
                "version": "1.0",
                "metrics": [{"type": "class", "class_name": class_name, "parameters": {}}],
            }

            with open(config_file, 'w') as f:
                json.dump(config, f)

            metrics = load_metrics_config(str(config_file))
            assert len(metrics) == 1
            assert isinstance(metrics[0], Metric)
            return  # Success!
        except (TypeError, ValueError):
            continue  # Try next class

    pytest.skip("No metric classes can be instantiated without parameters")


def test_load_metrics_config_invalid_format(tmp_path):
    """Test error for invalid file format"""
    config_file = tmp_path / "metrics.txt"
    config_file.write_text("invalid")

    with pytest.raises(ValueError, match="Unsupported config file format"):
        load_metrics_config(str(config_file))


def test_load_metrics_config_missing_metrics_key(tmp_path):
    """Test error for missing 'metrics' key"""
    config_file = tmp_path / "metrics.json"

    with open(config_file, 'w') as f:
        json.dump({"version": "1.0"}, f)

    with pytest.raises(ValueError, match="must contain 'metrics' key"):
        load_metrics_config(str(config_file))


def test_load_metrics_config_empty_metrics(tmp_path):
    """Test error for empty metrics list"""
    config_file = tmp_path / "metrics.json"

    config = {"version": "1.0", "metrics": []}

    with open(config_file, 'w') as f:
        json.dump(config, f)

    with pytest.raises(ValueError, match="contains no valid metrics"):
        load_metrics_config(str(config_file))
