"""
Unit tests for publish.py

Tests the OpenTelemetry OTLP metrics publishing functionality.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from publish import create_and_push_metrics, get_overall_scores, publish_metrics


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def evaluation_scores_file(temp_dir):
    """Create a test evaluation scores file"""
    test_file = Path(temp_dir) / "evaluation_scores.json"
    test_data = {
        "overall_scores": {"faithfulness": 0.85, "answer_relevancy": 0.90},
        "individual_results": [],
        "total_tokens": {"input_tokens": 0, "output_tokens": 0},
        "total_cost": 0.0,
    }

    with open(test_file, "w") as f:
        json.dump(test_data, f)

    return test_file


@pytest.fixture
def realistic_scores_file(temp_dir):
    """Create a realistic evaluation scores file"""
    test_file = Path(temp_dir) / "evaluation_scores.json"
    test_data = {
        "overall_scores": {
            "faithfulness": 0.8523,
            "answer_relevancy": 0.9012,
            "context_precision": 0.7845,
            "context_recall": 0.8234,
        },
        "individual_results": [
            {
                "user_input": "What is the weather?",
                "response": "It is sunny.",
                "faithfulness": 0.85,
                "answer_relevancy": 0.90,
            }
        ],
        "total_tokens": {"input_tokens": 0, "output_tokens": 0},
        "total_cost": 0.0,
    }

    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=2)

    return test_file


# TestGetOverallScores tests
def test_loads_overall_scores(evaluation_scores_file):
    """Test that get_overall_scores loads the overall_scores section"""
    scores = get_overall_scores(str(evaluation_scores_file))

    assert scores["faithfulness"] == 0.85
    assert scores["answer_relevancy"] == 0.90


def test_file_not_found(temp_dir):
    """Test behavior when file doesn't exist"""
    with pytest.raises(FileNotFoundError):
        get_overall_scores(str(Path(temp_dir) / "nonexistent.json"))


# TestCreateAndPushMetrics tests
def test_creates_gauges_for_each_metric(monkeypatch):
    """Test that a Gauge is created for each metric"""
    overall_scores = {"faithfulness": 0.85, "answer_relevancy": 0.90}

    # Mock the meter and gauge
    create_gauge_calls = []

    class MockGauge:
        def set(self, value, attributes):
            pass

    class MockMeter:
        def create_gauge(self, name, unit=None, description=None):
            create_gauge_calls.append({"name": name, "unit": unit, "description": description})
            return MockGauge()

    mock_meter = MockMeter()

    def mock_get_meter(*args, **kwargs):
        return mock_meter

    # Mock the provider
    class MockProvider:
        def force_flush(self):
            pass

        def shutdown(self):
            pass

    def mock_provider_init(**kwargs):
        return MockProvider()

    # Mock the exporter
    class MockExporter:
        _preferred_temporality = {}
        _preferred_aggregation = {}

    def mock_exporter_init(endpoint):
        return MockExporter()

    monkeypatch.setattr("publish.metrics.get_meter", mock_get_meter)
    monkeypatch.setattr("publish.MeterProvider", mock_provider_init)
    monkeypatch.setattr("publish.OTLPMetricExporter", mock_exporter_init)

    create_and_push_metrics(
        overall_scores=overall_scores,
        workflow_name="test-workflow",
        otlp_endpoint="localhost:4318",
    )

    # Verify create_gauge was called for each metric
    assert len(create_gauge_calls) == 2

    # Verify gauge names
    gauge_names = [call["name"] for call in create_gauge_calls]
    assert "ragas_evaluation_faithfulness" in gauge_names
    assert "ragas_evaluation_answer_relevancy" in gauge_names


def test_sets_gauge_values(monkeypatch):
    """Test that gauge values are set correctly"""
    overall_scores = {"faithfulness": 0.85}

    # Mock the meter and gauge
    set_calls = []

    class MockGauge:
        def set(self, value, attributes):
            set_calls.append({"value": value, "attributes": attributes})

    class MockMeter:
        def create_gauge(self, name, unit=None, description=None):
            return MockGauge()

    mock_meter = MockMeter()

    def mock_get_meter(*args, **kwargs):
        return mock_meter

    # Mock the provider
    class MockProvider:
        def force_flush(self):
            pass

        def shutdown(self):
            pass

    def mock_provider_init(**kwargs):
        return MockProvider()

    # Mock the exporter
    class MockExporter:
        _preferred_temporality = {}
        _preferred_aggregation = {}

    def mock_exporter_init(endpoint):
        return MockExporter()

    monkeypatch.setattr("publish.metrics.get_meter", mock_get_meter)
    monkeypatch.setattr("publish.MeterProvider", mock_provider_init)
    monkeypatch.setattr("publish.OTLPMetricExporter", mock_exporter_init)

    create_and_push_metrics(
        overall_scores=overall_scores,
        workflow_name="test-workflow",
        otlp_endpoint="localhost:4318",
    )

    # Verify gauge.set was called with correct value and attributes
    assert len(set_calls) == 1
    assert set_calls[0]["value"] == 0.85
    assert set_calls[0]["attributes"] == {"workflow_name": "test-workflow"}


def test_pushes_via_otlp(monkeypatch):
    """Test that metrics are pushed via OTLP"""
    overall_scores = {"faithfulness": 0.85}

    # Mock the meter and gauge
    class MockGauge:
        def set(self, value, attributes):
            pass

    class MockMeter:
        def create_gauge(self, name, unit=None, description=None):
            return MockGauge()

    mock_meter = MockMeter()

    def mock_get_meter(*args, **kwargs):
        return mock_meter

    # Mock the provider
    force_flush_calls = []
    shutdown_calls = []

    class MockProvider:
        def force_flush(self):
            force_flush_calls.append(True)

        def shutdown(self):
            shutdown_calls.append(True)

    def mock_provider_init(**kwargs):
        return MockProvider()

    # Mock the exporter
    class MockExporter:
        _preferred_temporality = {}
        _preferred_aggregation = {}

    exporter_calls = []

    def mock_exporter_init(endpoint):
        exporter_calls.append({"endpoint": endpoint})
        return MockExporter()

    monkeypatch.setattr("publish.metrics.get_meter", mock_get_meter)
    monkeypatch.setattr("publish.MeterProvider", mock_provider_init)
    monkeypatch.setattr("publish.OTLPMetricExporter", mock_exporter_init)

    create_and_push_metrics(
        overall_scores=overall_scores,
        workflow_name="test-workflow",
        otlp_endpoint="localhost:4318",
    )

    # Verify OTLPMetricExporter was initialized with correct endpoint
    assert len(exporter_calls) == 1
    assert exporter_calls[0]["endpoint"] == "http://localhost:4318/v1/metrics"

    # Verify force_flush and shutdown were called
    assert len(force_flush_calls) == 1
    assert len(shutdown_calls) == 1


def test_handles_push_error(monkeypatch):
    """Test error handling when OTLP export fails"""
    overall_scores = {"faithfulness": 0.85}

    # Mock the meter and gauge
    class MockGauge:
        def set(self, value, attributes):
            pass

    class MockMeter:
        def create_gauge(self, name, unit=None, description=None):
            return MockGauge()

    mock_meter = MockMeter()

    def mock_get_meter(*args, **kwargs):
        return mock_meter

    # Mock the provider to raise an exception on force_flush
    shutdown_calls = []

    class MockProvider:
        def force_flush(self):
            raise Exception("Connection refused")

        def shutdown(self):
            shutdown_calls.append(True)

    def mock_provider_init(**kwargs):
        return MockProvider()

    # Mock the exporter
    class MockExporter:
        _preferred_temporality = {}
        _preferred_aggregation = {}

    def mock_exporter_init(endpoint):
        return MockExporter()

    monkeypatch.setattr("publish.metrics.get_meter", mock_get_meter)
    monkeypatch.setattr("publish.MeterProvider", mock_provider_init)
    monkeypatch.setattr("publish.OTLPMetricExporter", mock_exporter_init)

    with pytest.raises(Exception, match="Connection refused"):
        create_and_push_metrics(
            overall_scores=overall_scores,
            workflow_name="test-workflow",
            otlp_endpoint="localhost:4318",
        )

    # Verify shutdown is still called in finally block
    assert len(shutdown_calls) == 1


# TestPublishMetrics tests
def test_publish_metrics_calls_create_and_push(evaluation_scores_file, monkeypatch):
    """Test that publish_metrics calls create_and_push_metrics"""
    create_push_calls = []

    def mock_create_push(overall_scores, workflow_name, otlp_endpoint):
        create_push_calls.append(
            {
                "overall_scores": overall_scores,
                "workflow_name": workflow_name,
                "otlp_endpoint": otlp_endpoint,
            }
        )

    monkeypatch.setattr("publish.create_and_push_metrics", mock_create_push)

    publish_metrics(
        input_file=str(evaluation_scores_file),
        workflow_name="test-workflow",
        otlp_endpoint="localhost:4318",
    )

    # Verify create_and_push_metrics was called
    assert len(create_push_calls) == 1

    # Verify parameters
    assert create_push_calls[0]["overall_scores"]["faithfulness"] == 0.85
    assert create_push_calls[0]["overall_scores"]["answer_relevancy"] == 0.90
    assert create_push_calls[0]["workflow_name"] == "test-workflow"
    assert create_push_calls[0]["otlp_endpoint"] == "localhost:4318"


def test_publish_metrics_with_empty_scores(temp_dir, monkeypatch):
    """Test behavior when overall_scores is empty"""
    # Create file with empty overall_scores
    test_data = {"overall_scores": {}, "individual_results": []}

    empty_file = Path(temp_dir) / "empty_scores.json"
    with open(empty_file, "w") as f:
        json.dump(test_data, f)

    create_push_calls = []

    def mock_create_push(overall_scores, workflow_name, otlp_endpoint):
        create_push_calls.append(True)

    monkeypatch.setattr("publish.create_and_push_metrics", mock_create_push)

    publish_metrics(
        input_file=str(empty_file),
        workflow_name="test-workflow",
        otlp_endpoint="localhost:4318",
    )

    # Verify create_and_push_metrics was NOT called
    assert len(create_push_calls) == 0


# TestIntegrationWithTestData tests
def test_publish_realistic_scores(realistic_scores_file, monkeypatch):
    """Test publishing realistic evaluation scores"""
    # Mock the meter and gauge
    create_gauge_calls = []

    class MockGauge:
        def set(self, value, attributes):
            pass

    class MockMeter:
        def create_gauge(self, name, unit=None, description=None):
            create_gauge_calls.append({"name": name})
            return MockGauge()

    mock_meter = MockMeter()

    def mock_get_meter(*args, **kwargs):
        return mock_meter

    # Mock the provider
    class MockProvider:
        def force_flush(self):
            pass

        def shutdown(self):
            pass

    def mock_provider_init(**kwargs):
        return MockProvider()

    # Mock the exporter
    class MockExporter:
        _preferred_temporality = {}
        _preferred_aggregation = {}

    exporter_calls = []

    def mock_exporter_init(endpoint):
        exporter_calls.append(True)
        return MockExporter()

    monkeypatch.setattr("publish.metrics.get_meter", mock_get_meter)
    monkeypatch.setattr("publish.MeterProvider", mock_provider_init)
    monkeypatch.setattr("publish.OTLPMetricExporter", mock_exporter_init)

    publish_metrics(
        input_file=str(realistic_scores_file),
        workflow_name="weather-assistant-test",
        otlp_endpoint="localhost:4318",
    )

    # Verify OTLPMetricExporter was called
    assert len(exporter_calls) == 1

    # Verify 4 metrics were created (faithfulness, answer_relevancy, context_precision, context_recall)
    assert len(create_gauge_calls) == 4
