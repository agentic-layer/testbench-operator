"""
Unit tests for publish.py

Tests the OpenTelemetry OTLP metrics publishing functionality.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from publish import create_and_push_metrics, get_overall_scores, publish_metrics


class TestGetOverallScores(unittest.TestCase):
    """Test the get_overall_scores function"""

    def setUp(self):
        """Set up temporary directory and test file"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "evaluation_scores.json"

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_loads_overall_scores(self):
        """Test that get_overall_scores loads the overall_scores section"""
        test_data = {
            "overall_scores": {"faithfulness": 0.85, "answer_relevancy": 0.90},
            "individual_results": [],
            "total_tokens": {"input_tokens": 0, "output_tokens": 0},
            "total_cost": 0.0,
        }

        with open(self.test_file, "w") as f:
            json.dump(test_data, f)

        scores = get_overall_scores(str(self.test_file))

        self.assertEqual(scores["faithfulness"], 0.85)
        self.assertEqual(scores["answer_relevancy"], 0.90)

    def test_file_not_found(self):
        """Test behavior when file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            get_overall_scores(str(Path(self.temp_dir) / "nonexistent.json"))


class TestCreateAndPushMetrics(unittest.TestCase):
    """Test the create_and_push_metrics function"""

    @patch("publish.OTLPMetricExporter")
    @patch("publish.MeterProvider")
    @patch("publish.metrics.get_meter")
    def test_creates_gauges_for_each_metric(self, mock_get_meter, mock_provider_class, mock_exporter_class):
        """Test that a Gauge is created for each metric"""
        overall_scores = {"faithfulness": 0.85, "answer_relevancy": 0.90}

        # Mock the meter and gauge
        mock_gauge = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = mock_gauge
        mock_get_meter.return_value = mock_meter

        # Mock the provider
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        create_and_push_metrics(
            overall_scores=overall_scores,
            workflow_name="test-workflow",
            otlp_endpoint="localhost:4318",
        )

        # Verify create_gauge was called for each metric
        self.assertEqual(mock_meter.create_gauge.call_count, 2)

        # Verify gauge names
        gauge_calls = mock_meter.create_gauge.call_args_list
        gauge_names = [call[1]["name"] for call in gauge_calls]
        self.assertIn("ragas_evaluation_faithfulness", gauge_names)
        self.assertIn("ragas_evaluation_answer_relevancy", gauge_names)

    @patch("publish.OTLPMetricExporter")
    @patch("publish.MeterProvider")
    @patch("publish.metrics.get_meter")
    def test_sets_gauge_values(self, mock_get_meter, mock_provider_class, mock_exporter_class):
        """Test that gauge values are set correctly"""
        overall_scores = {"faithfulness": 0.85}

        # Mock the meter and gauge
        mock_gauge = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = mock_gauge
        mock_get_meter.return_value = mock_meter

        # Mock the provider
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        create_and_push_metrics(
            overall_scores=overall_scores,
            workflow_name="test-workflow",
            otlp_endpoint="localhost:4318",
        )

        # Verify gauge.set was called with correct value and attributes
        mock_gauge.set.assert_called_once_with(0.85, {"workflow_name": "test-workflow"})

    @patch("publish.OTLPMetricExporter")
    @patch("publish.MeterProvider")
    @patch("publish.metrics.get_meter")
    def test_pushes_via_otlp(self, mock_get_meter, mock_provider_class, mock_exporter_class):
        """Test that metrics are pushed via OTLP"""
        overall_scores = {"faithfulness": 0.85}

        # Mock the meter and gauge
        mock_gauge = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = mock_gauge
        mock_get_meter.return_value = mock_meter

        # Mock the provider
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        create_and_push_metrics(
            overall_scores=overall_scores,
            workflow_name="test-workflow",
            otlp_endpoint="localhost:4318",
        )

        # Verify OTLPMetricExporter was initialized with correct endpoint
        mock_exporter_class.assert_called_once_with(endpoint="http://localhost:4318/v1/metrics")

        # Verify force_flush and shutdown were called
        mock_provider.force_flush.assert_called_once()
        mock_provider.shutdown.assert_called_once()

    @patch("publish.OTLPMetricExporter")
    @patch("publish.MeterProvider")
    @patch("publish.metrics.get_meter")
    def test_handles_push_error(self, mock_get_meter, mock_provider_class, mock_exporter_class):
        """Test error handling when OTLP export fails"""
        # Mock the provider to raise an exception on force_flush
        mock_provider = MagicMock()
        mock_provider.force_flush.side_effect = Exception("Connection refused")
        mock_provider_class.return_value = mock_provider

        # Mock the meter
        mock_gauge = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = mock_gauge
        mock_get_meter.return_value = mock_meter

        overall_scores = {"faithfulness": 0.85}

        with self.assertRaises(Exception) as context:
            create_and_push_metrics(
                overall_scores=overall_scores,
                workflow_name="test-workflow",
                otlp_endpoint="localhost:4318",
            )

        self.assertIn("Connection refused", str(context.exception))

        # Verify shutdown is still called in finally block
        mock_provider.shutdown.assert_called_once()


class TestPublishMetrics(unittest.TestCase):
    """Test the publish_metrics function"""

    def setUp(self):
        """Set up temporary directory and test file"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "evaluation_scores.json"

        # Create test evaluation scores file
        test_data = {
            "overall_scores": {"faithfulness": 0.85, "answer_relevancy": 0.90},
            "individual_results": [],
            "total_tokens": {"input_tokens": 0, "output_tokens": 0},
            "total_cost": 0.0,
        }

        with open(self.test_file, "w") as f:
            json.dump(test_data, f)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("publish.create_and_push_metrics")
    def test_publish_metrics_calls_create_and_push(self, mock_create_push):
        """Test that publish_metrics calls create_and_push_metrics"""
        publish_metrics(
            input_file=str(self.test_file),
            workflow_name="test-workflow",
            otlp_endpoint="localhost:4318",
        )

        # Verify create_and_push_metrics was called
        mock_create_push.assert_called_once()

        # Verify parameters (positional arguments)
        call_args = mock_create_push.call_args
        self.assertEqual(call_args[0][0]["faithfulness"], 0.85)
        self.assertEqual(call_args[0][0]["answer_relevancy"], 0.90)
        self.assertEqual(call_args[0][1], "test-workflow")
        self.assertEqual(call_args[0][2], "localhost:4318")

    @patch("publish.create_and_push_metrics")
    def test_publish_metrics_with_empty_scores(self, mock_create_push):
        """Test behavior when overall_scores is empty"""
        # Create file with empty overall_scores
        test_data = {"overall_scores": {}, "individual_results": []}

        empty_file = Path(self.temp_dir) / "empty_scores.json"
        with open(empty_file, "w") as f:
            json.dump(test_data, f)

        publish_metrics(
            input_file=str(empty_file),
            workflow_name="test-workflow",
            otlp_endpoint="localhost:4318",
        )

        # Verify create_and_push_metrics was NOT called
        mock_create_push.assert_not_called()


class TestIntegrationWithTestData(unittest.TestCase):
    """Integration tests with realistic evaluation data"""

    def setUp(self):
        """Set up temporary directory with realistic test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "evaluation_scores.json"

        # Create realistic evaluation scores
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

        with open(self.test_file, "w") as f:
            json.dump(test_data, f, indent=2)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("publish.OTLPMetricExporter")
    @patch("publish.MeterProvider")
    @patch("publish.metrics.get_meter")
    def test_publish_realistic_scores(self, mock_get_meter, mock_provider_class, mock_exporter_class):
        """Test publishing realistic evaluation scores"""
        # Mock the meter and gauge
        mock_gauge = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = mock_gauge
        mock_get_meter.return_value = mock_meter

        # Mock the provider
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        publish_metrics(
            input_file=str(self.test_file),
            workflow_name="weather-assistant-test",
            otlp_endpoint="localhost:4318",
        )

        # Verify OTLPMetricExporter was called
        mock_exporter_class.assert_called_once()

        # Verify 4 metrics were created (faithfulness, answer_relevancy, context_precision, context_recall)
        self.assertEqual(mock_meter.create_gauge.call_count, 4)


if __name__ == "__main__":
    unittest.main()
