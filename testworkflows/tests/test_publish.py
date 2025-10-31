"""
Unit tests for publish.py

Tests the Prometheus metrics publishing functionality.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from prometheus_client import CollectorRegistry, Gauge

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from publish import (
    get_overall_scores,
    create_and_push_metrics,
    publish_metrics
)


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
            "overall_scores": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.90
            },
            "individual_results": [],
            "total_tokens": {"input_tokens": 0, "output_tokens": 0},
            "total_cost": 0.0
        }

        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)

        scores = get_overall_scores(str(self.test_file))

        self.assertEqual(scores['faithfulness'], 0.85)
        self.assertEqual(scores['answer_relevancy'], 0.90)

    def test_returns_empty_dict_if_no_overall_scores(self):
        """Test that empty dict is returned if overall_scores is missing"""
        test_data = {
            "individual_results": [],
            "total_tokens": {"input_tokens": 0, "output_tokens": 0}
        }

        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)

        scores = get_overall_scores(str(self.test_file))

        self.assertEqual(scores, {})

    def test_file_not_found(self):
        """Test behavior when file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            get_overall_scores(str(Path(self.temp_dir) / "nonexistent.json"))


class TestCreateAndPushMetrics(unittest.TestCase):
    """Test the create_and_push_metrics function"""

    @patch('publish.push_to_gateway')
    @patch('publish.Gauge')
    def test_creates_gauges_for_each_metric(self, mock_gauge_class, mock_push):
        """Test that a Gauge is created for each metric"""
        overall_scores = {
            'faithfulness': 0.85,
            'answer_relevancy': 0.90
        }

        # Mock the Gauge instances
        mock_gauge_instance = MagicMock()
        mock_gauge_class.return_value = mock_gauge_instance

        create_and_push_metrics(
            overall_scores=overall_scores,
            workflow_name="test-workflow",
            pushgateway_url="localhost:9091"
        )

        # Verify Gauge was called for each metric
        self.assertEqual(mock_gauge_class.call_count, 2)

        # Verify gauge names
        gauge_calls = mock_gauge_class.call_args_list
        gauge_names = [call[0][0] for call in gauge_calls]
        self.assertIn('ragas_evaluation_faithfulness', gauge_names)
        self.assertIn('ragas_evaluation_answer_relevancy', gauge_names)

    @patch('publish.push_to_gateway')
    @patch('publish.Gauge')
    def test_sets_gauge_values(self, mock_gauge_class, mock_push):
        """Test that gauge values are set correctly"""
        overall_scores = {
            'faithfulness': 0.85
        }

        # Mock the Gauge instance and labels
        mock_gauge_instance = MagicMock()
        mock_labels = MagicMock()
        mock_gauge_instance.labels.return_value = mock_labels
        mock_gauge_class.return_value = mock_gauge_instance

        create_and_push_metrics(
            overall_scores=overall_scores,
            workflow_name="test-workflow",
            pushgateway_url="localhost:9091"
        )

        # Verify labels were set with workflow_name
        mock_gauge_instance.labels.assert_called_with(workflow_name="test-workflow")

        # Verify gauge value was set
        mock_labels.set.assert_called_with(0.85)

    @patch('publish.push_to_gateway')
    def test_pushes_to_gateway(self, mock_push):
        """Test that metrics are pushed to Pushgateway"""
        overall_scores = {
            'faithfulness': 0.85
        }

        create_and_push_metrics(
            overall_scores=overall_scores,
            workflow_name="test-workflow",
            pushgateway_url="localhost:9091"
        )

        # Verify push_to_gateway was called
        mock_push.assert_called_once()

        # Verify push parameters
        call_args = mock_push.call_args
        self.assertEqual(call_args[0][0], "localhost:9091")
        self.assertEqual(call_args[1]['job'], 'ragas_evaluation')

    @patch('publish.push_to_gateway')
    def test_handles_push_error(self, mock_push):
        """Test error handling when push fails"""
        mock_push.side_effect = Exception("Connection refused")

        overall_scores = {
            'faithfulness': 0.85
        }

        with self.assertRaises(Exception) as context:
            create_and_push_metrics(
                overall_scores=overall_scores,
                workflow_name="test-workflow",
                pushgateway_url="localhost:9091"
            )

        self.assertIn("Connection refused", str(context.exception))


class TestPublishMetrics(unittest.TestCase):
    """Test the publish_metrics function"""

    def setUp(self):
        """Set up temporary directory and test file"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "evaluation_scores.json"

        # Create test evaluation scores file
        test_data = {
            "overall_scores": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.90
            },
            "individual_results": [],
            "total_tokens": {"input_tokens": 0, "output_tokens": 0},
            "total_cost": 0.0
        }

        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('publish.create_and_push_metrics')
    def test_publish_metrics_calls_create_and_push(self, mock_create_push):
        """Test that publish_metrics calls create_and_push_metrics"""
        publish_metrics(
            input_file=str(self.test_file),
            workflow_name="test-workflow",
            pushgateway_url="localhost:9091"
        )

        # Verify create_and_push_metrics was called
        mock_create_push.assert_called_once()

        # Verify parameters (positional arguments)
        call_args = mock_create_push.call_args
        self.assertEqual(call_args[0][0]['faithfulness'], 0.85)
        self.assertEqual(call_args[0][0]['answer_relevancy'], 0.90)
        self.assertEqual(call_args[0][1], "test-workflow")
        self.assertEqual(call_args[0][2], "localhost:9091")

    @patch('publish.create_and_push_metrics')
    def test_publish_metrics_with_empty_scores(self, mock_create_push):
        """Test behavior when overall_scores is empty"""
        # Create file with empty overall_scores
        test_data = {
            "overall_scores": {},
            "individual_results": []
        }

        empty_file = Path(self.temp_dir) / "empty_scores.json"
        with open(empty_file, 'w') as f:
            json.dump(test_data, f)

        publish_metrics(
            input_file=str(empty_file),
            workflow_name="test-workflow",
            pushgateway_url="localhost:9091"
        )

        # Verify create_and_push_metrics was NOT called
        mock_create_push.assert_not_called()

    @patch('publish.create_and_push_metrics')
    def test_publish_metrics_default_pushgateway_url(self, mock_create_push):
        """Test that default pushgateway URL is used when not provided"""
        publish_metrics(
            input_file=str(self.test_file),
            workflow_name="test-workflow",
            pushgateway_url="localhost:9091"
        )

        # Verify the URL was passed correctly (third positional argument)
        call_args = mock_create_push.call_args
        self.assertEqual(call_args[0][2], "localhost:9091")


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
                "context_recall": 0.8234
            },
            "individual_results": [
                {
                    "user_input": "What is the weather?",
                    "response": "It is sunny.",
                    "faithfulness": 0.85,
                    "answer_relevancy": 0.90
                }
            ],
            "total_tokens": {
                "input_tokens": 0,
                "output_tokens": 0
            },
            "total_cost": 0.0
        }

        with open(self.test_file, 'w') as f:
            json.dump(test_data, f, indent=2)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('publish.push_to_gateway')
    def test_publish_realistic_scores(self, mock_push):
        """Test publishing realistic evaluation scores"""
        publish_metrics(
            input_file=str(self.test_file),
            workflow_name="weather-assistant-test",
            pushgateway_url="localhost:9091"
        )

        # Verify push was called
        mock_push.assert_called_once()

    def test_load_realistic_scores(self):
        """Test loading realistic evaluation scores"""
        scores = get_overall_scores(str(self.test_file))

        # Verify all scores are present
        self.assertIn('faithfulness', scores)
        self.assertIn('answer_relevancy', scores)
        self.assertIn('context_precision', scores)
        self.assertIn('context_recall', scores)

        # Verify values are floats
        for score in scores.values():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestPrometheusGaugeCreation(unittest.TestCase):
    """Test Prometheus Gauge creation and configuration"""

    def test_gauge_with_labels(self):
        """Test creating a Gauge with labels"""
        registry = CollectorRegistry()

        gauge = Gauge(
            'ragas_evaluation_test_metric',
            'Test metric description',
            labelnames=['workflow_name'],
            registry=registry
        )

        # Set a value with label
        gauge.labels(workflow_name='test-workflow').set(0.85)

        # Verify the gauge was created
        self.assertIsNotNone(gauge)

    def test_multiple_gauges_in_registry(self):
        """Test creating multiple gauges in the same registry"""
        registry = CollectorRegistry()

        gauge1 = Gauge(
            'ragas_evaluation_faithfulness',
            'Faithfulness score',
            labelnames=['workflow_name'],
            registry=registry
        )

        gauge2 = Gauge(
            'ragas_evaluation_answer_relevancy',
            'Answer relevancy score',
            labelnames=['workflow_name'],
            registry=registry
        )

        # Set values
        gauge1.labels(workflow_name='test').set(0.85)
        gauge2.labels(workflow_name='test').set(0.90)

        # Both gauges should be in the registry
        metrics = list(registry.collect())
        self.assertGreaterEqual(len(metrics), 2)


if __name__ == '__main__':
    unittest.main()
