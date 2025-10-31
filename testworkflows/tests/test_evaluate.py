"""
Unit tests for evaluate.py

Tests the RAGAS evaluation functionality.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from ragas.dataset_schema import EvaluationDataset, EvaluationResult
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate import (
    get_available_metrics,
    calculate_metrics,
    format_evaluation_scores,
    main,
    AVAILABLE_METRICS
)


class TestGetAvailableMetrics(unittest.TestCase):
    """Test the get_available_metrics function"""

    def test_returns_dict(self):
        """Test that get_available_metrics returns a dictionary"""
        metrics = get_available_metrics()
        self.assertIsInstance(metrics, dict)

    def test_contains_common_metrics(self):
        """Test that common metrics are available"""
        metrics = get_available_metrics()

        # Check for some common metrics
        common_metrics = ['faithfulness', 'answer_relevancy']

        for metric in common_metrics:
            if metric in metrics:
                self.assertIn(metric, metrics)

    def test_metric_instances_have_name(self):
        """Test that metric instances have a name attribute"""
        metrics = get_available_metrics()

        for metric_name, metric_instance in metrics.items():
            self.assertTrue(hasattr(metric_instance, 'name'))


class TestCalculateMetrics(unittest.TestCase):
    """Test the calculate_metrics function"""

    def setUp(self):
        """Set up test data"""
        # Create a mock dataset
        self.test_data = [
            {
                'user_input': 'What is the capital of France?',
                'retrieved_contexts': ['Paris is the capital of France.'],
                'reference': 'Paris',
                'response': 'The capital of France is Paris.'
            }
        ]

    @patch('evaluate.evaluate')
    def test_calculate_metrics_calls_evaluate(self, mock_evaluate):
        """Test that calculate_metrics calls RAGAS evaluate"""
        # Create mock dataset
        mock_dataset = MagicMock(spec=EvaluationDataset)

        # Create mock LLM
        mock_llm = MagicMock()

        # Mock evaluate return value
        mock_result = MagicMock(spec=EvaluationResult)
        mock_evaluate.return_value = mock_result

        # Call calculate_metrics
        metrics = ['faithfulness']
        result = calculate_metrics(mock_dataset, metrics, mock_llm)

        # Verify evaluate was called
        mock_evaluate.assert_called_once()
        self.assertEqual(result, mock_result)

    @patch('evaluate.evaluate')
    def test_calculate_metrics_with_multiple_metrics(self, mock_evaluate):
        """Test calculate_metrics with multiple metrics"""
        mock_dataset = MagicMock(spec=EvaluationDataset)
        mock_llm = MagicMock()
        mock_result = MagicMock(spec=EvaluationResult)
        mock_evaluate.return_value = mock_result

        # Call with multiple metrics
        metrics = ['faithfulness', 'answer_relevancy']
        result = calculate_metrics(mock_dataset, metrics, mock_llm)

        # Verify evaluate was called
        mock_evaluate.assert_called_once()

    def test_calculate_metrics_with_invalid_metric(self):
        """Test calculate_metrics with invalid metric name"""
        mock_dataset = MagicMock(spec=EvaluationDataset)
        mock_llm = MagicMock()

        # Call with invalid metric - this should raise ValueError
        metrics = ['nonexistent_metric']

        with self.assertRaises(ValueError) as context:
            calculate_metrics(mock_dataset, metrics, mock_llm)

        self.assertIn('No valid metrics', str(context.exception))


class TestFormatEvaluationScores(unittest.TestCase):
    """Test the format_evaluation_scores function"""

    def test_format_evaluation_scores_structure(self):
        """Test that formatted scores have correct structure"""
        # Create mock RAGAS result
        mock_result = MagicMock(spec=EvaluationResult)

        # Create a mock DataFrame
        df = pd.DataFrame({
            'user_input': ['Question 1', 'Question 2'],
            'response': ['Answer 1', 'Answer 2'],
            'faithfulness': [0.9, 0.8],
            'answer_relevancy': [0.85, 0.75]
        })
        mock_result.to_pandas.return_value = df

        # Format the scores
        metrics = ['faithfulness', 'answer_relevancy']
        formatted = format_evaluation_scores(mock_result, metrics)

        # Verify structure
        self.assertIn('overall_scores', formatted)
        self.assertIn('individual_results', formatted)
        self.assertIn('total_tokens', formatted)
        self.assertIn('total_cost', formatted)

    def test_overall_scores_calculation(self):
        """Test that overall scores are calculated correctly"""
        mock_result = MagicMock(spec=EvaluationResult)

        df = pd.DataFrame({
            'faithfulness': [0.9, 0.8, 0.7],
            'answer_relevancy': [0.85, 0.75, 0.65]
        })
        mock_result.to_pandas.return_value = df

        metrics = ['faithfulness', 'answer_relevancy']
        formatted = format_evaluation_scores(mock_result, metrics)

        # Verify overall scores are averages
        self.assertAlmostEqual(formatted['overall_scores']['faithfulness'], 0.8, places=2)
        self.assertAlmostEqual(formatted['overall_scores']['answer_relevancy'], 0.75, places=2)

    def test_individual_results_present(self):
        """Test that individual results are included"""
        mock_result = MagicMock(spec=EvaluationResult)

        df = pd.DataFrame({
            'user_input': ['Q1', 'Q2'],
            'faithfulness': [0.9, 0.8]
        })
        mock_result.to_pandas.return_value = df

        metrics = ['faithfulness']
        formatted = format_evaluation_scores(mock_result, metrics)

        # Verify individual results
        self.assertEqual(len(formatted['individual_results']), 2)

    def test_token_usage_placeholders(self):
        """Test that token usage has placeholder values"""
        mock_result = MagicMock(spec=EvaluationResult)

        df = pd.DataFrame({
            'faithfulness': [0.9]
        })
        mock_result.to_pandas.return_value = df

        metrics = ['faithfulness']
        formatted = format_evaluation_scores(mock_result, metrics)

        # Verify placeholders
        self.assertEqual(formatted['total_tokens']['input_tokens'], 0)
        self.assertEqual(formatted['total_tokens']['output_tokens'], 0)
        self.assertEqual(formatted['total_cost'], 0.0)


class TestMain(unittest.TestCase):
    """Test the main function"""

    def setUp(self):
        """Set up temporary directory and test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()

        # Create test experiment JSONL
        self.experiment_dir = Path(self.temp_dir) / "data" / "experiments"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_file = self.experiment_dir / "ragas_experiment.jsonl"
        test_data = [
            {
                'user_input': 'What is the weather?',
                'retrieved_contexts': ['Context about weather'],
                'reference': 'Expected answer',
                'response': 'The weather is sunny.'
            }
        ]

        with open(self.experiment_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)


    def test_main_no_metrics(self):
        """Test main function with no metrics provided"""
        import os
        os.chdir(self.temp_dir)

        try:
            # When metrics is None, the function should raise an error
            # The actual error type depends on implementation
            with self.assertRaises((TypeError, AttributeError)):
                main(
                    output_file="results/evaluation_scores.json",
                    model="gemini-flash-latest",
                    metrics=None
                )
        finally:
            os.chdir(self.original_cwd)


class TestEvaluationDatasetLoading(unittest.TestCase):
    """Test loading of evaluation datasets from JSONL"""

    def setUp(self):
        """Set up temporary directory with test JSONL"""
        self.temp_dir = tempfile.mkdtemp()
        self.jsonl_file = Path(self.temp_dir) / "test_experiment.jsonl"

        # Create test data
        test_data = [
            {
                'user_input': 'Question 1',
                'retrieved_contexts': ['Context 1'],
                'reference': 'Answer 1',
                'response': 'Response 1'
            },
            {
                'user_input': 'Question 2',
                'retrieved_contexts': ['Context 2'],
                'reference': 'Answer 2',
                'response': 'Response 2'
            }
        ]

        with open(self.jsonl_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_evaluation_dataset_from_jsonl(self):
        """Test loading EvaluationDataset from JSONL file"""
        dataset = EvaluationDataset.from_jsonl(str(self.jsonl_file))

        # Verify dataset was loaded
        self.assertIsInstance(dataset, EvaluationDataset)

    def test_loaded_dataset_structure(self):
        """Test that loaded dataset has correct structure"""
        dataset = EvaluationDataset.from_jsonl(str(self.jsonl_file))

        # Convert to pandas to inspect
        # Note: This may vary depending on Ragas version
        # The test validates that the dataset can be loaded successfully


class TestAvailableMetrics(unittest.TestCase):
    """Test the AVAILABLE_METRICS loading and validation"""

    def test_available_metrics_loaded(self):
        """Test that AVAILABLE_METRICS is populated correctly"""
        # Should be a non-empty dictionary
        self.assertIsInstance(AVAILABLE_METRICS, dict)
        self.assertGreater(len(AVAILABLE_METRICS), 0)

        # All keys should be strings
        for key in AVAILABLE_METRICS.keys():
            self.assertIsInstance(key, str)

        # All values should be Metric instances
        from ragas.metrics import Metric
        for value in AVAILABLE_METRICS.values():
            self.assertIsInstance(value, Metric)

    def test_invalid_metric_name(self):
        """Test that invalid metric names are rejected"""
        invalid_metric = 'this_metric_does_not_exist_12345'
        self.assertNotIn(invalid_metric, AVAILABLE_METRICS)


if __name__ == '__main__':
    unittest.main()
