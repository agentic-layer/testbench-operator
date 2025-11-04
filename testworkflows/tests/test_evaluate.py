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
    format_evaluation_scores,
    main,
    AVAILABLE_METRICS
)


class TestFormatEvaluationScores(unittest.TestCase):
    """Test the format_evaluation_scores function"""

    def test_overall_scores_calculation(self):
        """Test that overall scores are calculated correctly"""
        mock_result = MagicMock(spec=EvaluationResult)

        df = pd.DataFrame({
            'faithfulness': [0.9, 0.8, 0.7],
            'answer_relevancy': [0.85, 0.75, 0.65]
        })
        mock_result.to_pandas.return_value = df

        # Mock _repr_dict with calculated averages
        mock_result._repr_dict = {'faithfulness': 0.8, 'answer_relevancy': 0.75}

        # Mock token usage methods
        mock_token_usage = MagicMock()
        mock_token_usage.input_tokens = 100
        mock_token_usage.output_tokens = 50
        mock_result.total_tokens.return_value = mock_token_usage
        mock_result.total_cost.return_value = 0.001

        formatted = format_evaluation_scores(mock_result, 5.0/1e6, 15.0/1e6)

        # Verify overall scores are correct
        self.assertAlmostEqual(formatted.overall_scores['faithfulness'], 0.8, places=2)
        self.assertAlmostEqual(formatted.overall_scores['answer_relevancy'], 0.75, places=2)

    def test_individual_results_present(self):
        """Test that individual results are included"""
        mock_result = MagicMock(spec=EvaluationResult)

        df = pd.DataFrame({
            'user_input': ['Q1', 'Q2'],
            'faithfulness': [0.9, 0.8]
        })
        mock_result.to_pandas.return_value = df

        # Mock _repr_dict for overall scores
        mock_result._repr_dict = {'faithfulness': 0.85}

        # Mock token usage methods
        mock_token_usage = MagicMock()
        mock_token_usage.input_tokens = 100
        mock_token_usage.output_tokens = 50
        mock_result.total_tokens.return_value = mock_token_usage
        mock_result.total_cost.return_value = 0.001

        formatted = format_evaluation_scores(mock_result, 5.0/1e6, 15.0/1e6)

        # Verify individual results
        self.assertEqual(len(formatted.individual_results), 2)

    def test_token_usage_placeholders(self):
        """Test that token usage is returned correctly"""
        mock_result = MagicMock(spec=EvaluationResult)

        df = pd.DataFrame({
            'faithfulness': [0.9]
        })
        mock_result.to_pandas.return_value = df

        # Mock _repr_dict for overall scores
        mock_result._repr_dict = {'faithfulness': 0.9}

        # Mock token usage methods with specific values
        mock_token_usage = MagicMock()
        mock_token_usage.input_tokens = 150
        mock_token_usage.output_tokens = 75
        mock_result.total_tokens.return_value = mock_token_usage
        mock_result.total_cost.return_value = 0.002

        formatted = format_evaluation_scores(mock_result, 5.0/1e6, 15.0/1e6)

        # Verify token usage is captured correctly
        self.assertEqual(formatted.total_tokens['input_tokens'], 150)
        self.assertEqual(formatted.total_tokens['output_tokens'], 75)
        self.assertEqual(formatted.total_cost, 0.002)


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


if __name__ == '__main__':
    unittest.main()
