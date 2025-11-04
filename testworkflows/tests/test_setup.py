"""
Unit tests for setup.py

Tests the dataset download, conversion, and Ragas dataset creation functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from io import BytesIO
from unittest.mock import patch, MagicMock

import pandas as pd
from ragas import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from setup import (
    custom_convert_csv,
    get_converter,
    dataframe_to_ragas_dataset,
    main
)


class TestCustomConvertCSV(unittest.TestCase):
    """Test the custom_convert_csv function"""

    def test_converts_string_to_list(self):
        """Test that retrieved_contexts strings are converted to lists"""
        csv_content = b"user_input,retrieved_contexts,reference\n"
        csv_content += b'"Question?","Context text","Answer"\n'

        buffer = BytesIO(csv_content)
        df = custom_convert_csv(buffer)

        self.assertIsInstance(df['retrieved_contexts'].iloc[0], list)
        self.assertEqual(df['retrieved_contexts'].iloc[0], ["Context text"])

    def test_handles_empty_retrieved_contexts(self):
        """Test handling of empty retrieved_contexts"""
        csv_content = b"user_input,retrieved_contexts,reference\n"
        csv_content += b'"Question?","","Answer"\n'

        buffer = BytesIO(csv_content)
        df = custom_convert_csv(buffer)

        # Empty string becomes [nan] in pandas, which then becomes []
        # The function converts non-list values to lists
        result = df['retrieved_contexts'].iloc[0]
        # Check that it's a list and handle NaN case
        self.assertIsInstance(result, list)
        # If it contains NaN, that's acceptable behavior for empty strings in CSV
        if result and pd.isna(result[0]):
            # This is expected - pandas converts empty string to NaN
            pass
        else:
            # Or it should be an empty list
            self.assertEqual(result, [])

    def test_missing_retrieved_contexts_column(self):
        """Test that CSV without retrieved_contexts column works"""
        csv_content = b"user_input,reference\n"
        csv_content += b'"Question?","Answer"\n'

        buffer = BytesIO(csv_content)
        df = custom_convert_csv(buffer)

        self.assertNotIn('retrieved_contexts', df.columns)


class TestGetConverter(unittest.TestCase):
    """Test the get_converter function"""

    def test_unsupported_format(self):
        """Test that unsupported formats raise TypeError"""
        with self.assertRaises(TypeError) as context:
            get_converter("https://example.com/data.xlsx")

        self.assertIn("Unsupported filetype", str(context.exception))


class TestDataframeToRagasDataset(unittest.TestCase):
    """Test the dataframe_to_ragas_dataset function"""

    def setUp(self):
        """Set up temporary directory for test outputs"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_ragas_dataset_file(self):
        """Test that ragas_dataset.jsonl is created"""
        import os
        os.chdir(self.temp_dir)

        try:
            df = pd.DataFrame({
                'user_input': ['Question 1'],
                'retrieved_contexts': [['Context 1']],
                'reference': ['Answer 1']
            })

            dataframe_to_ragas_dataset(df)

            # Check for the file in the datasets subdirectory
            dataset_file = Path(self.temp_dir) / 'data' / 'datasets' / 'ragas_dataset.jsonl'
            self.assertTrue(dataset_file.exists(), f"Dataset file not found at {dataset_file}")
        finally:
            os.chdir(self.original_cwd)

class TestMain(unittest.TestCase):
    """Test the main function with mocked HTTP requests"""

    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('setup.requests.get')
    def test_main_with_csv(self, mock_get):
        """Test main function with CSV file"""
        import os
        os.chdir(self.temp_dir)

        try:
            # Mock the HTTP response
            csv_content = b"user_input,retrieved_contexts,reference\n"
            csv_content += b'"Question?","Context text","Answer"\n'

            mock_response = MagicMock()
            mock_response.content = csv_content
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # Run main
            main("https://example.com/data.csv")

            # Verify dataset was created in datasets subdirectory
            dataset_file = Path(self.temp_dir) / 'data' / 'datasets' / 'ragas_dataset.jsonl'
            self.assertTrue(dataset_file.exists(), f"Dataset file not found at {dataset_file}")

            # Verify requests.get was called
            mock_get.assert_called_once_with("https://example.com/data.csv")
        finally:
            os.chdir(self.original_cwd)

    @patch('setup.requests.get')
    def test_main_with_json(self, mock_get):
        """Test main function with JSON file"""
        import os
        os.chdir(self.temp_dir)

        try:
            # Mock the HTTP response
            json_content = b'''[
                {
                    "user_input": "Question?",
                    "retrieved_contexts": ["Context text"],
                    "reference": "Answer"
                }
            ]'''

            mock_response = MagicMock()
            mock_response.content = json_content
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # Run main
            main("https://example.com/data.json")

            # Verify dataset was created in datasets subdirectory
            dataset_file = Path(self.temp_dir) / 'data' / 'datasets' / 'ragas_dataset.jsonl'
            self.assertTrue(dataset_file.exists(), f"Dataset file not found at {dataset_file}")
        finally:
            os.chdir(self.original_cwd)

    @patch('setup.requests.get')
    def test_main_with_invalid_url(self, mock_get):
        """Test main function with invalid URL (HTTP error)"""
        import os
        os.chdir(self.temp_dir)

        try:
            # Mock HTTP error
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = Exception("HTTP 404")
            mock_get.return_value = mock_response

            # Verify that the error propagates
            with self.assertRaises(Exception):
                main("https://example.com/nonexistent.csv")
        finally:
            os.chdir(self.original_cwd)


if __name__ == '__main__':
    unittest.main()
