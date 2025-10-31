"""
Unit tests for run.py

Tests the agent query execution and experiment functionality.
"""

import unittest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from pydantic import BaseModel
from ragas import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run import initialize_client, run_agent_experiment, main


class TestInitializeClient(unittest.TestCase):
    """Test the initialize_client function"""

    @patch('run.ClientFactory')
    @patch('run.minimal_agent_card')
    async def test_initialize_client_creates_client(self, mock_agent_card, mock_factory):
        """Test that initialize_client creates a client correctly"""
        # Mock the agent card
        mock_card = MagicMock()
        mock_agent_card.return_value = mock_card

        # Mock the factory and client
        mock_client = MagicMock()
        mock_factory_instance = MagicMock()
        mock_factory_instance.create.return_value = mock_client
        mock_factory.return_value = mock_factory_instance

        # Mock httpx client and logger
        mock_httpx_client = AsyncMock()
        mock_logger = MagicMock()

        # Call the function
        result = await initialize_client(
            "http://test-agent:8000",
            mock_httpx_client,
            mock_logger
        )

        # Verify
        mock_agent_card.assert_called_once_with("http://test-agent:8000")
        mock_factory_instance.create.assert_called_once_with(mock_card)
        self.assertEqual(result, mock_client)

    def test_initialize_client_sync(self):
        """Synchronous wrapper for async test"""
        asyncio.run(self.test_initialize_client_creates_client())


class TestRunAgentExperiment(unittest.TestCase):
    """Test the run_agent_experiment function"""

    def setUp(self):
        """Set up temporary directory for experiment outputs"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('run.logging.getLogger')
    @patch('run.initialize_client')
    async def test_run_agent_experiment_success(self, mock_init_client, mock_get_logger):
        """Test successful agent query execution"""
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mock the client
        mock_client = MagicMock()

        # Create a mock task response
        mock_task = MagicMock()
        mock_task.model_dump.return_value = {
            'artifacts': [{
                'parts': [{'text': 'Agent response text'}]
            }],
            'history': []
        }

        # Mock the async generator returned by send_message
        async def mock_send_message(message):
            yield (mock_task, None)

        mock_client.send_message = mock_send_message
        mock_init_client.return_value = mock_client

        # Create test row
        test_row = {
            'user_input': 'What is the weather?',
            'retrieved_contexts': ['Context about weather'],
            'reference': 'Expected answer'
        }

        # Mock httpx AsyncClient
        with patch('run.httpx.AsyncClient') as mock_httpx:
            mock_httpx_instance = AsyncMock()
            mock_httpx_instance.__aenter__.return_value = mock_httpx_instance
            mock_httpx_instance.__aexit__.return_value = None
            mock_httpx.return_value = mock_httpx_instance

            # Create an async function that mimics the experiment function
            async def test_experiment_func(row, agent_url):
                return await run_agent_experiment.func(row, agent_url=agent_url)

            # Call the function
            result = await test_experiment_func(
                test_row,
                agent_url="http://test-agent:8000"
            )

        # Verify result structure
        self.assertIn('user_input', result)
        self.assertIn('retrieved_contexts', result)
        self.assertIn('reference', result)
        self.assertIn('response', result)
        self.assertEqual(result['user_input'], 'What is the weather?')
        self.assertEqual(result['response'], 'Agent response text')

    @patch('run.logging.getLogger')
    @patch('run.initialize_client')
    async def test_run_agent_experiment_error(self, mock_init_client, mock_get_logger):
        """Test agent query with error handling"""
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mock client that raises an error
        mock_init_client.side_effect = Exception("Connection failed")

        # Create test row
        test_row = {
            'user_input': 'What is the weather?',
            'retrieved_contexts': ['Context'],
            'reference': 'Answer'
        }

        # Mock httpx AsyncClient
        with patch('run.httpx.AsyncClient') as mock_httpx:
            mock_httpx_instance = AsyncMock()
            mock_httpx_instance.__aenter__.return_value = mock_httpx_instance
            mock_httpx_instance.__aexit__.return_value = None
            mock_httpx.return_value = mock_httpx_instance

            # Create an async function that mimics the experiment function
            async def test_experiment_func(row, agent_url):
                return await run_agent_experiment.func(row, agent_url=agent_url)

            # Call the function
            result = await test_experiment_func(
                test_row,
                agent_url="http://test-agent:8000"
            )

        # Verify error is captured in response
        self.assertIn('response', result)
        self.assertIn('ERROR', result['response'])
        self.assertIn('Connection failed', result['response'])

class TestMain(unittest.TestCase):
    """Test the main function"""

    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('run.run_agent_experiment.arun')
    @patch('run.Dataset.load')
    async def test_main_execution(self, mock_dataset_load, mock_arun):
        """Test main function execution flow"""
        import os
        os.chdir(self.temp_dir)

        try:
            # Create a mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=2)
            mock_dataset_load.return_value = mock_dataset

            # Mock experiment results
            mock_experiment = MagicMock()
            mock_arun.return_value = mock_experiment

            # Run main
            await main("http://test-agent:8000")

            # Verify Dataset.load was called
            mock_dataset_load.assert_called_once()

            # Verify experiment was run
            mock_arun.assert_called_once()
        finally:
            os.chdir(self.original_cwd)

    def test_main_sync(self):
        """Synchronous wrapper for async test"""
        asyncio.run(self.test_main_execution())






if __name__ == '__main__':
    unittest.main()
