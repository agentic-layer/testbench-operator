"""
Unit tests for run.py

Tests the agent query execution and experiment functionality.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run import initialize_client, main, run_agent_experiment


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmp = tempfile.mkdtemp()
    original_cwd = Path.cwd()
    yield tmp, original_cwd
    shutil.rmtree(tmp, ignore_errors=True)


# TestInitializeClient tests
@pytest.mark.asyncio
async def test_initialize_client_creates_client(monkeypatch):
    """Test that initialize_client creates a client correctly"""

    # Mock the agent card
    class MockCard:
        pass

    mock_card = MockCard()

    def mock_agent_card(url):
        return mock_card

    # Mock the factory and client
    class MockClient:
        pass

    mock_client = MockClient()

    class MockFactory:
        def create(self, card):
            return mock_client

    def mock_factory_init(config=None):
        return MockFactory()

    monkeypatch.setattr("run.minimal_agent_card", mock_agent_card)
    monkeypatch.setattr("run.ClientFactory", mock_factory_init)

    # Call the function
    result = await initialize_client("http://test-agent:8000")

    # Verify
    assert result == mock_client


# TestRunAgentExperiment tests
@pytest.mark.asyncio
async def test_run_agent_experiment_success(monkeypatch):
    """Test successful agent query execution"""

    # Mock the client
    class MockTask:
        def model_dump(self, **kwargs):
            return {
                "artifacts": [{"parts": [{"text": "Agent response text"}]}],
                "history": [],
            }

    mock_task = MockTask()

    class MockClient:
        async def send_message(self, message):
            yield (mock_task, None)

    mock_client = MockClient()

    async def mock_init_client(url):
        return mock_client

    # Mock httpx AsyncClient
    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_httpx_client():
        return MockAsyncClient()

    monkeypatch.setattr("run.initialize_client", mock_init_client)
    monkeypatch.setattr("run.httpx.AsyncClient", mock_httpx_client)

    # Create test row
    test_row = {
        "user_input": "What is the weather?",
        "retrieved_contexts": ["Context about weather"],
        "reference": "Expected answer",
    }

    # Call the function
    result = await run_agent_experiment.func(test_row, agent_url="http://test-agent:8000")

    # Verify result structure
    assert "user_input" in result
    assert "retrieved_contexts" in result
    assert "reference" in result
    assert "response" in result
    assert result["user_input"] == "What is the weather?"
    assert result["response"] == "Agent response text"


@pytest.mark.asyncio
async def test_run_agent_experiment_error(monkeypatch):
    """Test agent query with error handling"""

    # Mock client that raises an error
    async def mock_init_client(url):
        raise Exception("Connection failed")

    # Mock httpx AsyncClient
    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_httpx_client():
        return MockAsyncClient()

    monkeypatch.setattr("run.initialize_client", mock_init_client)
    monkeypatch.setattr("run.httpx.AsyncClient", mock_httpx_client)

    # Create test row
    test_row = {
        "user_input": "What is the weather?",
        "retrieved_contexts": ["Context"],
        "reference": "Answer",
    }

    # Call the function
    result = await run_agent_experiment.func(test_row, agent_url="http://test-agent:8000")

    # Verify error is captured in response
    assert "response" in result
    assert "ERROR" in result["response"]
    assert "Connection failed" in result["response"]


# TestMain tests
@pytest.mark.asyncio
async def test_main_execution(temp_dir, monkeypatch):
    """Test main function execution flow"""

    tmp, original_cwd = temp_dir
    os.chdir(tmp)

    try:
        # Create a mock dataset
        class MockDataset:
            def __len__(self):
                return 2

        mock_dataset = MockDataset()

        def mock_dataset_load(path, backend):
            return mock_dataset

        # Mock experiment results
        class MockExperiment:
            pass

        mock_experiment = MockExperiment()

        async def mock_arun(*args, **kwargs):
            return mock_experiment

        calls_to_load = []
        calls_to_arun = []

        def mock_dataset_load_tracked(**kwargs):
            calls_to_load.append(kwargs)
            return mock_dataset

        async def mock_arun_tracked(*args, **kwargs):
            calls_to_arun.append({"args": args, "kwargs": kwargs})
            return mock_experiment

        monkeypatch.setattr("run.Dataset.load", mock_dataset_load_tracked)
        monkeypatch.setattr("run.run_agent_experiment.arun", mock_arun_tracked)

        # Run main
        await main("http://test-agent:8000")

        # Verify Dataset.load was called
        assert len(calls_to_load) == 1

        # Verify experiment was run
        assert len(calls_to_arun) == 1
    finally:
        os.chdir(original_cwd)
