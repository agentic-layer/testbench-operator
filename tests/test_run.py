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

from run import (
    a2a_message_to_ragas,
    initialize_client,
    main,
    single_turn_experiment,
    validate_multi_turn_input,
)


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


# TestSingleTurnExperiment tests
@pytest.mark.asyncio
async def test_single_turn_experiment_success(monkeypatch):
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
    result = await single_turn_experiment.func(
        test_row, agent_url="http://test-agent:8000", workflow_name="test-workflow"
    )

    # Verify result structure
    assert "user_input" in result
    assert "retrieved_contexts" in result
    assert "reference" in result
    assert "response" in result
    assert result["user_input"] == "What is the weather?"
    assert result["response"] == "Agent response text"


@pytest.mark.asyncio
async def test_single_turn_experiment_error(monkeypatch):
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
    result = await single_turn_experiment.func(
        test_row, agent_url="http://test-agent:8000", workflow_name="test-workflow"
    )

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

            def __getitem__(self, index):
                # Return single-turn format for detection
                return {"user_input": "Test question", "retrieved_contexts": [], "reference": "Answer"}

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
        monkeypatch.setattr("run.single_turn_experiment.arun", mock_arun_tracked)

        # Run main
        await main("http://test-agent:8000", "test-workflow")

        # Verify Dataset.load was called
        assert len(calls_to_load) == 1

        # Verify experiment was run
        assert len(calls_to_arun) == 1

        # Verify workflow_name was passed through to arun
        assert calls_to_arun[0]["kwargs"]["workflow_name"] == "test-workflow"
    finally:
        os.chdir(original_cwd)


# Test helper functions
def test_a2a_message_to_ragas_human():
    """Test conversion of A2A user message to RAGAS HumanMessage"""
    from a2a.types import Message, Part, Role, TextPart

    # Create A2A user message
    a2a_msg = Message(
        role=Role.user,
        parts=[Part(TextPart(text="Hello, how are you?"))],
        message_id="test123",
    )

    # Convert to RAGAS
    ragas_msg = a2a_message_to_ragas(a2a_msg)

    # Verify
    from ragas.messages import HumanMessage

    assert isinstance(ragas_msg, HumanMessage)
    assert ragas_msg.content == "Hello, how are you?"


def test_a2a_message_to_ragas_ai():
    """Test conversion of A2A agent message to RAGAS AIMessage"""
    from a2a.types import Message, Part, Role, TextPart

    # Create A2A agent message
    a2a_msg = Message(
        role=Role.agent,
        parts=[Part(TextPart(text="I'm doing well, thank you!"))],
        message_id="test456",
    )

    # Convert to RAGAS
    ragas_msg = a2a_message_to_ragas(a2a_msg)

    # Verify
    from ragas.messages import AIMessage

    assert isinstance(ragas_msg, AIMessage)
    assert ragas_msg.content == "I'm doing well, thank you!"
    assert ragas_msg.tool_calls is None


def test_a2a_message_to_ragas_with_tool_calls():
    """Test tool call extraction from metadata"""
    from a2a.types import Message, Part, Role, TextPart

    # Create A2A agent message with tool calls in metadata
    a2a_msg = Message(
        role=Role.agent,
        parts=[Part(TextPart(text="Let me check the weather"))],
        message_id="test789",
        metadata={"tool_calls": [{"name": "get_weather", "args": {"location": "NYC"}}]},
    )

    # Convert to RAGAS
    ragas_msg = a2a_message_to_ragas(a2a_msg)

    # Verify
    from ragas.messages import AIMessage

    assert isinstance(ragas_msg, AIMessage)
    assert ragas_msg.content == "Let me check the weather"
    assert ragas_msg.tool_calls is not None
    assert len(ragas_msg.tool_calls) == 1
    assert ragas_msg.tool_calls[0].name == "get_weather"
    assert ragas_msg.tool_calls[0].args == {"location": "NYC"}


def test_a2a_message_to_ragas_multi_part():
    """Test text extraction from multiple parts"""
    from a2a.types import Message, Part, Role, TextPart

    # Create message with multiple text parts
    a2a_msg = Message(
        role=Role.agent,
        parts=[Part(TextPart(text="Hello")), Part(TextPart(text="World"))],
        message_id="test",
    )

    # Convert to RAGAS
    ragas_msg = a2a_message_to_ragas(a2a_msg)

    # Verify text parts are concatenated
    from ragas.messages import AIMessage

    assert isinstance(ragas_msg, AIMessage)
    assert ragas_msg.content == "Hello World"


def test_validate_multi_turn_input_success():
    """Test validation with valid multi-turn input"""
    user_input = [
        {"content": "Hello", "type": "human"},
        {"content": "Hi there!", "type": "ai"},
        {"content": "How are you?", "type": "human"},
    ]

    result = validate_multi_turn_input(user_input)

    assert result == user_input


def test_validate_multi_turn_input_invalid_type():
    """Test validation rejects non-list input"""
    with pytest.raises(ValueError, match="must be list"):
        validate_multi_turn_input("not a list")  # type: ignore


def test_validate_multi_turn_input_missing_fields():
    """Test validation catches missing content/type fields"""
    # Missing content
    with pytest.raises(ValueError, match="missing 'content' field"):
        validate_multi_turn_input([{"type": "human"}])

    # Missing type
    with pytest.raises(ValueError, match="missing 'type' field"):
        validate_multi_turn_input([{"content": "Hello"}])


def test_validate_multi_turn_input_invalid_message_type():
    """Test validation catches invalid message types"""
    with pytest.raises(ValueError, match="has invalid type"):
        validate_multi_turn_input([{"content": "Hello", "type": "invalid"}])


@pytest.mark.asyncio
async def test_main_detects_multi_turn(temp_dir, monkeypatch):
    """Test main calls multi_turn_experiment for list user_input"""
    tmp, original_cwd = temp_dir
    os.chdir(tmp)

    try:
        # Create a mock dataset with multi-turn format
        class MockDataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                # Return multi-turn format for detection
                return {
                    "user_input": [{"content": "Hello", "type": "human"}],
                    "reference": "Answer",
                }

        mock_dataset = MockDataset()

        calls_to_multi_turn = []

        async def mock_multi_turn_arun(*args, **kwargs):
            calls_to_multi_turn.append({"args": args, "kwargs": kwargs})
            return None

        def mock_dataset_load(**kwargs):
            return mock_dataset

        monkeypatch.setattr("run.Dataset.load", mock_dataset_load)
        monkeypatch.setattr("run.multi_turn_experiment.arun", mock_multi_turn_arun)

        # Run main
        await main("http://test-agent:8000", "test-workflow")

        # Verify multi_turn_experiment was called
        assert len(calls_to_multi_turn) == 1
        assert calls_to_multi_turn[0]["kwargs"]["workflow_name"] == "test-workflow"
    finally:
        os.chdir(original_cwd)
