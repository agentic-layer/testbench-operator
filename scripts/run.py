import argparse
import asyncio
import logging
from logging import Logger
from uuid import uuid4

import httpx
from a2a.client.client import Client, ClientConfig
from a2a.client.client_factory import ClientFactory, minimal_agent_card
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    TextPart,
)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from otel_setup import setup_otel
from pydantic import BaseModel
from ragas import Dataset, experiment
from ragas.messages import AIMessage, HumanMessage, ToolCall

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)


def a2a_message_to_ragas(message: Message) -> HumanMessage | AIMessage:
    """
    Convert A2A Message to RAGAS message format.

    Handles:
    - Text extraction from multiple parts
    - Role mapping (user → human, agent → ai)
    - Tool call extraction from metadata
    - Metadata preservation

    Args:
        message: A2A Message object

    Returns:
        HumanMessage or AIMessage

    Raises:
        ValueError: If role is not user or agent
    """
    # Extract text from all TextPart objects
    text_parts = []
    for part in message.parts:
        # Part is a wrapper - access the actual part inside
        actual_part = part.root if hasattr(part, "root") else part
        if hasattr(actual_part, "text"):
            text_parts.append(actual_part.text)

    content = " ".join(text_parts) if text_parts else ""

    # Map role
    if message.role == Role.user:
        return HumanMessage(content=content, metadata=message.metadata)
    elif message.role == Role.agent:
        # Extract tool calls from metadata if present
        tool_calls = None
        if message.metadata and "tool_calls" in message.metadata:
            # Parse tool calls from metadata
            tool_calls_data = message.metadata["tool_calls"]
            tool_calls = [ToolCall(name=tc["name"], args=tc["args"]) for tc in tool_calls_data]

        return AIMessage(content=content, metadata=message.metadata, tool_calls=tool_calls)
    else:
        raise ValueError(f"Unsupported message role: {message.role}")


def validate_multi_turn_input(user_input: list) -> list[dict]:
    """
    Validate and normalize multi-turn user_input.

    Expected format: [{"content": "...", "type": "human"}, {"content": "...", "type": "ai"}, ...]

    Args:
        user_input: List of message dictionaries

    Returns:
        Validated list of message dicts

    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(user_input, list):
        raise ValueError(f"Multi-turn user_input must be list, got {type(user_input)}")

    if not user_input:
        raise ValueError("Multi-turn user_input cannot be empty")

    for i, msg in enumerate(user_input):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {i} must be dict, got {type(msg)}")

        if "content" not in msg:
            raise ValueError(f"Message {i} missing 'content' field")

        if "type" not in msg:
            raise ValueError(f"Message {i} missing 'type' field")

        if msg["type"] not in ("human", "ai", "tool"):
            raise ValueError(f"Message {i} has invalid type: {msg['type']}")

    return user_input


async def initialize_client(agent_url: str) -> Client:
    """Initialize the A2A client with a minimal agent card."""
    logger.info(f"Initializing A2A client for: {agent_url}")

    # Create a minimal agent card with the provided URL
    agent_card: AgentCard = minimal_agent_card(agent_url)

    config: ClientConfig = ClientConfig()
    factory: ClientFactory = ClientFactory(config)
    client: Client = factory.create(agent_card)

    logger.info("A2A client initialized successfully")

    return client


@experiment()
async def single_turn_experiment(row, agent_url: str, workflow_name: str) -> dict[str, str | list]:
    """
    Single-turn experiment function that processes each row from the dataset.

    Sends a single user message to the agent and captures the response.

    Args:
        row: A dictionary containing 'user_input' (str), 'retrieved_contexts', and 'reference' fields
        agent_url: The URL of the agent to query
        workflow_name: Name of the test workflow for span labeling

    Returns:
        Dictionary with original row data plus 'response' and 'trace_id'
    """

    # Get tracer for creating spans
    tracer = trace.get_tracer("testbench.run")

    # Create span for this test case
    # Span name includes user_input preview for debugging
    user_input_preview = row.get("user_input", "")[:50]
    span_name = f"query_agent: {user_input_preview}"

    with tracer.start_as_current_span(span_name) as span:
        # Extract Trace ID from current span context
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, "032x")  # 32-char hex string

        # Add span attributes for filtering/debugging in Tempo UI
        span.set_attribute("test.user_input", row.get("user_input", ""))
        span.set_attribute("test.reference", row.get("reference", ""))
        span.set_attribute("agent.url", agent_url)
        span.set_attribute("workflow.name", workflow_name)

        try:
            async with httpx.AsyncClient():
                client = await initialize_client(agent_url)

                # Get the input from the row
                input_text = row.get("user_input")

                message = Message(
                    role=Role.user,
                    parts=[Part(TextPart(text=input_text))],
                    message_id=uuid4().hex,
                )

                logger.info(f"Processing: {input_text}")

                async for response in client.send_message(message):
                    # Client returns tuples, extract the task/message
                    if isinstance(response, tuple):
                        task, _ = response
                        if task:
                            artifacts: list = task.model_dump(mode="json", include={"artifacts"}).get("artifacts", [])

                            # Extract the model response
                            if artifacts and artifacts[0].get("parts"):
                                output_text = artifacts[0]["parts"][0].get("text", "")
                            else:
                                logger.warning("No text found in artifacts")
                    else:
                        logger.warning(f"Unexpected response: {response}")

            # Mark span as successful
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            logger.error(f'Error processing input "{row.get("user_input")}": {str(e)}')
            output_text = f"ERROR: {str(e)}"

            # Record exception in span for debugging
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, description=str(e)))

        # Return the original row data plus results AND trace_id
        result: dict[str, str | list] = {
            **row,
            "response": output_text,
            "trace_id": trace_id,
        }

        return result


@experiment()
async def multi_turn_experiment(row, agent_url: str, workflow_name: str) -> dict[str, list | str]:
    """
    Multi-turn experiment function for conversational interactions.

    Processes a conversation by:
    1. Extracting human messages from input
    2. Sequentially querying agent for each turn
    3. Maintaining context_id across turns
    4. Extracting full conversation history from final task
    5. Converting to RAGAS MultiTurnSample format

    Args:
        row: Dictionary with 'user_input' (list of message dicts) and 'reference'
        agent_url: URL of the agent to query
        workflow_name: Name of the test workflow for span labeling

    Returns:
        Dictionary with 'user_input' (list of RAGAS messages), 'reference', 'trace_id'
    """
    # Get tracer for creating spans
    tracer = trace.get_tracer("testbench.run")

    # Create parent span for entire conversation
    user_input_preview = str(row.get("user_input", []))[:100]
    span_name = f"query_agent_multi_turn: {user_input_preview}"

    with tracer.start_as_current_span(span_name) as span:
        # Extract trace ID
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, "032x")

        # Add span attributes
        span.set_attribute("test.turn_count", len(row.get("user_input", [])))
        span.set_attribute("test.reference", row.get("reference", ""))
        span.set_attribute("agent.url", agent_url)
        span.set_attribute("workflow.name", workflow_name)
        span.set_attribute("test.conversation_type", "multi_turn")

        try:
            # Validate input format
            user_input = validate_multi_turn_input(row.get("user_input"))

            async with httpx.AsyncClient():
                client = await initialize_client(agent_url)

                # Extract only human messages (agent messages are from dataset, not sent)
                human_messages = [msg for msg in user_input if msg.get("type") == "human"]

                if not human_messages:
                    raise ValueError("No human messages found in user_input")

                context_id = None
                conversation_messages = []

                # Sequentially query agent for each human turn
                for turn_idx, human_msg in enumerate(human_messages):
                    # Create child span for this turn
                    turn_span_name = f"turn_{turn_idx + 1}: {human_msg['content'][:50]}"
                    with tracer.start_as_current_span(turn_span_name) as turn_span:
                        turn_span.set_attribute("turn.index", turn_idx + 1)
                        turn_span.set_attribute("turn.content", human_msg["content"])

                        # Create A2A message
                        message = Message(
                            role=Role.user,
                            parts=[TextPart(text=human_msg["content"])],
                            message_id=uuid4().hex,
                            context_id=context_id,  # None for first turn, preserved after
                        )
                        conversation_messages.append({"content": human_msg["content"], "type": "human"})

                        logger.info(f"Turn {turn_idx + 1}/{len(human_messages)}: {human_msg['content']}")

                        # Send message and get response
                        agent_response_text = ""
                        async for response in client.send_message(message):
                            if isinstance(response, tuple):
                                task, _ = response
                                if task:
                                    # Capture context_id from first response
                                    if not context_id:
                                        context_id = task.context_id
                                        logger.info(f"Captured context_id: {context_id}")
                                        span.set_attribute("conversation.context_id", context_id)

                                    # Extract agent response from artifacts (same approach as single_turn_experiment)
                                    artifacts: list = task.model_dump(mode="json", include={"artifacts"}).get(
                                        "artifacts", []
                                    )
                                    if artifacts and artifacts[0].get("parts"):
                                        agent_response_text = artifacts[0]["parts"][0].get("text", "")

                        # Add agent response to conversation
                        if agent_response_text:
                            conversation_messages.append({"content": agent_response_text, "type": "ai"})
                            logger.info(f"Agent response: {agent_response_text[:100]}...")
                        else:
                            logger.warning(f"Empty agent response for turn {turn_idx + 1}")

                # Validate we got responses
                if len(conversation_messages) < 2:
                    raise ValueError(f"Incomplete conversation: only {len(conversation_messages)} messages")

                # Use the manually built conversation
                user_input_serialized = conversation_messages

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("conversation.message_count", len(conversation_messages))

        except Exception as e:
            logger.error(f"Error processing multi-turn conversation: {str(e)}")

            # Record exception in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, description=str(e)))

            # Return minimal result
            return {
                **row,
                "user_input": row.get("user_input"),
                "trace_id": trace_id,
            }

        # Return result in MultiTurnSample format
        result = {
            **row,
            "user_input": user_input_serialized,
            "trace_id": trace_id,
        }

        return result


async def main(agent_url: str, workflow_name: str) -> None:
    """Main function to load Dataset and run appropriate Experiment."""

    # Initialize OpenTelemetry tracing
    setup_otel()

    # Load existing dataset
    logger.info("Loading dataset from data/datasets/ragas_dataset.jsonl")
    dataset: Dataset[BaseModel] = Dataset.load(name="ragas_dataset", backend="local/jsonl", root_dir="./data")
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Detect dataset type by inspecting first row
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    first_row = dataset[0]
    is_multi_turn = isinstance(first_row.get("user_input"), list)

    if is_multi_turn:
        logger.info("Detected multi-turn dataset")
        logger.info("Starting multi-turn experiment...")
        await multi_turn_experiment.arun(
            dataset, name="ragas_experiment", agent_url=agent_url, workflow_name=workflow_name
        )
    else:
        logger.info("Detected single-turn dataset")
        logger.info("Starting single-turn experiment...")
        await single_turn_experiment.arun(
            dataset, name="ragas_experiment", agent_url=agent_url, workflow_name=workflow_name
        )

    logger.info("Experiment completed successfully")
    logger.info("Results saved to data/experiments/ragas_experiment.jsonl")


if __name__ == "__main__":
    # Parse parameters the script was called with
    parser = argparse.ArgumentParser(
        description="Runs all queries from the Ragas dataset through the agent at the provided URL"
    )
    parser.add_argument("url", help="URL to agent")
    parser.add_argument(
        "workflow_name",
        nargs="?",
        default="local-test",
        help="Name of the test workflow (e.g., 'weather-assistant-test'). Default: 'local-test'",
    )
    args = parser.parse_args()

    # Call main with parsed arguments
    asyncio.run(main(args.url, args.workflow_name))
