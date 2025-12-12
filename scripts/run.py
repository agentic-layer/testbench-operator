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

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)


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
async def run_agent_experiment(row, agent_url: str, workflow_name: str) -> dict[str, str | list]:
    """
    Experiment function that processes each row from the dataset.

    Args:
        row: A dictionary containing 'user_input', 'retrieved_contexts', and 'reference' fields
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


async def main(agent_url: str, workflow_name: str) -> None:
    """Main function to load Ragas Dataset and run Experiment."""

    # Initialize OpenTelemetry tracing
    setup_otel()

    # Load existing Ragas dataset
    logger.info("Loading Ragas dataset from data/datasets/ragas_dataset.jsonl")
    dataset: Dataset[BaseModel] = Dataset.load(name="ragas_dataset", backend="local/jsonl", root_dir="./data")
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Run the experiment
    logger.info("Starting experiment...")
    await run_agent_experiment.arun(dataset, name="ragas_experiment", agent_url=agent_url, workflow_name=workflow_name)

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
