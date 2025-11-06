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
)
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
async def run_agent_experiment(row, agent_url: str) -> dict[str, str | list]:
    """
    Experiment function that processes each row from the dataset.

    Args:
        row: A dictionary containing 'user_input', 'retrieved_contexts', and 'reference' fields
        agent_url: The URL of the agent to query

    Returns:
        Dictionary with original row data plus 'response'
    """

    try:
        async with httpx.AsyncClient():
            client = await initialize_client(agent_url)

            # Get the input from the row
            input_text = row.get("user_input")

            message = Message(
                role="user",
                parts=[{"kind": "text", "text": input_text}],
                messageId=uuid4().hex,
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

    except Exception as e:
        logger.error(f'Error processing input "{row.get("user_input")}": {str(e)}')
        output_text = f"ERROR: {str(e)}"

    # Return the original row data plus the results
    result: dict[str, str | list] = {
        **row,
        "response": output_text,
    }

    return result


async def main(agent_url: str) -> None:
    """Main function to load Ragas Dataset and run Experiment."""

    # Load existing Ragas dataset
    logger.info("Loading Ragas dataset from data/datasets/ragas_dataset.jsonl")
    dataset: Dataset[BaseModel] = Dataset.load(name="ragas_dataset", backend="local/jsonl", root_dir="./data")
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Run the experiment
    logger.info("Starting experiment...")
    await run_agent_experiment.arun(dataset, name="ragas_experiment", agent_url=agent_url)

    logger.info("Experiment completed successfully")
    logger.info("Results saved to data/experiments/ragas_experiment.jsonl")


if __name__ == "__main__":
    # Parse parameter the script was called with (URL)
    parser = argparse.ArgumentParser(
        description="Runs all queries from the Ragas dataset through the agent at the provided URL"
    )
    parser.add_argument("url", help="URL to agent")
    args = parser.parse_args()

    # Call main using the parsed URL
    asyncio.run(main(args.url))
