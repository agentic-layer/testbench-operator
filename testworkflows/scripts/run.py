import json
import asyncio
import argparse
import httpx

import logging
from logging import Logger

from uuid import uuid4

from pydantic import BaseModel
from ragas import Dataset, experiment, Experiment

from a2a.client.client_factory import ClientFactory, minimal_agent_card
from a2a.client.client import ClientConfig, Client

from a2a.types import (
    Message, AgentCard,
)


def extract_tool_calls(history: list[dict]) -> list[dict]:
    """
    Extracts the tool calls from the 'history' block of the agents response

    Returns a list[dict] with the 'name' of the called tool as well as the 'args'
    """

    tool_calls: list[dict] = []

    for message in history:
        # Skip if not a message or missing parts
        if not isinstance(message, dict) or 'parts' not in message:
            continue

        parts = message.get('parts', [])
        if not isinstance(parts, list):
            continue

        for part in parts:
            # Check if this part contains a function call
            if not isinstance(part, dict):
                continue

            # Look for data parts with function_call metadata
            if part.get('kind') == 'data':
                metadata: dict = part.get('metadata', {})
                data: dict = part.get('data', {})

                # Check if this is a function call
                if metadata.get('adk_type') == 'function_call' and isinstance(data, dict):
                    tool_call: dict[str, dict | None] = {
                        'name': data.get('name'),
                        'args': data.get('args', {}),
                        'id': data.get('id')
                    }

                    # Only add if there's a name
                    if tool_call['name']:
                        tool_calls.append(tool_call)

    return tool_calls


async def initialize_client(agent_url: str, httpx_client: httpx.AsyncClient, logger: Logger) -> Client:
    """Initialize the A2A client with a minimal agent card."""
    logger.info(f'Initializing A2A client for: {agent_url}')

    # Create a minimal agent card with the provided URL
    agent_card: AgentCard = minimal_agent_card(agent_url)

    config: ClientConfig = ClientConfig(httpx_client=httpx_client)
    factory: ClientFactory = ClientFactory(config)
    client: Client = factory.create(agent_card)

    logger.info('A2A client initialized successfully')

    return client


@experiment()
async def run_agent_experiment(row, agent_url: str) -> dict[str, str | list | list]:
    """
    Experiment function that processes each row from the dataset.

    Args:
        row: A dictionary containing 'input' and 'ground_truth' fields
        agent_url: The URL of the agent to query

    Returns:
        Dictionary with original row data plus 'output' and optionally 'tool_calls'
    """
    logger: Logger = logging.getLogger(__name__)

    async with httpx.AsyncClient() as httpx_client:
        client = await initialize_client(agent_url, httpx_client, logger)

        # Get the input from the row
        input_text = row.get("input")

        message = Message(
            role='user',
            parts=[
                {
                    'kind': 'text',
                    'text': input_text
                }
            ],
            messageId = uuid4().hex
        )

        logger.info(f'Processing: {input_text}')

        output_text: str = ""
        tool_calls: list = []

        async for response in client.send_message(message):
            # Client returns tuples, extract the task/message
            if isinstance(response, tuple):
                task, _ = response
                if task:
                    artifacts: list = task.model_dump(mode='json', include={'artifacts'}).get('artifacts',[])
                    history: list = task.model_dump(mode='json', include={'history'}).get('history',[])

                    # Extract the model response
                    if artifacts and artifacts[0].get('parts'):
                        output_text = artifacts[0]['parts'][0].get('text', '')
                    else:
                        logger.warning("No text found in artifacts")

                    # Extract tool calls from history
                    tool_calls = extract_tool_calls(history)
                    if tool_calls:
                        logger.info(f"Tool calls detected: {json.dumps(tool_calls, indent=2)}")
            elif hasattr(response, 'model_dump'):
                logger.debug(response.model_dump(mode='json', exclude_none=True))
            else:
                logger.warning(f'Unexpected response: {response}')

        # Return the original row data plus the results
        # Always include tool_calls field for consistent schema (empty list if none)
        result: dict[str, str | list | list] = {
            **row,
            "output": output_text,
            "tool_calls": tool_calls if tool_calls else []
        }

        return result


async def main(agent_url: str) -> None:
    """Main function to load Ragas Dataset and run Experiment."""

    # Set up logger & get logger instance
    logging.basicConfig(level=logging.INFO)
    logger: Logger = logging.getLogger(__name__)

    # Load dataset from JSON file
    logger.info('Loading dataset from data/dataset.json')
    with open('data/dataset.json') as dataset_file:
        parsed_data: dict[str, list[dict]] = json.load(dataset_file)
        dataset_samples: list[dict] = parsed_data["dataset"]

    # Create Ragas Dataset in datasets/ragas_dataset.csv
    dataset: Dataset[BaseModel] = Dataset(name="ragas_dataset", backend="local/csv", root_dir="./data")

    # Append all samples to the dataset
    for sample in dataset_samples:
        dataset.append(sample)

    # Save the dataset
    dataset.save()
    logger.info(f'Dataset saved with {len(dataset_samples)} samples')

    # Run the experiment
    logger.info('Starting experiment...')
    results: Experiment = await run_agent_experiment.arun(
        dataset,
        name = "ragas_experiment",
        agent_url = agent_url
    )

    logger.info('Experiment completed successfully')

    # Extract results and save to results.json
    result_dataset: list[dict] = []
    for result in results:
        result_dataset.append(result)

    # Wrap dataset with "results" block
    wrapped_result_dataset: dict[str, list[dict]] = {"results": result_dataset}

    # Write wrapped dataset to data/results.json
    with open('data/results.json', 'w') as results_file:
        json.dump(wrapped_result_dataset, results_file, indent=2)

    logger.info('Results saved to data/results.json')


if __name__ == '__main__':
    # Parse parameter the script was called with (URL)
    parser = argparse.ArgumentParser(description="Runs all queries from data/dataset.json through the agent at the provided URL")
    parser.add_argument('url', help='URL to agent')
    args = parser.parse_args()

    # Call main using the parsed URL
    asyncio.run(main(args.url))