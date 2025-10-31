import argparse
import json
import logging
from logging import Logger
from typing import Any
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)


def get_overall_scores(file_path: str) -> dict[str, Any]:
    """Load the evaluation_scores.json file and return the 'overall_scores' metrics."""
    with open(file_path, 'r') as file:
        return json.load(file).get("overall_scores", {})


def create_and_push_metrics(
    overall_scores: dict[str, float],
    workflow_name: str,
    pushgateway_url: str
) -> None:
    """
    Create Prometheus Gauge metrics for each overall score and push to Pushgateway.

    Args:
        overall_scores: Dictionary of metric names to scores
        workflow_name: Name of the test workflow (used as label to distinguish workflows)
        pushgateway_url: URL of the Prometheus Pushgateway
    """
    # Create a new registry for this push
    registry: CollectorRegistry = CollectorRegistry()

    for metric_name, score in overall_scores.items():
        # Create a Gauge with workflow_name as a label
        gauge = Gauge(
            f'ragas_evaluation_{metric_name}',
            f'Overall {metric_name} score from RAGAS evaluation',
            labelnames = ['workflow_name'],
            registry = registry
        )

        # Set the gauge value with the workflow_name label
        gauge.labels(workflow_name=workflow_name).set(score)
        logger.info(f"Set metric 'evaluation_{metric_name}{{workflow_name=\"{workflow_name}\"}}' to {score}")


    # Push metrics to Pushgateway
    try:
        logger.info(f"Pushing metrics to Pushgateway at {pushgateway_url}...")
        push_to_gateway(
            pushgateway_url,
            job = 'ragas_evaluation',
            registry = registry
        )
        logger.info("✓ Metrics successfully pushed to Pushgateway")
    except Exception as e:
        logger.error(f"✗ Error pushing metrics to Pushgateway: {e}")
        raise


def publish_metrics(
    input_file: str,
    workflow_name: str | None = None,
    pushgateway_url: str | None = None
) -> None:
    """
    Publish evaluation metrics to Prometheus Pushgateway.

    Args:
        input_file: Path to the evaluation scores
        workflow_name: Name of the test workflow (e.g., 'weather-assistant-test').
        pushgateway_url: URL of Prometheus Pushgateway (e.g., 'localhost:9091').
    """

    # Load overall scores from the evaluation file
    logger.info(f"Loading evaluation scores from {input_file}...")
    overall_scores = get_overall_scores(input_file)

    if not overall_scores:
        logger.warning("No overall scores found in evaluation_scores.json")
        return

    # Create and push Prometheus metrics
    logger.info(f"Creating Prometheus metrics for {len(overall_scores)} scores...")
    logger.info(f"Workflow: {workflow_name}")
    create_and_push_metrics(overall_scores, workflow_name, pushgateway_url)

    logger.info("Published metrics:")
    for metric_name, score in overall_scores.items():
        logger.info(f"  - ragas_evaluation_{metric_name}{{workflow_name=\"{workflow_name}\"}}: {score}")


if __name__ == "__main__":
    """
    Main function to publish Prometheus metrics to Prometheus Pushgateway.

    Args:
        workflow_name: Name of the test workflow
        pushgateway_url: (OPTIONAL) URL to the Prometheus Pushgateway (default: localhost:9091)

    Examples:
            python3 scripts/publish.py weather-assistant-test
            python3 scripts/publish.py weather-assistant-test pushgateway-url.com
    """

    parser = argparse.ArgumentParser(
        description = "Publish RAGAS evaluation metrics to Prometheus Pushgateway"
    )
    parser.add_argument(
        "workflow_name",
        help = "Name of the test workflow (e.g., 'weather-assistant-test')"
    )
    parser.add_argument(
        "pushgateway_url",
        nargs = "?",
        default = "localhost:9091",
        help = "URL of Prometheus Pushgateway (default: localhost:9091)"
    )

    args = parser.parse_args()

    # Call 'publish_metrics' with hardcoded input file and specified 'workflow_name' & 'pushgateway_url'
    publish_metrics(
        input_file = "results/evaluation_scores.json",
        workflow_name = args.workflow_name,
        pushgateway_url = args.pushgateway_url
    )
