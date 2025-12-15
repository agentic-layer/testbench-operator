import argparse
import json
import logging
from logging import Logger

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)


def get_overall_scores(file_path: str) -> dict[str, float]:
    """Load the evaluation_scores.json file and return the 'overall_scores' metrics."""
    with open(file_path, "r") as file:
        return json.load(file).get("overall_scores", {})


def create_and_push_metrics(overall_scores: dict[str, float], workflow_name: str, otlp_endpoint: str) -> None:
    """
    Create OpenTelemetry metrics for each overall score and push via OTLP.

    Args:
        overall_scores: Dictionary of metric names to scores
        workflow_name: Name of the test workflow (used as label to distinguish workflows)
        otlp_endpoint: URL of the OTLP endpoint (e.g., 'http://localhost:4318')
    """
    # Ensure the endpoint has the correct protocol
    if not otlp_endpoint.startswith("http://") and not otlp_endpoint.startswith("https://"):
        otlp_endpoint = f"http://{otlp_endpoint}"

    # Create OTLP exporter
    exporter = OTLPMetricExporter(endpoint=f"{otlp_endpoint}/v1/metrics")

    # Create a metric reader that exports immediately
    reader = PeriodicExportingMetricReader(
        exporter=exporter,
        export_interval_millis=1000,  # Export every second
    )

    # Create resource with workflow metadata
    resource = Resource.create({"service.name": "ragas-evaluation", "workflow.name": workflow_name})

    # Create MeterProvider with the exporter and resource
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    # Get a meter
    meter = metrics.get_meter("ragas.evaluation", "1.0.0")

    # Create and record metrics
    try:
        logger.info(f"Pushing metrics to OTLP endpoint at {otlp_endpoint}...")

        for metric_name, score in overall_scores.items():
            # Create a Gauge
            gauge = meter.create_gauge(
                name=f"ragas_evaluation_{metric_name}",
                description=f"Overall {metric_name} score from RAGAS evaluation",
                unit="1",
            )

            # Set the gauge value with workflow_name as an attribute
            gauge.set(score, {"workflow_name": workflow_name})
            logger.info(f"Set metric 'ragas_evaluation_{metric_name}{{workflow_name=\"{workflow_name}\"}}' to {score}")

        # Force flush to ensure metrics are sent
        provider.force_flush()

        logger.info("✓ Metrics successfully pushed via OTLP")
    except Exception as e:
        logger.error(f"✗ Error pushing metrics via OTLP: {e}")
        raise
    finally:
        # Shutdown the provider
        provider.shutdown()

    logger.info("Published metrics:")
    for metric_name, score in overall_scores.items():
        logger.info(f'  - ragas_evaluation_{metric_name}{{workflow_name="{workflow_name}"}}: {score}')


def publish_metrics(input_file: str, workflow_name: str, otlp_endpoint: str) -> None:
    """
    Publish evaluation metrics via OpenTelemetry OTLP.

    Args:
        input_file: Path to the evaluation scores
        workflow_name: Name of the test workflow (e.g., 'weather-assistant-test').
        otlp_endpoint: URL of the OTLP endpoint (e.g., 'http://localhost:4318').
    """

    # Load overall scores from the evaluation file
    logger.info(f"Loading evaluation scores from {input_file}...")
    overall_scores = get_overall_scores(input_file)

    if not overall_scores:
        logger.warning("No overall scores found in evaluation_scores.json")
        return

    # Create and push OpenTelemetry metrics
    logger.info(f"Creating OpenTelemetry metrics for {len(overall_scores)} scores...")
    logger.info(f"Workflow: {workflow_name}")
    create_and_push_metrics(overall_scores, workflow_name, otlp_endpoint)


if __name__ == "__main__":
    """
    Main function to publish metrics via OpenTelemetry OTLP.

    Args:
        workflow_name: Name of the test workflow
        otlp_endpoint: (OPTIONAL) URL to the OTLP endpoint (default: localhost:4318)

    Examples:
            python3 scripts/publish.py weather-assistant-test
            python3 scripts/publish.py weather-assistant-test http://localhost:4318
    """

    parser = argparse.ArgumentParser(description="Publish RAGAS evaluation metrics via OpenTelemetry OTLP")
    parser.add_argument(
        "workflow_name",
        help="Name of the test workflow (e.g., 'weather-assistant-test')",
    )
    parser.add_argument(
        "otlp_endpoint",
        nargs="?",
        default="localhost:4318",
        help="URL of the OTLP HTTP endpoint (default: localhost:4318)",
    )

    args = parser.parse_args()

    # Call 'publish_metrics' with hardcoded input file and specified 'workflow_name' & 'otlp_endpoint'
    publish_metrics(
        input_file="data/results/evaluation_scores.json",
        workflow_name=args.workflow_name,
        otlp_endpoint=args.otlp_endpoint,
    )
