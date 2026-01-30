import argparse
import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from logging import Logger
from typing import Any, Union

import ragas.metrics.collections as metrics_module
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai import AsyncOpenAI
from ragas import Experiment, experiment
from ragas.backends import LocalJSONLBackend
from ragas.llms import llm_factory
from ragas.metrics.collections import BaseMetric

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)


class MetricsRegistry:
    """Registry for RAGAS metrics discovery and management."""

    def __init__(self):
        """Initialize registry and discover available metrics."""
        self._classes: dict[str, type[BaseMetric]] = {}
        self._discover_metrics()

    def _discover_metrics(self) -> None:
        """
        Discover metric classes from Ragas.

        Populates _classes dictionary with available BaseMetric subclasses.
        """
        for name, obj in inspect.getmembers(metrics_module):
            if name.startswith("_"):
                continue

            if inspect.isclass(obj) and issubclass(obj, BaseMetric) and obj is not BaseMetric:
                self._classes[name] = obj

    def get_class(self, name: str) -> type[BaseMetric]:
        """
        Get metric class by name.

        Args:
            name: Class name

        Returns:
            BaseMetric class type

        Raises:
            ValueError: If class not found
        """
        if name not in self._classes:
            raise ValueError(f"Unknown class '{name}'.\nAvailable: {', '.join(sorted(self._classes.keys()))}")
        return self._classes[name]

    def instantiate_metric(self, class_name: str, parameters: dict[str, Any], llm: Any) -> BaseMetric:
        """
        Instantiate metric class with custom parameters.

        Args:
            class_name: Name of metric class
            parameters: Dictionary of constructor parameters
            llm: LLM wrapper to include in metric instantiation

        Returns:
            BaseMetric instance

        Raises:
            ValueError: If class not found or instantiation fails
        """
        metric_class = self.get_class(class_name)

        try:
            # Add llm to parameters for metrics that accept it
            params_with_llm = {**parameters, "llm": llm}
            return metric_class(**params_with_llm)
        except TypeError as e:
            sig = inspect.signature(metric_class.__init__)
            raise ValueError(f"Invalid parameters for {class_name}: {e}\nExpected signature: {sig}")

    def list_classes(self) -> list[str]:
        """Return sorted list of available class names."""
        return sorted(self._classes.keys())

    @classmethod
    def create_default(cls) -> "MetricsRegistry":
        """Factory method for default registry with auto-discovery."""
        return cls()


def load_metrics_config(config_path: str) -> list[dict]:
    """
    Load metrics configuration from JSON or YAML file.

    Returns raw metric definitions without instantiation.

    Args:
        config_path: Path to configuration file

    Returns:
        List of metric definition dictionaries

    Raises:
        ValueError: If config file invalid or can't be loaded
    """
    # File parsing
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            config = json.load(f)
    elif config_path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ValueError(
                "YAML support requires 'pyyaml' package.\n"
                "Install with: uv add pyyaml\n"
                "Or use JSON format instead: metrics.json"
            )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}\nSupported formats: .json, .yaml, .yml")

    # Validation
    if "metrics" not in config:
        raise ValueError("Config file must contain 'metrics' key")

    if not isinstance(config["metrics"], list):
        raise ValueError("'metrics' must be a list")

    if not config["metrics"]:
        raise ValueError("Config file contains no valid metrics")

    # Return raw definitions
    return config["metrics"]


def instantiate_metric(metric_def: dict, llm: Any, registry: MetricsRegistry) -> BaseMetric:
    """
    Instantiate a single metric from its definition.

    Args:
        metric_def: Metric definition dictionary
        llm: LLM wrapper to pass to metric
        registry: MetricsRegistry for class lookup

    Returns:
        Instantiated BaseMetric

    Raises:
        ValueError: If definition is invalid
    """
    if "type" not in metric_def:
        raise ValueError("Metric definition must include 'type' field")

    metric_type = metric_def["type"]

    if metric_type == "class":
        if "class_name" not in metric_def:
            raise ValueError("Class type requires 'class_name' field")

        class_name = metric_def["class_name"]
        parameters = metric_def.get("parameters", {})
        return registry.instantiate_metric(class_name, parameters, llm)
    else:
        raise ValueError(f"Unknown metric type '{metric_type}'.\nSupported types: 'class'")


@dataclass
class EvaluationScores:
    """Evaluation scores and results."""

    overall_scores: dict[str, float]
    individual_results: list[dict[str, Any]]
    total_tokens: dict[str, int]
    total_cost: float


def format_experiment_results(
    experiment_file: str,
    metric_definitions: list[dict],
) -> EvaluationScores:
    """
    Format experiment results into the expected EvaluationScores structure.

    Reads the experiment results JSONL file produced by @experiment() and:
    1. Calculates overall scores (mean of each metric)
    2. Extracts individual results with all fields
    3. Sets token usage and cost to zero (tracking not yet implemented)

    Args:
        experiment_file: Path to experiment results JSONL file
        metric_definitions: List of metric definition dicts from config

    Returns:
        EvaluationScores with overall_scores, individual_results, total_tokens, total_cost
    """
    # Load all experiment results
    individual_results = []
    with open(experiment_file, "r") as f:
        for line in f:
            if line.strip():
                individual_results.append(json.loads(line))

    if not individual_results:
        raise ValueError(f"No results found in {experiment_file}")

    # Calculate overall scores (mean of each metric)
    # Extract metric names from definitions
    metric_names = []
    for metric_def in metric_definitions:
        if metric_def.get("type") == "class" and "class_name" in metric_def:
            metric_names.append(metric_def["class_name"])

    overall_scores = {}

    for metric_name in metric_names:
        # Collect all non-None scores for this metric
        scores = [r[metric_name] for r in individual_results if r.get(metric_name) is not None]
        if scores:
            overall_scores[metric_name] = sum(scores) / len(scores)
        else:
            logger.warning(f"No valid scores found for metric: {metric_name}")
            overall_scores[metric_name] = 0.0

    # TODO: Phase 4 - Extract token usage from experiment results if available
    # For now, set to zero as we don't yet know how @experiment() tracks tokens
    logger.info("Token usage tracking not yet implemented for @experiment() pattern")
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    total_cost = 0.0

    return EvaluationScores(
        overall_scores=overall_scores,
        individual_results=individual_results,
        total_tokens=total_tokens,
        total_cost=total_cost,
    )


@experiment()
async def evaluation_experiment(
    row: dict[str, Any],
    metric_definitions: list[dict],
    llm: Any,  # LangchainLLMWrapper - using Any to avoid mypy type alias issue
    registry: MetricsRegistry,
) -> dict[str, Any]:
    """
    Evaluate a single sample using RAGAS metrics.

    This function is decorated with @experiment() to enable automatic result tracking
    and batch processing across the dataset.

    Args:
        row: Dataset row containing user_input, response, retrieved_contexts, reference
        metric_definitions: List of metric definition dicts from config
        llm: LLM wrapper for metric calculation
        registry: MetricsRegistry for instantiation

    Returns:
        Dictionary with original row data plus metric scores
    """
    result = dict(row)
    result["individual_results"] = {}

    # Instantiate and calculate each metric for this row
    for metric_def in metric_definitions:
        try:
            # Instantiate metric from definition with LLM
            metric = instantiate_metric(metric_def, llm, registry)

            # Get the parameters that ascore expects
            sig = inspect.signature(metric.ascore)
            expected_params = set(sig.parameters.keys())

            # Filter row to only include fields that ascore expects
            filtered_params = {k: v for k, v in row.items() if k in expected_params}

            if "user_input" in filtered_params and isinstance(filtered_params["user_input"], list):
                filtered_params["user_input"] = map_user_input(filtered_params["user_input"])

            if "reference_tool_calls" in filtered_params and isinstance(filtered_params["reference_tool_calls"], list):
                filtered_params["reference_tool_calls"] = map_reference_tool_calls(
                    filtered_params["reference_tool_calls"]
                )

            # Calculate metric_result with only the required parameters
            metric_result = await metric.ascore(**filtered_params)  # type: ignore[call-arg]
            result["individual_results"][metric.name] = metric_result.value
        except Exception as e:
            metric_name = metric_def.get("class_name", "unknown")
            logger.warning(f"Failed to calculate {metric_name} for row: {e}")
            result[metric_name] = None

    return result


def map_user_input(user_input: list[Any]) -> list[Union[HumanMessage, AIMessage, ToolMessage]]:
    """
    Map input message dicts to appropriate LangChain message types.

    Args:
        user_input: List of message dictionaries with 'type' and 'content' fields

    Returns:
        List of LangChain message objects (HumanMessage, AIMessage, ToolMessage)
    """
    mapped_messages: list[Union[HumanMessage, AIMessage, ToolMessage]] = []
    for input_msg in user_input:
        if isinstance(input_msg, dict) and "type" in input_msg:
            msg_type = input_msg["type"]
            content = input_msg.get("content", "")

            if msg_type == "human":
                mapped_messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                mapped_messages.append(AIMessage(content=content))
            elif msg_type == "tool":
                mapped_messages.append(ToolMessage(content=content, tool_call_id=input_msg.get("tool_call_id", "")))
            else:
                logger.warning(f"Unknown message type '{msg_type}', treating as human message")
                mapped_messages.append(HumanMessage(content=content))
        else:
            # If not a dict with type, treat as human message
            logger.warning(f"Invalid message format, treating as human message: {input_msg}")
            content_str = str(input_msg.get("content", "")) if isinstance(input_msg, dict) else str(input_msg)
            mapped_messages.append(HumanMessage(content=content_str))
    return mapped_messages


def map_reference_tool_calls(referenced_tool_calls):
    """
    Map reference tool call dicts to proper structure expected by RAGAS metrics.

    Converts tool call dictionaries to a standardized format with name, args, and id fields.
    """
    mapped_tool_calls = []
    for tool_call in referenced_tool_calls:
        if isinstance(tool_call, dict):
            # Ensure tool call has required fields
            mapped_call = {
                "name": tool_call.get("name", ""),
                "args": tool_call.get("args", {}),
                "id": tool_call.get("id", ""),
            }
            mapped_tool_calls.append(mapped_call)
        else:
            # If not a dict, keep original
            mapped_tool_calls.append(tool_call)
    return mapped_tool_calls


async def main(
    model: str,
    metrics_config: str,
    cost_per_input_token: float = 5.0 / 1e6,
    cost_per_output_token: float = 15.0 / 1e6,
) -> None:
    """
    Main function to evaluate results using RAGAS metrics.

    Args:
        model: Model name to use for evaluation
        metrics_config: Path to metrics configuration file (JSON or YAML)
        cost_per_input_token: Cost per input token
        cost_per_output_token: Cost per output token
    """
    # Load metric definitions from configuration file
    logger.info(f"Loading metrics from config: {metrics_config}")
    metric_definitions = load_metrics_config(metrics_config)
    logger.info(f"Loaded {len(metric_definitions)} metric definitions")

    # Create LLM client using the AI-Gateway
    # Setting a placeholder for the api_key since we instantiate a ChatOpenAI object,
    # but the AI-Gateway actually uses Gemini under the hood.
    # Not setting api_key here results in an OpenAIError
    ragas_llm: AsyncOpenAI = AsyncOpenAI(api_key="Placeholder->NotUsed")
    llm = llm_factory(model, client=ragas_llm)  # type: ignore[arg-type]

    dataset = Experiment.load(name="ragas_experiment", backend=LocalJSONLBackend(root_dir="./data"))

    # Extract metric names from definitions for logging
    metric_names = [d.get("class_name", "unknown") for d in metric_definitions]
    logger.info(f"Calculating metrics: {', '.join(metric_names)}...")

    # Create registry for metric instantiation
    registry = MetricsRegistry.create_default()

    # Run evaluation experiment - this will process each row and save results automatically
    # Metrics are instantiated per-row inside evaluation_experiment()
    # EvaluationDataset is compatible with Dataset[Any] at runtime
    await evaluation_experiment.arun(
        dataset=dataset,  # type: ignore[arg-type]
        name="ragas_evaluation",
        metric_definitions=metric_definitions,
        llm=llm,
        registry=registry,
    )

    logger.info("Evaluation experiment completed")
    logger.info("Evaluation scores saved to './data/experiments/ragas_evaluation.jsonl'")


if __name__ == "__main__":
    # Create registry for help text generation
    registry = MetricsRegistry.create_default()

    # Parse the parameters (model and metrics-config) evaluate.py was called with
    parser = argparse.ArgumentParser(
        description="Evaluate results using RAGAS metrics via configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""

Available metric classes (configurable via --metrics-config):
  {", ".join(registry.list_classes())}

Examples:
  python3 scripts/evaluate.py gemini-2.5-flash-lite --metrics-config examples/metrics_simple.json
  python3 scripts/evaluate.py gemini-2.5-flash-lite --metrics-config examples/metrics_advanced.json

Config file format (JSON):
  {{
    "version": "1.0",
    "metrics": [
      {{
        "type": "class",
        "class_name": "AspectCritic",
        "parameters": {{"name": "harmfulness", "definition": "Is this harmful?"}}
      }}
    ]
  }}
        """,
    )

    parser.add_argument(
        "model",
        type=str,
        help="Model name to use for evaluation (e.g., gemini-2.5-flash-lite)",
    )

    parser.add_argument(
        "--metrics-config",
        type=str,
        default="config/metrics.json",
        help="Path to metrics configuration file (JSON or YAML). Default: config/metrics.json",
    )

    parser.add_argument(
        "--cost-per-input",
        type=float,
        default=5.0 / 1e6,
        help="Cost per input token (default: 5.0/1M = $0.000005 for typical GPT-4 pricing)",
    )

    parser.add_argument(
        "--cost-per-output",
        type=float,
        default=15.0 / 1e6,
        help="Cost per output token (default: 15.0/1M = $0.000015 for typical GPT-4 pricing)",
    )

    args = parser.parse_args()

    # Run evaluation with the 'model' and 'metrics_config' provided as parameters, 'output_file' is hardcoded
    asyncio.run(
        main(
            model=args.model,
            metrics_config=args.metrics_config,
            cost_per_input_token=args.cost_per_input,
            cost_per_output_token=args.cost_per_output,
        )
    )
