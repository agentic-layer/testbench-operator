import argparse
import inspect
import json
import logging
import os
from argparse import ArgumentError
from dataclasses import asdict, dataclass
from logging import Logger
from typing import Any

import ragas.metrics as metrics_module
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ragas import evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.dataset_schema import EvaluationDataset, EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Metric

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)


def get_available_metrics() -> dict[str, Metric]:
    """
    Loads all Metric classes from Ragas
    Returns a dict mapping metric names to metric instances.
    """
    available_metrics: dict[str, Metric] = {}

    # Iterate through all members of the metrics module
    for name, obj in inspect.getmembers(metrics_module):
        # Check if it's a class and is a subclass of Metric (but not Metric itself)
        if inspect.isclass(obj) and issubclass(obj, Metric) and obj is not Metric:
            try:
                # Instantiate the metric
                metric_instance = obj()
                # Use the metric's name attribute
                metric_name = metric_instance.name
                available_metrics[metric_name] = metric_instance
            except Exception:
                # Skip metrics that can't be instantiated without parameters
                logger.info(f"Exception encountered: {Exception}")
                pass

    return available_metrics


# Get all available metrics
AVAILABLE_METRICS = get_available_metrics()


def convert_metrics(metrics: list[str]) -> list:
    """
    Map metric names to actual metric objects

    Args:
        metrics: List of metric names as strings (e.g., ["faithfulness", "answer_relevancy"])

    Returns:
        List containing metric objects
    """

    # Map metric names to actual metric objects
    metric_objects = []
    for metric_name in metrics:
        if metric_name in AVAILABLE_METRICS:
            metric_objects.append(AVAILABLE_METRICS[metric_name])
        else:
            logger.warning(f"Unknown metric '{metric_name}', skipping...")
            logger.warning(f"Available metrics: {', '.join(AVAILABLE_METRICS.keys())}")

    if not metric_objects:
        raise ValueError("No valid metrics provided for evaluation")

    return metric_objects


@dataclass
class EvaluationScores:
    """Evaluation scores and results."""

    overall_scores: dict[str, float]
    individual_results: list[dict[str, Any]]
    total_tokens: dict[str, int]
    total_cost: float


def format_evaluation_scores(
    ragas_result: EvaluationResult,
    cost_per_input_token: float,
    cost_per_output_token: float,
    experiment_file: str,
) -> EvaluationScores:
    """
    Format the RAGAS evaluation results.

    Args:
        ragas_result: The result object from RAGAS evaluate()
        cost_per_input_token: Cost per input token
        cost_per_output_token: Cost per output token
        experiment_file: Path to experiment JSONL file (to extract trace_ids)

    Returns:
        Formatted dictionary matching the required structure
    """

    # Load trace_ids from experiment file (RAGAS drops custom fields during processing)
    trace_ids = []
    with open(experiment_file, "r") as f:
        for line in f:
            data = json.loads(line)
            trace_ids.append(data.get("trace_id"))

    # Calculate overall scores (mean of each metric)
    overall_scores = ragas_result._repr_dict

    # Build individual results
    individual_results = ragas_result.to_pandas().to_dict(orient="records")

    # Merge trace_ids back into individual_results (preserve by row order)
    for i, result in enumerate(individual_results):
        if i < len(trace_ids):
            result["trace_id"] = trace_ids[i]
        else:
            logger.warning(f"No trace_id found for result {i}")
            result["trace_id"] = None

    # Extract token usage and calculate cost using TokenUsageParser
    # Check if token usage data was collected (some metrics don't use LLMs or use separate LLM instances)
    if ragas_result.cost_cb and hasattr(ragas_result.cost_cb, "usage_data") and ragas_result.cost_cb.usage_data:
        token_usage = ragas_result.total_tokens()
        # Handle both single TokenUsage and list of TokenUsage
        if isinstance(token_usage, list):
            input_tokens = sum(usage.input_tokens for usage in token_usage)
            output_tokens = sum(usage.output_tokens for usage in token_usage)
        else:
            input_tokens = token_usage.input_tokens
            output_tokens = token_usage.output_tokens

        total_tokens = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        total_cost = ragas_result.total_cost(
            cost_per_input_token=cost_per_input_token,
            cost_per_output_token=cost_per_output_token,
        )
    else:
        # No token usage data collected (e.g., non-LLM metrics or Nvidia metrics using separate LLM instances)
        logger.info("No token usage data collected for these metrics")
        total_tokens = {"input_tokens": 0, "output_tokens": 0}
        total_cost = 0.0

    return EvaluationScores(
        overall_scores=overall_scores,
        individual_results=individual_results,
        total_tokens=total_tokens,
        total_cost=total_cost,
    )


def main(
    output_file: str,
    model: str,
    metrics: list[str] | None = None,
    cost_per_input_token: float = 5.0 / 1e6,
    cost_per_output_token: float = 15.0 / 1e6,
) -> None:
    """
    Main function to evaluate results using RAGAS metrics.

    Args:
        output_file: Path to save evaluation_scores.json
        model: Model name to use for evaluation
        metrics: List of metric names to calculate
    """
    # Check if any metrics were provided
    if metrics is None:
        raise ArgumentError(argument=metrics, message="No metrics were provided as arguments")

    # Create LLM client using the AI-Gateway
    # Setting a placeholder for the api_key since we instantiate a ChatOpenAI object,
    # but the AI-Gateway actually uses Gemini under the hood.
    # Not setting api_key here results in an OpenAIError
    ragas_llm: ChatOpenAI = ChatOpenAI(model=model, api_key=SecretStr("Placeholder->NotUsed"))
    llm = LangchainLLMWrapper(ragas_llm)  # type: ignore[arg-type]

    dataset = EvaluationDataset.from_jsonl("data/experiments/ragas_experiment.jsonl")

    # Calculate metrics
    logger.info(f"Calculating metrics: {', '.join(metrics)}...")
    ragas_result = evaluate(
        dataset=dataset,
        metrics=convert_metrics(metrics),
        llm=llm,
        token_usage_parser=get_token_usage_for_openai,
    )

    # Ensure we have an EvaluationResult (not an Executor)
    if not isinstance(ragas_result, EvaluationResult):
        raise TypeError(f"Expected EvaluationResult, got {type(ragas_result)}")

    # Format results
    logger.info("Formatting evaluation scores...")
    evaluation_scores = format_evaluation_scores(
        ragas_result,
        cost_per_input_token=cost_per_input_token,
        cost_per_output_token=cost_per_output_token,
        experiment_file="data/experiments/ragas_experiment.jsonl",
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to file
    with open(output_file, "w") as f:
        json.dump(asdict(evaluation_scores), f, indent=2)

    logger.info(f"Evaluation scores saved to {output_file}")
    logger.info(f"Overall scores: {evaluation_scores.overall_scores}")


if __name__ == "__main__":
    # Parse the parameters (model and metrics) evaluate.py was called with
    parser = argparse.ArgumentParser(
        description="Evaluate results using RAGAS metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
            Available metrics: {", ".join(AVAILABLE_METRICS.keys())}

            Examples:
            python3 scripts/evaluate.py gemini-2.5-flash-lite faithfulness
            python3 scripts/evaluate.py gemini-2.5-flash-lite faithfulness context_precision context_recall
        """,
    )

    parser.add_argument(
        "model",
        type=str,
        help="Model name to use for evaluation (e.g., gemini-2.5-flash-lite)",
    )

    parser.add_argument(
        "metrics",
        nargs="+",
        choices=list(AVAILABLE_METRICS.keys()),
        help="At least one (or more) metrics to evaluate (e.g., faithfulness, answer_relevancy)",
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

    # Run evaluation with the 'model' and 'metrics' provided as parameters, 'output_file' is hardcoded
    main(
        output_file="data/results/evaluation_scores.json",
        model=args.model,
        metrics=args.metrics,
        cost_per_input_token=args.cost_per_input,
        cost_per_output_token=args.cost_per_output,
    )
