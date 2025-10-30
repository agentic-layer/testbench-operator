import argparse
import inspect
import json
import logging
import os
from pydantic import BaseModel
from argparse import ArgumentError
from logging import Logger
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import evaluate, Experiment
from ragas.dataset_schema import EvaluationDataset, EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Metric
import ragas.metrics as metrics_module
from langchain_google_genai import ChatGoogleGenerativeAI


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
                pass

    return available_metrics


# Get all available metrics
AVAILABLE_METRICS = get_available_metrics()


def calculate_metrics(dataset: EvaluationDataset, metrics: list[str], llm: Any) -> dict[str, Any]:
    """
    Calculate RAGAS metrics on the dataset.

    Args:
        dataset: Ragas EvaluationDataset
        metrics: List of metric names as strings (e.g., ["faithfulness", "answer_relevancy"])
        llm: Language model for evaluation

    Returns:
        Dictionary containing evaluation results
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

    # Run RAGAS evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metric_objects,
        llm=llm,
    )

    return result


def format_evaluation_scores(
    ragas_result: EvaluationResult,
    metrics: list[str]
) -> dict[str, Any]:

    """
    Format the RAGAS evaluation results.

    Args:
        ragas_result: The result object from RAGAS evaluate()
        experiment: Ragas Experiment containing the original results
        metrics: List of metric names used

    Returns:
        Formatted dictionary matching the required structure
    """

    # Get the pandas DataFrame from RAGAS result
    df = ragas_result.to_pandas()

    # Calculate overall scores (mean of each metric)
    overall_scores = {}
    for metric in metrics:
        if metric in df.columns:
            overall_scores[metric] = float(df[metric].mean())

    # Build individual results
    individual_results = ragas_result.to_pandas().to_dict(orient="records")

    # Ragas doesn't have TokenUsageParser for Google Gemini - setting placeholders instead
    total_tokens = {
        "input_tokens": 0,
        "output_tokens": 0
    }
    total_cost = 0.0


    return {
        "overall_scores": overall_scores,
        "individual_results": individual_results,
        "total_tokens": total_tokens,
        "total_cost": total_cost
    }


def main(
    output_file: str,
    model: str,
    metrics: list[str] = None
) -> None:
    """
    Main function to evaluate results using RAGAS metrics.

    Args:
        output_file: Path to save evaluation_scores.json
        model: Model name to use for evaluation
        metrics: List of metric names to calculate
    """

    ragas_llm = ChatOpenAI(
        model=model
    )
    llm = LangchainLLMWrapper(ragas_llm)

    dataset = EvaluationDataset.from_jsonl('data/experiments/ragas_experiment.jsonl')

    # Calculate metrics
    logger.info(f"Calculating metrics: {', '.join(metrics)}...")
    ragas_result = calculate_metrics(dataset, metrics, llm)

    # Format results
    logger.info("Formatting evaluation scores...")
    evaluation_scores = format_evaluation_scores(ragas_result, metrics)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(evaluation_scores, f, indent=2)

    logger.info(f"Evaluation scores saved to {output_file}")
    logger.info(f"Overall scores: {evaluation_scores['overall_scores']}")


if __name__ == "__main__":
    # Set up logger & get logger instance
    logging.basicConfig(level=logging.INFO)
    logger: Logger = logging.getLogger(__name__)

    # Parse the parameters (model and metrics) evaluate.py was called with
    parser = argparse.ArgumentParser(
        description = "Evaluate results using RAGAS metrics",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog=f"""
            Available metrics: {', '.join(AVAILABLE_METRICS.keys())}

            Examples:
            python3 scripts/evaluate.py gemini-flash-latest faithfulness
            python3 scripts/evaluate.py gemini-flash-latest faithfulness context_precision context_recall
        """
    )

    parser.add_argument(
        'model',
        type = str,
        help = 'Model name to use for evaluation (e.g., gemini-flash-latest)'
    )

    parser.add_argument(
        'metrics',
        nargs = '+',
        choices = list(AVAILABLE_METRICS.keys()),
        help = 'At least one (or more) metrics to evaluate (e.g., faithfulness, answer_relevancy)'
    )

    args = parser.parse_args()

    # Run evaluation with the 'model' and 'metrics' provided as parameters, 'output_file' is hardcoded
    main(
        output_file = "results/evaluation_scores.json",
        model = args.model,
        metrics = args.metrics
    )
