import argparse
import inspect
import json
import logging
import os
from argparse import ArgumentError
from logging import Logger
from typing import Any
from dotenv import load_dotenv
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.cost import get_token_usage_for_openai
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


def load_results(file_path: str) -> dict[str, Any]:
    # Load the results.json file from the data directory
    with open(file_path, 'r') as f:
        return json.load(f)


def prepare_ragas_dataset(results: list[dict]) -> Dataset:
    """
    Convert results to RAGAS dataset format.

    RAGAS expects:
    - user_input: The input question
    - response: The agent's response
    - contexts: List of context strings (from ground_truth or other sources)
    - ground_truth: The expected answer (optional, needed for some metrics)
    """
    data = {
        "user_input": [],
        "response": [],
        "contexts": [],
        "ground_truth": []
    }

    for result in results:
        data["user_input"].append(result.get("input", ""))
        data["response"].append(result.get("output", ""))

        # Use ground_truth as context (wrap in list as RAGAS expects list of contexts)
        ground_truth = result.get("ground_truth", "")
        data["contexts"].append([ground_truth] if ground_truth else [""])
        data["ground_truth"].append(ground_truth)

    return Dataset.from_dict(data)


def calculate_metrics(dataset: Dataset, metrics: list[str], llm: Any) -> dict[str, Any]:
    """
    Calculate RAGAS metrics on the dataset.

    Args:
        dataset: RAGAS-formatted dataset
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

    # Run RAGAS evaluation with token usage parser
    result = evaluate(
        dataset=dataset,
        metrics=metric_objects,
        llm=llm,
        token_usage_parser=get_token_usage_for_openai,
    )

    return result


def format_evaluation_scores(
    ragas_result: Any,
    results: list[dict],
    metrics: list[str]
) -> dict[str, Any]:

    """
    Format the RAGAS evaluation results.

    Args:
        ragas_result: The result object from RAGAS evaluate()
        results: Original results list from results.json
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
    individual_results = []
    for idx, result in enumerate(results):
        individual_result = {
            "question": result.get("input", ""),
            "answer": result.get("output", ""),
            "contexts": [result.get("ground_truth", "")],
            "scores": {}
        }

        # Add scores for each metric
        for metric in metrics:
            if metric in df.columns and idx < len(df):
                individual_result["scores"][metric] = float(df.iloc[idx][metric])

        individual_results.append(individual_result)

    # Extract token usage from RAGAS result
    total_tokens = {
        "input_tokens": ragas_result.total_tokens()[0],
        "output_tokens": ragas_result.total_tokens()[1]
    }

    # Calculate the total cost
    # ''cost_per_input_token' and 'cost_per_output_token' can be adjusted as needed
    total_cost = ragas_result.total_cost(
        cost_per_input_token=0.00003,  # $0.03 per 1K input tokens
        cost_per_output_token=0.00006   # $0.06 per 1K output tokens
    )


    return {
        "overall_scores": overall_scores,
        "individual_results": individual_results,
        "total_tokens": total_tokens,
        "total_cost": total_cost
    }


def main(
    input_file: str,
    output_file: str,
    model: str,
    metrics: list[str] = None
) -> None:
    """
    Main function to evaluate results using RAGAS metrics.

    Args:
        input_file: Path to results.json
        output_file: Path to save evaluation_scores.json
        model: Model name to use for evaluation
        metrics: List of metric names to calculate
    """
    # Load environment variables
    load_dotenv()

    # Check if any metrics were provided
    if metrics is None:
        raise ArgumentError("No metrics were provided as arguments")

    # Initialize the LLM for RAGAS evaluation
    # Using Google Generative AI (Gemini) via API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")



    # Create LLM client using the AI Gateway (environment variables 'OPEN_API_KEY' and 'OPENAI_API_BASE' need to be set)
    ragas_llm = ChatOpenAI(
        model=model
    )
    llm = LangchainLLMWrapper(ragas_llm)

    # Load results
    logger.info(f"Loading results from {input_file}...")
    data = load_results(input_file)
    results = data.get("results", [])

    if not results:
        raise ValueError("No results found in input file")

    logger.info(f"Found {len(results)} results to evaluate")

    # Prepare dataset for RAGAS
    logger.info("Preparing dataset for RAGAS evaluation...")
    dataset = prepare_ragas_dataset(results)

    # Calculate metrics
    logger.info(f"Calculating metrics: {', '.join(metrics)}...")
    ragas_result = calculate_metrics(dataset, metrics, llm)

    # Format results
    logger.info("Formatting evaluation scores...")
    evaluation_scores = format_evaluation_scores(ragas_result, results, metrics)

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

    # Run evaluation with the 'model' and 'metrics' provided as parameters, 'input_file' and 'output_file' are hardcoded
    main(
        input_file = "data/results.json",
        output_file = "results/evaluation_scores.json",
        model = args.model,
        metrics = args.metrics
    )
