"""
End-to-end test script that runs all scripts in the correct order:
1. setup.py - Downloads, converts and saves Ragas Dataset to data/datasets/ragas_dataset.jsonl
2. run.py - Runs agent queries on the dataset and saves Ragas Experiment to data/experiments/ragas_experiment.jsonl
3. evaluate.py - Evaluates results using RAGAS metrics and saves result to results/evaluation_scores.json
4. publish.py - Publishes metrics via OpenTelemetry OTLP

Usage:
    python test_e2e.py --dataset-url <URL> --agent-url <URL> --model <MODEL> \
        --metrics <METRIC1> [METRIC2 ...] --workflow-name <NAME> \
        [--otlp-endpoint <URL>]

Example:
    python test_e2e.py \
        --dataset-url "https://example.com/dataset.csv" \
        --agent-url "http://localhost:8000" \
        --model "gemini-flash-latest" \
        --metrics faithfulness answer_relevancy \
        --workflow-name "weather-assistant-test"
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class E2ETestRunner:
    """Manages the end-to-end test execution pipeline."""

    def __init__(
        self,
        dataset_url: str,
        agent_url: str,
        model: str,
        metrics: List[str],
        workflow_name: str,
        otlp_endpoint: str = "localhost:4318",
    ):
        self.dataset_url = dataset_url
        self.agent_url = agent_url
        self.model = model
        self.metrics = metrics
        self.workflow_name = workflow_name
        self.otlp_endpoint = otlp_endpoint

        # Define script paths
        self.scripts_dir = Path(__file__).parent.parent / "scripts"
        self.setup_script = self.scripts_dir / "setup.py"
        self.run_script = self.scripts_dir / "run.py"
        self.evaluate_script = self.scripts_dir / "evaluate.py"
        self.publish_script = self.scripts_dir / "publish.py"

        # Define expected output files
        self.dataset_file = Path("./data/datasets/ragas_dataset.jsonl")
        self.results_file = Path("./data/experiments/ragas_experiment.jsonl")
        self.evaluation_file = Path("./results/evaluation_scores.json")

    def verify_scripts_exist(self) -> bool:
        """Verify that all required scripts exist."""
        logger.info("Verifying all scripts exist...")
        scripts = [
            self.setup_script,
            self.run_script,
            self.evaluate_script,
            self.publish_script,
        ]

        missing_scripts = [script for script in scripts if not script.exists()]

        if missing_scripts:
            logger.error("Missing scripts:")
            for script in missing_scripts:
                logger.error(f"  - {script}")
            return False

        logger.info("✓ All scripts found")
        return True

    def run_command(self, command: List[str], step_name: str) -> bool:
        """
        Run a command and handle output/errors.

        Args:
            command: List of command arguments
            step_name: Name of the step for logging

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Step: {step_name}")
        logger.info(f"Command: {' '.join(command)}")
        logger.info(f"{'=' * 60}\n")

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            # Log stdout if present
            if result.stdout:
                logger.info("Output:")
                for line in result.stdout.strip().split("\n"):
                    logger.info(f"  {line}")

            logger.info(f"✓ {step_name} completed successfully\n")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {step_name} failed with exit code {e.returncode}")

            if e.stdout:
                logger.error("Standard output:")
                for line in e.stdout.strip().split("\n"):
                    logger.error(f"  {line}")

            if e.stderr:
                logger.error("Error output:")
                for line in e.stderr.strip().split("\n"):
                    logger.error(f"  {line}")

            return False

        except Exception as e:
            logger.error(f"✗ Unexpected error in {step_name}: {e}")
            return False

    def verify_file_exists(self, file_path: Path, step_name: str) -> bool:
        """Verify that an expected output file was created."""
        if file_path.exists():
            logger.info(f"✓ Verified {file_path} was created by {step_name}")
            return True
        else:
            logger.error(f"✗ Expected file {file_path} not found after {step_name}")
            return False

    def run_setup(self) -> bool:
        """Run setup.py to download and convert dataset."""
        command = ["python3", str(self.setup_script), self.dataset_url]
        success = self.run_command(command, "1. Setup - Download Dataset")

        if success:
            return self.verify_file_exists(self.dataset_file, "setup.py")
        return False

    def run_agent_queries(self) -> bool:
        """Run run.py to execute agent queries on the dataset."""
        command = ["python3", str(self.run_script), self.agent_url]
        success = self.run_command(command, "2. Run - Execute Agent Queries")

        if success:
            return self.verify_file_exists(self.results_file, "run.py")
        return False

    def run_evaluation(self) -> bool:
        """Run evaluate.py to evaluate results using RAGAS metrics."""
        command = ["python3", str(self.evaluate_script), self.model, *self.metrics]
        success = self.run_command(command, "3. Evaluate - Calculate RAGAS Metrics")

        if success:
            return self.verify_file_exists(self.evaluation_file, "evaluate.py")
        return False

    def run_publish(self) -> bool:
        """Run publish.py to publish metrics via OpenTelemetry OTLP."""
        command = [
            "python3",
            str(self.publish_script),
            self.workflow_name,
            self.otlp_endpoint,
        ]
        return self.run_command(command, "4. Publish - Push Metrics via OTLP")

    def run_full_pipeline(self) -> bool:
        """Execute the complete E2E test pipeline."""
        logger.info("\n" + "=" * 60)
        logger.info("Starting E2E Test Pipeline")
        logger.info("=" * 60 + "\n")

        # Verify all scripts exist before starting
        if not self.verify_scripts_exist():
            logger.error("Cannot proceed - missing required scripts")
            return False

        # Run each step in order
        steps = [
            ("Setup", self.run_setup),
            ("Run", self.run_agent_queries),
            ("Evaluate", self.run_evaluation),
            ("Publish", self.run_publish),
        ]

        for step_name, step_func in steps:
            if not step_func():
                logger.error(f"\n{'=' * 60}")
                logger.error(f"E2E Test FAILED at step: {step_name}")
                logger.error(f"{'=' * 60}\n")
                return False

        # All steps completed successfully
        logger.info("\n" + "=" * 60)
        logger.info("E2E Test Pipeline COMPLETED SUCCESSFULLY")
        logger.info("=" * 60 + "\n")

        logger.info("Summary:")
        logger.info(f"  ✓ Ragas Dataset created: {self.dataset_file}")
        logger.info(f"  ✓ Ragas Experiment saved: {self.results_file}")
        logger.info(f"  ✓ Evaluation completed: {self.evaluation_file}")
        logger.info(f"  ✓ Metrics published to: {self.otlp_endpoint}")
        logger.info(f"  ✓ Workflow name: {self.workflow_name}")

        return True


def main():
    """Parse arguments and run the E2E test"""
    parser = argparse.ArgumentParser(
        description="End-to-end test runner for the agent testing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                  # Basic usage
                  python test_e2e.py \\
                      --dataset-url "https://example.com/data.csv" \\
                      --agent-url "http://localhost:8000" \\
                      --model "gemini-flash-latest" \\
                      --metrics faithfulness answer_relevancy \\
                      --workflow-name "my-test"

                  # With custom OTLP endpoint
                  python test_e2e.py \\
                      --dataset-url "https://example.com/data.csv" \\
                      --agent-url "http://localhost:8000" \\
                      --model "gemini-flash-latest" \\
                      --metrics faithfulness \\
                      --workflow-name "my-test" \\
                      --otlp-endpoint "http://otlp.example.com:4318"
               """,
    )

    parser.add_argument(
        "--dataset-url",
        required=True,
        help="URL to the dataset in .csv / .json / .parquet format",
    )

    parser.add_argument("--agent-url", required=True, help="URL to agent")

    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use for evaluation (e.g., gemini-flash-latest)",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="At least one (or more) metrics to evaluate (e.g., faithfulness, answer_relevancy)",
    )

    parser.add_argument(
        "--workflow-name",
        required=True,
        help="Name of the test workflow (e.g., 'weather-assistant-test')",
    )

    parser.add_argument(
        "--otlp-endpoint",
        default="localhost:4318",
        help="URL of the OTLP HTTP endpoint (default: localhost:4318)",
    )

    args = parser.parse_args()

    # Create and run the test pipeline
    runner = E2ETestRunner(
        dataset_url=args.dataset_url,
        agent_url=args.agent_url,
        model=args.model,
        metrics=args.metrics,
        workflow_name=args.workflow_name,
        otlp_endpoint=args.otlp_endpoint,
    )

    success = runner.run_full_pipeline()

    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
