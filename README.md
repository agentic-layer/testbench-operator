# Agentic Layer Test Bench - Automated Agent Evaluation System

An automated evaluation and testing system for AI agents using the **RAGAS** (Retrieval Augmented Generation Assessment)
framework. This system downloads test datasets, executes queries through agents via the **A2A** protocol, evaluates
responses using configurable metrics, and publishes results to **OpenTelemetry** for monitoring.

----

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Dataset Requirements](#dataset-requirements)
- [Testing](#testing)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

----

## Overview

This project provides a complete pipeline for evaluating AI agent performance:

- **Automated Testing**: Run predefined test queries through your agents
- **Multi-Format Support**: Accept datasets in CSV, JSON, or Parquet formats
- **Flexible Evaluation**: Configure multiple RAGAS metrics for comprehensive assessment
- **Observability**: Publish metrics to OpenTelemetry endpoints for monitoring and analysis
- **Type-Safe**: Built with type hints and validated with MyPy
- **Limitation**: Currently only support SingleTurnSample Metrics (see [Available Metrics](#available-metrics))

----

## Architecture

```
Input Dataset (user_input, retrieved_contexts, reference)
        |
        v
 [1. setup.py] - Downloads & converts to RAGAS JSONL format
        |
        v
data/datasets/ragas_dataset.jsonl
        |
        v
 [2. run.py] - Executes queries via A2A protocol, adds responses
        |              ^
        |              |
        |         Agent URL
        v
data/experiments/ragas_experiment.jsonl
        |
        v
 [3. evaluate.py] - Calculates RAGAS metrics
        |              ^
        |              |
        |         LLM Model
        v
results/evaluation_scores.json
        |
        v
 [4. publish.py] - Publishes to OTLP endpoint
        |
        v
OpenTelemetry Collector
```

### Key Design Principles

- **RAGAS-Native Format**: Uses RAGAS column names (`user_input`, `response`, `retrieved_contexts`, `reference`)
  throughout
- **JSONL Backend**: Internal storage uses JSONL for native list support
- **Format-Aware Input**: Intelligent handling of CSV (list conversion), JSON, and Parquet formats

----

## Prerequisites

- **Python 3.13+**
- **API Key**: `OPENAI_API_KEY` environment variable (required for LLM-based evaluation)
- **OTLP Endpoint**: Optional, defaults to `localhost:4318`

----

## Getting Started

### With Tilt and Local Kubernetes

```shell
Start Tilt in the project root to set up the local Kubernetes environment:
tilt up
```

Run the RAGAS evaluation workflow with minimal setup:

```shell
kubectl testkube run testworkflow ragas-evaluation-workflow \
    --config datasetUrl="http://data-server.data-server:8000/dataset.csv" \
    --config agentUrl="http://agent-gateway-krakend.agent-gateway-krakend:10000/weather-agent" \
    --config metrics="nv_accuracy context_recall" \
    --config workflowName="Testworkflow-Name" \
    --config image="ghcr.io/agentic-layer/testbench/testworkflows:latest" \
    -n testkube
```

Run the RAGAS evaluation workflow with all optional parameters:

```shell
kubectl testkube run testworkflow ragas-evaluation-workflow \
    --config datasetUrl="http://data-server.data-server:8000/dataset.csv" \
    --config agentUrl="http://agent-gateway-krakend.agent-gateway-krakend:10000/weather-agent" \
    --config metrics="nv_accuracy context_recall"
    --config workflowName="Testworkflow-Name" \
    --config image="ghcr.io/agentic-layer/testbench/testworkflows:latest" \
    --config model="gemini/gemini-2.5-flash" \
    --config otlpEndpoint="http://otlp-endpoint:4093" \
    -n testkube
```

### Install dependencies using UV

```shell
# Install (dev & prod) dependencies with uv
uv sync
```

### Environment Setup

```shell
# Required for evaluation
export OPENAI_API_KEY="your-api-key-here"

# Optional: Configure custom OTLP endpoint
export OTLP_ENDPOINT="http://otlp-collector:4318"
```

The system automatically creates the required directories (`data/`, `results/`) on first run.

----

## Quick Start

Run the complete evaluation pipeline in 4 steps:

```shell
# 1. Download and prepare dataset
python3 scripts/setup.py "https://example.com/dataset.csv"

# 2. Execute queries through your agent
python3 scripts/run.py "http://localhost:8000"

# 3. Evaluate responses with RAGAS metrics
python3 scripts/evaluate.py gemini-2.5-flash-lite faithfulness answer_relevancy

# 4. Publish metrics to OpenTelemetry
python3 scripts/publish.py "my-agent-evaluation"
```

----

## Detailed Usage

### 1. setup.py - Dataset Preparation

Downloads and converts test datasets to RAGAS-native JSONL format.

**Syntax:**

```shell
python3 scripts/setup.py <dataset_url>
```

**Arguments:**

- `dataset_url` (required): URL to dataset file (`.csv`, `.json`, or `.parquet`)

**Required Dataset Schema:**

- See [Dataset Requirements](#dataset-requirements)

**Output:**

- `data/datasets/ragas_dataset.jsonl` - RAGAS Dataset in JSONL format

---

### 2. run.py - Agent Query Execution

Executes test queries through an agent using the A2A protocol and collects responses.

**Syntax:**

```shell
python3 scripts/run.py <agent_url>
```

**Arguments:**

- `agent_url` (required): URL to the agent's A2A endpoint

**Input:**

- `data/datasets/ragas_dataset.jsonl` (loaded automatically)

**Output:**

- `data/experiments/ragas_experiment.jsonl` - Agent responses with preserved context

**Output Schema:**

```jsonl
{"user_input": "What is X?", "retrieved_contexts": ["Context about X"], "reference": "X is...", "response": "Agent's answer"}
```

**Notes:**

- Uses asynchronous A2A client for efficient communication
- Preserves all original dataset fields
- Automatically handles response streaming

---

### 3. evaluate.py - RAGAS Metric Evaluation

Evaluates agent responses using configurable RAGAS metrics and calculates costs.

**Syntax:**

```shell
python3 scripts/evaluate.py <model> <metric1> [metric2 ...] [--cost-per-input COST] [--cost-per-output COST]
```

**Arguments:**

- `model` (required): Model name for evaluation (e.g., `gemini-2.5-flash-lite`, `gpt-4`)
- `metrics` (required): One or more RAGAS metric names
- `--cost-per-input` (optional): Cost per input token (default: 0.000005, i.e., $5 per 1M tokens)
- `--cost-per-output` (optional): Cost per output token (default: 0.000015, i.e., $15 per 1M tokens)

### **Available Metrics:**

| Metric                                    | Special required columns |
|-------------------------------------------|--------------------------|
| `faithfulness`                            | retrieved_contexts       |
| `context_precision`                       | retrieved_contexts       |
| `context_recall`                          | retrieved_contexts       |
| `context_entity_recall`                   | retrieved_contexts       |
| `context_utilization`                     | retrieved_contexts       |
| `llm_context_precision_with_reference`    | retrieved_contexts       |
| `llm_context_precision_without_reference` | retrieved_contexts       |
| `faithful_rate`                           | retrieved_contexts       |
| `relevance_rate`                          | retrieved_contexts       |
| `noise_sensitivity`                       | retrieved_contexts       |
| `factual_correctness`                     |                          |
| `domain_specific_rubrics`                 |                          |
| `nv_accuracy`                             |                          |
| `nv_context_relevance`                    | retrieved_contexts       |
| `nv_response_groundedness`                | retrieved_contexts       |
| `string_present`                          |                          |
| `exact_match`                             |                          |
| `summary_score`                           | reference_contexts       |
| `llm_sql_equivalence_with_reference`      | reference_contexts       |

**Input:**

- `data/experiments/ragas_experiment.jsonl` (loaded automatically)

**Examples:**

```shell
# Single metric
python3 scripts/evaluate.py gemini-2.5-flash-lite faithfulness

# Multiple metrics
python3 scripts/evaluate.py gemini-2.5-flash-lite faithfulness answer_relevancy context_precision

# Custom token costs
python3 scripts/evaluate.py gpt-4 faithfulness answer_correctness \
  --cost-per-input 0.00003 \
  --cost-per-output 0.00006
```

**Output:**

- `results/evaluation_scores.json` - Evaluation results with metrics, token usage, and costs

**Output Format:**

```json
{
  "overall_scores": {
    "faithfulness": 0.95,
    "answer_relevancy": 0.98
  },
  "individual_results": [
    {
      "user_input": "What is the capital of France?",
      "response": "Paris is the capital of France.",
      "faithfulness": 0.95,
      "answer_relevancy": 0.98
    }
  ],
  "total_tokens": {
    "input_tokens": 1500,
    "output_tokens": 500
  },
  "total_cost": 0.015
}
```

**Notes:**

- Currently only support **SingleTurnSample** Metrics (see [Available Metrics](#available-metrics))
- Dynamically discovers available metrics from `ragas.metrics` module
- Invalid metric names will show available options
- Token costs can be customized per model pricing

---

### 4. publish.py - Metrics Publishing

Publishes evaluation metrics to an OpenTelemetry OTLP endpoint for monitoring.

**Syntax:**

```shell
python3 scripts/publish.py <workflow_name> [otlp_endpoint]
```

**Arguments:**

- `workflow_name` (required): Name of the test workflow (used as metric label)
- `otlp_endpoint` (optional): OTLP HTTP endpoint URL (default: `localhost:4318`)

**Input:**

- `results/evaluation_scores.json` (loaded automatically)

**Published Metrics:**

Each RAGAS metric is published as a gauge with the workflow name as an attribute:

```
ragas_evaluation_faithfulness{workflow_name="weather-assistant-eval"} = 0.85
ragas_evaluation_answer_relevancy{workflow_name="weather-assistant-eval"} = 0.92
```

**Notes:**

- Sends metrics to `/v1/metrics` endpoint
- Uses resource with `service.name="ragas-evaluation"`
- Forces flush to ensure delivery before exit

----

## Dataset Requirements

### Schema

Your input dataset must contain these columns:

```
{
    "user_input": str,  # Test question/prompt
    "retrieved_contexts": [str],  # List of context strings (must be array type) (Optional but required by many metrics)
    "reference": str  # Ground truth answer
}
```

### Format-Specific Notes

**CSV Files:**

- `retrieved_contexts` must be formatted as quoted array strings
- Example: `"['Context 1', 'Context 2', 'Context 3']"`
- The system automatically parses these strings into Python lists

**JSON Files:**

```json
[
  {
    "user_input": "What is the capital of France?",
    "retrieved_contexts": [
      "Paris is a city in France.",
      "France is in Europe."
    ],
    "reference": "Paris"
  }
]
```

**Parquet Files:**

- Use native list/array columns for `retrieved_contexts`

### Example Dataset

```csv
user_input,retrieved_contexts,reference
"What is Python?","['Python is a programming language', 'Python was created by Guido van Rossum']","Python is a high-level programming language"
"What is AI?","['AI stands for Artificial Intelligence', 'AI systems can learn from data']","AI is the simulation of human intelligence by machines"
```

----

## Testing

### Unit Tests

Run all unit tests:

```shell
uv run poe test
```

### End-to-End Tests

Run the complete pipeline integration test:

```shell
uv run pytest tests_e2e/test_e2e.py -v
```

Or using the task runner:

```shell
uv run poe test_e2e
```

**Configuration via Environment Variables:**

```shell
export E2E_DATASET_URL="http://localhost:8000/dataset.json"
export E2E_AGENT_URL="http://localhost:11010"
export E2E_MODEL="gemini-2.5-flash-lite"
export E2E_METRICS="faithfulness,answer_relevancy"
export E2E_WORKFLOW_NAME="Test Workflow"
export E2E_OTLP_ENDPOINT="localhost:4318"

pytest tests_e2e/test_e2e.py -v
```

----

## Development

## Code Quality Standards

### Code Style:

- **Linting**: Ruff with 120 character line limit
- **Type Checking**: mypy for static type analysis
- **Security**: Bandit for security vulnerability detection
- **Import Organization**: import-linter for dependency management

### Development Commands:

This project uses `poethepoet` for task automation:

```shell
# Run all quality checks
uv run poe check

# Individual checks
uv run poe mypy          # Type checking
uv run poe ruff          # Linting and formatting
uv run poe bandit        # Security analysis
uv run poe lint-imports  # Import dependency validation
uv run poe test          # Execute test suite
uv run poe test_e2e      # Execute end-to-end tests

# Auto-formatting
uv run poe format        # Code formatting
uv run poe lint          # Auto-fix linting issues
```

----

## Troubleshooting

### "Source dataset is missing required columns"

**Problem**: Dataset doesn't have the required schema.

**Solution**:

- Verify your dataset has columns: `user_input`, `retrieved_contexts`, and `reference`
- Check that column names match exactly (case-sensitive)
- Ensure `retrieved_contexts` is formatted as a list (see Dataset Requirements)

Example fix for CSV:

```csv
# Wrong (missing columns)
question,context,answer

# Correct
user_input,retrieved_contexts,reference
```

### "No results found in experiment"

**Problem**: `evaluate.py` can't find experiment results.

**Solution**:

- Check if `data/experiments/ragas_experiment.jsonl` exists
- Verify `run.py` completed successfully without errors
- Ensure the agent URL was accessible during execution
- Check file permissions on the `data/` directory

### CSV List Conversion Issues

**Problem**: `retrieved_contexts` not parsing correctly from CSV.

**Solution**:

- Ensure lists are formatted as Python array strings: `"['item1', 'item2']"`
- Use proper quoting in CSV: wrap the entire array string in double quotes
- Consider using JSON or Parquet format for complex data types

Example:

```csv
user_input,retrieved_contexts,reference
"What is X?","['Context about X', 'More context']","X is..."
```

### Evaluation Metrics Fail

**Problem**: Certain metrics fail during evaluation.

**Solution**:

- Some metrics require the `reference` field (e.g., `context_precision`, `context_recall`)
- Verify your dataset includes all required fields for the metrics you're using
- Check the RAGAS documentation for metric-specific requirements

----

## Contributing

See [Contribution Guide](https://github.com/agentic-layer/testbench?tab=contributing-ov-file) for details on
contribution, and the process for submitting pull requests.
