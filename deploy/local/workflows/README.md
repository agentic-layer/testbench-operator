# Ragas Evaluation Workflow

## Prerequisites
- Local Kubernetes (e.g. kind) cluster
- Docker 
  - & an up-to-date Testbench docker container
- Testkube CLI


## Setup

```
# Build the docker container from your local files
docker build -t ghcr.io/agentic-layer/testbench/testworkflows:latest .

# Load the docker container in your kind cluster
kind load docker-image ghcr.io/agentic-layer/testbench/testworkflows:latest --name kind

# Optional apply the Templates manually (should be automatically applied via Tilt)
kubectl apply -f deploy/local/templates/
kubectl apply -f deploy/local/workflows/ragas-evaluation-workflow.yaml

```

## Usage
With minimal setup:
```
kubectl testkube run testworkflow ragas-evaluation-workflow \
    --config datasetUrl="http://example.com/dataset.csv" \
    --config agentUrl="http://ai-gateway-litellm:11010" \
    --config metrics="nv_accuracy context_recall"
    --config workflowName="Testworkflow-Name" \
    -n testkube
```


With all the optional parameters:
```
kubectl testkube run testworkflow ragas-evaluation-workflow \
    --config datasetUrl="http://example.com/dataset.csv" \
    --config agentUrl="http://ai-gateway-litellm:11010" \
    --config model="gemini-2.5-pro" \
    --config metrics="nv_accuracy context_recall"
    --config workflowName="Testworkflow-Name" \
    --config otlpEndpoint="http://otlp-endpoint:4093"
    -n testkube
```
