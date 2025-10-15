# Getting Started

## Prerequisites
- go version v1.24.0+
- docker version 17.03+.
- kubectl version v1.11.3+.
- Access to a Kubernetes v1.11.3+ cluster.

## Testkube Setup

**Install Testkube locally:**
```sh
# Install Testkube CLI
brew install testkube

# Install Testkube in your cluster
testkube init
```

**Run Testkube workflows:**
```sh
# Apply workflow configuration
kubectl apply -f config/samples/first_workflow.yaml

# Run a workflow
testkube run workflow <workflow-name>

# Check workflow status
testkube get workflows
```

# Contribution

See [Contribution Guide](https://github.com/agentic-layer/agent-runtime-operator?tab=contributing-ov-file) for details on contribution, and the process for submitting pull requests.


