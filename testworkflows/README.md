# Getting Started with Testkube

## Testkube Setup

**Install Testkube locally:**
```shell
# Install Testkube CLI
brew install testkube

# Install Testkube in your cluster
testkube init
```

**Run Testkube workflows:**
```shell
# Apply workflow configuration
kubectl apply -f first_workflow.yaml

# Run a workflow
testkube run workflow my-test

# Check workflow status
testkube get workflows
```
