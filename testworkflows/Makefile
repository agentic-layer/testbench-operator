# Testworkflows Docker Image Makefile

# Image configuration
IMAGE_NAME ?= testworkflows
IMAGE_TAG ?= latest

# Full image reference
IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

# Container name for persistent state
CONTAINER_NAME := testworkflows-runner

.PHONY: help
help: ## Display this help message
	@echo "Testworkflows Docker Image Management"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-15s %s\n", $$1, $$2 } /^##@/ { printf "\n%s\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

## Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE) .

## Run the container
.PHONY: run
run:
	docker start $(CONTAINER_NAME)

## Run an interactive shell in the container
.PHONY: shell
shell:
	docker run --rm -it $(IMAGE) /bin/bash

## Run an interactive Python shell in the container
.PHONY: python
python:
	docker run --rm -it $(IMAGE) python3

## Run pytest tests in the container
.PHONY: test
test:
	docker run --rm $(IMAGE) pytest tests/ -v

## Remove the persistent container
.PHONY: cleanup
cleanup:
	docker rm -f $(CONTAINER_NAME) || true

## Remove container and start fresh
.PHONY: reset
reset: cleanup
	@echo "Container removed. Run 'make setup' to start a new workflow."