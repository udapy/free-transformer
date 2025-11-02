.PHONY: help install install-dev test test-unit test-integration test-comparison
.PHONY: lint format type-check quality train-baseline train-free compare
.PHONY: generate-data clean clean-checkpoints clean-data clean-all
.PHONY: setup-env publish-test publish
.PHONY: docs-install docs-build docs-serve docs-serve-dev docs-deploy docs-clean docs-check

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Project variables
PROJECT_NAME := free-transformer
PYTHON_VERSION := 3.12
SRC_DIR := src/free_transformer
TEST_DIR := tests
EXAMPLES_DIR := examples
DATA_DIR := data
CHECKPOINT_DIR := checkpoints

# UV commands
UV := uv
UV_RUN := $(UV) run
UV_PIP := $(UV) pip

##@ General

help: ## Display this help message
	@echo "$(BLUE)Free Transformer - Makefile Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

setup-env: ## Create UV environment and install Python version
	@echo "$(BLUE)Setting up UV environment...$(NC)"
	@$(UV) python install $(PYTHON_VERSION)
	@echo "$(PYTHON_VERSION)" > .python-version
	@$(UV) venv --python $(PYTHON_VERSION)
	@echo "$(GREEN)Environment created successfully!$(NC)"
	@echo "Activate with: source .venv/bin/activate"

install: ## Install package dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@$(UV_PIP) install -e .
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing with development dependencies...$(NC)"
	@$(UV_PIP) install -e ".[dev]"
	@echo "$(GREEN)Development installation complete!$(NC)"

install-all: setup-env install-dev docs-install ## Full setup: create env and install all dependencies
	@echo "$(GREEN)Full installation complete!$(NC)"

sync: ## Sync dependencies with uv.lock
	@echo "$(BLUE)Syncing dependencies...$(NC)"
	@$(UV) sync
	@echo "$(GREEN)Dependencies synced!$(NC)"

##@ Code Quality

lint: ## Run linting with ruff
	@echo "$(BLUE)Running linter...$(NC)"
	@$(UV_RUN) ruff check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(UV_RUN) black $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@$(UV_RUN) isort $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@echo "$(GREEN)Code formatted!$(NC)"

format-check: ## Check code formatting without changes
	@echo "$(BLUE)Checking code format...$(NC)"
	@$(UV_RUN) black --check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@$(UV_RUN) isort --check-only $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	@$(UV_RUN) mypy $(SRC_DIR)

quality: lint type-check format-check ## Run all quality checks

##@ Testing

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	@$(UV_RUN) pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)Tests complete! Coverage report: htmlcov/index.html$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(UV_RUN) pytest $(TEST_DIR)/unit -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(UV_RUN) pytest $(TEST_DIR)/integration -v

test-comparison: ## Run Transformer vs Free Transformer comparison
	@echo "$(BLUE)Running comparison tests...$(NC)"
	@$(UV_RUN) pytest $(TEST_DIR)/test_comparison.py -v -s

test-fast: ## Run tests without coverage (faster)
	@echo "$(BLUE)Running fast tests...$(NC)"
	@$(UV_RUN) pytest $(TEST_DIR) -v -x

##@ Data

generate-data: ## Generate synthetic training data
	@echo "$(BLUE)Generating synthetic data...$(NC)"
	@mkdir -p $(DATA_DIR)
	@$(UV_RUN) python $(EXAMPLES_DIR)/generate_data.py \
		--output-dir $(DATA_DIR) \
		--vocab-size 10000 \
		--seq-length 512 \
		--num-train 50000 \
		--num-val 5000 \
		--seed 42
	@echo "$(GREEN)Data generated in $(DATA_DIR)/$(NC)"

generate-data-small: ## Generate small dataset for testing
	@echo "$(BLUE)Generating small test dataset...$(NC)"
	@mkdir -p $(DATA_DIR)
	@$(UV_RUN) python $(EXAMPLES_DIR)/generate_data.py \
		--output-dir $(DATA_DIR) \
		--vocab-size 1000 \
		--seq-length 128 \
		--num-train 1000 \
		--num-val 100 \
		--seed 42
	@echo "$(GREEN)Small dataset generated!$(NC)"

##@ Training

train-baseline: ## Train baseline Transformer
	@echo "$(BLUE)Training baseline Transformer...$(NC)"
	@mkdir -p $(CHECKPOINT_DIR)/baseline
	@$(UV_RUN) python $(EXAMPLES_DIR)/train_baseline.py \
		--config configs/baseline.yaml \
		--output-dir $(CHECKPOINT_DIR)/baseline
	@echo "$(GREEN)Baseline training complete!$(NC)"

train-free: ## Train Free Transformer
	@echo "$(BLUE)Training Free Transformer...$(NC)"
	@mkdir -p $(CHECKPOINT_DIR)/free
	@$(UV_RUN) python $(EXAMPLES_DIR)/train_free.py \
		--config configs/free_transformer.yaml \
		--output-dir $(CHECKPOINT_DIR)/free
	@echo "$(GREEN)Free Transformer training complete!$(NC)"

train-baseline-fsdp: ## Train baseline with FSDP (multi-GPU)
	@echo "$(BLUE)Training baseline with FSDP...$(NC)"
	@mkdir -p $(CHECKPOINT_DIR)/baseline_fsdp
	@$(UV_RUN) torchrun --nproc_per_node=auto \
		$(EXAMPLES_DIR)/train_baseline.py \
		--config configs/baseline.yaml \
		--use-fsdp \
		--output-dir $(CHECKPOINT_DIR)/baseline_fsdp

train-free-fsdp: ## Train Free Transformer with FSDP (multi-GPU)
	@echo "$(BLUE)Training Free Transformer with FSDP...$(NC)"
	@mkdir -p $(CHECKPOINT_DIR)/free_fsdp
	@$(UV_RUN) torchrun --nproc_per_node=auto \
		$(EXAMPLES_DIR)/train_free.py \
		--config configs/free_transformer.yaml \
		--use-fsdp \
		--output-dir $(CHECKPOINT_DIR)/free_fsdp

train-all: train-baseline train-free ## Train both models sequentially

##@ Evaluation

compare: ## Compare baseline and Free Transformer
	@echo "$(BLUE)Comparing models...$(NC)"
	@$(UV_RUN) python $(EXAMPLES_DIR)/eval_compare.py \
		--baseline-checkpoint $(CHECKPOINT_DIR)/baseline/model_final.pt \
		--free-checkpoint $(CHECKPOINT_DIR)/free/model_final.pt \
		--data-dir $(DATA_DIR) \
		--output-dir results/comparison
	@echo "$(GREEN)Comparison complete! Results in results/comparison/$(NC)"

evaluate-baseline: ## Evaluate baseline model
	@echo "$(BLUE)Evaluating baseline model...$(NC)"
	@$(UV_RUN) python $(EXAMPLES_DIR)/eval_compare.py \
		--baseline-checkpoint $(CHECKPOINT_DIR)/baseline/model_final.pt \
		--data-dir $(DATA_DIR) \
		--output-dir results/baseline

evaluate-free: ## Evaluate Free Transformer
	@echo "$(BLUE)Evaluating Free Transformer...$(NC)"
	@$(UV_RUN) python $(EXAMPLES_DIR)/eval_compare.py \
		--free-checkpoint $(CHECKPOINT_DIR)/free/model_final.pt \
		--data-dir $(DATA_DIR) \
		--output-dir results/free

##@ Quick Start

quick-start: install-dev generate-data-small test-fast ## Quick start: setup + small data + tests
	@echo "$(GREEN)Quick start complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Run 'make train-baseline' to train baseline model"
	@echo "  2. Run 'make train-free' to train Free Transformer"
	@echo "  3. Run 'make compare' to compare models"
	@echo "  4. Run 'make docs-serve' to view documentation"

demo: generate-data-small train-baseline train-free compare ## Full demo pipeline
	@echo "$(GREEN)Demo complete!$(NC)"

##@ Cleanup

clean: ## Remove Python cache files
	@echo "$(BLUE)Cleaning Python cache...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	@echo "$(GREEN)Cache cleaned!$(NC)"

clean-checkpoints: ## Remove all checkpoints
	@echo "$(BLUE)Removing checkpoints...$(NC)"
	@rm -rf $(CHECKPOINT_DIR)
	@echo "$(GREEN)Checkpoints removed!$(NC)"

clean-data: ## Remove generated data
	@echo "$(BLUE)Removing generated data...$(NC)"
	@rm -rf $(DATA_DIR)
	@echo "$(GREEN)Data removed!$(NC)"

clean-results: ## Remove evaluation results
	@echo "$(BLUE)Removing results...$(NC)"
	@rm -rf results
	@echo "$(GREEN)Results removed!$(NC)"

clean-all: clean clean-checkpoints clean-data clean-results docs-clean ## Remove everything (cache, checkpoints, data, docs)
	@echo "$(GREEN)Full cleanup complete!$(NC)"

clean-env: ## Remove virtual environment
	@echo "$(YELLOW)Warning: This will remove the .venv directory$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf .venv; \
		echo "$(GREEN)Environment removed!$(NC)"; \
	fi

##@ Publishing

build: ## Build distribution packages
	@echo "$(BLUE)Building package...$(NC)"
	@$(UV) build
	@echo "$(GREEN)Build complete! Packages in dist/$(NC)"

publish-test: build ## Publish to Test PyPI
	@echo "$(BLUE)Publishing to Test PyPI...$(NC)"
	@$(UV) publish --repository testpypi
	@echo "$(GREEN)Published to Test PyPI!$(NC)"

publish: build ## Publish to PyPI
	@echo "$(YELLOW)Warning: Publishing to production PyPI$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(UV) publish; \
		echo "$(GREEN)Published to PyPI!$(NC)"; \
	fi

##@ Docker

docker-build: ## Build Docker image for GPU
	@echo "$(BLUE)Building Docker image for GPU...$(NC)"
	@docker build -t $(PROJECT_NAME):demo .
	@echo "$(GREEN)Docker image built!$(NC)"

docker-build-cpu: ## Build Docker image for CPU
	@echo "$(BLUE)Building Docker image for CPU...$(NC)"
	@docker build -f Dockerfile.cpu -t $(PROJECT_NAME):cpu .
	@echo "$(GREEN)CPU Docker image built!$(NC)"

docker-demo: ## Run demo using Docker Compose
	@echo "$(BLUE)Running Free Transformer demo...$(NC)"
	@docker-compose up free-transformer-demo
	@echo "$(GREEN)Demo complete!$(NC)"

docker-interactive: ## Start interactive Docker container
	@echo "$(BLUE)Starting interactive container...$(NC)"
	@docker-compose up -d free-transformer-interactive
	@docker-compose exec free-transformer-interactive bash

docker-run-gpu: ## Run container with GPU support
	@echo "$(BLUE)Running Docker container with GPU...$(NC)"
	@docker run --gpus all -it --rm \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		$(PROJECT_NAME):demo

docker-run-cpu: ## Run container with CPU only
	@echo "$(BLUE)Running Docker container with CPU...$(NC)"
	@docker run -it --rm \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		$(PROJECT_NAME):cpu

docker-clean: ## Clean Docker containers and images
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	@docker-compose down --remove-orphans
	@docker rmi $(PROJECT_NAME):demo $(PROJECT_NAME):cpu 2>/dev/null || true
	@echo "$(GREEN)Docker resources cleaned!$(NC)"

##@ Documentation

docs-install: ## Install documentation dependencies
	@echo "$(BLUE)Installing documentation dependencies...$(NC)"
	@$(UV_PIP) install -e ".[docs]"
	@echo "$(GREEN)Documentation dependencies installed!$(NC)"

docs-build: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@$(UV_RUN) mkdocs build
	@echo "$(GREEN)Documentation built in site/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation (will find available port)$(NC)"
	@$(UV_RUN) python scripts/serve_docs.py

docs-serve-dev: ## Serve documentation with auto-reload
	@echo "$(BLUE)Serving documentation with auto-reload (will find available port)$(NC)"
	@$(UV_RUN) python scripts/serve_docs.py --dev

docs-serve-port: ## Serve documentation on specific port (usage: make docs-serve-port PORT=8001)
	@echo "$(BLUE)Serving documentation at http://127.0.0.1:$(or $(PORT),8000)$(NC)"
	@$(UV_RUN) mkdocs serve --dev-addr 127.0.0.1:$(or $(PORT),8000)

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(BLUE)Deploying documentation to GitHub Pages...$(NC)"
	@$(UV_RUN) mkdocs gh-deploy --force
	@echo "$(GREEN)Documentation deployed!$(NC)"

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation build...$(NC)"
	@rm -rf site/
	@echo "$(GREEN)Documentation build cleaned!$(NC)"

docs-check: ## Check documentation for issues
	@echo "$(BLUE)Checking documentation...$(NC)"
	@$(UV_RUN) mkdocs build --strict
	@echo "$(GREEN)Documentation check passed!$(NC)"

##@ CI/CD

ci: quality test docs-check ## Run CI pipeline locally
	@echo "$(GREEN)CI checks passed!$(NC)"

pre-commit: format lint test-fast ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

##@ Info

info: ## Display project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "  Name: $(PROJECT_NAME)"
	@echo "  Python: $(PYTHON_VERSION)"
	@echo "  UV version: $$($(UV) --version)"
	@echo ""
	@echo "$(BLUE)Directory Structure:$(NC)"
	@echo "  Source: $(SRC_DIR)"
	@echo "  Tests: $(TEST_DIR)"
	@echo "  Examples: $(EXAMPLES_DIR)"
	@echo "  Data: $(DATA_DIR)"
	@echo "  Checkpoints: $(CHECKPOINT_DIR)"

check-gpu: ## Check GPU availability
	@echo "$(BLUE)Checking GPU...$(NC)"
	@$(UV_RUN) python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
