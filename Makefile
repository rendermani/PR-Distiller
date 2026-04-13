.DEFAULT_GOAL := help
MODEL ?= qwen2.5-coder:7b-instruct
SHELL := /bin/bash

# Colors
CYAN  := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED   := \033[31m
RESET := \033[0m

.PHONY: help up down build logs run-model stop-model test lint preflight clean

## —— General ——————————————————————————————————————————————

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'

## —— Docker ———————————————————————————————————————————————

up: ## Start all services (backend, web-ui, ollama)
	docker compose up -d
	@echo ""
	@printf "  $(GREEN)Backend API$(RESET)  http://localhost:$${API_PORT:-8923}\n"
	@printf "  $(GREEN)Web UI$(RESET)       http://localhost:$${WEB_PORT:-4096}\n"
	@printf "  $(GREEN)Ollama$(RESET)       http://localhost:$${OLLAMA_PORT:-11434}\n"
	@echo ""
	@printf "  Run $(CYAN)make run-model$(RESET) to pull the default LLM.\n"

down: ## Stop all services
	docker compose down

build: ## Rebuild all Docker images
	docker compose build

logs: ## Tail logs from all services
	docker compose logs -f

## —— LLM Model ————————————————————————————————————————————

run-model: ## Pull and load the LLM into Ollama (MODEL=qwen2.5-coder:7b-instruct)
	@printf "$(CYAN)Pulling $(MODEL) into Ollama...$(RESET)\n"
	docker compose exec ollama ollama pull $(MODEL)
	@printf "$(GREEN)Model $(MODEL) ready.$(RESET)\n"

stop-model: ## Remove a model from Ollama (MODEL=qwen2.5-coder:7b-instruct)
	docker compose exec ollama ollama rm $(MODEL)

## —— Development ——————————————————————————————————————————

test: ## Run backend test suite
	cd backend && python -m pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	cd backend && python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

lint: ## Lint backend code
	cd backend && python -m py_compile api.py && \
		python -m py_compile settings.py && \
		python -m py_compile db/lightrag_manager.py && \
		python -m py_compile llm/extractor.py && \
		python -m py_compile pipeline/job_orchestrator.py

## —— Preflight Check ——————————————————————————————————————

preflight: ## Verify all requirements before first run
	@printf "$(CYAN)Running preflight checks...$(RESET)\n\n"
	@PASS=0; FAIL=0; WARN=0; \
	\
	printf "  Checking Docker...          "; \
	if command -v docker >/dev/null 2>&1; then \
		printf "$(GREEN)OK$(RESET) ($$(docker --version | head -1))\n"; \
		PASS=$$((PASS+1)); \
	else \
		printf "$(RED)MISSING$(RESET) — Install Docker: https://docs.docker.com/get-docker/\n"; \
		FAIL=$$((FAIL+1)); \
	fi; \
	\
	printf "  Checking Docker Compose...  "; \
	if docker compose version >/dev/null 2>&1; then \
		printf "$(GREEN)OK$(RESET) ($$(docker compose version --short))\n"; \
		PASS=$$((PASS+1)); \
	else \
		printf "$(RED)MISSING$(RESET) — Install Docker Compose v2\n"; \
		FAIL=$$((FAIL+1)); \
	fi; \
	\
	printf "  Checking .env file...       "; \
	if [ -f .env ]; then \
		printf "$(GREEN)OK$(RESET)\n"; \
		PASS=$$((PASS+1)); \
	else \
		printf "$(YELLOW)MISSING$(RESET) — Copy .env.example to .env and fill in values\n"; \
		WARN=$$((WARN+1)); \
	fi; \
	\
	printf "  Checking GITHUB_TOKEN...    "; \
	if [ -f .env ] && grep -qE '^GITHUB_TOKEN=.+' .env 2>/dev/null; then \
		printf "$(GREEN)SET$(RESET)\n"; \
		PASS=$$((PASS+1)); \
	else \
		printf "$(YELLOW)NOT SET$(RESET) — Required for crawling PR comments\n"; \
		WARN=$$((WARN+1)); \
	fi; \
	\
	printf "  Checking GPU (NVIDIA)...    "; \
	if command -v nvidia-smi >/dev/null 2>&1; then \
		printf "$(GREEN)AVAILABLE$(RESET) ($$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))\n"; \
		PASS=$$((PASS+1)); \
	else \
		printf "$(YELLOW)NOT FOUND$(RESET) — Ollama will run on CPU (slower). Remove docker-compose.override.yml if on Mac.\n"; \
		WARN=$$((WARN+1)); \
	fi; \
	\
	printf "  Checking ports...           "; \
	PORT_OK=true; \
	for port in $${API_PORT:-8923} $${WEB_PORT:-4096} $${OLLAMA_PORT:-11434}; do \
		if ss -tln 2>/dev/null | grep -q ":$$port " || lsof -i :$$port >/dev/null 2>&1; then \
			printf "$(RED)Port $$port in use$(RESET) "; \
			PORT_OK=false; \
		fi; \
	done; \
	if [ "$$PORT_OK" = true ]; then \
		printf "$(GREEN)OK$(RESET) (8923, 4096, 11434 free)\n"; \
		PASS=$$((PASS+1)); \
	else \
		printf "\n"; \
		FAIL=$$((FAIL+1)); \
	fi; \
	\
	printf "\n  ────────────────────────────\n"; \
	printf "  $(GREEN)$$PASS passed$(RESET)  $(YELLOW)$$WARN warnings$(RESET)  $(RED)$$FAIL failed$(RESET)\n\n"; \
	if [ $$FAIL -gt 0 ]; then \
		printf "  $(RED)Fix the failures above before running 'make up'.$(RESET)\n\n"; \
		exit 1; \
	else \
		printf "  $(GREEN)Ready! Run 'make up' to start.$(RESET)\n\n"; \
	fi

## —— Cleanup ——————————————————————————————————————————————

clean: ## Remove all containers, volumes, and build cache
	docker compose down -v --remove-orphans
	docker system prune -f
