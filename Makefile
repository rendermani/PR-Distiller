.DEFAULT_GOAL := help
MODEL ?= qwen3:8b
SHELL := /bin/bash

# Compose file combinations
COMPOSE_BASE   := -f docker-compose.yml
COMPOSE_OLLAMA := -f docker-compose.yml -f docker-compose.ollama.yml
COMPOSE_GPU    := -f docker-compose.yml -f docker-compose.ollama.yml -f docker-compose.gpu.yml

# Colors
CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
RESET  := \033[0m

.PHONY: help up up-linux-gpu up-cpu down build logs \
        run-model test test-coverage lint preflight clean

## —— General ——————————————————————————————————————————————

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2}'

## —— Start (pick one for your platform) ———————————————————

up: ## Start services (backend + web-ui in Docker, Ollama on host)
	@printf "$(CYAN)Checking host Ollama at localhost:11434...$(RESET)\n"
	@curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 || { \
		printf "$(RED)Host Ollama is not reachable.$(RESET) Install: https://ollama.com/download\n"; \
		printf "Then run: $(CYAN)ollama serve$(RESET) (or start the Ollama app)\n"; \
		exit 1; }
	LLM_API_BASE=http://host.docker.internal:11434/v1 docker compose $(COMPOSE_BASE) up -d
	@$(MAKE) --no-print-directory _print-urls

up-linux-gpu: ## Linux+NVIDIA: everything in Docker with GPU acceleration
	docker compose $(COMPOSE_GPU) up -d
	@$(MAKE) --no-print-directory _print-urls

up-cpu: ## Any OS: everything in Docker, CPU only (slow, for testing)
	docker compose $(COMPOSE_OLLAMA) up -d
	@$(MAKE) --no-print-directory _print-urls

down: ## Stop all services
	docker compose $(COMPOSE_GPU) down 2>/dev/null || docker compose $(COMPOSE_OLLAMA) down 2>/dev/null || docker compose $(COMPOSE_BASE) down

build: ## Rebuild all Docker images
	docker compose $(COMPOSE_BASE) build

logs: ## Tail logs from all services
	docker compose $(COMPOSE_BASE) logs -f

_print-urls:
	@echo ""
	@printf "  $(GREEN)Backend API$(RESET)  http://localhost:$${API_PORT:-8923}\n"
	@printf "  $(GREEN)Web UI$(RESET)       http://localhost:$${WEB_PORT:-4096}\n"
	@echo ""

## —— LLM Model ————————————————————————————————————————————

run-model: ## Pull the default LLM (host Ollama). MODEL=qwen3:8b
	@if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then \
		printf "$(CYAN)Pulling $(MODEL) into host Ollama...$(RESET)\n"; \
		ollama pull $(MODEL); \
	else \
		printf "$(CYAN)Pulling $(MODEL) into containerized Ollama...$(RESET)\n"; \
		docker compose $(COMPOSE_OLLAMA) exec ollama ollama pull $(MODEL); \
	fi
	@printf "$(GREEN)Model $(MODEL) ready.$(RESET)\n"

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
	OS=$$(uname -s); \
	printf "  Detected OS:                "; \
	case "$$OS" in \
		Darwin)  printf "$(GREEN)macOS$(RESET)\n";; \
		Linux)   printf "$(GREEN)Linux$(RESET)\n";; \
		MINGW*|MSYS*|CYGWIN*) printf "$(GREEN)Windows$(RESET)\n";; \
		*)       printf "$(YELLOW)$$OS (untested)$(RESET)\n";; \
	esac; \
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
	printf "  Checking host Ollama...     "; \
	if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then \
		printf "$(GREEN)RUNNING$(RESET) (use: make up-mac-ollama)\n"; \
		PASS=$$((PASS+1)); \
	else \
		printf "$(YELLOW)NOT RUNNING$(RESET) — install from https://ollama.com or use make up-cpu\n"; \
		WARN=$$((WARN+1)); \
	fi; \
	\
	if [ "$$OS" = "Linux" ]; then \
		printf "  Checking NVIDIA GPU...      "; \
		if command -v nvidia-smi >/dev/null 2>&1; then \
			printf "$(GREEN)AVAILABLE$(RESET) (use: make up-linux-gpu)\n"; \
			PASS=$$((PASS+1)); \
		else \
			printf "$(YELLOW)NOT FOUND$(RESET) — CPU only, use make up-cpu\n"; \
			WARN=$$((WARN+1)); \
		fi; \
	fi; \
	printf "  Checking ports...           "; \
	PORT_OK=true; \
	for port in $${API_PORT:-8923} $${WEB_PORT:-4096}; do \
		if ss -tln 2>/dev/null | grep -q ":$$port " || lsof -i :$$port >/dev/null 2>&1; then \
			printf "$(RED)Port $$port in use$(RESET) "; \
			PORT_OK=false; \
		fi; \
	done; \
	if [ "$$PORT_OK" = true ]; then \
		printf "$(GREEN)OK$(RESET) (8923, 4096 free)\n"; \
		PASS=$$((PASS+1)); \
	else \
		printf "\n"; \
		FAIL=$$((FAIL+1)); \
	fi; \
	\
	printf "\n  ────────────────────────────\n"; \
	printf "  $(GREEN)$$PASS passed$(RESET)  $(YELLOW)$$WARN warnings$(RESET)  $(RED)$$FAIL failed$(RESET)\n\n"; \
	if [ $$FAIL -gt 0 ]; then \
		printf "  $(RED)Fix the failures above before running make up.$(RESET)\n\n"; \
		exit 1; \
	else \
		printf "  $(GREEN)Ready! Pick a start command:$(RESET)\n"; \
		printf "    $(CYAN)make up$(RESET)              Mac/Win/Linux with host Ollama (recommended)\n"; \
		printf "    $(CYAN)make up-linux-gpu$(RESET)    Linux + NVIDIA, everything in Docker\n"; \
		printf "    $(CYAN)make up-cpu$(RESET)          CPU-only (any OS)\n\n"; \
	fi

## —— Cleanup ——————————————————————————————————————————————

clean: ## Remove all containers, volumes, and build cache
	docker compose $(COMPOSE_OLLAMA) down -v --remove-orphans
	docker system prune -f
