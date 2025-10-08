.PHONY: help setup clean bench bench-force logs clean-logs test-compose

help:
	@echo "Server Benchmark Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Install system dependencies, create venv, and install Python packages"
	@echo "  bench          - Run benchmarks in parallel"
	@echo "  bench-force    - Run benchmarks in parallel (force re-run, skip cached results)"
	@echo "  logs           - Show the latest benchmark log"
	@echo "  clean-logs     - Remove all log files"
	@echo "  clean          - Remove virtual environment and generated files"
	@echo "  test-compose   - Test docker-compose generation with sample config"

setup:
	@echo "Installing system dependencies..."
	@sudo apt update
	@sudo apt install -y make python3-venv
	@echo "Creating virtual environment..."
	@python3 -m venv venv
	@echo "Installing Python dependencies..."
	@./venv/bin/pip install -r requirements.txt
	@echo "✅ Setup complete!"

bench:
	@if [ ! -d "venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Running benchmarks in parallel..."
	./run_remote_benchmark.py --parallel

bench-force:
	@if [ ! -d "venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Running benchmarks in parallel (force mode)..."
	./run_remote_benchmark.py --parallel --force

logs:
	@if [ ! -d "logs" ]; then \
		echo "❌ No logs directory found."; \
		exit 1; \
	fi
	@LATEST_LOG=$$(ls -t logs/benchmark_*.log 2>/dev/null | head -1); \
	if [ -z "$$LATEST_LOG" ]; then \
		echo "❌ No log files found."; \
		exit 1; \
	fi; \
	echo "📝 Showing: $$LATEST_LOG"; \
	echo ""; \
	cat "$$LATEST_LOG"

clean-logs:
	@echo "Removing log files..."
	rm -rf logs/
	@echo "✅ Logs cleaned!"

clean:
	@echo "Removing virtual environment and generated files..."
	rm -rf venv/
	rm -f docker-compose.*.yml nginx.*.conf
	rm -rf __pycache__/ utils/__pycache__/
	@echo "✅ Clean complete!"

test-compose:
	@if [ ! -d "venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Testing docker-compose generation..."
	./venv/bin/python utils/generate_compose.py \
		--num-instances 1 \
		--tensor-parallel-size 4 \
		--container-name test \
		--model-path /test/model \
		--model-name test-model \
		--hf-directory /hf \
		--hf-token test \
		--extra-args "--enable-expert-parallel --swap-space 16" \
		--output /tmp/test-compose.yml
	@echo "✅ Generated: /tmp/test-compose.yml"
	@echo ""
	@cat /tmp/test-compose.yml
