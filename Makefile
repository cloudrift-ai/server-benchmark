.PHONY: help setup clean bench bench-force bench-kernels bench-kernels-tune test-compose lint format git-sha-guard \
	serve-models serve-config serve-config-guard serve-goldens serve-warm serve-image serve-verify serve-push

help:
	@echo "Server Benchmark Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Install system dependencies, create venv, and install Python packages"
	@echo "  lint           - Run linter and format checks"
	@echo "  format         - Auto-format code and fix lint violations"
	@echo "  bench          - Run benchmarks in parallel"
	@echo "  bench-force    - Run benchmarks in parallel (force re-run, skip cached results)"
	@echo "  bench-kernels  - Run per-kernel perf comparison vs PyTorch (tests/perf/, requires CUDA)"
	@echo "  wheel          - Build the emmy wheel into dist/"
	@echo "  vllm-emmy-image - Build the vLLM + emmy serving image (docker/vllm-emmy)"
	@echo "  vllm-emmy-push  - Push the serving image to Docker Hub (cloudriftai/)"
	@echo "  serve-goldens / serve-warm / serve-image / serve-verify / serve-push  MODEL=<hf-id>"
	@echo "                  - Check goldens, warm (on the target GPU), bake, verify, push a"
	@echo "                    prebuilt per-model serving image (docker/vllm-emmy-serve)"
	@echo "  serve-models    - List the models with a pinned release config"
	@echo "  clean          - Remove virtual environment and generated files"
	@echo "  test-compose   - Test docker-compose generation with sample config"

setup:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3.12 -m venv venv --prompt "emmy"; \
		echo "Installing Python dependencies..."; \
		./venv/bin/pip install -e ".[dev]"; \
	fi

setup-ci:
	python3.12 -m venv venv --prompt "emmy"
	./venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
	./venv/bin/pip install -e ".[compile,test,image]"

lint: setup
	./venv/bin/ruff check
	./venv/bin/ruff format --check

format: setup
	./venv/bin/ruff format
	./venv/bin/ruff check --fix

# Compile CUDA kernels at -Xcicc -O1: ~3x faster suite (dodges the cicc/LLVM unroll
# blowup on big register-tile kernels). This is the CORRECTNESS lane — -O1 changes
# runtime perf, not numerics, and the deployable perf tests (tests/perf, -m perf) run
# at -O3 via `make bench-kernels`. Override with EMMY_NVCC_FLAGS= to test at -O3.
test: setup
	EMMY_NVCC_FLAGS="-Xcicc -O1" ./venv/bin/pytest tests/ -v -n auto --dist=loadgroup

# The name the docs reference; the stock (no tune DB) lane is the default.
bench-kernels: bench-kernels-clean

bench-kernels-clean: setup
	@rm -f /tmp/emmy-gpu.lock
	./venv/bin/pytest tests/perf/ -m perf -n 4 --dist=loadgroup -v -p no:randomly --no-header

bench-kernels-tuned: setup
	@rm -f /tmp/emmy-gpu.lock
	@test -f ~/.cache/emmy/tune-kernels.db || (echo "The kernel tuning DB not foud; run make tune-kernels"; exit 1)
	EMMY_TUNE_DB=~/.cache/emmy/tune-kernels.db ./venv/bin/pytest tests/perf/ -m perf -n 4 --dist=loadgroup -v -p no:randomly --no-header

tune-kernels: setup
	@rm -f /tmp/emmy-gpu.lock
	@rm -f ~/.cache/emmy/tune-kernels.db
	EMMY_TUNE=1 EMMY_TUNE_DB=~/.cache/emmy/tune-kernels.db ./venv/bin/pytest tests/perf/ -m perf -n 4 --dist=loadgroup -v -p no:randomly --no-header

# --- vLLM + emmy serving image (emmy/serving, docker/vllm-emmy) ---
VLLM_VERSION ?= v0.23.0
VLLM_EMMY_TAG ?= cloudriftai/vllm-emmy:$(patsubst v%,%,$(VLLM_VERSION))-$(shell git rev-parse --short HEAD)

wheel: setup
	./venv/bin/pip install --quiet build
	rm -rf dist build && ./venv/bin/python -m build --wheel -o dist/ .

# Image tags embed the short sha; an empty rev-parse (e.g. root over a synced tree without
# git safe.directory) would silently tag "...:0.23.0-" — fail loudly instead.
git-sha-guard:
	@test -n "$(shell git rev-parse --short HEAD)" || \
		(echo "ERROR: git rev-parse returned empty — image tag would be malformed."; \
		 echo "  likely fix: git config --global --add safe.directory $(CURDIR)"; exit 1)

vllm-emmy-image: wheel git-sha-guard
	docker build -f docker/vllm-emmy/Dockerfile --build-arg VLLM_VERSION=$(VLLM_VERSION) \
		-t $(VLLM_EMMY_TAG) .

vllm-emmy-push: vllm-emmy-image
	docker push $(VLLM_EMMY_TAG)

# --- per-model prebuilt-kernel serving image (docker/vllm-emmy-serve): warm on the
# --- target GPU, bake cubins + model snapshot, verify zero-compile/zero-download start ---
#
# Parameterized by MODEL (an HF id). The slug derived from it (model_slug.sh) names BOTH the
# published image and the pinned config, so onboarding a model is one new file:
#
#   make serve-warm  MODEL=google/gemma-4-12B-it     -> models/gemma-4-12b-it.env
#                                                    -> cloudriftai/vllm-emmy-gemma-4-12b-it
SERVE_DIR := docker/vllm-emmy-serve
MODEL ?= google/gemma-4-12B-it
MODEL_SLUG := $(shell $(SERVE_DIR)/model_slug.sh '$(MODEL)')
SERVE_CONFIG := $(SERVE_DIR)/models/$(MODEL_SLUG).env
SERVE_TAG ?= cloudriftai/vllm-emmy-$(MODEL_SLUG):$(patsubst v%,%,$(VLLM_VERSION))-$(shell git rev-parse --short HEAD)

# `-include`, not `include`: a missing config must fail inside the serve-* targets with a
# usable message, not abort every `make test` in a tree whose MODEL has no config yet.
-include $(SERVE_CONFIG)
# The config is also `source`d by warm.sh/verify.sh, so values with spaces carry double
# quotes for bash's sake. Make keeps them verbatim — strip them once here rather than at
# every use site, where a missed one silently word-splits the GPU name into four args.
# `subst`, not `patsubst`: patsubst matches per WORD, and a quoted multi-word value is
# several words to make, so `"%"` never matches and the quotes survive into the argv.
SERVE_GPU_NAME := $(subst ",,$(SERVE_GPU))

# What a `make serve-* MODEL=<id>` would act on. The release workflow prints this first, so
# the model / card / tag under test are on the record before any multi-hour step starts.
serve-config: serve-config-guard
	@echo "MODEL      = $(MODEL)"
	@echo "slug       = $(MODEL_SLUG)"
	@echo "config     = $(SERVE_CONFIG)"
	@echo "image tag  = $(SERVE_TAG)"
	@echo "base image = $(VLLM_EMMY_TAG)"
	@echo "target GPU = $(SERVE_GPU_NAME)"
	@echo "serve      = --max-model-len $(SERVE_MAX_MODEL_LEN) --max-num-batched-tokens $(SERVE_MAX_NUM_BATCHED_TOKENS) --gpu-memory-utilization $(SERVE_GPU_MEM_UTIL) (decode bucket $(SERVE_DECODE_BUCKET))"

serve-models:
	@echo "Models with a pinned release config ($(SERVE_DIR)/models/):"
	@ls -1 $(SERVE_DIR)/models/*.env 2>/dev/null | sed 's|.*/||; s|\.env$$||; s|^|  |' || echo "  (none)"

serve-config-guard:
	@test -f "$(SERVE_CONFIG)" || ( \
		echo "ERROR: no pinned config for MODEL=$(MODEL) (expected $(SERVE_CONFIG))."; \
		echo "  A release config is per (model, GPU) and must be headroom-swept on the card —"; \
		echo "  see $(SERVE_DIR)/ARCHITECTURE.md, then add the file. Existing:"; \
		ls -1 $(SERVE_DIR)/models/*.env 2>/dev/null | sed 's|.*/||; s|\.env$$||; s|^|    |'; \
		exit 1)

# The goldens are the top tier of the fork-resolution evidence hierarchy — without them the
# warm bakes cold-greedy picks (catastrophic on unseeded projection shapes) into cubins and
# the pack, where nothing downstream revisits them. Gate the warm on coverage existing.
serve-goldens: serve-config-guard
	./venv/bin/python scripts/check_serving_goldens.py --model "$(SERVE_MODEL)" --gpu "$(SERVE_GPU_NAME)"

serve-warm: serve-config-guard
	BASE_IMAGE=$(VLLM_EMMY_TAG) MODEL="$(MODEL)" $(SERVE_DIR)/warm.sh

serve-image: git-sha-guard serve-config-guard
	@test -n "$$(ls -A $(SERVE_DIR)/warm/hf 2>/dev/null)" -a -n "$$(ls -A $(SERVE_DIR)/warm/cubin 2>/dev/null)" || \
		(echo "$(SERVE_DIR)/warm/ is empty — run 'make serve-warm MODEL=$(MODEL)' on the target GPU first"; exit 1)
	@mkdir -p $(SERVE_DIR)/warm/pack  # pack is optional (COPY needs the dir); empty -> boot falls back to full compile
	docker run --rm --user $$(id -u):$$(id -g) -v $(PWD)/$(SERVE_DIR)/warm/hf:/hf \
		-v $(PWD)/$(SERVE_DIR)/reshard_snapshot.py:/reshard.py \
		--entrypoint python3 $(VLLM_EMMY_TAG) /reshard.py /hf
	$(SERVE_DIR)/split_hf.sh
	docker build -f $(SERVE_DIR)/Dockerfile \
		--build-arg BASE_IMAGE=$(VLLM_EMMY_TAG) \
		--build-arg MODEL=$(SERVE_MODEL) \
		--build-arg MAX_MODEL_LEN=$(SERVE_MAX_MODEL_LEN) \
		--build-arg MAX_NUM_BATCHED_TOKENS=$(SERVE_MAX_NUM_BATCHED_TOKENS) \
		--build-arg GPU_MEM_UTIL=$(SERVE_GPU_MEM_UTIL) \
		--build-arg DECODE_BUCKET=$(SERVE_DECODE_BUCKET) \
		-t $(SERVE_TAG) $(SERVE_DIR)

serve-verify: serve-config-guard
	IMAGE=$(SERVE_TAG) MODEL="$(MODEL)" $(SERVE_DIR)/verify.sh

serve-push: serve-config-guard
	docker push $(SERVE_TAG)

bench: setup
	@echo "Running benchmarks..."
	./venv/bin/emmy bench recipes/*

bench-force: setup
	@echo "Running benchmarks (force mode)..."
	./venv/bin/emmy bench recipes/* --force

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
