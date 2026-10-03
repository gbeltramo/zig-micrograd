.PHONY: lock install-deps build-wheel install-wheel-only install-everything test style clean zig-run zig-test

ARCH := $(shell uname -m)
BUFFER_DATE := $(shell date -u -d '3 days ago' '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -v-3d '+%Y-%m-%dT%H:%M:%SZ')

all: lock install-deps build-wheel install-wheel-only test

lock:
	uv lock
	uv pip compile pyproject.toml --all-extras --no-annotate --exclude-newer $(BUFFER_DATE) -o requirements_$(ARCH).txt

install-deps:
	uv sync --no-install-project

build-wheel:
	python -m build --verbose --wheel --no-isolation .

install-wheel-only:
	pip install --no-deps --force-reinstall ./dist/zig_micrograd-*.whl

test:
	pytest -v tests/

style:
	ruff format zig_micrograd/
	ruff check --fix zig_micrograd/
	ruff format tests/
	ruff check --fix tests/
	ruff format scripts/
	ruff check --fix scripts/

clean:
	rm -r ./build/ || true
	rm -r ./dist/ || true
	rm -r ./.zig-cache/ || true
	rm -r ./zig-out/ || true
	rm zig_micrograd/_core*.so || true
	find tests -name "__pycache__" -type d | xargs rm -r || true
	find zig_micrograd -name "__pycache__" -type d | xargs rm -r || true

# ----------
# Other commands

# Train a MLP on MNIST digits
train-MNIST:
	python scripts/train_mnist.py

# Run and test the zig code on its own
zig-run:
	python -m ziglang build run --summary all

zig-test:
	python -m ziglang build test --summary all
