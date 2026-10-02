.PHONY: lock download build install-wheel-only install-everything test style clean

ARCH := $(shell uname -m)
BUFFER_DATE := $(shell date -u -d '3 days ago' '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -v-3d '+%Y-%m-%dT%H:%M:%SZ')

lock:
	uv lock
	uv pip compile pyproject.toml --all-extras --no-annotate --exclude-newer $(BUFFER_DATE) -o requirements_$(ARCH).txt

build:
	python -m build -v .

install-wheel:
	pip install --upgrade ./dist/zig_micrograd-0.1.0-cp314-cp314-linux_x86_64.whl

install:
	uv sync

test:
	pytest -v tests/

style:
	ruff format zig_micrograd/
	ruff check --fix zig_micrograd/
	ruff format tests/
	ruff check --fix tests/

clean:
	rm -r ./build/ || true
	rm -r ./dist/ || true
	rm -r ./.zig-cache/ || true
	rm -r ./zig-out/ || true
