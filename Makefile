.PHONY: help install test build
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## sync the dev environment (uv)
	uv sync

test:  ## run the packaging contract tests (no GPU/vLLM)
	uv run pytest -q

build:  ## build the sdist + wheel into dist/
	uv build
