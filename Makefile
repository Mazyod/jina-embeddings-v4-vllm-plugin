.PHONY: install test deploy reference offline parity report
install:
	uv sync
test:
	uv run pytest -q
deploy:
	uv run modal deploy src/jinav4_vllm/modal_app/app.py
reference:
	uv run modal run src/jinav4_vllm/modal_app/reference.py
offline:
	uv run modal run src/jinav4_vllm/modal_app/offline.py
parity:
	uv run python -m jinav4_vllm.eval.report
