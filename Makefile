.PHONY: help install test package extract bake reference offline serve collect smoke parity revalidate e2e

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## sync the dev environment (uv)
	uv sync

test:  ## run the local pure-logic test suite (no GPU/vLLM)
	uv run pytest -q

package:  ## build the installable plugin wheel -> src/jinav4_vllm/vllm_plugin/dist
	cd src/jinav4_vllm/vllm_plugin && uv build

# --- GPU validation loop on Modal (requires `modal` configured + a huggingface-secret) ---

extract:  ## extract the retrieval projector -> artifacts/projector/retrieval.npz
	uv run modal run src/jinav4_vllm/modal_app/app.py::extract_projector

bake:  ## bake the drop-in checkpoint (override MIN_PIXELS=.. MAX_PIXELS=.. for image fidelity)
	uv run modal run "src/jinav4_vllm/modal_app/app.py::bake_checkpoint" --min-pixels $(or $(MIN_PIXELS),0) --max-pixels $(or $(MAX_PIXELS),0)

reference:  ## HF ground-truth embeddings -> artifacts/reference
	uv run modal run src/jinav4_vllm/modal_app/reference.py::reference_text
	uv run modal run src/jinav4_vllm/modal_app/reference.py::reference_image

offline:  ## vLLM in-process embeddings -> artifacts/offline
	uv run modal run src/jinav4_vllm/modal_app/offline.py::offline_text
	uv run modal run src/jinav4_vllm/modal_app/offline.py::offline_image

serve:  ## deploy the native vLLM OpenAI server with the plugin (serve_c)
	uv run modal deploy src/jinav4_vllm/modal_app/serve_c.py

collect:  ## capture served embeddings for parity:  make collect URL=https://...
	uv run python -m jinav4_vllm.modal_app.collect $(URL)

smoke:  ## /pooling contract smoke (dim 128, L2-normalized):  make smoke URL=https://...
	uv run python -m jinav4_vllm.modal_app.collect $(URL) --smoke

parity:  ## element-wise parity table: reference vs offline vs served -> reports/parity.md
	uv run python -m jinav4_vllm.eval.report

revalidate:  ## regenerate vLLM API facts after an upgrade (see docs/COMPAT.md)
	uv run modal run src/jinav4_vllm/modal_app/revalidate.py::recon_qwen25_api

e2e:  ## full GPU validation: extract -> reference -> offline -> parity
	$(MAKE) extract && $(MAKE) reference && $(MAKE) offline && $(MAKE) parity
