# Deploying Jina v4 multi-vector on the vLLM OpenAI server (Variant C)

This is the operator runbook for serving **Jina Embeddings v4 multi-vector** (128-dim/token,
ColBERT-style) **multimodal** (text + image) embeddings from a **stock vLLM OpenAI server**. The
server's `/pooling` endpoint returns final L2-normalized `[n,128]` per-token multivectors, matching
canonical Jina output (per-token cosine ≈ 0.999 text, ≈ 0.992–0.997 image; bf16 floor).

It works by adding a small **out-of-tree model plugin** (`JinaV4MultiVector`) to vLLM. The plugin
follows vLLM's in-tree ColQwen3/ColPali pattern: Qwen2.5-VL backbone + a ColBERT projection applied
in-engine. Pin the vLLM version — the model class touches vLLM internals; re-validate on upgrades.

---

## 0. Components

| Piece | Path | Why |
|---|---|---|
| Plugin package | repo root (`src/jina_v4_vllm_plugin/`, published as `jina-v4-vllm-plugin`) | registers `JinaV4MultiVector` (entry point) + model class |
| Image chat template | shipped in the `jina-v4-vllm-plugin` package (`chat_template_path()`) | makes multimodal `/pooling` emit Jina's exact image prompt |
| Projector weights | `artifacts/projector/retrieval.npz` (`W[128,2048]`,`b[128]`) | the 128-dim head — NOT in the vLLM checkpoint |
| Dockerfile | `deploy/Dockerfile` | extends official `vllm/vllm-openai` with the plugin |
| Bake script | `deploy/bake_checkpoint.py` | builds the fully drop-in checkpoint (Mode B) |

To (re)generate `retrieval.npz` from scratch: extract the retrieval-effective `multi_vector_projector`
(base + retrieval-LoRA, merged) from `jinaai/jina-embeddings-v4` — see `src/jinav4_vllm/projector/`.

---

## Two deployment modes

- **Mode A — original checkpoint + flags.** Serve `jinaai/jina-embeddings-v4-vllm-retrieval` and pass
  `--hf-overrides`, `--chat-template`, and the projector via env/file. Simplest to set up; nothing to
  rebuild if Jina ships a new checkpoint.
- **Mode B — baked checkpoint (recommended for production).** Bake the projector + `architectures` +
  chat template into a self-contained checkpoint, then `vllm serve <repo>` needs **no extra flags**.
  Cleanest drop-in; the artifact fully describes itself.

---

## 1. Build the image (both modes)

```bash
# from the repo root (build context = repo root):  docker build -f research/deploy/Dockerfile -t <img> .
docker build -f deploy/Dockerfile --build-arg VLLM_TAG=v0.22.0 -t jina-v4-mv-vllm:0.22.0 .
```

For **Mode A**, also bake the projector + chat template into the image: uncomment the two
`COPY/ENV` lines for `retrieval.npz` in `deploy/Dockerfile` before building (or mount them at runtime).

---

## 2A. Run — Mode A (original checkpoint)

```bash
docker run --gpus all -p 8000:8000 \
  -v $PWD/artifacts/projector/retrieval.npz:/opt/retrieval.npz \
  -e JINA_MV_PROJECTOR=/opt/retrieval.npz \
  jina-v4-mv-vllm:0.22.0 \
  jinaai/jina-embeddings-v4-vllm-retrieval \
  --runner pooling --pooler-config.task token_embed \
  --hf-overrides '{"architectures":["JinaV4MultiVector"]}' \
  --chat-template /opt/jina_image_chat_template.jinja \
  --served-model-name jina-v4 --max-model-len 4096
```

## 2B. Bake + Run — Mode B (drop-in checkpoint)

```bash
# Bake once (needs: huggingface_hub safetensors torch numpy). Optionally --push to your HF org.
python deploy/bake_checkpoint.py \
  --src jinaai/jina-embeddings-v4-vllm-retrieval \
  --npz artifacts/projector/retrieval.npz \
  --out ./jina-v4-mv-baked \
  # --push your-org/jina-v4-mv-vllm

# Serve it — no --hf-overrides, no --chat-template, no projector env var:
docker run --gpus all -p 8000:8000 \
  -v $PWD/jina-v4-mv-baked:/models/jina-v4-mv-baked \
  jina-v4-mv-vllm:0.22.0 \
  /models/jina-v4-mv-baked \
  --runner pooling --pooler-config.task token_embed \
  --served-model-name jina-v4 --max-model-len 4096
# (or pass the HF repo id from --push instead of the local path)
```

The server is ready when `GET /health` returns 200. Per-token output is on **`/pooling`**
(`/v1/embeddings` only returns one pooled vector per input).

---

## 3. Client usage

### Text (query / passage) — use `input`
Prefix queries with `Query: ` and passages with `Passage: ` (Jina's retrieval convention).

```bash
curl -s http://localhost:8000/pooling -H 'Content-Type: application/json' -d '{
  "model": "jina-v4",
  "input": ["Query: overview of climate change impacts on coastal cities"]
}'
# -> {"data":[{"data": [[...128 floats...], ...n tokens...], "index":0}], ...}
```

### Image — use chat-style `messages` with `image_url`

```bash
B64=$(base64 -w0 page.png)
curl -s http://localhost:8000/pooling -H 'Content-Type: application/json' -d "{
  \"model\": \"jina-v4\",
  \"messages\": [{\"role\":\"user\",\"content\":[
    {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${B64}\"}},
    {\"type\":\"text\",\"text\":\"Describe the image.\"}]}]
}'
```

### Python: embed + late-interaction (MaxSim) scoring

```python
import base64, numpy as np, requests
BASE = "http://localhost:8000"

def embed_text(text, kind="query"):
    p = ("Query: " if kind == "query" else "Passage: ") + text
    r = requests.post(f"{BASE}/pooling", json={"model": "jina-v4", "input": [p]}, timeout=120)
    return np.array(r.json()["data"][0]["data"], dtype=np.float32)   # [n,128], L2-normalized

def embed_image(path):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Describe the image."}]}]
    r = requests.post(f"{BASE}/pooling", json={"model": "jina-v4", "messages": msg}, timeout=120)
    return np.array(r.json()["data"][0]["data"], dtype=np.float32)   # [m,128]

def maxsim(query_mv, doc_mv):           # ColBERT late interaction
    return float((query_mv @ doc_mv.T).max(axis=1).sum())

q = embed_text("climate impact on coastal cities", "query")
score = maxsim(q, embed_image("page.png"))   # text query vs image document
```

Vectors are already L2-normalized, so MaxSim uses plain dot products.

---

## 3b. Image fidelity (max resolution) — IMPORTANT

Jina v4 uses the Qwen2.5-VL **dynamic-resolution** image processor, and the checkpoint's default
**caps resolution low** — too low for dense documents. Number of image tokens ≈
`(resized_H · resized_W) / (28·28·merge)`; raise `max_pixels` for higher fidelity (more patch tokens,
more compute). `pixels` = resized H·W (multiples of `28·28 = 784`).

Reference presets (`src/jinav4_vllm/common/imaging.py`): `PRESET_MAX_STANDARD = 1_003_520`
(Qwen default), `PRESET_MAX_HIFI = 3_211_264` (high-fidelity docs).

**Set it in whichever mode you run:**
- **Mode A (flags):** add `--mm-processor-kwargs '{"min_pixels":200704,"max_pixels":3211264}'` to the
  serve command (the Modal `serve_c` function adds this automatically from
  `JINA_IMAGE_MIN_PIXELS` / `JINA_IMAGE_MAX_PIXELS`).
- **Mode B (baked, recommended):** bake it in →
  `python deploy/bake_checkpoint.py --out ./jina-v4-mv-baked --min-pixels 200704 --max-pixels 3211264`
  (writes `min_pixels`/`max_pixels` + `size` into the checkpoint's `preprocessor_config.json`). Then
  no flag is needed at serve time; an env/flag still overrides if set.
- **Offline engine:** `LLM(..., mm_processor_kwargs={"min_pixels":...,"max_pixels":...})` (the repo's
  offline harness reads the same env vars).

**Parity caveat:** changing `max_pixels` changes the image-token COUNT, so the **reference and the
served/offline side must use the same values** — otherwise the per-token alignment check fails (R2).
The repo defaults leave these UNSET (checkpoint default) so the verified parity demo stays green; to
re-verify at a raised resolution, set the same `JINA_IMAGE_MAX_PIXELS` for `reference_*` and
`offline_*` and re-run `eval.report`.

## 4. Operational notes

- **Pin vLLM.** The plugin's model class subclasses `Qwen2_5_VLForConditionalGeneration` and uses
  `pooler_for_token_embed` — internal APIs. On a vLLM upgrade, follow the revalidation checklist in
  `docs/COMPAT.md` (the `src/jinav4_vllm/modal_app/revalidate.py` jobs regenerate the needed API
  facts in one cheap run), and smoke-test that `/pooling` returns dim 128 (`make smoke URL=…`).
- **Parity is bf16, not bit-exact.** Canonical Jina `encode_*` is itself bf16 and vLLM runs bf16 with
  different kernels → ~0.999 per-token cosine (text) / ~0.992–0.997 (image). This is expected; details
  in `docs/VALIDATION.md`.
- **Without the plugin, the checkpoint won't load** (`architectures: JinaV4MultiVector` is unknown).
  That's intended — the model requires the plugin. Keep them versioned together.
- **token_ids:** the `/pooling` HTTP response returns only the `[n,dim]` matrix, not token ids. If you
  need them (e.g., to drop special tokens before MaxSim), the offline engine exposes
  `output.prompt_token_ids`. In practice MaxSim over all tokens is the standard ColBERT scoring.
- **Cold start** is ~40–70s (model load + torch.compile + CUDA-graph capture). Persist a vLLM compile
  cache volume and/or use GPU memory snapshots to reduce it.

## 5. Smoke test

```bash
# dim must be 128
curl -s http://localhost:8000/pooling -H 'Content-Type: application/json' \
  -d '{"model":"jina-v4","input":["Query: hello"]}' \
  | python3 -c "import sys,json,numpy as np; d=json.load(sys.stdin); print('dim', np.array(d['data'][0]['data']).shape[1])"
```
