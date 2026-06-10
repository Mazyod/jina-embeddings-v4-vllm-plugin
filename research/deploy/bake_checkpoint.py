#!/usr/bin/env python3
"""Bake a fully self-contained, drop-in checkpoint for Variant C (runs anywhere, no Modal).

Produces: the vLLM Jina-v4 checkpoint + the multi_vector_projector tensors + `architectures`
override + the Jina image chat template. Then a stock vLLM OpenAI server (with the
jina-v4-vllm-plugin installed) serves it with NO --hf-overrides, NO --chat-template, NO env var:

    vllm serve <out_dir> --runner pooling --pooler-config.task token_embed

Usage:
    python deploy/bake_checkpoint.py \
        --src jinaai/jina-embeddings-v4-vllm-retrieval \
        --npz artifacts/projector/retrieval.npz \
        --out ./jina-v4-mv-baked \
        [--push your-org/jina-v4-mv-vllm]   # optional: upload to the HF Hub

Requires: pip install huggingface_hub safetensors torch numpy
"""
from __future__ import annotations
import argparse, json, os, shutil

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

TEMPLATE = (
    "{%- for message in messages -%}\n"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}\n"
    "{%- if message['content'] is string -%}\n"
    "{{- message['content'] -}}\n"
    "{%- else -%}\n"
    "{%- for item in message['content'] -%}\n"
    "{%- if item['type'] == 'image' or item['type'] == 'image_url' or 'image' in item -%}"
    "{{- '<|vision_start|><|image_pad|><|vision_end|>' -}}\n"
    "{%- elif item['type'] == 'text' -%}{{- item['text'] -}}{%- endif -%}\n"
    "{%- endfor -%}\n"
    "{%- endif -%}\n"
    "{{- '<|im_end|>\\n' -}}\n"
    "{%- endfor -%}\n"
)


def bake(src: str, npz: str, out_dir: str, min_pixels: int = 0, max_pixels: int = 0) -> dict:
    repo = src if os.path.isdir(src) else snapshot_download(src)
    os.makedirs(out_dir, exist_ok=True)

    for fn in os.listdir(repo):
        s = os.path.join(repo, fn)
        if os.path.isfile(s):
            shutil.copy2(os.path.realpath(s), os.path.join(out_dir, fn))

    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
        weight_map, metadata = index["weight_map"], index.get("metadata", {})
    else:
        weight_map, metadata = {}, {}
        with safe_open(os.path.join(out_dir, "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                weight_map[k] = "model.safetensors"

    proj = np.load(npz)
    tensors = {
        "multi_vector_projector.weight": torch.from_numpy(proj["W"]).to(torch.float32),
        "multi_vector_projector.bias": torch.from_numpy(proj["b"]).to(torch.float32),
    }
    save_file(tensors, os.path.join(out_dir, "model-projector.safetensors"))
    for k in tensors:
        weight_map[k] = "model-projector.safetensors"
    metadata["total_size"] = int(metadata.get("total_size", 0)) + sum(
        t.numel() * 4 for t in tensors.values())
    json.dump({"metadata": metadata, "weight_map": weight_map}, open(index_path, "w"), indent=2)

    cfg_path = os.path.join(out_dir, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["architectures"] = ["JinaV4MultiVector"]
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    # image fidelity: bake min/max pixels into the image processor config (drop-in)
    if min_pixels or max_pixels:
        pp_path = os.path.join(out_dir, "preprocessor_config.json")
        pp = json.load(open(pp_path)) if os.path.exists(pp_path) else {}
        size = pp.get("size", {}) if isinstance(pp.get("size"), dict) else {}
        if min_pixels:
            pp["min_pixels"] = int(min_pixels); size["shortest_edge"] = int(min_pixels)
        if max_pixels:
            pp["max_pixels"] = int(max_pixels); size["longest_edge"] = int(max_pixels)
        if size:
            pp["size"] = size
        json.dump(pp, open(pp_path, "w"), indent=2)

    open(os.path.join(out_dir, "chat_template.jinja"), "w").write(TEMPLATE)
    # The Qwen2.5-VL processor template (chat_template.json) wins for multimodal; overwrite it too.
    json.dump({"chat_template": TEMPLATE}, open(os.path.join(out_dir, "chat_template.json"), "w"), indent=2)
    tok_path = os.path.join(out_dir, "tokenizer_config.json")
    if os.path.exists(tok_path):
        tok = json.load(open(tok_path))
        tok["chat_template"] = TEMPLATE
        json.dump(tok, open(tok_path, "w"), indent=2)

    return {"out_dir": out_dir, "architectures": cfg["architectures"],
            "files": sorted(os.listdir(out_dir))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="jinaai/jina-embeddings-v4-vllm-retrieval")
    ap.add_argument("--npz", default="artifacts/projector/retrieval.npz")
    ap.add_argument("--out", default="./jina-v4-mv-baked")
    ap.add_argument("--min-pixels", type=int, default=0,
                    help="image processor min_pixels (0 = keep checkpoint default)")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="image processor max_pixels — raise for higher image fidelity")
    ap.add_argument("--push", default=None, help="optional HF repo id to upload the baked checkpoint")
    args = ap.parse_args()

    meta = bake(args.src, args.npz, args.out, args.min_pixels, args.max_pixels)
    print(json.dumps(meta, indent=2))

    if args.push:
        from huggingface_hub import HfApi
        HfApi().create_repo(args.push, exist_ok=True, private=True)
        HfApi().upload_folder(folder_path=args.out, repo_id=args.push)
        print(f"pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
